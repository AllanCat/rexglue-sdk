/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2020 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 *
 * @modified    Tom Clay, 2026 - Adapted for ReXGlue runtime
 */

#include <array>
#include <cstring>

#include <rex/assert.h>
#include <rex/audio/conversion.h>
#include <rex/audio/flags.h>
#include <rex/audio/sdl/sdl_audio_driver.h>
#include <rex/cvar.h>
#include <rex/dbg.h>
#include <rex/logging.h>

REXCVAR_DEFINE_BOOL(audio_mute, false, "Audio", "Mute audio output");
// Volume as an integer percentage (100 = 1.0, 250 = 2.5x, etc.).
// Default 250 compensates for the 5.1->stereo downmix normalisation factor
// (1/2.5) applied when the game outputs stereo-only front-left/front-right.
REXCVAR_DEFINE_UINT32(audio_volume_pct, 250, "Audio",
    "Master volume as integer percentage (100=normal, 250=default 2.5x to "
    "compensate for 5.1->stereo downmix when game outputs stereo content).");

namespace rex::audio::sdl {

SDLAudioDriver::SDLAudioDriver(memory::Memory* memory, rex::thread::Semaphore* semaphore)
    : AudioDriver(memory), semaphore_(semaphore) {}

SDLAudioDriver::~SDLAudioDriver() {
  assert_true(frames_queued_.empty());
  assert_true(frames_unused_.empty());
}

bool SDLAudioDriver::Initialize() {
  SDL_version ver = {};
  SDL_GetVersion(&ver);
  if ((ver.major < 2) || (ver.major == 2 && ver.minor == 0 && ver.patch < 8)) {
    REXAPU_WARN(
        "SDL library version {}.{}.{} is outdated. "
        "You may experience choppy audio.",
        ver.major, ver.minor, ver.patch);
  }

  // Prevent SDL from interfering with timer resolution (causes FPS drops)
  SDL_SetHintWithPriority(SDL_HINT_TIMER_RESOLUTION, "0", SDL_HINT_OVERRIDE);

  // Set audio category for proper OS audio handling
  SDL_SetHint(SDL_HINT_AUDIO_CATEGORY, "playback");

  // Set app name for audio device identification
#if !SDL_VERSION_ATLEAST(2, 0, 14)
#define SDL_HINT_AUDIO_DEVICE_APP_NAME "SDL_AUDIO_DEVICE_APP_NAME"
#endif
  SDL_SetHint(SDL_HINT_AUDIO_DEVICE_APP_NAME, "rexglue");

  if (SDL_InitSubSystem(SDL_INIT_AUDIO) < 0) {
    REXAPU_ERROR("SDL_InitSubSystem(SDL_INIT_AUDIO) failed: {}", SDL_GetError());
    return false;
  }
  sdl_initialized_ = true;

  SDL_AudioSpec desired_spec = {};
  SDL_AudioSpec obtained_spec;
  desired_spec.freq = frame_frequency_;
  desired_spec.format = AUDIO_F32;
  desired_spec.channels = frame_channels_;
  desired_spec.samples = channel_samples_;
  desired_spec.callback = SDLCallback;
  desired_spec.userdata = this;
  // Allow the hardware to decide between 5.1 and stereo
  int allowed_change = SDL_AUDIO_ALLOW_CHANNELS_CHANGE;
  for (int i = 0; i < 2; i++) {
    sdl_device_id_ = SDL_OpenAudioDevice(nullptr, 0, &desired_spec, &obtained_spec, allowed_change);
    if (sdl_device_id_ <= 0) {
      REXAPU_ERROR("SDL_OpenAudioDevice() failed: {}", SDL_GetError());
      return false;
    }
    if (obtained_spec.channels == 2 || obtained_spec.channels == 6) {
      break;
    }
    // If the system is 4 or 7.1, let SDL convert
    allowed_change = 0;
    SDL_CloseAudioDevice(sdl_device_id_);
    sdl_device_id_ = -1;
  }
  if (sdl_device_id_ <= 0) {
    REXAPU_ERROR("Failed to get a compatible SDL Audio Device.");
    return false;
  }
  sdl_device_channels_ = obtained_spec.channels;

  SDL_PauseAudioDevice(sdl_device_id_, 0);

  return true;
}

void SDLAudioDriver::SubmitFrame(uint32_t frame_ptr) {
  const auto input_frame = memory_->TranslateVirtual<float*>(frame_ptr);
  float* output_frame;
  {
    std::unique_lock<std::mutex> guard(frames_mutex_);
    if (frames_unused_.empty()) {
      output_frame = new float[frame_samples_];
    } else {
      output_frame = frames_unused_.top();
      frames_unused_.pop();
    }
  }

  std::memcpy(output_frame, input_frame, frame_samples_ * sizeof(float));

  {
    std::unique_lock<std::mutex> guard(frames_mutex_);
    // If a backlog has built up (e.g. after a pause/resume or scheduler spike)
    // drop the oldest frames so that latency cannot creep beyond 2× the normal
    // pre-buffer window.  The dropped frames end up on the unused stack so no
    // memory is leaked.
    constexpr size_t kMaxQueueDepth = 32;
    while (frames_queued_.size() >= kMaxQueueDepth) {
      frames_unused_.push(frames_queued_.front());
      frames_queued_.pop();
    }
    frames_queued_.push(output_frame);
  }
}

void SDLAudioDriver::Shutdown() {
  if (sdl_device_id_ > 0) {
    SDL_CloseAudioDevice(sdl_device_id_);
    sdl_device_id_ = -1;
  }
  if (sdl_initialized_) {
    SDL_QuitSubSystem(SDL_INIT_AUDIO);
    sdl_initialized_ = false;
  }
  std::unique_lock<std::mutex> guard(frames_mutex_);
  while (!frames_unused_.empty()) {
    delete[] frames_unused_.top();
    frames_unused_.pop();
  }
  while (!frames_queued_.empty()) {
    delete[] frames_queued_.front();
    frames_queued_.pop();
  }
}

void SDLAudioDriver::SDLCallback(void* userdata, Uint8* stream, int len) {
  SCOPE_profile_cpu_f("apu");
  if (!userdata || !stream) {
    REXAPU_ERROR("SDLAudioDriver::SDLCallback called with nullptr.");
    return;
  }
  const auto driver = static_cast<SDLAudioDriver*>(userdata);
  assert_true(len ==
              static_cast<int>(sizeof(float) * channel_samples_ * driver->sdl_device_channels_));

  // Pop a frame pointer under the lock but release the lock before the
  // conversion.  This minimises contention: SubmitFrame() on the audio worker
  // thread no longer stalls behind the SDL callback thread while float
  // conversion and volume math are running.
  float* buffer = nullptr;
  {
    std::unique_lock<std::mutex> guard(driver->frames_mutex_);
    if (!driver->frames_queued_.empty()) {
      buffer = driver->frames_queued_.front();
      driver->frames_queued_.pop();
    }
  }

  if (!buffer) {
    // Underrun — output silence.  Do NOT release the semaphore: we want the
    // audio worker to keep the game producing frames as fast as possible to
    // refill the queue.
    std::memset(stream, 0, len);
    return;
  }

  if (REXCVAR_GET(audio_mute)) {
    std::memset(stream, 0, len);
  } else {
    // Conversion and volume are done with the mutex released.
    switch (driver->sdl_device_channels_) {
      case 2:
        conversion::sequential_6_BE_to_interleaved_2_LE(reinterpret_cast<float*>(stream), buffer,
                                                        channel_samples_);
        break;
      case 6:
        conversion::sequential_6_BE_to_interleaved_6_LE(reinterpret_cast<float*>(stream), buffer,
                                                        channel_samples_);
        break;
      default:
        assert_unhandled_case(driver->sdl_device_channels_);
        break;
    }
    // Apply master volume.
    const float volume = REXCVAR_GET(audio_volume_pct) * 0.01f;
    if (volume != 1.0f) {
      auto* fout = reinterpret_cast<float*>(stream);
      const int sample_count = len / static_cast<int>(sizeof(float));
      for (int i = 0; i < sample_count; i++) {
        fout[i] *= volume;
      }
    }
  }

  // Return the buffer to the pool and signal the game to produce another frame.
  {
    std::unique_lock<std::mutex> guard(driver->frames_mutex_);
    driver->frames_unused_.push(buffer);
  }
  auto ret = driver->semaphore_->Release(1, nullptr);
  assert_true(ret);
}

}  // namespace rex::audio::sdl
