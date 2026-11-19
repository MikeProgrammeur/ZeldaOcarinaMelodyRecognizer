
import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from scipy.signal import stft

# ============================================================
# CONFIGURATION
# ============================================================

SAMPLE_RATE = 16000
RECORD_DURATION = 5  # seconds

# Silence detection - time domain
SILENCE_DETECION_SMOOTHING_WINDOWS = 10  # ms
SILENCE_DETECTION_REFERENCE_PERCENTILE = 99
SILENCE_DETECTION_THRESHOLD_FACTOR = 0.1

# STFT
STFT_NPERSEG = 512
STFT_NOVERLAP = 256
STFT_NFTT = 1024

# Spectral detection
SPECTRAL_RATIO_THRESHOLD = 150.0
MIN_FRAME_RMS_FACTOR = 0.03


# ============================================================
# FIRST SILENCE DETECTOR: TIME DOMAIN
# ============================================================

def removeSilences(x: np.ndarray, fs: int) -> np.ndarray:
    x_res = x.copy()

    window_size = max(
        1,
        int(SILENCE_DETECION_SMOOTHING_WINDOWS * fs / 1000)
    )

    envelope = np.convolve(
        np.abs(x.astype(float)),
        np.ones(window_size) / window_size,
        mode="same"
    )

    reference_amplitude = np.percentile(
        envelope,
        SILENCE_DETECTION_REFERENCE_PERCENTILE
    )

    silence_threshold = (
        SILENCE_DETECTION_THRESHOLD_FACTOR * reference_amplitude
    )

    silence_mask = envelope < silence_threshold
    x_res[silence_mask] = 0

    return x_res


# ============================================================
# RECORDING
# ============================================================

print(f"Recording for {RECORD_DURATION} seconds...")

recording = sd.rec(
    int(RECORD_DURATION * SAMPLE_RATE),
    samplerate=SAMPLE_RATE,
    channels=1,
    dtype="float32"
)

sd.wait()

signal = recording[:, 0]

print("Recording finished.")


# ============================================================
# FIRST DETECTION
# ============================================================

cleaned_signal = removeSilences(signal, SAMPLE_RATE)


# ============================================================
# STFT
# ============================================================

frequencies, times, Zxx = stft(
    signal,
    fs=SAMPLE_RATE,
    nperseg=STFT_NPERSEG,
    noverlap=STFT_NOVERLAP,
    nfft=STFT_NFTT
)

magnitude = np.abs(Zxx)
# print(frequencies)

# Ignore DC component
magnitude_no_dc = magnitude[20:256, :] # look between 300hz and 4khz

# Spectral crest factor: max / median
spectral_max = np.max(magnitude_no_dc, axis=0)
spectral_median = np.median(magnitude_no_dc, axis=0)

spectral_ratio = spectral_max / (spectral_median + 1e-12)


# ============================================================
# SECOND DETECTION: SPECTRAL
# ============================================================

# Frame RMS estimated from STFT magnitude
frame_rms = np.sqrt(np.mean(magnitude**2, axis=0))

# Minimum energy threshold to avoid classifying silence as tonal
rms_reference = np.percentile(frame_rms, 99)
rms_threshold = MIN_FRAME_RMS_FACTOR * rms_reference

# Keep frames with sufficient energy AND a dominant spectral peak
active_frames = (
    (frame_rms > rms_threshold)
    & (spectral_ratio > SPECTRAL_RATIO_THRESHOLD)
)

# Convert frame-level mask to sample-level mask
sample_times = np.arange(len(signal)) / SAMPLE_RATE

sample_mask = np.interp(
    sample_times,
    times,
    active_frames.astype(float),
    left=0,
    right=0
) > 0.5

spectrally_cleaned_signal = signal.copy()
spectrally_cleaned_signal[~sample_mask] = 0


# ============================================================
# DISPLAY
# ============================================================

fig, axes = plt.subplots(5, 1, figsize=(15, 16))

# 1. Original signal
axes[0].plot(sample_times, signal)
axes[0].set_title("1. Original audio signal")
axes[0].set_ylabel("Amplitude")

# 2. Time-domain silence removal
axes[1].plot(sample_times, cleaned_signal)
axes[1].set_title("2. Signal after time-domain silence removal")
axes[1].set_ylabel("Amplitude")

# 3. STFT
mesh = axes[2].pcolormesh(
    times,
    frequencies,
    20 * np.log10(magnitude + 1e-10),
    shading="auto",
    cmap="magma"
)
axes[2].set_title("3. STFT magnitude (dB)")
axes[2].set_ylabel("Frequency (Hz)")
fig.colorbar(mesh, ax=axes[2], label="Magnitude (dB)")

# 4. Spectral statistic
axes[3].plot(times, spectral_ratio, label="Max / Median")
axes[3].axhline(
    SPECTRAL_RATIO_THRESHOLD,
    color="red",
    linestyle="--",
    label="Spectral threshold"
)
axes[3].set_title("4. Spectral crest factor")
axes[3].set_ylabel("Max / Median")
axes[3].set_xlabel("Time (s)")
axes[3].legend()
axes[3].set_ylim(bottom=0)

# 5. Signal after spectral detection
axes[4].plot(sample_times, spectrally_cleaned_signal)
axes[4].set_title("5. Signal after spectral silence detection")
axes[4].set_ylabel("Amplitude")
axes[4].set_xlabel("Time (s)")

for ax in axes:
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
