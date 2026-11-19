import numpy as np
from math import gcd
from scipy.ndimage import median_filter
import scipy.signal as sig
from skimage.filters import threshold_otsu

import ruptures as rpt

def midi_array_conversion(x : np.ndarray):
    return 69 + 12*np.log2((x+1e-6)/440)

def ReLU(x : np.ndarray):
    return x * (x > 0)

SILENCE_DETECTION_SMOOTHING_WINDOWS = 10 # ms
SILENCE_DETECTION_REFERENCE_PERCENTILE = 99 # between 0 and 100
SILENCE_DETECTION_THRESHOLD_FACTOR = 0.1 # fraction of the threshold, 0.1 for 10% is appropriate

def removeSilences(x : np.ndarray, fs : int) -> np.ndarray:
    x_res = x.copy()
    window_size = int(SILENCE_DETECTION_SMOOTHING_WINDOWS * fs / 1000) # ms to s conversion
    envelope = np.convolve(a=np.abs(x.astype(float)), v=np.ones(window_size)/window_size, mode='same')

    # Estimate the reference amplitude and define the silence threshold
    reference_amplitude = np.percentile(envelope, SILENCE_DETECTION_REFERENCE_PERCENTILE)
    silence_threshold = SILENCE_DETECTION_THRESHOLD_FACTOR * reference_amplitude

    # Identify non-silent samples
    silence_mask = envelope < silence_threshold

    # Remove silent samples for testing
    x_res[silence_mask] = 0
    return x_res

CLEAN_MAGNITUDE_RATIO_THRESHOLD = 50
CLEAN_MAGNITUDE_FREQ_RANGE = [300,4000]

def cleanMagnitude(
    magnitude: np.ndarray,
    frequencies: np.ndarray,
    freq_range: tuple[float, float] = CLEAN_MAGNITUDE_FREQ_RANGE,
    ratio_threshold: float = 10.0,
    min_peak: float = 1e-3,
    eps: float = 1e-12
) -> np.ndarray:
    """ Remove STFT frames that do not exhibit sufficient spectral concentration. """
    magnitude_clean = magnitude.copy()

    # Select frequency range
    if freq_range is None:
        freq_mask = np.ones(len(frequencies), dtype=bool)
    else:
        f_min, f_max = freq_range
        freq_mask = (frequencies >= f_min) & (frequencies <= f_max)

    if not np.any(freq_mask):
        raise ValueError("The selected frequency range contains no STFT bins.")

    selected_magnitude = magnitude[freq_mask, :]

    # Spectral peak and median for each time frame
    spectral_max = np.max(selected_magnitude, axis=0)
    spectral_median = np.median(selected_magnitude, axis=0)

    # Spectral concentration
    spectral_ratio = spectral_max / (spectral_median + eps)

    # Keep only frames with sufficient energy and spectral concentration
    active_frames = (
        (spectral_max > min_peak) &
        (spectral_ratio > ratio_threshold)
    )

    magnitude_clean[:, ~active_frames] = 0

    return magnitude_clean

# Resampling
TARGET_RATE = 16000  # Target sampling rate in Hz, 8 kHz is enough because no one will sing higher than 8kHz

# STFT
STFT_NPERSEG = 512  # Number of samples per STFT segment, one segment is then 512/8000 = 64ms
STFT_NOVERLAP = 256  # If None default overlap is 50%
STFT_NFTT = 1024 # we can artificially increase the frequential resolution by k if nfft = k * nperseg

# Change point detection (PELT)
PELT_MODEL = "l1"  # Cost function
PELT_MIN_SIZE = 3  # Minimum segment length in frames, one segment is 512/8000 = 0.064 = 64ms
PELT_JUMP = 1  # Breakpoint search step, 1 achieves best accuracy
PELT_PENALTY = 10  # Change point penalty

MAX_SILENCE_DURATION = 1  # Maximum tolerated silence: 500 ms

MIDI_TONE_SILENCE_THRESHOLD = 60 # Every note below this threshold is going to be removed in the midi tone sequence before applying diff

def estimate_melody_diff(
    signal: np.ndarray,
    sample_rate: int,
    target_rate: int = TARGET_RATE,
    nperseg: int = STFT_NPERSEG,
    noverlap: int | None = STFT_NOVERLAP,
    nfft:int = STFT_NFTT,
    pelt_model: str = PELT_MODEL,
    pelt_min_size: int = PELT_MIN_SIZE,
    pelt_jump: int = PELT_JUMP,
    pelt_penalty: float = PELT_PENALTY,
    return_debug: bool = False
) -> np.ndarray:
    """Estimate the melody's interval sequence from an audio signal."""
    # [1] NORMALISE THE SIGNAL
    signal = signal/np.max(np.abs(signal))

    # [2] DOWNSAMPLE THE SIGNAL, to avoid high noise and to reduce computation cost
    if int(target_rate) != int(sample_rate):
        g = gcd(sample_rate, target_rate)
        signal = sig.resample_poly(
            signal,
            target_rate // g,
            sample_rate // g
        )
    
    # [3] DETECT AND DELETE BACKGROUND NOISE
    signal = removeSilences(signal,target_rate)

    # [4] COMPUTE THE STFT
    f, t, Zxx = sig.stft(
        signal,
        fs=target_rate,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft
    )
    magnitude = np.abs(Zxx)
    
    # [5] Clean the magnitude using a spectral flatness estimator to remove residual noise, 
    magnitude_clean = cleanMagnitude(
        magnitude,
        f,
        freq_range=(300, 3000),
        ratio_threshold=10.0,
        min_peak=1e-3
    )
    
    # [6] ESTIMATE PITCH, MIDI CONVERSION, SMOOTHING, using median filtering
    dominant_freq = f[np.argmax(magnitude_clean, axis=0)]
    dominant_freq_midi = ReLU(midi_array_conversion(dominant_freq))
    
    raw_midi = dominant_freq_midi.copy()
    dominant_freq_midi = median_filter(raw_midi, size=5)

    # [7] TONE SEQUENCE ESTIMATION 
    # [7.1] FIND BREAKPOINTS
    algo = rpt.Pelt(
        model=pelt_model,
        min_size=pelt_min_size,
        jump=pelt_jump
    )
    algo.fit(dominant_freq_midi.reshape(-1, 1))
    breakpoints = algo.predict(pen=pelt_penalty)
    
    if return_debug:
        segment_midi = np.zeros_like(dominant_freq_midi) # segment midi contains silences

        for start, end in zip([0] + breakpoints[:-1], breakpoints):
            segment_midi[start:end] = np.mean(dominant_freq_midi[start:end])

    # # [7.2] COMPUTE EACH TONE BETWEEN TWO BREAKPOINTS
    notes = np.array([
    np.mean(dominant_freq_midi[start:end])
        if np.mean(dominant_freq_midi[start:end]) > MIDI_TONE_SILENCE_THRESHOLD
        else 1000 # arbitrarly high value so a melody can not contain a very long silence, but not np.inf because the melody detection raise an error if there is infinite value
        if (end - start) * (nperseg - noverlap) / target_rate > MAX_SILENCE_DURATION
        else 0
        for start, end in zip([0] + breakpoints[:-1], breakpoints)
    ])
    
    
    # [8] COMPUTE MIDI TONE DIFF, diff ensure the song is recognized even if there is a constant shift by any number of tones
    notes = notes[(notes>50)] # we remove silences and very low tones, but we keep long silences, labeled as 1000
    midi_diff = np.diff(notes)

    if return_debug:
        debug = {
            "signal": signal,
            "sample_rate": target_rate,
            "frequencies": f,
            "times": t,
            "stft": Zxx,
            "magnitude": np.abs(Zxx),
            "raw_midi": raw_midi,       # Before median filtering
            "filtered_midi": dominant_freq_midi,
            "breakpoints": breakpoints,
            "notes": notes,
            "midi_diff": midi_diff,
            "segment_midi": segment_midi,
        }
        return midi_diff, debug

    return midi_diff
