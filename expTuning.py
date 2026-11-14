
import numpy as np
import scipy.io.wavfile as wavfile
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tkinter import Tk, filedialog

from utils.toneSequenceEstimation import estimate_melody_diff

# ================= SELECT FILE =================

root = Tk()
root.withdraw()

file = filedialog.askopenfilename(
    title="Select a song",
    filetypes=[("WAV files", "*.wav")]
)

root.destroy()

if not file:
    raise RuntimeError("No file selected")

# ================= RUN ESTIMATOR =================

songRate, songData = wavfile.read(file)

if songData.ndim == 2:
    songData = songData[:, 0]

_, debug = estimate_melody_diff(
    signal=songData.astype(float),
    sample_rate=songRate,
    return_debug=True
)

# ================= EXTRACT DATA =================

signal = debug["signal"]
rate = debug["sample_rate"]

f = debug["frequencies"]
t = debug["times"]

magnitude = debug["magnitude"]

raw_midi = debug["raw_midi"]
filtered_midi = debug["filtered_midi"]
segment_midi = debug["segment_midi"]

breakpoints = debug["breakpoints"]

# ================= VISUALIZATION =================

fig = make_subplots(
    rows=3,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.06,
    subplot_titles=[
        "STFT",
        "Audio signal",
        "MIDI pitch estimation"
    ]
)

# STFT
fig.add_trace(
    go.Heatmap(
        x=t,
        y=f,
        z=20 * np.log10(magnitude + 1e-10),
        colorscale="Viridis",
        colorbar=dict(title="dB"),
        name="STFT"
    ),
    row=1, col=1
)

# Signal
signal_t = np.arange(len(signal)) / rate

fig.add_trace(
    go.Scattergl(
        x=signal_t,
        y=signal,
        mode="lines",
        name="Signal"
    ),
    row=2, col=1
)

# Raw pitch
fig.add_trace(
    go.Scatter(
        x=t,
        y=raw_midi,
        mode="lines",
        name="Raw MIDI",
        line=dict(width=1, color="lightgray")
    ),
    row=3, col=1
)

# Filtered pitch
fig.add_trace(
    go.Scatter(
        x=t,
        y=filtered_midi,
        mode="lines",
        name="Filtered MIDI",
        line=dict(width=1.5, color="blue")
    ),
    row=3, col=1
)

# PELT segment estimates
fig.add_trace(
    go.Scatter(
        x=t,
        y=segment_midi,
        mode="lines",
        name="PELT segments",
        line=dict(width=3, color="orange")
    ),
    row=3, col=1
)

# Breakpoints
for bp in breakpoints[:-1]:
    for row in [1, 2, 3]:
        fig.add_vline(
            x=t[bp],
            line_dash="dash",
            line_color="red",
            line_width=1,
            row=row,
            col=1
        )

fig.update_yaxes(title_text="Frequency (Hz)", row=1, col=1)
fig.update_yaxes(title_text="Amplitude", row=2, col=1)
fig.update_yaxes(title_text="MIDI", row=3, col=1)
fig.update_xaxes(title_text="Time (s)", row=3, col=1)

fig.update_layout(
    title=file.split("/")[-1],
    height=1000,
    showlegend=True
)

fig.show()

# ================= RESULTS =================

print("\nDetected notes:", debug["notes"])
print("MIDI differences:", debug["midi_diff"])
print("Breakpoints:", breakpoints)
