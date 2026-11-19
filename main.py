import numpy as np

from typing import Any, List
import time

import sounddevice as sd
from sounddevice import CallbackFlags

import utils.songDictionaries as songDictionaries
from utils.toneSequenceEstimation import estimate_melody_diff #(signal: np.ndarray, sample_rate: int, target_rate: int = TARGET_RATE, nperseg: int = STFT_NPERSEG, noverlap: int | None = STFT_NOVERLAP, pelt_model: str = PELT_MODEL, pelt_min_size: int = PELT_MIN_SIZE, pelt_jump: int = PELT_JUMP, pelt_penalty: float = PELT_PENALTY)
from utils.sequenceMatchMetric import identify_melody #(measured, target_dic, detection_threshold, loss="mae", huber_delta=1.0):
from utils.keyboardInterraction import read_confirmsong_key, init_button_to_key_dict, waitForKey, simulateSong, init_song_to_key_dict

# Initializing the key for user interraction with the script
confirm_song_key = read_confirmsong_key()
button_to_key_dict = init_button_to_key_dict()
if confirm_song_key is None or button_to_key_dict is None:
    raise ValueError('The JSON is not correctly configured it must look like : {"A": "key1", "Up": "key2", "Right": "key3", "Down": "key4", "Left": "key5", "ConfirmSong": "key6"}')
song_to_key_dict = init_song_to_key_dict(songDictionaries.song_dict_button, button_to_key_dict)

CONST_DETECTION_THRESHOLD = 1.5

SAMPLE_RATE = 16000 # Sampling rate
CHUNK_DURATION = 2 # Duration of one chunk in seconds, the program is an infinite loop, consisting of recording a chunk, put it in a finite sized queue, and restart
CHUNK_DURATION_MS = int(CHUNK_DURATION*1000)
AUDIO_BUFFER_SIZE = 3

DEBUG = True

FRENCH = True
#########

import matplotlib.pyplot as plt

if DEBUG :
    plt.ion()

    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    line_original, = axes[0].plot([], [])
    line_clean, = axes[1].plot([], [])
    line_pitch, = axes[2].plot([], [])
    line_filtered, = axes[2].plot([], [])

    axes[0].set_title("Original signal")
    axes[1].set_title("Signal after silence removal")
    axes[2].set_title("Estimated pitch")

    plt.tight_layout() 
    plt.show(block=False)

def update_plot(signal_original, sample_rate, debug):

    t_original = np.arange(len(signal_original)) / sample_rate
    line_original.set_data(t_original, signal_original)
    axes[0].relim()
    axes[0].autoscale_view()

    signal_clean = debug["signal"]
    t_clean = np.arange(len(signal_clean)) / debug["sample_rate"]
    line_clean.set_data(t_clean, signal_clean)
    axes[1].relim()
    axes[1].autoscale_view()

    times = debug["times"]
    line_pitch.set_data(times, debug["raw_midi"])
    line_filtered.set_data(times, debug["filtered_midi"])

    axes[2].relim()
    axes[2].autoscale_view()

    # Refresh breakpoint markers
    for artist in axes[2].lines[2:]:
        artist.remove()

    for bp in debug["breakpoints"][:-1]:
        if bp < len(times):
            axes[2].axvline(times[bp], color="red", linestyle="--", alpha=0.7)

    fig.canvas.draw_idle()
    fig.canvas.flush_events()
#########

def append_finite_size(
    queue : List,
    element : Any, # python can store Any type in a List
    size_queue : int
    ) -> None:
    """This function appends an element {element} into a list {queue} as if it were a finite size queue of size {size_queue}"""
    if len(queue)>=size_queue:
        _ = queue.pop(0)
    queue.append(element)

audio_buffer_queue = [] # use append or pop(0) to use list as a queue


chunk_size = int(SAMPLE_RATE * CHUNK_DURATION)

def callbackCustom(
    indata: np.ndarray,
    frames: int,
    time : Any, 
    status: CallbackFlags) -> None: # time is from time Any because from sounddevice import CData gives this error : [ImportError: cannot import name 'CData' from 'sounddevice']
    """The classical callback function does not offer the possibilty to get access to indata, this method export indata into a list using a global pointer"""
    append_finite_size(queue = audio_buffer_queue, element = indata.squeeze().copy(), size_queue=AUDIO_BUFFER_SIZE) # we use the global list pointer to export indata out of the range of the callback function 

song_dict_note_midi_diff_local = songDictionaries.song_dict_note_midi_diff

if FRENCH:
    print("Enregistrement en cours... Ctrl+C pour arrêter.")
else :
    print("Start recording... Press Ctrl+C to stop")

try:
    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=chunk_size,
        callback=callbackCustom
    ):
        while True:
            computation_start = time.time()
            if audio_buffer_queue:
                print("New 2s chunk")
                # estimated_pitch_midi_diff = estimate_melody_diff(signal=np.concatenate(audio_buffer_queue), sample_rate=SAMPLE_RATE)
                if DEBUG:
                    estimated_pitch_midi_diff, debug = estimate_melody_diff(
                        signal=np.concatenate(audio_buffer_queue),
                        sample_rate=SAMPLE_RATE,
                        return_debug=True
                    )
                    #print(estimated_pitch_midi_diff)
                else:
                    estimated_pitch_midi_diff = estimate_melody_diff(
                        signal=np.concatenate(audio_buffer_queue),
                        sample_rate=SAMPLE_RATE
                    )
                identified_melody_name, identified_melody_score = identify_melody(estimated_pitch_midi_diff, song_dict_note_midi_diff_local, CONST_DETECTION_THRESHOLD)
                
                if identified_melody_name is not None : # A song has been recognized
                    print(f"The melody {identified_melody_name} has been recognized with an error of {identified_melody_score}")
                    # A song is recognized we launch the song confirmation routine :
                    print(f"Waiting for confirmation : you have 3 seconds to press {confirm_song_key}")
                    confirmation_key_pressed = waitForKey(confirm_song_key, 3000) 
                    if confirmation_key_pressed: # we simulate the song
                        try:
                            simulateSong(identified_melody_name, song_to_key_dict)
                        except:
                            print(f"The song {identified_melody_name} can not be played")
                    else :
                        print("Time limit reached, song playing will be cancelled")
                    # If a song is recognized we empty the buffer t ensure we don not detect several times the same melody
                    audio_buffer_queue = []
                        
                    
                if DEBUG and audio_buffer_queue:
                    print(int(1000*np.max(audio_buffer_queue[0]))/1000)
                    update_plot(np.concatenate(audio_buffer_queue), SAMPLE_RATE, debug)
                    
            computation_stop = time.time()
            
            computation_time_ms = int(1000*(computation_stop - computation_start))
            
            if computation_time_ms > CHUNK_DURATION_MS:
                print(f"No sounddevice.sleep because the chunk duration is {CHUNK_DURATION_MS}ms and the computation took {computation_time_ms}ms, WARNING we lost {computation_time_ms-CHUNK_DURATION_MS} RECOMMENDATION increase chunk duration or reduce algorithm computational costs")
            else:
                sd.sleep(CHUNK_DURATION_MS-computation_time_ms)
            
except KeyboardInterrupt:
    print("\nEnregistrement arrêté.")
    if FRENCH:
        print("\nEnregistrement arrêté. Fin du processus.")
    else :
        print("Stop recording. Process killed")

