# Maths and Vectorization
import numpy as np

# Audio
from scipy.io import wavfile

import songDictionaries
from utils.toneSequenceEstimation import estimate_melody_diff #(signal: np.ndarray, sample_rate: int, target_rate: int = TARGET_RATE, nperseg: int = STFT_NPERSEG, noverlap: int | None = STFT_NOVERLAP, pelt_model: str = PELT_MODEL, pelt_min_size: int = PELT_MIN_SIZE, pelt_jump: int = PELT_JUMP, pelt_penalty: float = PELT_PENALTY)
from utils.sequenceMatchMetric import match_sequence #(measured, target, loss="mae", huber_delta=1.0)

CONST_DETECTION_THRESHOLD = 2

song_dict_note_midi_diff_local = songDictionaries.song_dict_note_midi_diff

def test_song(file: str,
              song_midi_diff_dictionnary,
              dic_entry_target: str):

    songRate, songData = wavfile.read(file)
    songData = 1.0 * songData[:, 0].squeeze()

    array_diff_est = estimate_melody_diff(signal=songData, sample_rate=songRate)

    errors = []
    for key, value in song_midi_diff_dictionnary.items():
        currentError = match_sequence(
            array_diff_est, value
        )
        errors.append((key, currentError))

    minErrorSong, minErrorValue = min(errors, key=lambda x: x[1])

    print("\n" + "\n".join(
        f"{'✓' if name == dic_entry_target else ' '} {name:<22} {error:.4f}"
        for name, error in sorted(errors, key=lambda x: x[1])
    ))

    print(f"\n {'✅' if minErrorSong == dic_entry_target else '❌'}Detected: {minErrorSong} | Expected: {dic_entry_target}")
    
    return (minErrorSong == dic_entry_target and minErrorValue < CONST_DETECTION_THRESHOLD)
    
BoleroOfFireTest = test_song(file='Data/AllSongsWhistled/wav/BoleroOfFire.wav',
          song_midi_diff_dictionnary=song_dict_note_midi_diff_local,
          dic_entry_target='Bolero of Fire')

EponaSongTest = test_song(file='Data/AllSongsWhistled/wav/EponaSong.wav',
          song_midi_diff_dictionnary=song_dict_note_midi_diff_local,
          dic_entry_target='Epona Song')

MinuetOfForestTest = test_song(file='Data/AllSongsWhistled/wav/MinuetOfForest.wav',
          song_midi_diff_dictionnary=song_dict_note_midi_diff_local,
          dic_entry_target='Minuet of Forest')

NocturneOfShadowTest = test_song(file='Data/AllSongsWhistled/wav/NocturneOfShadow.wav',
          song_midi_diff_dictionnary=song_dict_note_midi_diff_local,
          dic_entry_target='Nocturne of Shadow')

PreludeOfLightTest = test_song(file='Data/AllSongsWhistled/wav/PreludeOfLight.wav',
          song_midi_diff_dictionnary=song_dict_note_midi_diff_local,
          dic_entry_target='Prelude of Light')

RequiemOfSpiritTest = test_song(file='Data/AllSongsWhistled/wav/RequiemOfSpirit.wav',
          song_midi_diff_dictionnary=song_dict_note_midi_diff_local,
          dic_entry_target='Requiem of Spirit')

SariaSongTest = test_song(file='Data/AllSongsWhistled/wav/SariaSong.wav',
          song_midi_diff_dictionnary=song_dict_note_midi_diff_local,
          dic_entry_target='Saria Song')

SerenadeOfWaterTest = test_song(file='Data/AllSongsWhistled/wav/SerenadeOfWater.wav',
          song_midi_diff_dictionnary=song_dict_note_midi_diff_local,
          dic_entry_target='Serenade of Water')

SongOfStormTest = test_song(file='Data/AllSongsWhistled/wav/SongOfStorm.wav',
          song_midi_diff_dictionnary=song_dict_note_midi_diff_local,
          dic_entry_target='Song of Storms')

SongOfTimeTest = test_song(file='Data/AllSongsWhistled/wav/SongOfTime.wav',
          song_midi_diff_dictionnary=song_dict_note_midi_diff_local,
          dic_entry_target='Song of Time')

SunSongTest = test_song(file='Data/AllSongsWhistled/wav/SunSong.wav',
          song_midi_diff_dictionnary=song_dict_note_midi_diff_local,
          dic_entry_target='Sun Song')

ZeldaLullabyTest = test_song(file='Data/AllSongsWhistled/wav/ZeldaLullaby.wav',
          song_midi_diff_dictionnary=song_dict_note_midi_diff_local,
          dic_entry_target='Zelda Lullaby')

assert BoleroOfFireTest
assert EponaSongTest
assert MinuetOfForestTest
assert NocturneOfShadowTest
assert PreludeOfLightTest
assert RequiemOfSpiritTest
assert SariaSongTest
assert SerenadeOfWaterTest
assert SongOfStormTest
assert SongOfTimeTest
assert SunSongTest
assert ZeldaLullabyTest