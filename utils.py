import os
import librosa
from scipy.io.wavfile import write
from defaults import RAW_AUDIO_PATH, SAVE_PATH

def save_audio(waveform, sample_rate, audio_name):
    write(f"{SAVE_PATH}/{audio_name}.wav", sample_rate, waveform)

def load_audio(audio_id):
    waveform, sample_rate = librosa.load(os.path.join(RAW_AUDIO_PATH, f"{audio_id}.mp3"))

    return waveform, sample_rate

def get_audio_segments(waveform, sample_rate, start, end):
    start_pos = int(start * sample_rate)
    end_pos = int(end * sample_rate)
    waveform_segment = waveform[start_pos : end_pos]

    return waveform_segment