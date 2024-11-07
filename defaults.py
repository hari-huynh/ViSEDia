# Dataset configs
RAW_AUDIO_PATH = "raw_audio"
SAVE_PATH = "ViSEDia"
MAX_LENGTH = 30
MIN_DURATION = 3
N_CONSECUTIVE_UTTERANCE = 3
PAD_DURATION = 0.5
ORIG_SAMPLE_RATE = 22050
TARGET_SAMPLE_RATE = 16000

speaker2id = {
    "Speaker 1": 0,
    "Speaker 2": 1,
    "Speaker 3": 2,
    "Speaker 4": 3
}

id2speaker = {v: k for k, v in speaker2id.items()}

label2id = {
    'Angry': 0,   # angry
    'Neutral': 1,   # neutral
    'Happy': 2,   # happy
    'Sad': 3    # sad
}

id2label = {v: k for k, v in label2id.items()}