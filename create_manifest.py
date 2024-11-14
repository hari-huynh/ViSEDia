import os
import pandas as pd
import json
# from defaults import MAX_LENGTH, SAVE_PATH, MIN_DURATION, N_CONSECUTIVE_UTTERANCE, PAD_DURATION
from utils import load_audio, save_audio, get_audio_segments
import numpy as np
import matplotlib.pyplot as plt
import yaml


# Load configs
with open("./ViSEDia/configs.yaml", "r") as file:
    conf = yaml.safe_load(file)

# Make folder if not exist
os.makedirs(conf['save_path'], exist_ok=True)

data = pd.read_csv("process_emotion_transcript.csv")

# Get relevant informations
audio_ids = ["-".join(path.split("/")[-1].split("-")[1:])[:-4] for path in data["audio"]]
raw_diarizations = [eval(dia) for dia in data["label"]]
raw_emotions = [eval(emo) for emo in data["sentiment"]]
raw_transcriptions = [eval(trans) for trans in data["transcription"]]

# Sort metadata
diarizations, emotions, transcriptions = [], [], []
for dia, emo, trans in zip(raw_diarizations, raw_emotions, raw_transcriptions):
    # Sort diarization results by start time
    indexed_data = [(i, item) for i, item in enumerate(dia)]
    sorted_data = sorted(indexed_data, key=lambda x: x[1]["start"])
    sorted_indices = [item[0] for item in sorted_data]

    diarizations.append([dia[i] for i in sorted_indices])
    emotions.append([emo[i] for i in sorted_indices])
    transcriptions.append([trans[i] for i in sorted_indices])

spk2id = {
    "Speaker 1": 0.2,
    "Speaker 2": 0.4,
    "Speaker 3": 0.6,
    "Speaker 4": 0.8
}

colors = ["red", "green", "blue", "violet"]
def plot_diarization(diarization):
    plt.figure(figsize=(20, 8))

    # diarization = diarization[:10]
    for dia in diarization:
        spk = dia["labels"][0]
        spk_id = spk2id[spk]

        plt.plot([dia["start"], dia["end"]], [spk_id, spk_id], marker="|", c=colors[int(spk.split()[-1])])

    plt.ylim([0.15, 0.85])
    plt.xlabel("Time (s)")
    plt.show()

def merge_segments(diarization):
    non_overlap_diarizations = []
    durations = []

    start_time, end_time = diarization[0]["start"], diarization[0]["end"]

    seg_id = 0
    non_overlap_diarizations.append([diarization[0]])

    for idx in range(1, len(diarization)):
        # Check if have overlap
        if (start_time <= diarization[idx]["start"] <= end_time):
            non_overlap_diarizations[seg_id].append(diarization[idx])
            end_time = max(end_time, diarization[idx]["end"])
        else:
            durations.append([start_time, end_time])
            seg_id += 1
            start_time, end_time = diarization[idx]["start"],  diarization[idx]["end"]
            non_overlap_diarizations.append([diarization[idx]])

    return non_overlap_diarizations, durations

def merge_consecutive_short_segments(diarizations, durations):
    # Get short segments
    consecutive_durations, consecutive_segments = [], []
    short_utterance_mask = [1 if duration[1] - duration[0] < conf['min_duration']  else 0 for i, duration in enumerate(durations)]

    idx = 0
    while idx < len(short_utterance_mask):
        if short_utterance_mask[idx] == 0:
            consecutive_segments.append(diarizations[idx])
            consecutive_durations.append(durations[idx])
            idx += 1
        else:
            if idx >= 1:
                start_time = durations[idx][0] - min(conf['pad_duration'], durations[idx][0] - durations[idx - 1][0])
            else:
                start_time = durations[idx][0] - min(conf['pad_duration'], durations[idx][0])

            consecutive_segments.append(diarizations[idx])

            idx += 1
            try:
                while short_utterance_mask[idx] == 1:
                    consecutive_segments[-1].append(diarizations[idx][0])
                    idx += 1
            except Exception:
                break

            end_time = durations[idx - 1][1] + min(conf['pad_duration'], durations[idx][1] - durations[idx - 1][1])
            consecutive_durations.append([start_time, end_time])
    return consecutive_segments, consecutive_durations

manifests = {}

for audio_id, diarization, emotion, transcript in zip(audio_ids, diarizations, emotions, transcriptions):
    pos = 0
    non_overlaps, duration = merge_segments(diarization)
    diarization, duration = merge_consecutive_short_segments(non_overlaps, duration)

    # Load audio file
    waveform, sample_rate = load_audio(audio_id)

    for idx in range(len(duration)):
        key = f"{audio_id}-{idx}"
        start_time, end_time = duration[idx][0], duration[idx][1]

        # Process audio
        processed_audio = get_audio_segments(
            waveform, sample_rate, start_time, end_time
        )

        # Save audio segments
        save_audio(processed_audio, sample_rate, key)

        manifests[key] = {
            "wav": f"datafolder/{key}.wav",
            "duration": end_time - start_time,
            "speaker": [dia["labels"][0] for dia in diarization[idx]],
            "emotion": [
                {
                    "start": dia["start"] - start_time,
                    "end": dia["end"] - start_time
                }
            for dia in diarization[idx]],
            "transcript": [],
        }

        for j in range(pos, pos + len(diarization[idx])):
            manifests[key]["emotion"][j - pos]["emo"] = emotion[j]
            manifests[key]["transcript"].append(transcript[j])

        pos += len(diarization[idx])

# Save manifest file
with open(os.path.join(conf['save_path'], "ViSEDia_manifest.json"), "w", encoding="utf-8") as f:
    json.dump(manifests, f, ensure_ascii=False, indent=4)
