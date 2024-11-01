import os
import pandas as pd
import json
from defaults import MAX_LENGTH, SAVE_PATH
from utils import load_audio, save_audio, get_audio_segments

# Make folder if not exist
os.makedirs(SAVE_PATH, exist_ok=True)

data = pd.read_csv("process_emotion_transcript.csv")

# Get relevant informations
audio_ids = [path.split("/")[-1].split("-")[1][:-4] for path in data["audio"]]
diarizations = [eval(dia) for dia in data["label"]]
emotions = [eval(emo) for emo in data["sentiment"]]
transcriptions = [eval(trans) for trans in data["transcription"]]

manifests = {}

for audio_id, diarization, emotion, transcript in zip(audio_ids, diarizations, emotions, transcriptions):
    # Load MP3 audio
    waveform, sample_rate = load_audio(audio_id)

    for idx in range(len(diarization)):
        key = f"{audio_id}#{idx}"
        manifests[key] = {
            "wav": f"datafolder/{audio_id}#{idx}.wav",
            "speaker": [],
            "emotion": [],
            "transcript": [],
        }

        manifests[key]["speaker"].append(diarization[idx]["labels"][0])
        manifests[key]["emotion"].append({
            "emo": emotion[idx],
            "start": diarization[idx]["start"],
            "end": diarization[idx]["end"]
        })
        manifests[key]["transcript"].append(transcript[idx])

        pos = idx + 1

        start_time = diarization[idx]["start"]

        try:
            while (diarization[pos]["end"] - start_time <= MAX_LENGTH):
                manifests[key]["speaker"].append(diarization[pos]["labels"][0])
                manifests[key]["emotion"].append({
                    "emo": emotion[pos],
                    "start": diarization[pos]["start"],
                    "end": diarization[pos]["end"]
                })
                manifests[key]["transcript"].append(transcript[pos])

                pos = pos + 1

            # Edit start and end time
            end_time = manifests[key]["emotion"][-1]["end"]
            manifests[key]["duration"] = round(end_time - start_time, 3)
            manifests[key]["emotion"] = [{
                "emo": emo_dict["emo"],
                "start": round(emo_dict["start"] - start_time, 3),
                "end": round(emo_dict["end"] - start_time, 3),
            } for emo_dict in manifests[key]["emotion"]]

            # Process audio
            processed_audio = get_audio_segments(
                waveform, sample_rate, start_time, end_time
            )

            # Save audio segments
            save_audio(processed_audio, sample_rate, f"{audio_id}#{idx}")

        except IndexError:
            break


# Save manifest file
with open(os.path.join(SAVE_PATH, "ViSED_manifest.json"), "w", encoding="utf-8") as f:
    json.dump(manifests, f, ensure_ascii=False, indent=4)


