import torch
import torchaudio
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.nn.functional import pad
import json
from defaults import speaker2id, id2speaker, label2id, id2label

# Check if two audio segment overlapped
def get_overlap(a, b):
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


import torch

def get_multilabel(data, win_len=0.02, stride=0.02):
    emo_list = data["emotion"]
    speakers = data["speaker"]
    duration = data["duration"]

    # Calculation num frames of labels
    num_frames = int(duration / stride) + 1

    start = 0.0
    frame_multilabels = torch.zeros(num_frames, 4, 4)    # Batch x Num Frames x Num Speaker x Num Emotions
    intervals = []    # List [start, end] pairs
    emotions = []

    for idx, emo in enumerate(emo_list):
        emotions.append(emo["emo"])
        intervals.append([emo["start"], emo["end"]])

    for i in range(num_frames):
        win_start = start + i * stride
        win_end = win_start + win_len

        # Make sure that every sample exists in a window
        if win_end > duration:
            win_end = duration
            win_start = max(duration - win_end, 0)

        for j in range(len(intervals)):
            if get_overlap([win_start, win_end], intervals[j]) >= 0.5 * (win_end - win_start):
                spk_id = speaker2id[speakers[j]]
                emo_id = label2id[emotions[j]]
                frame_multilabels[i, spk_id, emo_id] = 1
                break

        if win_end >= duration:
            break

    return frame_multilabels

class ViSEDia(Dataset):
    def __init__(self, dataset_path, id2label, label2id, transform):
        self.dataset_path = dataset_path
        self.audio_metadata = self.load_audio_metadata()
        self.id2label = id2label
        self.label2id = label2id
        self.transform = transform

    def load_audio_metadata(self):
        with open(f"{self.dataset_path}/ViSEDia_manifest.json", "r") as file:
          all_dict = json.load(file)

        return list(all_dict.values())

    def __len__(self):
        return len(self.audio_metadata)

    def __getitem__(self, idx):
        # Load audio data (waveform, sample_rate)
        metadata = self.audio_metadata[idx]
        audio_path = metadata["wav"]

        waveform, sample_rate = torchaudio.load(audio_path.replace("datafolder", self.dataset_path))
        waveform = self.transform(waveform)
        waveform = waveform.squeeze()
        frame_multilabels = get_multilabel(metadata)
        # labels_tensor = torch.tensor([self.label2id[label] for label in frame_labels])
        # transcription = self.audio_metadata[idx]["transcription"]

        return {"input_ids": waveform, "labels": frame_multilabels}


def my_collate_fn(batch):
    waveforms, labels = [], []
    for data in batch:
        waveform = data["input_ids"]
        label = data["labels"]

        waveforms.append(waveform)
        labels.append(label)

    waveforms = pad_sequence(waveforms, batch_first=True, padding_value=0)
    labels = pad_sequence(labels, batch_first=True, padding_value=0)

    waveforms = waveforms.clone().detach()
    labels = labels.clone().detach()

    return {
        "input_ids": waveforms,
        "labels": labels
    }