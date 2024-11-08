import torch
import numpy as np
from transformers import WavLMConfig
from defaults import id2label, TARGET_SAMPLE_RATE
from data.visedia_dataset import SED_Model
from torchaudio.transforms import Resample
import argparse



def parse_args():
    parser = argparse.ArgumentParser(description="Load model checkpoint and predict")

    parser.add_argument(
        "--checkpoint_path", 
        type=str, 
        required=True, 
        help="Path to the checkpoint directory"
    )
    parser.add_argument(
        "--audio_path", 
        type=str, 
        required=False, 
        default=None,
        help="Path to the audio file for prediction (optional)"
    )
    parser.add_argument(
        "--threshold", 
        type=float, 
        default=0.5,
        help="Threshold for prediction (default: 0.5)"
    )
    
    return parser.parse_args()
    


def load_model_from_checkpoint(checkpoint_path):
    config = WavLMConfig()
    model = SED_Model(config)
    model = model.from_pretrained(checkpoint_path)
    # model.eval()
    return model

def predict(model, waveform, sr, threshold=0.5):
    resampler = Resample(sr, TARGET_SAMPLE_RATE)
    waveform = resampler(waveform)

    model.eval()
    
    with torch.no_grad():
        out = model(waveform)   
        out = out.squeeze(0).permute(1, 2, 0)   # Num Speaker x Num Emotion x Frames
    
    probs = torch.sigmoid(out)
    mask = probs > threshold
    
    results = []
    stride = 1
    window_length = 1
    
    for spk_id in range(mask.shape[0]):
        for emo_id in range(mask.shape[1]):
            tensor = mask[spk_id, emo_id, :].int().tolist()
    
            frame_start = None
            for idx, val in enumerate(tensor):
                if (val == 1) and frame_start is None:
                    frame_start = idx
                elif (val == 0) and frame_start is not None:
                    # Tính thời gian bắt đầu và kết thúc
                    start = round(stride * 0.02 * frame_start, 2)
                    end = round(start + window_length * 0.02 * ((idx - 1) - frame_start), 2)
                    frame_start = None
                    
                    results.append({
                        "start": start,
                        "end": end, 
                        "emotion": id2label[emo_id]
                    })
        
            if frame_start is not None:
                start = round(stride * 0.02 * frame_start, 2)
                end = round(start + window_length * 0.02 * ((len(tensor) - 1) - frame_start), 2)
                results.append({
                    "start": start,
                    "end": end, 
                    "emotion": id2label[emo_id]
                })
        
    return results

if __name__ == "__main__":
    args = parse_args()

    model = load_model_from_checkpoint(args.checkpoint_path)

    if args.audio_path:
        import torchaudio
        waveform, sample_rate = torchaudio.load(args.audio_path)
    else:
        waveform = torch.randn(1, 16000)  
        sample_rate = 16000

    results = predict(model, waveform, sample_rate, threshold=args.threshold)
    
    print("_________________Prediction Results_________________________")
    print(results)