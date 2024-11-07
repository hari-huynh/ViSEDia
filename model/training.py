from transformers import TrainingArguments, Trainer
from transformers import WavLMConfig
 , TARGET_SAMPLE_RATE
from data.visedia_dataset import SED_Model, my_collate_fn, ViSEDia
from torchaudio.transforms import Resample
import yaml
import argparse

# Load configs
with open("configs.yaml", "r") as file:
    conf = yaml.safe_load(file)

id2label = {int(k): v for k, v in conf["id2emo"].items()}
label2id = {v: k for k, v in id2label.items()}
ORIG_SAMPLE_RATE = conf["orig_sample_rate"]
TARGET_SAMPLE_RATE = conf["target_sample_rate"]

def parse_args():
    parser = argparse.ArgumentParser(description="Train ViSEDia model")

    # Tham số cho Trainer
    parser.add_argument("--dataset-dir", type=str)
    parser.add_argument("--output_dir", type=str, default="./ViSEDia_ckpt", help="Output directory for model checkpoints")
    parser.add_argument("--num_train_epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate for training")
    parser.add_argument("--logging_steps", type=int, default=15, help="Number of steps between logging")
    parser.add_argument("--train_batch_size", type=int, default=1, help="Training batch size")
    parser.add_argument("--eval_batch_size", type=int, default=1, help="Evaluation batch size")
    # parser.add_argument("--save_steps", type=int, default=500, help="Number of steps between model checkpoints")
    # parser.add_argument("--eval_steps", type=int, default=500, help="Number of steps between evaluations")
    # parser.add_argument("--warmup_steps", type=int, default=0, help="Number of warmup steps for learning rate scheduler")
    # parser.add_argument("--logging_dir", type=str, default="./logs", help="Directory for logging")
    # parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    return parser.parse_args()

def main():
    
    args = parse_args()
    config = WavLMConfig()
    model = SED_Model(config)

    resampler = Resample(orig_freq=ORIG_SAMPLE_RATE, new_freq=TARGET_SAMPLE_RATE)

    train_data = ViSEDia(
        dataset_path = args.dataset_dir,
        label2id = label2id,
        id2label = id2label,
        transform = resampler
    )

    training_args = TrainingArguments(
        output_dir=  args.output_dir , # "./ViSEDia_ckpt",
        num_train_epochs=args.num_train_epochs,
        learning_rate = args.learning_rate , #1e-5,
        logging_steps = args.logging_steps,
        dataloader_pin_memory = True,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        report_to = 'none'
    )

    trainer = Trainer(
        model = model,
        args = training_args,
        train_dataset = train_data,
        data_collator = my_collate_fn
    )

    trainer.train()
    
if __name__ == "__main__":
    main()
