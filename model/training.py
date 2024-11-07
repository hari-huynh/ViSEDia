from transformers import TrainingArguments, Trainer
from transformers import WavLMConfig
from defaults import  label2id, id2label, ORIG_SAMPLE_RATE, TARGET_SAMPLE_RATE, SAVE_PATH
from data.visedia_dataset import SED_Model, my_collate_fn, ViSEDia
from torchaudio.transforms import Resample


config = WavLMConfig()
model = SED_Model(config)

resampler = Resample(orig_freq=ORIG_SAMPLE_RATE, new_freq=TARGET_SAMPLE_RATE)

train_data = ViSEDia(
    dataset_path = SAVE_PATH,
    label2id = label2id,
    id2label = id2label,
    transform = resampler
)

training_args = TrainingArguments(
    output_dir="./ViSEDia_ckpt",
    num_train_epochs=5,
    learning_rate = 1e-5,
    logging_steps = 15,
    dataloader_pin_memory = True,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    report_to = 'none'
)

trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset = train_data,
    data_collator = my_collate_fn
)

trainer.train()