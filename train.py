# run: CUDA_VISIBLE_DEVICES="0" python train.py
# Will resume previous runs via loading checkpoints and models

import os
import datetime
import shutil
import torch
import logging
import argparse


from trainer import Trainer, TrainerArgs
from TTS.tts.configs.tacotron2_config import Tacotron2Config
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.tacotron2 import Tacotron2
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def parse_timestamp(folder_name):
    timestamp_str = folder_name.split('-')[1:4]
    timestamp_str = '-'.join(timestamp_str).replace('+', ':')
    return datetime.datetime.strptime(timestamp_str, '%B-%d-%Y_%I:%M%p')

def save_checkpoint_cb(trainer):
    trainer.save_checkpoint()

def extract_model_number(filename):
    parts = filename.split('_')
    if len(parts) > 2:
        try:
            return int(parts[2].split('.')[0])  # Extract the number and convert to int
        except ValueError:
            return -1  # In case of non-integer value
    return -1
    

def load_best_model_from_previous_runs(output_path):
    run_folders = [d for d in os.listdir(output_path) if os.path.isdir(os.path.join(output_path, d)) and d.startswith("run-")]
    best_model = None
    best_model_path = None

    for folder in run_folders:
        checkpoint_files = [f for f in os.listdir(os.path.join(output_path, folder)) if f.startswith("best_model") and f.endswith(".pth")]
        for file in checkpoint_files:
            file_path = os.path.join(output_path, folder, file)
            model_number = extract_model_number(file)
            if best_model is None or model_number > best_model:
                best_model = model_number
                best_model_path = file_path

    return best_model_path

output_path = os.path.dirname(os.path.abspath(__file__))

# Log the output path
logging.info(f"Output path: {output_path}")

dataset_config = BaseDatasetConfig(
    formatter="ljspeech",
    meta_file_train="transcriptions.txt",
    path=os.path.join(output_path, "mark")
)

parser = argparse.ArgumentParser(description='Train Tacotron2')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
parser.add_argument('--test-epochs', type=int, default=1000, help='Number of epochs for testing')
parser.add_argument('--skip-checkpoints', action='store_true', help='Skip loading checkpoints for batch size testing')


args = parser.parse_args()

config = Tacotron2Config(
    batch_size=args.batch_size if args.batch_size is not None else 64,
    eval_batch_size=16,
    num_loader_workers=4,
    num_eval_loader_workers=4,
    run_eval=True,
    test_delay_epochs=-1,
    epochs=args.test_epochs if args.test_epochs is not None else 500,
    text_cleaner="phoneme_cleaners",
    use_phonemes=True,
    phoneme_language="en-us",
    phoneme_cache_path=os.path.join(output_path, "phoneme_cache"),
    print_step=25,
    print_eval=False,
    mixed_precision=True,
    output_path=output_path,
    datasets=[dataset_config],
)

ap = AudioProcessor.init_from_config(config)

tokenizer, config = TTSTokenizer.init_from_config(config)

train_samples, eval_samples = load_tts_samples(
    dataset_config,
    eval_split=True,
    eval_split_max_size=config.eval_split_max_size,
    eval_split_size=config.eval_split_size,
)

model = Tacotron2(config, ap, tokenizer, speaker_manager=None)

trainer_args = TrainerArgs()

if not args.skip_checkpoints:
    checkpoint_folders = [d for d in os.listdir(output_path) if os.path.isdir(os.path.join(output_path, d)) and d.startswith("run-")]
    best_model_path = load_best_model_from_previous_runs(output_path)
    if best_model_path:
        logging.info(f"Loading best model from previous run: {best_model_path}")
        checkpoint = torch.load(best_model_path)
        
        trainer_args = TrainerArgs(restore_path=best_model_path)
    if checkpoint_folders:
        logging.info("Run folders exist, looking for the most recent")

        sorted_folders = sorted(checkpoint_folders, key=parse_timestamp)
        current_run_folder = sorted_folders[-1]
        if len(sorted_folders) > 1:
            previous_run_folder = sorted_folders[-2]
            previous_checkpoint_folder_path = os.path.join(output_path, previous_run_folder)
            checkpoint_files = [f for f in os.listdir(previous_checkpoint_folder_path) if f.endswith(".pth")]
            if checkpoint_files:
                latest_checkpoint_file = max(checkpoint_files, key=lambda x: os.path.getmtime(os.path.join(previous_checkpoint_folder_path, x)))
                src = os.path.join(previous_checkpoint_folder_path, latest_checkpoint_file)
                dst = os.path.join(output_path, current_run_folder, latest_checkpoint_file)
                shutil.copy(src, dst)
                checkpoint = torch.load(dst)
                trainer_args = TrainerArgs(restore_path=dst)

trainer = Trainer(
    trainer_args,
    config,
    output_path,
    model=model,
    train_samples=train_samples,
    eval_samples=eval_samples,
    callbacks={"on_epoch_end": save_checkpoint_cb}
)

# Log the start of training
logging.info("Starting training process...")
trainer.fit()
logging.info("Training complete.")
