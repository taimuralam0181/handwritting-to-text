import json
from dataclasses import dataclass
from pathlib import Path

import torch
from datasets import Dataset
from PIL import Image
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
    TrOCRProcessor,
    VisionEncoderDecoderModel,
)


def _read_metadata(metadata_path: Path, sample_limit: int | None = None):
    """Read exported JSONL metadata records from the prepared handwriting dataset."""

    records = []
    with metadata_path.open('r', encoding='utf-8') as metadata_file:
        for line in metadata_file:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
            if sample_limit is not None and len(records) >= sample_limit:
                break
    return records


def _load_split_from_metadata(metadata_path: Path, sample_limit: int | None = None):
    """Build a Hugging Face Dataset from local JSONL metadata."""

    records = _read_metadata(metadata_path, sample_limit=sample_limit)
    if not records:
        raise ValueError(f'No dataset records found in {metadata_path}')
    return Dataset.from_list(records)


def _build_processor(base_model_dir: str):
    """Load the TrOCR processor from a local or Hugging Face model path."""

    return TrOCRProcessor.from_pretrained(base_model_dir, use_fast=False)


def _build_model(base_model_dir: str):
    """Load and configure the TrOCR model for sequence generation training."""

    model = VisionEncoderDecoderModel.from_pretrained(base_model_dir)
    model.config.decoder_start_token_id = 2
    model.config.pad_token_id = 1
    model.config.eos_token_id = 2
    model.config.max_length = 128
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 2
    model.config.length_penalty = 1.0
    model.config.num_beams = 4

    original_forward = model.forward

    def forward_without_num_items_in_batch(*args, **kwargs):
        kwargs.pop('num_items_in_batch', None)
        return original_forward(*args, **kwargs)

    model.forward = forward_without_num_items_in_batch
    return model


def _preprocess_record(record, processor, max_target_length: int):
    """Convert one handwriting sample into model-ready tensors."""

    image = Image.open(record['image_path']).convert('RGB')
    pixel_values = processor(images=image, return_tensors='pt').pixel_values[0]
    labels = processor.tokenizer(
        record['text'],
        padding='max_length',
        max_length=max_target_length,
        truncation=True,
    ).input_ids
    labels = [token if token != processor.tokenizer.pad_token_id else -100 for token in labels]

    return {
        'pixel_values': pixel_values,
        'labels': labels,
    }


@dataclass
class OCRDataCollator:
    """Batch collator for image-to-text OCR fine-tuning."""

    processor: TrOCRProcessor

    def __call__(self, features):
        pixel_values = torch.stack(
            [
                feature['pixel_values']
                if isinstance(feature['pixel_values'], torch.Tensor)
                else torch.tensor(feature['pixel_values'], dtype=torch.float32)
                for feature in features
            ]
        )
        labels = torch.tensor([feature['labels'] for feature in features], dtype=torch.long)
        return {
            'pixel_values': pixel_values,
            'labels': labels,
        }


class OCRSeq2SeqTrainer(Seq2SeqTrainer):
    """Compatibility wrapper for trainer/model argument mismatches."""

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        inputs.pop('num_items_in_batch', None)
        return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)


class ProgressCallback(TrainerCallback):
    """Report training progress back to the app."""

    def __init__(self, progress_callback=None):
        self.progress_callback = progress_callback

    def _emit(self, progress, message):
        if self.progress_callback:
            self.progress_callback(progress, message)

    def on_train_begin(self, args, state, control, **kwargs):
        self._emit(35, 'Model training started.')

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.max_steps:
            progress = 35 + int((state.global_step / max(state.max_steps, 1)) * 55)
            self._emit(min(progress, 90), f'Training step {state.global_step} of {state.max_steps}')

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        self._emit(92, 'Validation completed.')

    def on_train_end(self, args, state, control, **kwargs):
        self._emit(96, 'Saving the trained model.')


def train_handwriting_model(
    dataset_dir: str,
    base_model_dir: str,
    output_dir: str,
    epochs: int = 1,
    train_batch_size: int = 2,
    eval_batch_size: int = 2,
    learning_rate: float = 5e-5,
    max_target_length: int = 128,
    train_limit: int | None = None,
    eval_limit: int | None = None,
    progress_callback=None,
):
    """
    Fine-tune a TrOCR handwriting model using a prepared local dataset.

    Expects dataset_dir to contain:
    - train/metadata.jsonl
    - validation/metadata.jsonl
    """

    dataset_path = Path(dataset_dir)
    train_metadata_path = dataset_path / 'train' / 'metadata.jsonl'
    validation_metadata_path = dataset_path / 'validation' / 'metadata.jsonl'

    if not train_metadata_path.exists():
        raise FileNotFoundError(f'Train metadata not found: {train_metadata_path}')
    if not validation_metadata_path.exists():
        raise FileNotFoundError(f'Validation metadata not found: {validation_metadata_path}')

    if progress_callback:
        progress_callback(10, 'Loading processor and base model.')

    processor = _build_processor(base_model_dir)
    model = _build_model(base_model_dir)

    if progress_callback:
        progress_callback(18, 'Loading prepared training and validation metadata.')
    train_dataset = _load_split_from_metadata(train_metadata_path, sample_limit=train_limit)
    validation_dataset = _load_split_from_metadata(validation_metadata_path, sample_limit=eval_limit)

    if progress_callback:
        progress_callback(25, 'Preprocessing images and target texts for training.')
    train_dataset = train_dataset.map(
        lambda record: _preprocess_record(record, processor, max_target_length),
        remove_columns=train_dataset.column_names,
    )
    validation_dataset = validation_dataset.map(
        lambda record: _preprocess_record(record, processor, max_target_length),
        remove_columns=validation_dataset.column_names,
    )

    training_arguments = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        predict_with_generate=True,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        logging_strategy='steps',
        logging_steps=10,
        learning_rate=learning_rate,
        num_train_epochs=epochs,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        greater_is_better=False,
        fp16=False,
        report_to=[],
        remove_unused_columns=False,
    )

    trainer = OCRSeq2SeqTrainer(
        model=model,
        tokenizer=processor,
        args=training_arguments,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=OCRDataCollator(processor=processor),
        callbacks=[ProgressCallback(progress_callback=progress_callback)],
    )

    trainer.train()
    if progress_callback:
        progress_callback(98, 'Saving model files.')
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)

    return {
        'output_dir': str(Path(output_dir).resolve()),
        'train_samples': len(train_dataset),
        'validation_samples': len(validation_dataset),
        'epochs': epochs,
        'train_batch_size': train_batch_size,
        'eval_batch_size': eval_batch_size,
        'learning_rate': learning_rate,
    }
