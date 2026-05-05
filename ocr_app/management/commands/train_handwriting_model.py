import json

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError


class Command(BaseCommand):
    help = "Fine-tune the local handwriting OCR model using the prepared dataset."

    def add_arguments(self, parser):
        parser.add_argument(
            '--dataset-dir',
            default=settings.HANDWRITING_DATASET_DIR,
            help='Prepared dataset directory containing train and validation metadata files.',
        )
        parser.add_argument(
            '--base-model-dir',
            default=settings.OCR_LOCAL_MODEL_DIR,
            help='Base TrOCR model path or Hugging Face model id.',
        )
        parser.add_argument(
            '--output-dir',
            default=settings.OCR_FINETUNED_MODEL_DIR,
            help='Directory where the fine-tuned model will be saved.',
        )
        parser.add_argument(
            '--epochs',
            type=int,
            default=1,
            help='Number of fine-tuning epochs.',
        )
        parser.add_argument(
            '--train-batch-size',
            type=int,
            default=2,
            help='Training batch size per device.',
        )
        parser.add_argument(
            '--eval-batch-size',
            type=int,
            default=2,
            help='Validation batch size per device.',
        )
        parser.add_argument(
            '--learning-rate',
            type=float,
            default=5e-5,
            help='Optimizer learning rate.',
        )
        parser.add_argument(
            '--train-limit',
            type=int,
            default=None,
            help='Optional limit on training samples for quick tests.',
        )
        parser.add_argument(
            '--eval-limit',
            type=int,
            default=None,
            help='Optional limit on validation samples for quick tests.',
        )

    def handle(self, *args, **options):
        try:
            from ocr_app.training_utils import train_handwriting_model

            summary = train_handwriting_model(
                dataset_dir=options['dataset_dir'],
                base_model_dir=options['base_model_dir'],
                output_dir=options['output_dir'],
                epochs=options['epochs'],
                train_batch_size=options['train_batch_size'],
                eval_batch_size=options['eval_batch_size'],
                learning_rate=options['learning_rate'],
                train_limit=options['train_limit'],
                eval_limit=options['eval_limit'],
            )
        except Exception as exc:
            raise CommandError(f'Failed to train handwriting model: {exc}') from exc

        self.stdout.write(self.style.SUCCESS('Handwriting model fine-tuning completed successfully.'))
        self.stdout.write(json.dumps(summary, indent=2))
