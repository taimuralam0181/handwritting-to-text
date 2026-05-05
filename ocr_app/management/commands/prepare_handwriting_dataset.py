import json

from django.core.management.base import BaseCommand, CommandError

from ocr_app.dataset_utils import prepare_handwriting_dataset


class Command(BaseCommand):
    help = "Download and prepare a handwriting dataset for OCR training and evaluation."

    def add_arguments(self, parser):
        parser.add_argument(
            '--dataset-profile',
            default='iam_line',
            help='Named dataset profile. Example: iam_line or emnist_letters.',
        )
        parser.add_argument(
            '--dataset-id',
            default=None,
            help='Hugging Face dataset id. Default uses HANDWRITING_DATASET_ID from settings.',
        )
        parser.add_argument(
            '--output-dir',
            default=None,
            help='Directory where the prepared dataset will be stored.',
        )
        parser.add_argument(
            '--limit',
            type=int,
            default=None,
            help='Optional per-split sample limit for quick testing.',
        )

    def handle(self, *args, **options):
        try:
            summary = prepare_handwriting_dataset(
                dataset_profile=options['dataset_profile'],
                dataset_id=options['dataset_id'],
                output_dir=options['output_dir'],
                sample_limit=options['limit'],
            )
        except Exception as exc:
            raise CommandError(f'Failed to prepare handwriting dataset: {exc}') from exc

        self.stdout.write(self.style.SUCCESS('Handwriting dataset prepared successfully.'))
        self.stdout.write(json.dumps(summary, indent=2))
