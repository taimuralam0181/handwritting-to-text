import json

from django.core.management.base import BaseCommand

from ocr_app.dataset_utils import get_handwriting_dataset_profiles


class Command(BaseCommand):
    help = "List available handwriting dataset profiles for OCR training."

    def handle(self, *args, **options):
        profiles = get_handwriting_dataset_profiles()
        self.stdout.write(json.dumps(profiles, indent=2))
