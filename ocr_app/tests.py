import shutil
import tempfile
from unittest.mock import patch

from django.contrib.auth.models import User
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, override_settings
from django.urls import reverse
from PIL import Image
import io

from .models import UploadedImage


class CorrectionMemoryTests(TestCase):
    def setUp(self):
        self.media_root = tempfile.mkdtemp()
        self.settings_override = override_settings(MEDIA_ROOT=self.media_root)
        self.settings_override.enable()
        self.user = User.objects.create_user(username='reader', password='test-password-123')
        self.client.force_login(self.user)

    def tearDown(self):
        self.settings_override.disable()
        shutil.rmtree(self.media_root, ignore_errors=True)

    def _image_upload(self, name='sample.png'):
        output = io.BytesIO()
        image = Image.new('L', (200, 80), color='white')
        for x in range(30, 170):
            image.putpixel((x, 40), 0)
        image.save(output, format='PNG')
        return SimpleUploadedFile(name, output.getvalue(), content_type='image/png')

    @patch('ocr_app.views.detect_target_type', return_value='line')
    @patch('ocr_app.views.extract_and_correct_text')
    def test_same_image_uses_saved_correction(self, extract_text, _detect_target):
        extract_text.return_value = ('wrong text', 'wrong text', 'local', '')
        self.client.post(
            reverse('home'),
            {
                'action': 'upload',
                'title': '',
                'ocr_engine': 'smart',
                'extraction_mode': 'both',
                'target_type': 'auto',
                'image': self._image_upload(),
            },
        )
        first_upload = UploadedImage.objects.get()
        first_upload.user_corrected_text = 'correct handwritten text'
        first_upload.predicted_text = 'correct handwritten text'
        first_upload.correction_applied = True
        first_upload.save()

        extract_text.reset_mock()
        self.client.post(
            reverse('home'),
            {
                'action': 'upload',
                'title': '',
                'ocr_engine': 'smart',
                'extraction_mode': 'both',
                'target_type': 'auto',
                'image': self._image_upload('same-image.png'),
            },
        )

        latest = UploadedImage.objects.order_by('-uploaded_at').first()
        self.assertEqual(latest.predicted_text, 'correct handwritten text')
        self.assertEqual(latest.prediction_source, 'memory')
        extract_text.assert_not_called()
