import shutil
import tempfile
from unittest.mock import patch

from django.contrib.auth.models import User
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, override_settings
from django.urls import reverse
from PIL import Image
import io
import json

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


class OCRApiTests(TestCase):
    def setUp(self):
        self.media_root = tempfile.mkdtemp()
        self.settings_override = override_settings(MEDIA_ROOT=self.media_root)
        self.settings_override.enable()
        self.user = User.objects.create_user(username='api-user', password='test-password-123')

    def tearDown(self):
        self.settings_override.disable()
        shutil.rmtree(self.media_root, ignore_errors=True)

    def _image_upload(self):
        output = io.BytesIO()
        Image.new('RGB', (120, 60), color='white').save(output, format='PNG')
        return SimpleUploadedFile('api-sample.png', output.getvalue(), content_type='image/png')

    def _token(self):
        response = self.client.post(
            reverse('api_login'),
            data=json.dumps({'username': 'api-user', 'password': 'test-password-123'}),
            content_type='application/json',
        )
        return response.json()['token']

    def test_login_returns_bearer_token(self):
        token = self._token()
        self.assertTrue(token)

    def test_upload_list_requires_token(self):
        response = self.client.get(reverse('api_uploads'))
        self.assertEqual(response.status_code, 401)

    @patch('ocr_app.api.detect_target_type', return_value='line')
    @patch('ocr_app.api.extract_and_correct_text')
    def test_ocr_upload_returns_json_prediction(self, extract_text, _detect_target):
        extract_text.return_value = ('raw text', 'corrected text', 'local', '')
        response = self.client.post(
            reverse('api_ocr'),
            data={
                'image': self._image_upload(),
                'target_type': 'auto',
                'extraction_mode': 'both',
            },
            HTTP_AUTHORIZATION=f'Bearer {self._token()}',
        )

        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.json()['predicted_text'], 'corrected text')
        self.assertEqual(response.json()['target_type'], 'line')
        self.assertEqual(response.json()['prediction_source'], 'local')
