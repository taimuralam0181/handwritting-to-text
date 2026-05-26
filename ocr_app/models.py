from django.db import models


class UploadedImage(models.Model):
    """Stores uploaded handwriting images for preview/testing."""

    OCR_ENGINE_CHOICES = [
        ('smart', 'Smart Hybrid OCR (Recommended)'),
        ('local', 'Local Tesseract OCR'),
        ('ai_local', 'Local AI Handwriting Model'),
        ('api', 'API Key Model'),
    ]
    PREDICTION_SOURCE_CHOICES = [
        ('gemini', 'Gemini'),
        ('api', 'API OCR'),
        ('local', 'Local OCR'),
        ('local_ai', 'Local AI OCR'),
        ('unknown', 'Unknown'),
    ]

    title = models.CharField(max_length=200, blank=True)
    image = models.ImageField(upload_to='uploaded_images/')
    ocr_engine = models.CharField(max_length=20, choices=OCR_ENGINE_CHOICES, default='smart')
    prediction_source = models.CharField(max_length=20, choices=PREDICTION_SOURCE_CHOICES, default='unknown', blank=True)
    prediction_notes = models.TextField(blank=True)
    raw_ocr_text = models.TextField(blank=True)
    predicted_text = models.TextField(blank=True)
    user_corrected_text = models.TextField(blank=True)
    correction_applied = models.BooleanField(default=False)
    added_to_training_set = models.BooleanField(default=False)
    uploaded_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-uploaded_at']

    def __str__(self):
        return self.title or f"Image {self.pk}"

    def is_gemini_prediction(self):
        """Return whether the latest prediction came from Gemini."""

        return self.prediction_source == 'gemini'
