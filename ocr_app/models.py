from django.db import models


class UploadedImage(models.Model):
    """Stores uploaded handwriting images for preview/testing."""

    OCR_ENGINE_CHOICES = [
        ('smart', 'Smart Hybrid OCR (Recommended)'),
        ('local', 'Local Tesseract OCR'),
        ('ai_local', 'Local AI Handwriting Model'),
        ('api', 'API Key Model'),
    ]

    title = models.CharField(max_length=200, blank=True)
    image = models.ImageField(upload_to='uploaded_images/')
    ocr_engine = models.CharField(max_length=20, choices=OCR_ENGINE_CHOICES, default='smart')
    raw_ocr_text = models.TextField(blank=True)
    predicted_text = models.TextField(blank=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-uploaded_at']

    def __str__(self):
        return self.title or f"Image {self.pk}"
