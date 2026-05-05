from django import forms
from django.core.exceptions import ValidationError

from .models import UploadedImage


class ImageUploadForm(forms.ModelForm):
    """Simple form for uploading a handwriting image."""

    max_file_size = 5 * 1024 * 1024
    allowed_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['ocr_engine'].initial = 'smart'
        self.fields['ocr_engine'].choices = [
            ('smart', 'Smart Hybrid OCR (Recommended)'),
        ]
        self.fields['ocr_engine'].widget = forms.HiddenInput()

    class Meta:
        model = UploadedImage
        fields = ['title', 'ocr_engine', 'image']
        widgets = {
            'title': forms.TextInput(
                attrs={
                    'class': 'form-control',
                    'placeholder': 'Optional image title',
                }
            ),
            'ocr_engine': forms.Select(
                attrs={
                    'class': 'form-select',
                }
            ),
            'image': forms.ClearableFileInput(
                attrs={
                    'class': 'form-control',
                    'accept': 'image/*',
                }
            ),
        }

    def clean_image(self):
        """Allow only image files with a reasonable size for demo usage."""

        image = self.cleaned_data.get('image')
        if not image:
            raise ValidationError('Please select an image file to upload.')

        file_name = image.name.lower()
        if not any(file_name.endswith(ext) for ext in self.allowed_extensions):
            raise ValidationError('Only JPG, PNG, BMP, GIF, or WEBP images are allowed.')

        if image.size > self.max_file_size:
            raise ValidationError('Image size must be 5 MB or smaller.')

        return image


class TrainingForm(forms.Form):
    """One-click training form with no extra user input."""

    pass
