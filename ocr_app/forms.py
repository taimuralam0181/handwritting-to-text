from django import forms
from django.core.exceptions import ValidationError

from .models import UploadedImage


class ImageUploadForm(forms.ModelForm):
    """Simple form for uploading a handwriting image."""

    max_file_size = 5 * 1024 * 1024
    allowed_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
    EXTRACTION_MODE_CHOICES = [
        ('both', 'Use both and keep better result'),
        ('crop', 'Use auto crop'),
        ('full', 'Use full image'),
    ]
    TARGET_TYPE_CHOICES = [
        ('mixed', 'Mixed Text'),
        ('word', 'Single Word'),
        ('line', 'Single Line'),
        ('page', 'Full Page'),
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['ocr_engine'].initial = 'smart'
        self.fields['ocr_engine'].choices = [
            ('smart', 'Smart Hybrid OCR (Recommended)'),
        ]
        self.fields['ocr_engine'].widget = forms.HiddenInput()
        self.fields['extraction_mode'] = forms.ChoiceField(
            choices=self.EXTRACTION_MODE_CHOICES,
            initial='both',
            widget=forms.Select(
                attrs={
                    'class': 'form-select',
                }
            ),
        )
        self.fields['target_type'] = forms.ChoiceField(
            choices=self.TARGET_TYPE_CHOICES,
            initial='mixed',
            widget=forms.Select(
                attrs={
                    'class': 'form-select',
                }
            ),
        )

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


class PredictionCorrectionForm(forms.Form):
    """Let the user correct a bad prediction and save it for retraining."""

    image_id = forms.IntegerField(widget=forms.HiddenInput())
    corrected_text = forms.CharField(
        widget=forms.Textarea(
            attrs={
                'class': 'form-control',
                'rows': 4,
                'placeholder': 'Write the correct text here.',
            }
        )
    )
    start_training = forms.BooleanField(
        required=False,
        initial=True,
        widget=forms.CheckboxInput(
            attrs={
                'class': 'form-check-input',
            }
        ),
    )

    def clean_corrected_text(self):
        """Require a non-empty correction."""

        corrected_text = self.cleaned_data.get('corrected_text', '').strip()
        if not corrected_text:
            raise ValidationError('Please enter the correct text before saving.')
        return corrected_text


class MultiFileInput(forms.ClearableFileInput):
    """Allow selecting multiple image files in a single form field."""

    allow_multiple_selected = True


class MultiFileField(forms.FileField):
    """Make Django treat a multi-file input as a list of uploaded files."""

    widget = MultiFileInput

    def clean(self, data, initial=None):
        """Validate one or many uploaded files with the base FileField logic."""

        single_clean = super().clean
        if isinstance(data, (list, tuple)):
            if not data:
                raise ValidationError(self.error_messages['required'], code='required')
            return [single_clean(item, initial) for item in data]
        return single_clean(data, initial)


class CustomTrainingDatasetForm(forms.Form):
    """Collect user-provided handwriting samples and matching labels."""

    max_file_size = 5 * 1024 * 1024
    allowed_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}

    images = MultiFileField(
        widget=MultiFileInput(
            attrs={
                'class': 'form-control',
                'accept': 'image/*',
                'multiple': True,
            }
        )
    )
    texts = forms.CharField(
        widget=forms.Textarea(
            attrs={
                'class': 'form-control',
                'rows': 6,
                'placeholder': 'One line per image, same order as selected files.',
            }
        )
    )
    auto_segment = forms.BooleanField(
        required=False,
        widget=forms.CheckboxInput(
            attrs={
                'class': 'form-check-input',
            }
        )
    )

    def clean_images(self):
        """Validate every uploaded dataset image."""

        images = self.files.getlist('images')
        if not images:
            raise ValidationError('Please select at least one training image.')

        for image in images:
            file_name = image.name.lower()
            if not any(file_name.endswith(ext) for ext in self.allowed_extensions):
                raise ValidationError('Only JPG, PNG, BMP, GIF, or WEBP images are allowed.')
            if image.size > self.max_file_size:
                raise ValidationError('Each training image must be 5 MB or smaller.')

        return images

    def clean_texts(self):
        """Normalize text labels to one non-empty line per sample."""

        texts_raw = self.cleaned_data.get('texts', '')
        texts = [line.strip() for line in texts_raw.splitlines() if line.strip()]
        if not texts:
            raise ValidationError('Please enter one text label per image.')
        return texts

    def clean(self):
        """Ensure image and label counts match exactly."""

        cleaned_data = super().clean()
        images = cleaned_data.get('images') or []
        texts = cleaned_data.get('texts') or []
        auto_segment = cleaned_data.get('auto_segment')

        if auto_segment and len(images) == 1 and texts:
            return cleaned_data

        if images and texts and len(images) != len(texts):
            raise ValidationError(
                f'You selected {len(images)} image(s) but provided {len(texts)} text line(s). '
                'Keep one text line for each image in the same order.'
            )

        return cleaned_data
