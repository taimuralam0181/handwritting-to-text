from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('ocr_app', '0006_alter_uploadedimage_ocr_engine'),
    ]

    operations = [
        migrations.AddField(
            model_name='uploadedimage',
            name='prediction_source',
            field=models.CharField(
                blank=True,
                choices=[
                    ('gemini', 'Gemini'),
                    ('api', 'API OCR'),
                    ('local', 'Local OCR'),
                    ('local_ai', 'Local AI OCR'),
                    ('unknown', 'Unknown'),
                ],
                default='unknown',
                max_length=20,
            ),
        ),
    ]
