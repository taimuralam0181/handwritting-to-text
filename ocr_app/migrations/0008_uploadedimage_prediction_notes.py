from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('ocr_app', '0007_uploadedimage_prediction_source'),
    ]

    operations = [
        migrations.AddField(
            model_name='uploadedimage',
            name='prediction_notes',
            field=models.TextField(blank=True),
        ),
    ]
