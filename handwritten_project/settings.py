"""
Django settings for handwritten_project project.
"""

import os
from pathlib import Path

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent


# Quick-start development settings - unsuitable for production
# See https://docs.djangoproject.com/en/5.1/howto/deployment/checklist/

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = 'django-insecure-gi-w04p$!sgx_j@+ukzuf8=qp412x-8uwx*dlradxq2h4tb#4m'

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = True

# Local hosts for development and simple project demo use.
ALLOWED_HOSTS = ['127.0.0.1', 'localhost', 'testserver']


# Application definition

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'ocr_app',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'handwritten_project.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        # Root template directory for shared project templates.
        'DIRS': [BASE_DIR / 'templates'],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'handwritten_project.wsgi.application'


# Database
# https://docs.djangoproject.com/en/5.1/ref/settings/#databases

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}


# Password validation
# https://docs.djangoproject.com/en/5.1/ref/settings/#auth-password-validators

AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]


# Internationalization
# https://docs.djangoproject.com/en/5.1/topics/i18n/

LANGUAGE_CODE = 'en-us'

TIME_ZONE = 'UTC'

USE_I18N = True

USE_TZ = True


# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/5.1/howto/static-files/

# URL used to serve static files such as CSS.
STATIC_URL = '/static/'

# Additional static files directory for custom assets.
STATICFILES_DIRS = [BASE_DIR / 'static']

# URL used to access uploaded files in development.
MEDIA_URL = '/media/'

# Folder where uploaded files will be stored.
MEDIA_ROOT = BASE_DIR / 'media'

# Optional API-based OCR model settings.
# Keep these empty if you want to use only local Tesseract OCR.
OCR_API_URL = os.environ.get('OCR_API_URL', '')
OCR_API_KEY = os.environ.get('OCR_API_KEY', '')
OCR_API_MODEL = os.environ.get('OCR_API_MODEL', '')
OCR_API_TIMEOUT = int(os.environ.get('OCR_API_TIMEOUT', '60'))
OCR_TESSERACT_CMD = os.environ.get('OCR_TESSERACT_CMD', '')
OCR_LOCAL_MODEL_ID = os.environ.get('OCR_LOCAL_MODEL_ID', 'microsoft/trocr-small-handwritten')
OCR_LOCAL_MODEL_DIR = os.environ.get('OCR_LOCAL_MODEL_DIR', str(BASE_DIR / 'local_models' / 'trocr-small-handwritten'))
OCR_FINETUNED_MODEL_DIR = os.environ.get('OCR_FINETUNED_MODEL_DIR', str(BASE_DIR / 'local_models' / 'trocr-small-handwritten-finetuned'))
HANDWRITING_DATASET_ID = os.environ.get('HANDWRITING_DATASET_ID', 'Teklia/IAM-line')
HANDWRITING_DATASET_DIR = os.environ.get('HANDWRITING_DATASET_DIR', str(BASE_DIR / 'datasets' / 'iam_line'))
HANDWRITING_DATASET_PROFILES = {
    'custom_csv': {
        'dataset_id': 'local/custom_csv',
        'output_dir': str(BASE_DIR / 'datasets' / 'custom_csv_prepared'),
        'csv_path': str(BASE_DIR / 'datasets' / 'handwriting_dataset.csv'),
        'image_name_field': 'image_name',
        'text_field': 'text',
        'validation_split_ratio': 0.2,
        'test_split_ratio': 0.1,
        'description': 'Your own handwriting dataset from a local CSV file with image_name,text columns.',
    },
    'iam_line': {
        'dataset_id': 'Teklia/IAM-line',
        'output_dir': str(BASE_DIR / 'datasets' / 'iam_line'),
        'image_field': 'image',
        'text_field': 'text',
        'description': 'Line-level English handwriting OCR dataset. Best for notes, sentences, and document lines.',
    },
    'emnist_letters': {
        'dataset_id': 'tanganke/emnist_letters',
        'output_dir': str(BASE_DIR / 'datasets' / 'emnist_letters'),
        'image_field': 'image',
        'label_field': 'label',
        'label_offset': 1,
        'alphabet': 'ABCDEFGHIJKLMNOPQRSTUVWXYZ',
        'validation_split_ratio': 0.1,
        'description': 'Single handwritten letter dataset. Useful for isolated characters and uppercase alphabet samples.',
    },
}

# Default primary key field type
# https://docs.djangoproject.com/en/5.1/ref/settings/#default-auto-field

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'
