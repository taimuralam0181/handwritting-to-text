# Handwritten Image to Text Recognition

This is a beginner-friendly Django project setup for a university final year project demo.

Current stage:
- Django project setup completed
- Image upload form completed
- Uploaded image preview completed
- Local OCR output completed
- API key based model option added

## Project Name

`handwritten_project`

## App Name

`ocr_app`

## Required Software

- Python 3.12 or 3.13
- pip
- Virtual environment support
- Tesseract OCR for local OCR mode on Windows

## Install and Run

Open the project folder in VS Code terminal and run:

```bash
git clone https://github.com/taimuralam0181/handwritting-to-text.git
cd handwritting-to-text
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python manage.py migrate
python manage.py runserver
```

Then open:

```text
http://127.0.0.1:8000/
```

## Windows Local OCR Setup

The local OCR mode uses `pytesseract`, which requires the Tesseract OCR desktop app.

1. Install Tesseract OCR for Windows.
2. Make sure `tesseract.exe` exists at:

```text
C:\Program Files\Tesseract-OCR\tesseract.exe
```

3. If Tesseract is installed in a different folder, either add it to Windows `PATH` or set:

```text
set OCR_TESSERACT_CMD=C:\your\custom\path\tesseract.exe
```

4. Verify the installation:

```bash
"C:\Program Files\Tesseract-OCR\tesseract.exe" --version
```

The app now tries these options automatically:

- `OCR_TESSERACT_CMD`
- `C:\Program Files\Tesseract-OCR\tesseract.exe`
- `C:\Program Files (x86)\Tesseract-OCR\tesseract.exe`
- `%LOCALAPPDATA%\Tesseract-OCR\tesseract.exe`
- `tesseract` from Windows `PATH`

## What This Setup Includes

- Django project and app configuration
- Template rendering
- Static files setup
- Media files setup
- Home page for image upload
- Uploaded image preview after submit
- Local Tesseract OCR option
- Local AI handwriting model option
- API-based OCR model option

## API Model Configuration

If you want to use an external API-based OCR model, set these environment variables before running the server.

For Gemini SDK usage, install the default requirements and then set:

```bash
set OCR_API_KEY=your-gemini-api-key
set OCR_API_MODEL=gemini-2.0-flash
```

If `OCR_API_MODEL` is not set, the project now defaults to `gemini-2.0-flash`.

For a custom HTTP OCR endpoint, set:

```bash
set OCR_API_URL=https://your-api-endpoint
set OCR_API_KEY=your-secret-api-key
set OCR_API_MODEL=your-model-name
```

The API route sends JSON with:

```text
model
file_name
image_base64
```

Expected response JSON can include any one of these fields:

```text
predicted_text
text
result
output
```

## Important Note

This project now supports local OCR and an optional API-based OCR path.
You can later replace the generic API route with OpenAI, Hugging Face, or another OCR provider.

## Local AI Model

The project is configured to use this lighter handwritten model by default:

```text
microsoft/trocr-small-handwritten
```

Default local model folder:

```text
local_models/trocr-small-handwritten
```

Fine-tuned model output folder:

```text
local_models/trocr-small-handwritten-finetuned
```

## Handwriting Dataset

The project is configured to use this handwriting OCR dataset by default:

```text
Teklia/IAM-line
```

This is a line-level English handwriting dataset suitable for OCR model fine-tuning.

Prepare the dataset locally with:

```bash
python manage.py prepare_handwriting_dataset
```

Optional quick sample export:

```bash
python manage.py prepare_handwriting_dataset --limit 100
```

Default dataset output folder:

```text
datasets/iam_line
```

## Fine-Tune the Handwriting Model

If you want the full OCR / AI training environment, install the larger dependency set first:

```bash
pip install -r full_ocr_requirements.txt
```

After preparing the dataset, fine-tune the local handwritten OCR model with:

```bash
python manage.py train_handwriting_model --epochs 1 --train-limit 200 --eval-limit 50
```

Full training example:

```bash
python manage.py train_handwriting_model --epochs 3 --train-batch-size 2 --eval-batch-size 2
```

The app will automatically prefer the fine-tuned model if this folder exists:

```text
local_models/trocr-small-handwritten-finetuned
```
