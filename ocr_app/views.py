import csv
from pathlib import Path
import re
import threading
import uuid

import cv2
import numpy as np
from django.contrib import messages
from django.conf import settings
from django.core.exceptions import ValidationError
from django.http import JsonResponse
from django.shortcuts import redirect, render
from django.utils.text import slugify

from .forms import CustomTrainingDatasetForm, ImageUploadForm, PredictionCorrectionForm, TrainingForm
from .models import UploadedImage
from .services import _crop_primary_foreground_region, _crop_text_region, _detect_sparse_text_boxes, extract_and_correct_text
from .training_status import read_training_status, write_training_status


def _build_training_output_dir(dataset_profile: str) -> str:
    """Keep training outputs separate for each dataset profile."""

    if dataset_profile == 'iam_line':
        return settings.OCR_FINETUNED_MODEL_DIR

    return str(Path(settings.BASE_DIR) / 'local_models' / f'trocr-{dataset_profile}-finetuned')


def _resolve_training_preset(training_mode: str, form_data: dict):
    """Map a simple browser training mode to concrete training arguments."""

    presets = {
        'quick': {
            'epochs': 1,
            'train_limit': 50,
            'eval_limit': 20,
            'train_batch_size': 1,
            'eval_batch_size': 1,
        },
        'balanced': {
            'epochs': 2,
            'train_limit': 300,
            'eval_limit': 80,
            'train_batch_size': 1,
            'eval_batch_size': 1,
        },
        'full': {
            'epochs': 3,
            'train_limit': None,
            'eval_limit': None,
            'train_batch_size': 1,
            'eval_batch_size': 1,
        },
    }
    resolved = presets.get(training_mode, presets['quick']).copy()

    for key in ['epochs', 'train_limit', 'eval_limit', 'train_batch_size', 'eval_batch_size']:
        if form_data.get(key) not in (None, ''):
            resolved[key] = form_data[key]

    return resolved


def _custom_csv_dataset_ready() -> bool:
    """Check whether the local custom CSV dataset has matching image files."""

    csv_path = Path(settings.HANDWRITING_DATASET_PROFILES['custom_csv']['csv_path'])
    if not csv_path.exists():
        return False

    try:
        lines = csv_path.read_text(encoding='utf-8-sig').splitlines()
    except OSError:
        return False

    dataset_root = csv_path.parent
    for raw_line in lines[1:6]:
        if not raw_line.strip():
            continue
        image_name = raw_line.split(',', 1)[0].strip().strip('"')
        if not image_name:
            continue
        direct_image = dataset_root / image_name
        nested_image = dataset_root / 'images' / image_name
        if direct_image.exists() or nested_image.exists():
            return True

    return False


def _count_custom_csv_samples() -> int:
    """Return how many labeled custom handwriting samples are available."""

    csv_path = Path(settings.HANDWRITING_DATASET_PROFILES['custom_csv']['csv_path'])
    if not csv_path.exists():
        return 0

    try:
        with csv_path.open('r', encoding='utf-8-sig', newline='') as csv_file:
            reader = csv.DictReader(csv_file)
            return sum(1 for row in reader if (row.get('image_name') or '').strip() and (row.get('text') or '').strip())
    except OSError:
        return 0


def _store_custom_training_samples(images, texts):
    """Save user-uploaded handwriting samples into the custom CSV dataset."""

    config = settings.HANDWRITING_DATASET_PROFILES['custom_csv']
    csv_path = Path(config['csv_path'])
    dataset_root = csv_path.parent
    images_dir = dataset_root / 'images'
    images_dir.mkdir(parents=True, exist_ok=True)
    dataset_root.mkdir(parents=True, exist_ok=True)

    csv_exists = csv_path.exists()
    image_name_field = config.get('image_name_field', 'image_name')
    text_field = config.get('text_field', 'text')
    saved_count = 0

    with csv_path.open('a', encoding='utf-8', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=[image_name_field, text_field])
        if not csv_exists:
            writer.writeheader()

        for image, text in zip(images, texts):
            safe_stem = slugify(Path(image.name).stem) or 'sample'
            extension = Path(image.name).suffix.lower() or '.png'
            image_name = f'{safe_stem}-{uuid.uuid4().hex[:8]}{extension}'
            target_path = images_dir / image_name

            with target_path.open('wb') as destination:
                for chunk in image.chunks():
                    destination.write(chunk)

            writer.writerow({image_name_field: image_name, text_field: text})
            saved_count += 1

    return saved_count


def _split_uploaded_training_image(image, expected_segments: int):
    """Split one uploaded page image into isolated handwriting crops."""

    image_bytes = image.read()
    image.seek(0)
    image_array = np.frombuffer(image_bytes, np.uint8)
    color_image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if color_image is None:
        raise ValidationError('Could not read the uploaded training image for auto-segmentation.')

    gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    cropped_page = _crop_text_region(gray_image)
    enlarged_page, boxes = _detect_sparse_text_boxes(cropped_page)
    if len(boxes) != expected_segments:
        raise ValidationError(
            f'Auto-segmentation found {len(boxes)} text group(s), but you provided {expected_segments} label line(s). '
            'Crop the image more tightly or adjust the labels.'
        )

    segment_entries = []
    image_height, image_width = enlarged_page.shape[:2]
    for index, (x, y, w, h) in enumerate(sorted(boxes, key=lambda box: (box[1], box[0])), start=1):
        pad = 25
        roi = enlarged_page[max(y - pad, 0):min(y + h + pad, image_height), max(x - pad, 0):min(x + w + pad, image_width)]
        roi = _crop_primary_foreground_region(roi)
        success, encoded_image = cv2.imencode('.png', roi)
        if not success:
            raise ValidationError('Could not prepare one of the auto-segmented training crops.')
        segment_entries.append((f'{Path(image.name).stem}-segment-{index}.png', encoded_image.tobytes()))

    return segment_entries


def _build_custom_training_entries(images, texts, auto_segment: bool):
    """Normalize uploaded samples into save-ready image/text pairs."""

    if auto_segment and len(images) == 1:
        segmented_images = _split_uploaded_training_image(images[0], len(texts))
        return [
            {
                'name': segment_name,
                'bytes': segment_bytes,
                'text': text,
            }
            for (segment_name, segment_bytes), text in zip(segmented_images, texts)
        ]

    entries = []
    for image, text in zip(images, texts):
        image_bytes = image.read()
        image.seek(0)
        entries.append(
            {
                'name': image.name,
                'bytes': image_bytes,
                'text': text,
            }
        )
    return entries


def _store_custom_training_entries(entries):
    """Persist normalized custom training entries into the local CSV dataset."""

    config = settings.HANDWRITING_DATASET_PROFILES['custom_csv']
    csv_path = Path(config['csv_path'])
    dataset_root = csv_path.parent
    images_dir = dataset_root / 'images'
    images_dir.mkdir(parents=True, exist_ok=True)
    dataset_root.mkdir(parents=True, exist_ok=True)

    csv_exists = csv_path.exists()
    image_name_field = config.get('image_name_field', 'image_name')
    text_field = config.get('text_field', 'text')
    saved_count = 0

    with csv_path.open('a', encoding='utf-8', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=[image_name_field, text_field])
        if not csv_exists:
            writer.writeheader()

        for entry in entries:
            safe_stem = slugify(Path(entry['name']).stem) or 'sample'
            extension = Path(entry['name']).suffix.lower() or '.png'
            image_name = f'{safe_stem}-{uuid.uuid4().hex[:8]}{extension}'
            target_path = images_dir / image_name
            target_path.write_bytes(entry['bytes'])
            writer.writerow({image_name_field: image_name, text_field: entry['text']})
            saved_count += 1

    return saved_count


def _store_prediction_correction(uploaded_image, corrected_text: str) -> int:
    """Upsert one corrected prediction into the local custom dataset."""

    config = settings.HANDWRITING_DATASET_PROFILES['custom_csv']
    csv_path = Path(config['csv_path'])
    dataset_root = csv_path.parent
    images_dir = dataset_root / 'images'
    images_dir.mkdir(parents=True, exist_ok=True)
    dataset_root.mkdir(parents=True, exist_ok=True)

    image_name_field = config.get('image_name_field', 'image_name')
    text_field = config.get('text_field', 'text')
    safe_stem = slugify(Path(uploaded_image.image.name).stem) or f'uploaded-image-{uploaded_image.pk}'
    extension = Path(uploaded_image.image.name).suffix.lower() or '.png'
    correction_image_name = f'{safe_stem}-correction{extension}'
    corrected_text = corrected_text.strip()

    with uploaded_image.image.open('rb') as image_file:
        image_bytes = image_file.read()

    existing_rows = []
    if csv_path.exists():
        with csv_path.open('r', encoding='utf-8-sig', newline='') as csv_file:
            existing_rows = list(csv.DictReader(csv_file))

    def _row_base_name(image_name: str) -> str:
        stem = Path(image_name).stem
        return re.sub(r'-[0-9a-f]{8}$', '', stem)

    kept_rows = []
    stale_image_names = []
    for row in existing_rows:
        row_image_name = (row.get(image_name_field) or '').strip()
        if not row_image_name:
            continue

        row_base = _row_base_name(row_image_name)
        if row_image_name == correction_image_name or row_base == safe_stem:
            stale_image_names.append(row_image_name)
            continue

        kept_rows.append(row)

    kept_rows.append({image_name_field: correction_image_name, text_field: corrected_text})

    with csv_path.open('w', encoding='utf-8', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=[image_name_field, text_field])
        writer.writeheader()
        writer.writerows(kept_rows)

    for stale_image_name in stale_image_names:
        stale_path = images_dir / stale_image_name
        if stale_path.exists() and stale_path.name != correction_image_name:
            stale_path.unlink()

    (images_dir / correction_image_name).write_bytes(image_bytes)
    return 1


def _get_recommended_training_setup():
    """Recommended one-click training configuration for this project."""

    dataset_profiles = ['iam_line']
    if _custom_csv_dataset_ready():
        dataset_profiles.insert(0, 'custom_csv')

    return {
        'dataset_profiles': dataset_profiles,
        'training_mode': 'balanced',
        'epochs': 2,
        'train_limit': 300,
        'eval_limit': 80,
        'train_batch_size': 1,
        'eval_batch_size': 1,
    }


def _run_recommended_training():
    """Run one-click recommended training in the background."""

    from .dataset_utils import prepare_handwriting_dataset

    recommended_setup = _get_recommended_training_setup()
    dataset_profiles = recommended_setup['dataset_profiles']
    total_profiles = len(dataset_profiles)

    write_training_status(
        status='running',
        progress=5,
        dataset_profile=dataset_profiles[0],
        training_mode=recommended_setup['training_mode'],
        message='Preparing recommended training datasets.',
        output_dir='',
        completed_runs=[],
    )

    completed_runs = []

    try:
        from .training_utils import train_handwriting_model

        for index, dataset_profile in enumerate(dataset_profiles, start=1):
            dataset_output_dir = settings.HANDWRITING_DATASET_PROFILES[dataset_profile]['output_dir']
            output_dir = _build_training_output_dir(dataset_profile)
            progress_start = int(((index - 1) / total_profiles) * 100)
            progress_span = max(int(100 / total_profiles), 1)

            write_training_status(
                status='running',
                progress=max(progress_start, 5),
                dataset_profile=dataset_profile,
                training_mode=recommended_setup['training_mode'],
                message=f'Preparing dataset {index} of {total_profiles}: {dataset_profile}.',
                output_dir=output_dir,
                completed_runs=completed_runs,
            )

            dataset_summary = prepare_handwriting_dataset(
                dataset_profile=dataset_profile,
                output_dir=dataset_output_dir,
            )
            write_training_status(
                status='running',
                progress=min(progress_start + 10, 95),
                dataset_profile=dataset_profile,
                dataset_id=dataset_summary['dataset_id'],
                training_mode=recommended_setup['training_mode'],
                output_dir=output_dir,
                completed_runs=completed_runs,
                message=f'Dataset prepared. Starting training {index} of {total_profiles}: {dataset_profile}.',
            )

            def _emit_progress(progress, message, *, current_profile=dataset_profile, current_output=output_dir, current_dataset_id=dataset_summary['dataset_id']):
                scaled_progress = progress_start + int((progress / 100) * progress_span)
                write_training_status(
                    status='running',
                    progress=min(scaled_progress, 99),
                    dataset_profile=current_profile,
                    dataset_id=current_dataset_id,
                    training_mode=recommended_setup['training_mode'],
                    output_dir=current_output,
                    completed_runs=completed_runs,
                    message=f'{message} ({index}/{total_profiles})',
                )

            training_summary = train_handwriting_model(
                dataset_dir=dataset_output_dir,
                base_model_dir=settings.OCR_LOCAL_MODEL_DIR,
                output_dir=output_dir,
                epochs=recommended_setup['epochs'],
                train_batch_size=recommended_setup['train_batch_size'],
                eval_batch_size=recommended_setup['eval_batch_size'],
                train_limit=recommended_setup['train_limit'],
                eval_limit=recommended_setup['eval_limit'],
                progress_callback=_emit_progress,
            )

            completed_runs.append(
                {
                    'dataset_profile': dataset_profile,
                    'dataset_id': dataset_summary['dataset_id'],
                    'output_dir': training_summary['output_dir'],
                    'train_samples': training_summary['train_samples'],
                    'validation_samples': training_summary['validation_samples'],
                    'epochs': training_summary['epochs'],
                }
            )
    except Exception as exc:
        write_training_status(
            status='failed',
            progress=100,
            dataset_profile=dataset_profiles[0],
            training_mode=recommended_setup['training_mode'],
            completed_runs=completed_runs,
            message=f'Training failed: {exc}',
        )
        return

    write_training_status(
        status='completed',
        progress=100,
        dataset_profile=dataset_profiles[-1],
        dataset_id=completed_runs[-1]['dataset_id'],
        training_mode=recommended_setup['training_mode'],
        output_dir=completed_runs[-1]['output_dir'],
        train_samples=completed_runs[-1]['train_samples'],
        validation_samples=completed_runs[-1]['validation_samples'],
        epochs=completed_runs[-1]['epochs'],
        completed_runs=completed_runs,
        message=f'Training completed successfully for {len(completed_runs)} dataset(s).',
    )


def training_status(request):
    """Return current one-click training status for polling."""

    return JsonResponse(read_training_status())


def home(request):
    """
    Display the upload form and show the uploaded image preview.
    OCR logic will be added later.
    """

    form = ImageUploadForm()
    training_form = TrainingForm()
    custom_training_form = CustomTrainingDatasetForm()
    correction_form = PredictionCorrectionForm()
    uploaded_image = None
    training_summary = read_training_status()
    latest_ocr_engine = request.session.get('latest_ocr_engine', 'smart')
    latest_extraction_mode = request.session.get('latest_extraction_mode', 'both')
    latest_target_type = request.session.get('latest_target_type', 'mixed')
    form.fields['ocr_engine'].initial = latest_ocr_engine
    form.fields['extraction_mode'].initial = latest_extraction_mode
    form.fields['target_type'].initial = latest_target_type

    if request.method == 'POST':
        action = request.POST.get('action', 'upload')

        if action == 'train':
            current_status = read_training_status()
            if current_status.get('status') == 'running':
                messages.warning(request, 'Training is already running. Please wait until it finishes.')
            else:
                write_training_status(
                    status='queued',
                    progress=0,
                    message='Training request received. Starting background task.',
                )
                worker = threading.Thread(target=_run_recommended_training, daemon=True)
                worker.start()
                messages.success(request, 'Training started. Progress will update below.')
            return redirect('home')
        elif action == 'save_correction':
            correction_form = PredictionCorrectionForm(request.POST)
            if correction_form.is_valid():
                uploaded_image = UploadedImage.objects.filter(id=correction_form.cleaned_data['image_id']).first()
                if not uploaded_image:
                    messages.error(request, 'Could not find that uploaded image to save the correction.')
                    return redirect('home')

                corrected_text = correction_form.cleaned_data['corrected_text']
                saved_count = _store_prediction_correction(uploaded_image, corrected_text)
                uploaded_image.user_corrected_text = corrected_text
                uploaded_image.predicted_text = corrected_text
                uploaded_image.correction_applied = True
                uploaded_image.added_to_training_set = saved_count > 0
                uploaded_image.save(
                    update_fields=[
                        'user_corrected_text',
                        'predicted_text',
                        'correction_applied',
                        'added_to_training_set',
                    ]
                )

                request.session['latest_uploaded_image_id'] = uploaded_image.id
                if correction_form.cleaned_data.get('start_training'):
                    current_status = read_training_status()
                    if current_status.get('status') != 'running':
                        write_training_status(
                            status='queued',
                            progress=0,
                            message='Correction saved. Starting deep-learning retraining with the updated dataset.',
                        )
                        worker = threading.Thread(target=_run_recommended_training, daemon=True)
                        worker.start()
                        messages.success(
                            request,
                            'Correction saved to the training dataset and deep-learning retraining started.',
                        )
                    else:
                        messages.success(
                            request,
                            'Correction saved to the training dataset. Training is already running, so the new sample will be used in the next run.',
                        )
                else:
                    messages.success(request, 'Correction saved to the training dataset.')
                return redirect('home')

            messages.error(request, 'Could not save the correction. Please review the form and try again.')
        elif action == 'add_training_samples':
            custom_training_form = CustomTrainingDatasetForm(request.POST, request.FILES)
            if custom_training_form.is_valid():
                try:
                    entries = _build_custom_training_entries(
                        custom_training_form.cleaned_data['images'],
                        custom_training_form.cleaned_data['texts'],
                        custom_training_form.cleaned_data['auto_segment'],
                    )
                    saved_count = _store_custom_training_entries(entries)
                except ValidationError as exc:
                    custom_training_form.add_error(None, exc.message)
                else:
                    messages.success(
                        request,
                        f'{saved_count} custom training sample(s) added. Start training to fine-tune the model with your handwriting.',
                    )
                    return redirect('home')
            messages.error(request, 'Could not save the custom training samples. Please correct the form and try again.')
        else:
            form = ImageUploadForm(request.POST, request.FILES)
            if form.is_valid():
                uploaded_image = form.save()
                extraction_mode = form.cleaned_data.get('extraction_mode', 'both')
                target_type = form.cleaned_data.get('target_type', 'mixed')
                raw_text, corrected_text, prediction_source, prediction_notes = extract_and_correct_text(
                    uploaded_image,
                    uploaded_image.ocr_engine,
                    extraction_mode=extraction_mode,
                    target_type=target_type,
                )
                uploaded_image.raw_ocr_text = raw_text
                uploaded_image.predicted_text = corrected_text
                uploaded_image.prediction_source = prediction_source
                uploaded_image.prediction_notes = prediction_notes
                uploaded_image.save(update_fields=['raw_ocr_text', 'predicted_text', 'prediction_source', 'prediction_notes'])
                request.session['latest_uploaded_image_id'] = uploaded_image.id
                request.session['latest_ocr_engine'] = uploaded_image.ocr_engine
                request.session['latest_extraction_mode'] = extraction_mode
                request.session['latest_target_type'] = target_type
                messages.success(request, 'Image uploaded and prediction generated successfully.')
                return redirect('home')
            messages.error(request, 'Upload failed. Please correct the form and try again.')

    latest_uploaded_image_id = request.session.get('latest_uploaded_image_id')
    if latest_uploaded_image_id:
        uploaded_image = UploadedImage.objects.filter(id=latest_uploaded_image_id).first()
        if uploaded_image:
            correction_form = PredictionCorrectionForm(
                initial={
                    'image_id': uploaded_image.id,
                    'corrected_text': uploaded_image.user_corrected_text or uploaded_image.predicted_text,
                    'start_training': True,
                }
            )

    recent_uploads = UploadedImage.objects.all()[:6]

    context = {
        'form': form,
        'training_form': training_form,
        'custom_training_form': custom_training_form,
        'correction_form': correction_form,
        'training_summary': training_summary,
        'recommended_training_setup': _get_recommended_training_setup(),
        'custom_dataset_ready': _custom_csv_dataset_ready(),
        'custom_dataset_sample_count': _count_custom_csv_samples(),
        'latest_ocr_engine': latest_ocr_engine,
        'latest_extraction_mode': latest_extraction_mode,
        'latest_target_type': latest_target_type,
        'uploaded_image': uploaded_image,
        'recent_uploads': recent_uploads,
    }
    return render(request, 'ocr_app/home.html', context)
