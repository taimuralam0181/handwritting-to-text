from pathlib import Path
import threading

from django.contrib import messages
from django.conf import settings
from django.http import JsonResponse
from django.shortcuts import redirect, render

from .dataset_utils import prepare_handwriting_dataset
from .forms import ImageUploadForm, TrainingForm
from .models import UploadedImage
from .services import extract_and_correct_text
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
    uploaded_image = None
    training_summary = read_training_status()

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
        else:
            form = ImageUploadForm(request.POST, request.FILES)
            if form.is_valid():
                uploaded_image = form.save()
                raw_text, corrected_text = extract_and_correct_text(
                    uploaded_image,
                    uploaded_image.ocr_engine,
                )
                uploaded_image.raw_ocr_text = raw_text
                uploaded_image.predicted_text = corrected_text
                uploaded_image.save(update_fields=['raw_ocr_text', 'predicted_text'])
                request.session['latest_uploaded_image_id'] = uploaded_image.id
                messages.success(request, 'Image uploaded and prediction generated successfully.')
                return redirect('home')
            messages.error(request, 'Upload failed. Please correct the form and try again.')

    latest_uploaded_image_id = request.session.get('latest_uploaded_image_id')
    if latest_uploaded_image_id:
        uploaded_image = UploadedImage.objects.filter(id=latest_uploaded_image_id).first()

    recent_uploads = UploadedImage.objects.all()[:6]

    context = {
        'form': form,
        'training_form': training_form,
        'training_summary': training_summary,
        'recommended_training_setup': _get_recommended_training_setup(),
        'uploaded_image': uploaded_image,
        'recent_uploads': recent_uploads,
    }
    return render(request, 'ocr_app/home.html', context)
