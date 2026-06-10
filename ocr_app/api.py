import json
import threading
from functools import wraps

from django.conf import settings
from django.contrib.auth import authenticate
from django.contrib.auth.models import User
from django.core import signing
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

from .forms import ImageUploadForm
from .models import UploadedImage
from .services import detect_target_type, extract_and_correct_text
from .training_status import read_training_status, write_training_status
from .views import (
    _ensure_image_fingerprint,
    _find_saved_correction,
    _run_recommended_training,
    _store_prediction_correction,
)


TOKEN_SALT = 'ocr-app-api-token'
TOKEN_MAX_AGE = int(getattr(settings, 'OCR_API_TOKEN_MAX_AGE', 60 * 60 * 24 * 30))


def _json_body(request):
    if not request.body:
        return {}
    try:
        return json.loads(request.body.decode('utf-8'))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None


def _user_token(user):
    payload = {'user_id': user.pk, 'password': user.password[-12:]}
    return signing.dumps(payload, salt=TOKEN_SALT, compress=True)


def _user_from_token(token):
    try:
        payload = signing.loads(token, salt=TOKEN_SALT, max_age=TOKEN_MAX_AGE)
        user = User.objects.get(pk=payload['user_id'], is_active=True)
    except (signing.BadSignature, signing.SignatureExpired, User.DoesNotExist, KeyError, TypeError):
        return None

    if payload.get('password') != user.password[-12:]:
        return None
    return user


def api_auth_required(view):
    @wraps(view)
    def wrapped(request, *args, **kwargs):
        authorization = request.headers.get('Authorization', '')
        scheme, _, token = authorization.partition(' ')
        if scheme.lower() != 'bearer' or not token:
            return JsonResponse({'error': 'Bearer token is required.'}, status=401)

        user = _user_from_token(token.strip())
        if not user:
            return JsonResponse({'error': 'Token is invalid or expired.'}, status=401)

        request.api_user = user
        return view(request, *args, **kwargs)

    return wrapped


def _serialize_upload(uploaded_image, request):
    image_url = request.build_absolute_uri(uploaded_image.image.url) if uploaded_image.image else ''
    return {
        'id': uploaded_image.pk,
        'title': uploaded_image.title,
        'image_url': image_url,
        'ocr_engine': uploaded_image.ocr_engine,
        'prediction_source': uploaded_image.prediction_source,
        'prediction_notes': uploaded_image.prediction_notes,
        'raw_ocr_text': uploaded_image.raw_ocr_text,
        'predicted_text': uploaded_image.predicted_text,
        'user_corrected_text': uploaded_image.user_corrected_text,
        'correction_applied': uploaded_image.correction_applied,
        'added_to_training_set': uploaded_image.added_to_training_set,
        'uploaded_at': uploaded_image.uploaded_at.isoformat(),
    }


@csrf_exempt
@require_http_methods(['POST'])
def api_register(request):
    data = _json_body(request)
    if data is None:
        return JsonResponse({'error': 'Request body must be valid JSON.'}, status=400)

    username = str(data.get('username', '')).strip()
    password = str(data.get('password', ''))
    email = str(data.get('email', '')).strip()
    if not username or not password:
        return JsonResponse({'error': 'username and password are required.'}, status=400)
    if len(password) < 8:
        return JsonResponse({'error': 'Password must contain at least 8 characters.'}, status=400)
    if User.objects.filter(username__iexact=username).exists():
        return JsonResponse({'error': 'Username already exists.'}, status=409)

    user = User.objects.create_user(username=username, password=password, email=email)
    return JsonResponse(
        {
            'token': _user_token(user),
            'user': {'id': user.pk, 'username': user.username, 'email': user.email},
        },
        status=201,
    )


@csrf_exempt
@require_http_methods(['POST'])
def api_login(request):
    data = _json_body(request)
    if data is None:
        return JsonResponse({'error': 'Request body must be valid JSON.'}, status=400)

    user = authenticate(
        request,
        username=str(data.get('username', '')).strip(),
        password=str(data.get('password', '')),
    )
    if not user:
        return JsonResponse({'error': 'Invalid username or password.'}, status=401)

    return JsonResponse(
        {
            'token': _user_token(user),
            'expires_in': TOKEN_MAX_AGE,
            'user': {'id': user.pk, 'username': user.username, 'email': user.email},
        }
    )


@csrf_exempt
@api_auth_required
@require_http_methods(['POST'])
def api_ocr(request):
    form_data = request.POST.copy()
    form_data['ocr_engine'] = 'smart'
    form_data.setdefault('extraction_mode', 'both')
    form_data.setdefault('target_type', 'auto')
    form = ImageUploadForm(form_data, request.FILES)
    if not form.is_valid():
        return JsonResponse({'error': 'Invalid upload.', 'fields': form.errors.get_json_data()}, status=400)

    uploaded_image = form.save(commit=False)
    uploaded_image.user = request.api_user
    uploaded_image.save()

    extraction_mode = form.cleaned_data.get('extraction_mode', 'both')
    target_selection = form.cleaned_data.get('target_type', 'auto')
    target_type = detect_target_type(uploaded_image) if target_selection == 'auto' else target_selection
    saved_correction = _find_saved_correction(uploaded_image)

    if saved_correction:
        corrected_text = saved_correction.user_corrected_text or saved_correction.predicted_text
        raw_text = corrected_text
        prediction_source = 'memory'
        prediction_notes = 'Used your saved correction for this image.'
    else:
        raw_text, corrected_text, prediction_source, prediction_notes = extract_and_correct_text(
            uploaded_image,
            uploaded_image.ocr_engine,
            extraction_mode=extraction_mode,
            target_type=target_type,
        )

    if target_selection == 'auto':
        prediction_notes = f'Auto-detected target type: {target_type}. {prediction_notes}'.strip()

    uploaded_image.raw_ocr_text = raw_text
    uploaded_image.predicted_text = corrected_text
    uploaded_image.prediction_source = prediction_source
    uploaded_image.prediction_notes = prediction_notes
    uploaded_image.save(
        update_fields=[
            'raw_ocr_text',
            'predicted_text',
            'prediction_source',
            'prediction_notes',
            'image_fingerprint',
        ]
    )

    response = _serialize_upload(uploaded_image, request)
    response['target_type'] = target_type
    response['used_saved_correction'] = prediction_source == 'memory'
    return JsonResponse(response, status=201)


@api_auth_required
@require_http_methods(['GET'])
def api_uploads(request):
    uploads = UploadedImage.objects.filter(user=request.api_user)[:50]
    return JsonResponse({'results': [_serialize_upload(item, request) for item in uploads]})


@api_auth_required
@require_http_methods(['GET'])
def api_upload_detail(request, image_id):
    uploaded_image = UploadedImage.objects.filter(pk=image_id, user=request.api_user).first()
    if not uploaded_image:
        return JsonResponse({'error': 'Upload not found.'}, status=404)
    return JsonResponse(_serialize_upload(uploaded_image, request))


@csrf_exempt
@api_auth_required
@require_http_methods(['POST'])
def api_correct_upload(request, image_id):
    uploaded_image = UploadedImage.objects.filter(pk=image_id, user=request.api_user).first()
    if not uploaded_image:
        return JsonResponse({'error': 'Upload not found.'}, status=404)

    data = _json_body(request)
    if data is None:
        return JsonResponse({'error': 'Request body must be valid JSON.'}, status=400)
    corrected_text = str(data.get('corrected_text', '')).strip()
    if not corrected_text:
        return JsonResponse({'error': 'corrected_text is required.'}, status=400)

    saved_count = _store_prediction_correction(uploaded_image, corrected_text)
    _ensure_image_fingerprint(uploaded_image)
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
    return JsonResponse(_serialize_upload(uploaded_image, request))


@api_auth_required
@require_http_methods(['GET'])
def api_training_status(request):
    return JsonResponse(read_training_status())


@csrf_exempt
@api_auth_required
@require_http_methods(['POST'])
def api_start_training(request):
    if not request.api_user.is_staff:
        return JsonResponse({'error': 'Admin access is required.'}, status=403)

    current_status = read_training_status()
    if current_status.get('status') == 'running':
        return JsonResponse({'error': 'Training is already running.', 'status': current_status}, status=409)

    write_training_status(status='queued', progress=0, message='API training request received.')
    worker = threading.Thread(target=_run_recommended_training, daemon=True)
    worker.start()
    return JsonResponse(read_training_status(), status=202)
