from django.urls import path

from .api import (
    api_correct_upload,
    api_login,
    api_ocr,
    api_register,
    api_start_training,
    api_training_status,
    api_upload_detail,
    api_uploads,
)
from .views import ProjectLoginView, ProjectLogoutView, home, register, training_status

urlpatterns = [
    path('api/register/', api_register, name='api_register'),
    path('api/login/', api_login, name='api_login'),
    path('api/ocr/', api_ocr, name='api_ocr'),
    path('api/uploads/', api_uploads, name='api_uploads'),
    path('api/uploads/<int:image_id>/', api_upload_detail, name='api_upload_detail'),
    path('api/uploads/<int:image_id>/correction/', api_correct_upload, name='api_correct_upload'),
    path('api/training/status/', api_training_status, name='api_training_status'),
    path('api/training/start/', api_start_training, name='api_start_training'),
    path('login/', ProjectLoginView.as_view(), name='login'),
    path('logout/', ProjectLogoutView.as_view(), name='logout'),
    path('register/', register, name='register'),
    path('', home, name='home'),
    path('training-status/', training_status, name='training_status'),
]
