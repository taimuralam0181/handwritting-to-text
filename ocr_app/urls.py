from django.urls import path

from .views import home, training_status

urlpatterns = [
    path('', home, name='home'),
    path('training-status/', training_status, name='training_status'),
]
