from django.urls import path

from .views import ProjectLoginView, ProjectLogoutView, home, register, training_status

urlpatterns = [
    path('login/', ProjectLoginView.as_view(), name='login'),
    path('logout/', ProjectLogoutView.as_view(), name='logout'),
    path('register/', register, name='register'),
    path('', home, name='home'),
    path('training-status/', training_status, name='training_status'),
]
