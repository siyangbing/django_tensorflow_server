from django.urls import path

from tongdianzuocang import views

urlpatterns = [
    path('tdzc/', views.tongdianzuocang, name='tongdianzuocang'),
    # path('b/', views.backstage, name='backstage'),
]