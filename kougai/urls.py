from django.urls import path

from kougai import views

urlpatterns = [
    path('kg/', views.kougai, name='kougai'),
    # path('b/', views.backstage, name='backstage'),
]