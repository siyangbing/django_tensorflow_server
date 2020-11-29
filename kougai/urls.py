from django.urls import path

from kougai import views

urlpatterns = [
    path('kg/', views.kougai, name='eval_img'),
    # path('b/', views.backstage, name='backstage'),
]