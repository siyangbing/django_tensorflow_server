from django.urls import path

from fangfeizuocang import views

urlpatterns = [
    path('ffzc/', views.fangfeizuocang, name='fangfeizuocang'),
    # path('b/', views.backstage, name='backstage'),
]