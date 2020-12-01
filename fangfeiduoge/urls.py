from django.urls import path

from fangfeiduoge import views

urlpatterns = [
    path('ffdg/', views.fangfeiduoge, name='fangfeiduoge'),
    # path('t/', views.terminal, name='terminal'),
    # path('b/', views.backstage, name='backstage'),
]
