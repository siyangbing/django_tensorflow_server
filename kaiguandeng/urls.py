from django.urls import path

from kaiguandeng import views

urlpatterns = [
    path('t/', views.terminal, name='terminal'),
    path('b/', views.backstage, name='backstage'),
]
