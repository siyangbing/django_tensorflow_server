from django.urls import path

from kaiguandeng import views

urlpatterns = [
    path('base64/', views.base64_test, name='base64_test'),
    # path('t/', views.terminal, name='terminal'),
    # path('b/', views.backstage, name='backstage'),
]
