from django.urls import path

from kaiguandeng import views

urlpatterns = [
    path('kgd/', views.kaiguandeng, name='kaiguandeng'),
    # path('test/', views.test, name='test'),
]
