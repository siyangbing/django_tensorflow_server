from django.urls import path

from shiziluoding import views

urlpatterns = [
    path('szld/', views.shiziluoding, name='shiziluoding'),
    # path('b/', views.backstage, name='backstage'),
]