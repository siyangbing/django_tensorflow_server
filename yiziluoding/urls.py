from django.urls import path

from yiziluoding import views

urlpatterns = [
    path('yzld/', views.yiziluoding, name='yiziluoding'),
    # path('b/', views.backstage, name='backstage'),
]