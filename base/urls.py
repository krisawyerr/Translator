from django.contrib import admin
from django.urls import path
from . import views

urlpatterns = [
    path('', views.translate, name='translate'),
    path('clear/', views.clearChat, name='clearChat'),
]
