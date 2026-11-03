from django.urls import path
from predictor import views

urlpatterns = [
    path("", views.index, name="home"),
    path("predict/", views.predict_crop, name="predict_crop"),
    path("dashboard/", views.dashboard, name="dashboard"),
]
