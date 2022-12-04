import os
import joblib
from django.apps import AppConfig
from django.conf import settings


class ApiConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'main'
    MODEL_FILE = os.path.join(settings.MODELS, "rd_forest_clf.pkl.pkl")
    model = joblib.load(MODEL_FILE)
