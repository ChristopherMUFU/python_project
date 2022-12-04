import numpy as np
import pandas as pd
from .apps import *
from rest_framework.views import APIView
from rest_framework.response import Response
from sklearn.preprocessing import StandardScaler, RobustScaler

variables = [
            'n_tokens_title', 
            'n_tokens_content', 
            'average_token_length', 
            'global_rate_negative_words', 
            'kw_min_avg', 'num_self_hrefs', 
            'n_unique_tokens', 'kw_max_min', 
            'num_hrefs', 'kw_avg_max', 'kw_max_avg', 
            'global_subjectivity', 'rate_positive_words',
            'title_sentiment_polarity', 
            'self_reference_min_shares', 
            'avg_positive_polarity'
        ]

def variables_post_config(data, variables):
    return [[data[x] for x in variables]]

def to_scale(X, scaler):
    return scaler.fit_transform(X)

class Prediction(APIView):
    def post(self, request):
        data = request.data
        X = variables_post_config(data, variables)
        X_scaled = to_scale(X, RobustScaler())
        print(X_scaled)
        rd_forest_clf = ApiConfig.model
        #predict using independent variables
        PredictionMade = rd_forest_clf.predict(X_scaled)
        response_dict = {"Predicted popularity": PredictionMade}
        print(response_dict)
        return Response(response_dict, status=200)