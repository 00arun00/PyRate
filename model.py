import numpy as np
import warnings
from eval_metrics import Metric

class Model(object):
    '''
    Recomender  System model to be used

    **Note**
    This is a base class and cannot be used to make predictions
    '''
    def __call__(self,X):
        '''
        redirect to predict
        '''
        return self.predict(X)

    def __repr__(self):
        '''
        pretty print
        '''
        if hasattr(self,'model_name'):
            return f'{self.model_name}'
        else:
            return 'Not implemented'

    def predict(self,X):
        '''
        Predict Function

        **note**
        this is an abstract class it jsut predicsts random values

        Args:
            :X (numpy.ndarray): User, Item pairs to predict rating on

        Retruns:
            :predicted_rating (numpy.ndarray): predicted ratings
        '''
        warnings.warn("usng abstract class")
        predicted_rating = np.random.uniform(0,5,len(X)).reshape(-1,1)
        return  predicted_rating

    def set_eval_metric(self,metric):
        '''
        Sets evaluation metric

        Args:
            :metric (Metric): evaluation metric used
        '''
        assert isinstance(metric,Metric)
        self.eval_metric = metric

    def score(self,X,Y,metric='RMSE'):
        '''
        Predicts the score based on set eval metric

        Args:
            :X (numpy.ndarray): Input
            :Y (numpy.ndarray): Labels

        Retruns:
            :score (float):  score based on the selected eval metric
        '''
        y_pred = self.predict(X)
        if not hasattr(self,'eval_metric'):
            raise KeyError("Please add eval_metric")
        score = self.eval_metrics(y_pred,Y)

        return score
