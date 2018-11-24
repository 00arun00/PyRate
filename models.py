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

    def _predict_single_(self,x):
        '''
        Predicts single
        '''
        return np.random.uniform(0,5)

    def predict(self,X):
        '''
        Predict Function

        Args:
            :X (numpy.ndarray): User, Item pairs to predict rating on

        Retruns:
            :predicted_rating (numpy.ndarray): predicted ratings
        '''
        predicted_rating = np.array(list(map(self._predict_single_,X))).reshape(-1,1)
        return  predicted_rating

    def set_eval_metric(self,metric):
        '''
        Sets evaluation metric

        Args:
            :metric (Metric): evaluation metric used
        '''
        assert isinstance(metric,Metric)
        self.eval_metric = metric

    def score(self,X,Y):
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
        score = self.eval_metric(y_pred,Y)

        return score

    def fit(self,X,Y):
        '''
        Fits model to the data
        '''
        raise NotImplementedError('This is an abstract class')

class Baseline(Model):
    '''
    Baseline model
    '''
    def __init__(self):
        self.model_name = 'Baseline'
        self.alpha = 0
        self.fit_flag = False

    def __call__(self,X):
        '''
        redirect to predict
        '''
        return self.predict(X)

    def _predict_single_(self,X):
        if not self.fit_flag:
            warnings.warn(f'Model currently not fit, predicting 0 for all')
        return self.alpha


    def fit(self,X,Y):
        '''
        Fits model to the data
        '''
        self.alpha = np.mean(Y)
        self.fit_flag = True
