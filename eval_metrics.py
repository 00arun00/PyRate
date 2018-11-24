import numpy as np

class Metric(object):
    '''
    Abstarct class for evaluation metrics
    '''
    @staticmethod
    def score(Y_hat,Y):
        '''
        retruns the score based on the eval metric

        Args:
            :Y_hat (numpy.ndarray): Predicted values
            :Y (numpy.ndarray): Labels

        Returns:
            :error (float): Score
        '''
        raise NotImplementedError('Abstract class')

    def __call__(self,Y_hat,Y):
        return self.score(Y_hat,Y)

    def __repr__(self):
        if hasattr(self,'eval_metric'):
            return f'{self.eval_metric}'
        else:
            raise NotImplementedError('pretty print not implemented')

class RMSE(Metric):
    '''
    Root Mean Square Error
    '''
    def __init__(self):
        self.eval_metric = "RMSE"

    @staticmethod
    def score(Y_hat,Y):
        '''
        retruns the score based on root mean square

        Args:
            :Y_hat (numpy.ndarray): Predicted values
            :Y (numpy.ndarray): Labels

        Returns:
            :error (float): Score based on RMSE
        '''
        error = np.sqrt(np.mean((Y_hat-Y)**2))
        return error

class MSE(Metric):
    '''
    Mean Square Error
    '''
    def __init__(self):
        self.eval_metric = "MSE"

    @staticmethod
    def score(Y_hat,Y):
        '''
        retruns the score based on root mean square

        Args:
            :Y_hat (numpy.ndarray): Predicted values
            :Y (numpy.ndarray): Labels

        Returns:
            :error (float): Score based on MSE
        '''
        error = np.mean((Y_hat-Y)**2)
        return error

class SSE(Metric):
    '''
    Sum of Square Error
    '''
    def __init__(self):
        self.eval_metric = "SSE"

    @staticmethod
    def score(Y_hat,Y):
        '''
        retruns the score based on sum of squared error

        Args:
            :Y_hat (numpy.ndarray): Predicted values
            :Y (numpy.ndarray): Labels

        Returns:
            :error (float): Score based on SSE
        '''
        error = np.sum((Y_hat-Y)**2)
        return error

class MAE(Metric):
    '''
    Mean Absolute Error
    '''
    def __init__(self):
        self.eval_metric = "MAE"

    @staticmethod
    def score(Y_hat,Y):
        '''
        retruns the score based on mean absolute error

        Args:
            :Y_hat (numpy.ndarray): Predicted values
            :Y (numpy.ndarray): Labels

        Returns:
            :error (float): Score based on MAE
        '''
        error = np.mean(np.abs((Y_hat-Y)**2))
        return error

#aliases
rmse = RMSE 
mse = MSE
sse = SSE
mae = MAE
