from sklearn.ensemble import GradientBoostingRegressor
from model import *
from test import *

class GradBoost(Model):
    def __init__(self, params):
        super().__init__(params)
        self.n_estimators = params['n_estimators']
        self.loss = params['loss']
        self.lr = params['lr']
        self.subsample = params['subsample']
        self.max_depth = params['max_depth']
        self.d = params['d']
        self.sigma = params['sigma']
    
    def gen_model(self):
        self.model = GradientBoostingRegressor(n_estimators=self.n_estimators, 
                                               loss=self.loss, 
                                               learning_rate=self.lr,
                                               subsample=self.subsample,
                                               max_depth=self.max_depth,
                                               random_state=3)

if __name__ == '__main__':
    params = {
        'n_estimators':250,
        'loss':'huber',
        'lr':0.1,
        'subsample':0.9,
        'max_depth':10,
        'd':5,
        'sigma':1,
        'name':'AdaBoostRegressor'
    }

    test = Test(Model=GradBoost, params=params, tests=paper_tests, f='gb-paper-tests.json', plot=True)
    test.fixed_origin_tests(folder='gradient_boosting')

    test = Test(Model=GradBoost, params=params, tests=own_tests, f='gb-own-tests.json', plot=True)
    test.fixed_origin_tests(folder='gradient_boosting')

    test = Test(Model=GradBoost, params=params, tests=rolling_window_tests, f='gb-rolling-tests.json', plot=True)
    test.rolling_window_test(folder='gradient_boosting')