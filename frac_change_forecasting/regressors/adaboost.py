from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from model import *
from test import *
import math
import time
import numpy as np

class AdaBoost(Model):
    def __init__(self, params):
        super().__init__(params)
        self.n_estimators = params['n_estimators']
        self.max_depth = params['max_depth']
        self.d = params['d']
        self.sigma = params['sigma']
    
    def gen_model(self):
        self.model = AdaBoostRegressor(base_estimator=DecisionTreeRegressor(max_depth=self.max_depth),
                                       n_estimators=self.n_estimators,
                                       random_state=3)

if __name__ == '__main__':
    params = {
        'n_estimators':250,
        'max_depth':10,
        'd':5,
        'sigma':1,
        'name':'AdaBoostRegressor'
    }

    # test = Test(Model=AdaBoost, params=params, tests=paper_tests, f='ada-paper-tests.json', plot=True)
    # test.fixed_origin_tests(folder='adaboost')

    # test = Test(Model=AdaBoost, params=params, tests=own_tests, f='ada-own-tests.json', plot=True)
    # test.fixed_origin_tests(folder='adaboost')

    start = time.time()
    test = Test(Model=AdaBoost, params=params, tests=rolling_window_tests, f='ada-rolling-tests.json', plot=True)
    results = test.rolling_window_test(folder='adaboost')
    end = time.time()
    print(f'time elapsed: {(end-start)/60}')
    print(f'average R2 score: {np.average(list(results["R2"].values()))}')
    print(f'average MAPE: {np.average(list(results["MAPE"].values()))}')