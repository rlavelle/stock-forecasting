from sklearn.ensemble import RandomForestRegressor
from model import *
from test import *
import math
import time
import numpy as np

class RandomForest(Model):
    def __init__(self, params):
        super().__init__(params)
        self.n_estimators = params['n_estimators']
        self.max_depth = params['max_depth']
        self.max_samples = params['max_samples']
        self.d = params['d']
        self.sigma = params['sigma']
    
    def gen_model(self):
        self.model = RandomForestRegressor(n_estimators=self.n_estimators,
                                           max_depth=self.max_depth,
                                           max_samples=self.max_samples,
                                           random_state=3)

if __name__ == '__main__':
    params = {
        'n_estimators':1000,
        'max_depth':25,
        'max_samples':150,
        'd':5,
        'sigma':1,
        'name':'RandomForestRegressor'
    }

    # test = Test(Model=RandomForest, params=params, tests=paper_tests, f='rf-paper-tests.json', plot=True)
    # test.fixed_origin_tests(folder='random_forest')

    # test = Test(Model=RandomForest, params=params, tests=own_tests, f='rf-own-tests.json', plot=True)
    # test.fixed_origin_tests(folder='random_forest')

    start = time.time()
    test = Test(Model=RandomForest, params=params, tests=rolling_window_tests, f='rf-rolling-tests.json', plot=True)
    results = test.rolling_window_test(folder='random_forest')
    end = time.time()
    print(f'time elapsed: {(end-start)/60}')
    print(f'average R2 score: {np.average(list(results["R2"].values()))}')
    print(f'average MAPE: {np.average(list(results["MAPE"].values()))}')


