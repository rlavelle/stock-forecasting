from sklearn.ensemble import RandomForestRegressor
from model import *
from test import *

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

    test = Test(Model=RandomForest, params=params, tests=paper_tests, f='ghmm-paper-tests.json', plot=True)
    test.fixed_origin_tests(folder='random_forest')