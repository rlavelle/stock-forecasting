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
    pass