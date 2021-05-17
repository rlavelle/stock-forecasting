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
    pass