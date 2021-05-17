from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from model import *
from test import *

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
    pass