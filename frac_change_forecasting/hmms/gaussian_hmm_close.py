import sys
from test import *
from model import Model
from fastquant import get_stock_data
from hmmlearn import hmm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import time
from multiprocessing import Pool


class GHMM(Model):
    def __init__(self,params):
        super().__init__(params)
        self.n_components = params['n_components']
        self.algorithm = params['algorithm']
        self.n_iter = params['n_iter']
        self.d = params['d']
    
    def train(self, train_data):
        self.train_data = train_data
        self.train = train_data['close'].values
        self.train_obs = self.data_prep(train_data)
        self.model = hmm.GaussianHMM(n_components=self.n_components,
                                     algorithm=self.algorithm,
                                     n_iter=self.n_iter)

        self.model.fit(self.train_obs)
    
    def predict(self,test_data):
        # save for returning at the end
        test_close_prices = test_data['close'].values

        # add last training point ot testing data, first point is NaN
        last = self.train_data.iloc[-1].to_dict()
        row = pd.DataFrame(last, index=[0])
        row['dt'] = None
        test_data = test_data.reset_index()
        test_data = pd.concat([row,test_data], ignore_index=True)
        test_obs = self.data_prep(test_data).values

        # use a latency of d days. So observations start as training data
        observed = self.train_obs.iloc[-self.d:].values
        
        # list to hold predictions
        preds = []

        # sample 10k points from the fracChange distribution of the training data as our possible next days
        np.random.seed(seed=47)
        mean = self.train_obs['fracChange'].mean()
        std = self.train_obs['fracChange'].std()
        change = np.random.normal(loc=mean, scale=std, size=10000)
        
        # loop through all points we want to test
        for i in range(len(test_obs)):

            observations = [np.array([c]) for c in change]
            
            # compute all log likelihoods w/ their observations in parallel
            jump = int(len(observations)/20)
            with Pool(processes=20) as pool:
                results = pool.starmap(self.log_lik_calc, [(observed, observations[i:i+jump]) for i in range(0,len(observations),jump)])

            best = {'obs':None, 'log_lik':-math.inf}
            for log_liks in results:
                for obs,log_lik in log_liks:
                    if log_lik > best['log_lik']:
                        best['obs'],best['log_lik'] = obs,log_lik

            # stack the actual observation on so we can predict next day
            # drop the first thing in observed to shift our latency window `d`
            observed = np.vstack((observed,test_obs[i]))
            observed = observed[1:]

            # calculate the close value from best using the previous days close price instead of next days opening
            if i == 0:
                pred_close = best['obs'][0]*self.train[-1]+self.train[-1]
            else:
                pred_close = best['obs'][0]*test_close_prices[i-1]+test_close_prices[i-1]
            preds.append(pred_close)

            print(f'{i+1}/{len(test_data)}',end='\r',flush=True)
        print('DONE')
        return preds,test_close_prices
    
    def log_lik_calc(self, observed, observations):
        log_liks = []
        for o in observations:
            obs = np.vstack((observed,o))
            log_lik = self.model.score(obs)
            log_liks.append((o,log_lik))

        return log_liks
    
    def data_prep(self, data):
        df = pd.DataFrame(data=None, columns=['fracChange'])
        df['fracChange'] = data['close'].pct_change().iloc[1:]

        return df


if __name__ == "__main__":
    params = {'n_components': 2, 
              'algorithm': 'map', 
              'n_iter': 100, 
              'd': 5,
              'name':'GHMM-CLOSE'}
    
    # ghmm = GHMM(params=params)
    # train = ghmm.get_data('AAPL', '2020-01-01', '2021-01-01')
    # print(ghmm.data_prep(train))
    
    print('testing best found parameters paper tests')
    test = Test(Model=GHMM, params=params, tests=paper_tests, f='ghmm-close-paper-tests.json', plot=True)
    test.fixed_origin_tests()

    print('testing best found parameters own tests')
    test = Test(Model=GHMM, params=params, tests=own_tests, f='ghmm-close-own-tests.json', plot=True)
    test.fixed_origin_tests()

    print('testing')
    test = Test(Model=GHMM, params=params, tests=rolling_window_tests, f='ghmm-close-rolling-tests.json', plot=True)
    test.rolling_window_test()
