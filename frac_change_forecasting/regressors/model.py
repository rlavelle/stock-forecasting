from abc import ABC, abstractmethod
from fastquant import get_stock_data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from scipy.ndimage.filters import gaussian_filter

class Model(ABC):

    # params should be a dict of your parameters that you want to pass to the model
    # name should be a string (used for saving results)
    # params dict *must* include {'name':name}
    def __init__(self, params):
        self.model = None
        self.name = params['name']
    
    # wrapper model function for collecting fastquant data
    def get_data(self, ticker, start_date, end_date):
        return get_stock_data(ticker, start_date, end_date)

    # plotting function for price prediction
    def plot_prices(self, preds, title, folder):
        # generate predicted closing prices
        pred_close = []
        closes = self.test_data['close'].values
        opens = self.test_data['open'].values[1:]
        for i,pred in enumerate(preds):
            if i == 0:
                pred_close.append(pred*self.train_data['close'].values[-1]+self.train_data['close'].values[-1])
            else:
                pred_close.append(pred*closes[i-1]+closes[i-1])
        truth = self.test_data['close'].values[1:]

        fig, ax = plt.subplots(figsize=(15,5))
        ax.set_title(title)
        time = range(len(preds))
        ax.plot(time,truth,color='tab:blue',marker='s',markersize=2,linestyle='-',linewidth=1,label='actual')
        ax.plot(time,pred_close,color='tab:red',marker='s',markersize=2,linestyle='-',linewidth=1,label='preds')
        ax.set_xlabel('time')
        ax.set_ylabel('stock price ($)')
        ax.set_xticks(np.arange(0,len(preds)+10,10))
        ax.set_xlim(0,len(preds)+10)
        ax.xaxis.grid(True,ls='--')
        ax.yaxis.grid(True,ls='--')
        ax.legend()
        plt.savefig(f'../imgs/{folder}/{title}')

    # plotting function for fractional change 
    def plot_change(self, preds, actual, title, folder):
        fig, ax = plt.subplots(figsize=(15,5))
        ax.set_title(title)
        time = range(len(preds))
        ax.plot(time,actual,color='tab:blue',marker='s',markersize=2,linestyle='-',linewidth=1,label='actual')
        ax.plot(time,preds,color='tab:red',marker='s',markersize=2,linestyle='-',linewidth=1,label='preds')
        ax.set_xlabel('time')
        ax.set_ylabel('stock price ($)')
        ax.set_xticks(np.arange(0,len(preds)+10,10))
        ax.set_xlim(0,len(preds)+10)
        ax.xaxis.grid(True,ls='--')
        ax.yaxis.grid(True,ls='--')
        ax.legend()
        plt.savefig(f'../imgs/{folder}/{title}')

    # function to get error of the model based on preds and true values
    def mean_abs_percent_error(self, y_pred, y_true):
        return (1.0)/(len(y_pred))*((np.abs(y_pred-y_true)/np.abs(y_true))*100).sum()

    # percent change for data
    def data_prep(data):
        return data['close'].pct_change().iloc[1:].values

    # training function for the model, create the model, train it, and store in self.model
    def train(self, train_data):
        # save for later
        self.train_data = train_data

        # prep the data / pre process
        self.train_obs = self.data_prep(train_data)
        self.train_obs = gaussian_filter(self.train_obs, sigma=self.sigma)
        
        # build the x as the observation from (O_i,...,O_i+d)
        # y is O_i+d
        x_train, y_train = [],[]
        for i in range(self.d, len(self.train_obs)):
            x_train.append(self.train_obs[i-self.d:i])
            y_train.append(self.train_obs[i])

        x_train,y_train = np.array(x_train),np.array(y_train)
        y_train = np.reshape(y_train, (*y_train.shape,1))

        self.model.fit(x_train, y_train)
    
    # prediction function for the model, return the preds and y_true given the test data
    def predict(self, test_data):
        # add last row of training data to testing data
        last = self.train_data.iloc[-1].to_dict()
        row = pd.DataFrame(last, index=[0])
        row['dt'] = None
        self.test_data = self.test_data.reset_index()
        self.test_data = pd.concat([row,self.test_data], ignore_index=True)

        # convert the testing data and smooth
        test_obs = self.data_prep(self.test_data)
        test_labels = test_obs.copy()
        test_obs = gaussian_filter(test_obs, sigma=self.sigma)

        # add the last training observations to test observations for full test set predicting
        test_obs = np.concatenate((self.train_obs[-self.d:], test_obs), axis=0)

        # build the x as the observation from (O_i,...,O_i+d)
        # y is O_i+d
        x_test, y_test = [],[]
        index = 0
        for i in range(self.d, len(test_obs)):
            x_test.append(test_obs[i-self.d:i])
            y_test.append(test_labels[index])
            index += 1

        x_test,y_test = np.array(x_test),np.array(y_test)
        y_test = np.reshape(y_test, (*y_test.shape,1))

        # predict testing data
        preds = self.model.predict(x_test)

        return x_test,preds,y_test
        
    # generate the model and store in self.model
    @abstractmethod
    def gen_model(self):
        pass