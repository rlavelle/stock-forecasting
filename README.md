# P556FinalProject

## Notes for Dr. Williamson and Junyi

Work for Section 3.1 "Models for Fractional Change Prediction" can be found in `frac_change_forecasting`.
- Within `hmms`
    - `gmmhmm_base.py` contains the GMMHMM model.
    - `gaussian_hmm.py` contains the GHMM model.
    - `backtesting.ipynb`,  `gmmhmm_backtest.py` both contain the backtesting test for the GHMM.
    - `gmmhmm_slow.py` contains the origional work before the multithreading speed increase.
    - `gmmhmm_close_as_open.py` contains the experiment of using the gmmhmm model as true forecasting (using the predicted closing price as the next days opening price).
    - `gmmhmm_vol.py` contains the experiment for using volume as an additional dimension of the observation vector.
    - `gmmhmm_grid_search.py` contains the experiment for grid searching to find the optimal hyperparameters for the model. 
    - `data_explore.ipynb` is a data exploration notebook which birthed the idea for gaussian smoothing and sampling candidates from the fractional change distribution of a given stock.
    - `hmm_notebook.ipynb` is a notebook for experimenting with the hmm library to get the model working.
    - `simple_moving_average.py` is a baseline test model that just uses the simple moving average of the stock.
    - `test.py` is a streamlined testing file for the fixed origin and rolling window tests.
    - `model.py` is an abstract model class that allows for streamlined model creation and streamlined testing.
- Within `rnns`
    - `test.py` again contains the streamlined testing files and `model.py` again is the abstract model class which allows for streamlined model creation and testing. 
    - the remaining files contain the structure for all the LSTM models presented in section 3.1.

The `results` folder contains subfolders which hold the MAPE results in json files for the different tests. And the `imgs` folder contains the graphs for all of the models built in section 3.1, along with a random folder which can be ignored (contains random graphs from other experiments).

Work for Section 3.2 "Models for Raw Close Price Prediction" can be found in `raw_close_forecasting_rnns`.  
  - Within this folder:  
    - `lstm_X.py` contain the models we built.  
    - `X_train_predictor.py` contain the train/predict methods used on the models.  
    - `test.py` is the main driving code for running and testing our models  
    - Any other `*test*.py` files are entry points for test.py.  
    - `playgound.py` is a file used to test random bits of code and can be ignored.  
    - The `old` folder can be ignored. It contains early works that we've moved away from.  
    - The `results` folder contains the graphs for each of the tests organized by specific test name.  
      - The `excessive` folder contains many more graphs of tests we ran but realized was unnecessary.  

For both folders, `frac_change_forecasting` and `raw_close_forecasting_rnns`, you can find files with "\_backtest" in them. These files contain the code that uses the models for backtesting. Results from our tests can be found in the `results` folder under the parent folder. Either in the `Fastquant` or `Backtest Results` folders.

Work for our sentiment anaylsis proof of concept can be found in `sentiment`.

* Within `sentiment_analysis`
  * `driver.py` is a python script the drives the training and testing of the DNN and CNN networks used in the sentiment analysis of the tweet dataset.
  * `graphs` contains the learning curves for the networks tested.
  * `models` contains python files describing general network structure for DNN and CNN types.
  * `util/utils.py` contains utility functions that aid in the training of the neural networks such as training, evaluation, and iterator creation
  * `scraper` contains a script for scraping and pickling tweets from Twitter using the tweepy API.
  * `preprocessing` contains the scripts used for cleaning scraped tweets and generating csv files from them.
  * `data` contains the scraped tweets and csv files used in training and testing network architectures.

* `model_exploration.ipynb` is a playground notebook used for experimenting with the data, and attempting to find a correlation between previous day sentiment and next day fractional change.
* `sentiment_model.py` is a wrapper class for attempting to predict next day fractional change from previous days sentiment
* `sentiment_trend.py` is an experiment for predicting up and down trends based on previous days sentiment
* `tweet_collector.py` is a wrapper class for the twitter API, it also uses the flair library to classify tweets as positive and negative as they are collected.
* `twitter_playground.ipynb` is a notebook that was used for experimentation to get the tweet scraping working
* the `graphs` folder contains a few graphs of experimenting with using `sentiment_model.py` as a predictor for a stocks next day closing price

## Requirements

Make sure you are using python3.6 (highest version of python tensorflow works with). Create a virtual environment by running:

```
python3.6 -m venv proj_env
```

Once the environment is created run:

```
source proj_env/bin/activate
```

Then go ahead an install the requirements:

```
python3.6 -m pip install --upgrade pip
python3.6 -m pip install -r requirements.txt
```

To run it with a jupyter notebook:

```
python3.6 -m ipykernel install --user --name=proj_env
```
