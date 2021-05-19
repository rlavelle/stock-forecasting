# Stock Forecasting 
Attempting to predict next day closing prices of securities using a variety of models. 

The `frac_change_forecasting` contains 
- `hmms` which has all of the continuos hidden markov model variations, along with a simple moving average model.
- `imgs` contains the result graphs for all models in the `frac_change_forecasting` folder
- `regressors` contains the various regression models used for prediction
- `results` contains all the results in json files for the proposed models, regression results have both `R2` scores and `MAPE` scores
- `rnns` contains various recurrent neural network models 

The `gauss_smoothing` folder contains experiment work for using gaussian smoothing in an attempt to avoid overfitting when using LSTMs.

The `trading` folder contains scratch work that will be used to automate the sending of emails when a trading strategy is decided upon.


# Preliminary Results

## Mean Absolute Percentage Errors (MAPE)

|        |               | Regressor |                   | Simple Moving Average |
|--------|:-------------:|-----------|-------------------|-----------------------|
| Ticker | Random Forest | AdaBoost  | Gradient Boosting | D=3                   |
| AMZN   | 1.022         | 0.986     | 0.955             | 1.249                 |
| MSFT   | 1.024         | 1.007     | 1.002             | 1.175                 |
| GOOGL  | 1.242         | 1.188     | 1.179             | 1.394                 |
| DPZ    | 1.289         | 1.279     | 1.250             | 1.501                 |
| DIS    | 1.040         | 1.018     | 1.020             | 1.168                 |
| TMO    | 1.095         | 1.062     | 1.048             | 1.353                 |
| Avg    | 1.119         | 1.090     | 1.076             | 1.307                 |

# Graph Examples

## Simple Moving Average (d=3)
![](https://github.com/rlavelle/stock-forecasting/blob/master/frac_change_forecasting/imgs/sma/SMA-3%20IBM%20forcasted%20vs%20actual%20stock%20prices%202004-09-13%20to%202005-01-22.png)
![](https://github.com/rlavelle/stock-forecasting/blob/master/frac_change_forecasting/imgs/sma/SMA-3%20GOOGL%20forcasted%20vs%20actual%20stock%20prices%202015-01-02%20to%202016-01-02.png)

## Gradient Boosting Reegression
![](https://github.com/rlavelle/stock-forecasting/blob/master/frac_change_forecasting/imgs/gradient_boosting/GradientBoostRegressor%20IBM%20forcasted%20vs%20actual%20fractional%20price%20change%202004-09-13%20to%202005-01-22.png)
![](https://github.com/rlavelle/stock-forecasting/blob/master/frac_change_forecasting/imgs/gradient_boosting/GradientBoostRegressor%20IBM%20forcasted%20vs%20actual%20stock%20price%20(%24)%202004-09-13%20to%202005-01-22.png)
![](https://github.com/rlavelle/stock-forecasting/blob/master/frac_change_forecasting/imgs/gradient_boosting/GradientBoostRegressor%20GOOGL%20forcasted%20vs%20actual%20fractional%20price%20change%202015-01-02%20to%202016-01-02.png)
![](https://github.com/rlavelle/stock-forecasting/blob/master/frac_change_forecasting/imgs/gradient_boosting/GradientBoostRegressor%20GOOGL%20forcasted%20vs%20actual%20stock%20price%20(%24)%202015-01-02%20to%202016-01-02.png)


# Requirements

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
