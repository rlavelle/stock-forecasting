# Stock Forecasting 

## Preliminary Results
Mean Absolute Percentage Errors

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

### Simple Moving Average Graphs Examples (d=3)
![](https://github.com/rlavelle/stock-forecasting/blob/master/frac_change_forecasting/imgs/sma/SMA-3%20IBM%20forcasted%20vs%20actual%20stock%20prices%202004-09-13%20to%202005-01-22.png)
![](https://github.com/rlavelle/stock-forecasting/blob/master/frac_change_forecasting/imgs/sma/SMA-3%20AMZN%20forcasted%20vs%20actual%20stock%20prices%202015-01-02%20to%202016-01-02.png)
![](https://github.com/rlavelle/stock-forecasting/blob/master/frac_change_forecasting/imgs/sma/SMA-3%20GOOGL%20forcasted%20vs%20actual%20stock%20prices%202015-01-02%20to%202016-01-02.png)

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
