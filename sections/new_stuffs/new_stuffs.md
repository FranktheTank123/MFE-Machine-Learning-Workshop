# Learn Something New

## Long-term Forecasting in Machine Learning World
**Question**: Given a time-series $y_t$, What if we have want to forecast $H$ steps further in the future?

###Seasonal ARIMA?

* well-suited for short-term forecasts, not for longer term forecasts
* convergence of the autoregressive part

![](images/ARIMA_longterm.png)


### Let's use ML
Still assume **assume** $y_t$ follows some additive autoregressive models: 

$$y_{t+1} = f(y_t, ..., y_{t-n+1}) + \epsilon_t$$

* **Note I didn't assume stationarity here.** (Why?)
* $f(\cdot)$ can be any machine learning model with
    * $X = [y_t, ..., y_{t-n+1}]$
    * $Y = y_{t+1}$
* When $H=1$, any ML models **might** take care of.
* When $H>1$, things become more interesting. Three possible solutions presented here.

#### Solution 1: Iterated forecasting


#### Solution 2: $H$-step ahead forecasting

#### Solution 3: Multiple input multiple output (MIMO) models

### How to decide $n$?


## Hyperparameter Tuning

## Proper Backtesting

