import pickle
import numpy as np
import pandas as pd
import requests
from pmdarima import auto_arima
from statsmodels.tsa.stattools import adfuller
from datetime import date

def generate_model(value):


    URL = ("https://api.nasdaq.com/api/quote/{}/historical?assetclass=stocks&fromdate=2011-05-22&limit=9999&todate=" + date.today().strftime("%Y-%m-%d")).format(
        value)
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.76 Safari/537.36',
        "Upgrade-Insecure-Requests": "1", "DNT": "1",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8", "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate"}

    response = requests.get(URL, headers=headers)

    df = pd.DataFrame(response.json()['data']["tradesTable"]["rows"])
    df['date'] = df['date'].replace("/", "-")
    df['close'] = pd.to_numeric(df['close'].str[1:], downcast="float")
    con = df['date']
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df['year'] = df.index.year
    df['month'] = df.index.month
    df['day'] = df.index.day
    df.sample(5, random_state=0)
    test = df[1029:]
    train = df[:1028]


    def test_stationarity(timeseries):
        rolmean = timeseries.rolling(12).mean()
        rolstd = timeseries.rolling(12).std()
        adft = adfuller(timeseries, autolag='AIC')
        output = pd.Series(adft[0:4],
                           index=['Test Statistics', 'p-value', 'No. of lags used', 'Number of observations used'])
        for key, values in adft[4].items():
            output['critical value (%s)' % key] = values



    test_stationarity(train['close'])

    train_log = np.log(train['close'])
    test_log = np.log(test['close'])
    moving_avg = train_log.rolling(24).mean()

    train_log_moving_avg_diff = train_log - moving_avg
    train_log_moving_avg_diff.dropna(inplace=True), test_stationarity(train_log_moving_avg_diff)
    train_log_diff = train_log - train_log.shift(1)
    test_stationarity(train_log_diff.dropna())
    model = auto_arima(train_log, trace=True, error_action='ignore', suppress_warnings=True)
    model_fit = model.fit(train_log)
    forecast = model.predict(n_periods=len(test))
    forecast = pd.DataFrame(forecast, index=test_log.index, columns=['Prediction'])

    model_name = value + str(date.today()) + 'model.pkl'

    with open(model_name, 'wb') as pkl:
        pickle.dump(model_fit, pkl)

    return model_name

    # with open(value + str(date.today()) + 'model.pkl', 'rb') as pkl:
    #     mod = pickle.load(pkl)
    #     print(mod.predict(100))
    #
    # from math import sqrt
    # from sklearn.metrics import mean_squared_error
    #
    # rms = sqrt(mean_squared_error(test_log, forecast))
    # print("RMSE: ", rms)
