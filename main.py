import pickle

import requests

import stock_model_generator
from datetime import date, timedelta

base_url = "https://smart-trade-ai.herokuapp.com"


def post_predictions(stk, predictions, yesterdayPrice):
    day = date.today()
    stockAbv = stk["abbreviation"]
    for prediction in predictions:
        day = day + timedelta(days=1)

        d = day.day
        m = day.month
        y = day.year


        request_body = {
            "devKey": "avVNeJfwOyMGfvm1QYDBQcocBlfNeE7V",
            "stockAbbrev": stockAbv,
            "day": d,
            "month": m,
            "year": y,
            "closingPrice": lastPrice * (1 + (prediction/100))
        }

        response = requests.post(base_url+"/api/Prediction/Dev", json=request_body)
        print(response.status_code)


if __name__ == '__main__':
    stocks = requests.get("https://smart-trade-ai.herokuapp.com/api/Stock").json()
    for stock in stocks:
        file, lastPrice = stock_model_generator.generate_model(stock["abbreviation"])

        with open(file, 'rb') as pkl:
            mod = pickle.load(pkl)
            res = mod.predict(10)
            post_predictions(stock, res, lastPrice)
