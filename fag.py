import requests, csv, json, urllib
import pandas as pd
import time
from fake_useragent import UserAgent
from datetime import datetime

json_filename = 'feargreed.json'

def request_fear_and_greed():
    BASE_URL = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata/"
    START_DATE = '2020-09-19'
    END_DATE = '2024-06-12'
    ua = UserAgent()

    headers = {
    'User-Agent': ua.random,
    }

    r = requests.get(BASE_URL + START_DATE, headers = headers)
    data = r.json()

    with open(json_filename, 'w') as json_file:
        json.dump(data, json_file, indent=4)  # indent=4 for pretty forma

"""
fng_data = pd.read_csv('fear-greed.csv', usecols=['Date', 'Fear Greed'])
fng_data['Date'] = pd.to_datetime(fng_data['Date'], format='%Y-%m-%d')  

fng_data.set_index('Date', inplace=True)
missing_dates = []
all_dates = (pd.date_range(fng_data.index[0], END_DATE, freq='D'))
for date in all_dates:
	if date not in fng_data.index:
		missing_dates.append(date)
		#print(date)
		fng_data.loc[date] = [0]
fng_data.sort_index(inplace=True)


for data in ((data['fear_and_greed_historical']['data'])):
	x = int(data['x'])
	x = datetime.fromtimestamp(x / 1000).strftime('%Y-%m-%d')
	y = int(data['y'])
	fng_data.at[x, 'Fear Greed'] = y
#currently any days that do not have data points from cnn are filled with zeros, uncomment the following line to backfill
#fng_data['Fear Greed'].replace(to_replace=0, method='bfill')

fng_data.to_pickle('all_fng.pkl')
fng_data.to_csv('all_fng_csv.csv')
"""

def load_fear_and_greed():
    with open(json_filename, 'r') as json_file:
        data = json.load(json_file)
    #print(data)

    nd = {}

    columns = ['fear_and_greed_historical', 'market_momentum_sp500', 'market_momentum_sp125', 'stock_price_strength',
                'stock_price_breadth', 'put_call_options', 'market_volatility_vix', 'market_volatility_vix_50', 
                'junk_bond_demand', 'safe_haven_demand']

    for column in columns:
        d1 = data.get(column, {})
        d2 = d1.get('data', {})
        for row in d2:
            x = row.get('x', None)
            y = row.get('y', None)
            r = row.get('rating', None)
            dt = str(datetime.fromtimestamp(x / 1000).date())
            drow = nd.get(dt, None)
            if drow is None:
                drow = {}
                nd[dt] = drow
            drow[f'{column}_val'] = y
            drow[f'{column}_rating'] = r
            pass
        pass
    pass

    json_filename_simple = 'feargreed_simple.json'
    with open(json_filename_simple, 'w') as json_file:
        json.dump(nd, json_file, indent=4)  # indent=4 for pretty forma



load_fear_and_greed()