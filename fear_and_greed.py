import pandas as pd
import requests
#import mplfinance as mpf
import yfinance as yf

url = 'https://production.dataviz.cnn.io/index/fearandgreed/graphdata'

headers = dict()
headers['user-agent'] = 'Mozilla/5.0 (Windows NT 6.3; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/101.0.4951.54 Safari/537.36'

req = requests.get(url, headers=headers)

if req.status_code == 200:
    # Antwort von Zeichenkette in Objekt umwandeln
    resp = req.json()
    # Sind die gewünschten Daten vorhanden?
    if 'fear_and_greed_historical' in resp.keys() and 'data' in resp['fear_and_greed_historical'].keys():
        # Daten in pandas DataFrame umwandeln
        df = pd.DataFrame(resp['fear_and_greed_historical']['data'])
        # Index korrekt aufbereiten
        df = df.set_index(pd.to_datetime(df['x'].apply(lambda x: x), unit='ms'))
        # OHLC Wert setzen
        df['Close'] = df['y']
        df['Open'] = df['Close']
        df['High'] = df['Close']
        df['Low'] = df['Close']
