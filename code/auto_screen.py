import sqlite3 as sq
import pandas as pd
import yahoo_fin.stock_info as si
from os.path import join
from datetime import datetime

class StockDataFetcher:
    def __init__(self, file_path, rs_stocks, start_date, current_date, db_path):
        self.file_path = file_path
        self.rs_stocks = rs_stocks
        self.start_date = start_date
        self.current_date = current_date
        self.db_path = db_path
        self.historical_datas = {}
        
    def load_data(self):
        self.tv_data = pd.read_csv(join(self.file_path, self.tv_file))
        tickers_data = pd.read_csv(join(self.file_path, self.rs_stocks))
        self.tickers_list = tickers_data['Ticker'].tolist()
        
    def fetch_historical_data(self):
        for idx, symbol in enumerate(self.tickers_list, start=1):
            try:
                self.historical_datas[symbol] = si.get_data(symbol, start_date=self.start_date, end_date=self.current_date, index_as_date=True)
                print(f"Fetching data for {symbol}: {idx}/{len(self.tickers_list)}")
            except Exception as e:
                print(f"Error fetching data for {symbol}: {e}")
                continue
                
    def save_to_db(self):
        conn = sq.connect(self.db_path)
        try:
            df_db = pd.read_sql_query("SELECT * from price_action", conn)
        except sq.Error as e:  # Especificar a exceção
            print(f"SQL Error: {e}")
            df_db = pd.DataFrame(columns=['date', 'ticker', 'open', 'high', 'low', 'close', 'adjclose', 'volume'])
        
        all_data = pd.concat(self.historical_datas.values())
        all_data = all_data.reset_index().rename(columns={'index': 'date'})
        combined_data = pd.concat([df_db, all_data]).drop_duplicates(subset=['date', 'ticker'], keep='last')
        combined_data.to_sql('price_action', conn, if_exists='replace', index=False)
        conn.close()

class StockDataAnalyzer:
    def __init__(self, db_path, tv_file, rs_stocks):
        self.db_path = db_path
        self.tv_file = tv_file
        self.rs_stocks = rs_stocks
        
    @staticmethod    
    def minervini_criteria_basic(row):
        criteria = [
            row['Price'] > 5,
            row['Price'] < 300,
            row['Price'] > row['SMA50'],
            row['SMA50'] > row['SMA150'],
            row['SMA150'] > row['SMA200'],
            row['Percentile'] >= 70,
            row['Avg_Vol_90d'] > 100000,
        ]
        return 'yes' if all(criteria) else 'no'
    
    @staticmethod
    def minervini_criteria_full(row):
        criteria = [
            row['Price'] > 5,
            row['Price'] < 300,
            row['Price'] > row['SMA50'],
            row['SMA50'] > row['SMA150'],
            row['SMA150'] > row['SMA200'],
            row['Price'] >= row['52Week-Low'] * 1.25,
            row['Price'] >= row['52Week-High'] * 0.85,
            row['Percentile'] >= 70,
            row['SMA200_1M_Trend'] == 1,
            row['Avg_Vol_90d'] > 100000,
            row['Avg_Vol_10d'] <= row['Avg_Vol_30d']
        ]
        return 'yes' if all(criteria) else 'no'
    
    @staticmethod
    def power_trend(row, prev_power_trend):
        # Entry criteria
        if (
            row['lows_above_EMA21'] >= 10 and
            row['EMA21_above_SMA50'] >= 5 and
            row['SMA50_uptrend'] == 1 and
            row['close_above_open'] == 1
        ):
            return 'in'
        
        # Exit criteria
        if prev_power_trend == 'in' and (
            row['EMA21'] < row['SMA50'] or
            (row['close'] < row['high_last_week'] * 0.9 and row['close'] < row['SMA50'])
        ):
            return 'out'
        
        return prev_power_trend

    def load_data_from_db(self):
        conn = sq.connect(self.db_path)
        self.price_action = pd.read_sql_query("SELECT * FROM price_action", conn)
        conn.close()
        self.rs_stocks = pd.read_csv(join(self.file_path, self.rs_stocks))
        self.mkt_cap_data = pd.read_csv(join(self.file_path, self.tv_file))
    
    def generate_df(self):
        self.screen_df =  pd.merge(self.price_action, self.mkt_cap_data, left_on='ticker', right_on='Ticker', how='left')
        self.screen_df = pd.merge(self.screen_df, rs_stocks, left_on='Ticker', right_on='Ticker', how='left')

    def calculate_metrics(self):
        # Add the 'Price' column representing the latest close for each stock
        self.screen_df ['Price'] = self.screen_df .groupby('ticker')['close'].transform('last')

        # Calculate the 50-day Simple Moving Average (SMA50) for each stock.
        self.screen_df ['SMA50'] = self.screen_df .groupby('ticker')['close'].transform(lambda x: x.rolling(window=50).mean())

        # Calculate the 21-day Exponential Moving Average (EMA21) for each stock.
        self.screen_df ['EMA21'] = self.screen_df .groupby('ticker')['close'].transform(lambda x: x.ewm(span=21, adjust=False).mean())

        # Calculate the 150-day Simple Moving Average (SMA150) for each stock.
        self.screen_df ['SMA150'] = self.screen_df .groupby('ticker')['close'].transform(lambda x: x.rolling(window=150).mean())

        # Calculate the 200-day Simple Moving Average (SMA200) for each stock.
        self.screen_df ['SMA200'] = self.screen_df .groupby('ticker')['close'].transform(lambda x: x.rolling(window=200).mean())

        # Calculate the 90-day Average Volume for each stock.
        self.screen_df ['Avg_Vol_90d'] = self.screen_df .groupby('ticker')['volume'].transform(lambda x: x.rolling(window=90).mean())

        # Calculate the 30-day Average Volume for each stock.
        self.screen_df ['Avg_Vol_30d'] = self.screen_df .groupby('ticker')['volume'].transform(lambda x: x.rolling(window=30).mean())

        # Calculate the 10-day Average Volume for each stock.
        self.screen_df ['Avg_Vol_10d'] = self.screen_df .groupby('ticker')['volume'].transform(lambda x: x.rolling(window=10).mean())

        # Calculate the 52-week high (highest closing price in the last 252 trading days) for each stock.
        self.screen_df ['52Week-High'] = self.screen_df .groupby('ticker')['close'].transform(lambda x: x.rolling(window=252).max())

        # Calculate the 52-week low (lowest closing price in the last 252 trading days) for each stock.
        self.screen_df ['52Week-Low'] = self.screen_df .groupby('ticker')['close'].transform(lambda x: x.rolling(window=252).min())

        # Calculate the highest high and lowest low over the last 10 days
        self.screen_df ['10D_high'] = self.screen_df .groupby('ticker')['high'].transform(lambda x: x.rolling(window=10).max())
        self.screen_df ['10D_low'] = self.screen_df .groupby('ticker')['low'].transform(lambda x: x.rolling(window=10).min())

        # Calculate the range as a percentage of the highest high
        self.screen_df ['10D_range_pct'] = (self.screen_df ['10D_high'] - self.screen_df ['10D_low']) / self.screen_df ['10D_high'] * 100

        # Identify low-risk entries
        self.screen_df ['Low-Risk_Entry'] = (self.screen_df ['10D_range_pct'] < 8).astype(int)

        # Calculate the SMA200 trend over the last month
        self.screen_df ['SMA200_1M_Trend'] = self.screen_df .groupby('ticker')['SMA200'].transform(lambda x: x.diff(periods=21)).gt(0).astype(int)

        # Calculate criteria for PowerTrend
        self.screen_df ['lows_above_EMA21'] = self.screen_df .groupby('ticker').apply(lambda x: (x['low'] > x['EMA21']).rolling(window=10).sum()).reset_index(level=0, drop=True)
        self.screen_df ['EMA21_above_SMA50'] = self.screen_df .groupby('ticker').apply(lambda x: (x['EMA21'] > x['SMA50']).rolling(window=5).sum()).reset_index(level=0, drop=True)
        self.screen_df ['SMA50_uptrend'] = self.screen_df .groupby('ticker')['SMA50'].transform(lambda x: x.diff().rolling(window=21).sum()).gt(0).astype(int)
        self.screen_df ['close_above_open'] = (self.screen_df ['close'] > self.screen_df ['open']).astype(int)
        self.screen_df ['high_last_week'] = self.screen_df .groupby('ticker')['high'].transform(lambda x: x.rolling(window=5).max())

    def apply_criteria(self):
        # Apply the Minervini criteria to each row
        self.screen_df ['Minervini_basic'] = self.screen_df .apply(self.minervini_criteria_basic, axis=1)
        self.screen_df ['Minervini_full'] = self.screen_df .apply(self.minervini_criteria_full, axis=1)

        # Apply the PowerTrend criteria to each row
        prev_power_trend = 'out'
        power_trends = []

        for _, row in self.screen_df .iterrows():
            current_power_trend = self.power_trend(row, prev_power_trend)
            power_trends.append(current_power_trend)
            prev_power_trend = current_power_trend
            
        # Apply the Minervini criteria to each row
        self.screen_df ['PowerTrend'] = power_trends

    def filter_screen_df(self):
        metrics = ['Ticker', 'EMA21', 'SMA50', 'SMA150', 'SMA200', 'Avg_Vol_10d', 'Avg_Vol_30d', 'Avg_Vol_90d', '52Week-High', 
                '52Week-Low', 'Minervini_basic', 'Minervini_full', 'PowerTrend', 'Low-Risk_Entry', 'Market Capitalization',
                'Upcoming Earnings Date', 'Industry', 'Percentile', '1 Month Ago', '3 Months Ago', '6 Months Ago']
        self.screen_df = self.screen_df.drop_duplicates(subset='Ticker', keep='last')[metrics]
        return self.screen_df

    def save_to_excel_and_db(self):
        today = datetime.now().strftime('%Y-%m-%d')
        file_name = f'../output/screen_{today}.xlsx'
        self.screen_df.to_excel(file_name, engine='xlsxwriter')

        current_date = datetime.now().strftime('%d-%m-%Y')
        table_name = f"screen-{current_date}"

        conn = sq.connect(self.db_path)
        self.screen_df.to_sql(table_name, conn, if_exists='replace', index=False)
        conn.close()

if __name__ == "__main__":
    # Initialize and run StockDataFetcher
    file_path = 'C:/Users/rodri/Apps/trading/src'
    tv_file = 'TV.csv'
    tickers_file = 'Copy of NYSE_Stocks.xlsx'
    rs_stocks = 'rs_stocks.csv'
    current_date = datetime.today()
    start_date = current_date
    db_path = 'C:/Users/rodri/Apps/trading/market_data.db'
    
    fetcher = StockDataFetcher(file_path, tv_file, tickers_file, current_date)
    fetcher.load_data()
    fetcher.fetch_historical_data()
    fetcher.save_to_db()

    # Initialize and run StockDataAnalyzer
    analyzer = StockDataAnalyzer()
    analyzer.load_data_from_db()
    analyzer.generate_df()
    analyzer.calculate_metrics()
    analyzer.apply_criteria()
    analyzer.filter_screen_df()
    analyzer.save_to_excel_and_db()