import sqlite3 as sq
import pandas as pd
import yahoo_fin.stock_info as si
from os.path import join
from datetime import datetime

class StockDataFetcher:
    def __init__(self, src_path, tv_file, tickers_file, start_date, end_date):
        self.src_path = src_path
        self.tv_file = tv_file
        self.tickers_file = tickers_file
        self.start_date = start_date
        self.end_date = end_date
        self.tickers_list = []
        self.df = pd.DataFrame()
        self.historical_datas = {}
        
    def load_data(self):
        self.tv_data = pd.read_csv(join(self.src_path, self.tv_file))
        tickers_csv = pd.read_excel(join(self.src_path, self.tickers_file))
        self.tickers_list = tickers_csv['Symbol'].tolist()
        
    def create_dataframe(self):
        self.df = pd.DataFrame({'Ticker': self.tickers_list})
        self.df = pd.merge(self.df, self.tv_data[['Ticker', 'Market Capitalization', 'Upcoming Earnings Date']], on='Ticker', how='left')
        self.df.rename(columns={'Market Capitalization': 'Mkt Cap', 'Upcoming Earnings Date': 'Next Earnings Date'}, inplace=True)
        
    def fetch_historical_data(self):
        for idx, symbol in enumerate(self.tickers_list, start=1):
            try:
                self.historical_datas[symbol] = si.get_data(symbol, start_date=self.start_date, end_date=self.end_date, index_as_date=True)
                print(f"Fetching data for {symbol}: {idx}/{len(self.tickers_list)}")
            except Exception as e:
                print(f"Error fetching data for {symbol}: {e}")
                continue
                
    def save_to_db(self):
        conn = sq.connect('../market_data.db')
        try:
            df_db = pd.read_sql_query("SELECT * from price_action", conn)
        except sq.Error as e:  # Especificar a exceção
            print(f"SQL Error: {e}")
            df_db = pd.DataFrame(columns=['date', 'ticker', 'open', 'high', 'low', 'close', 'adjclose', 'volume'])
        
        all_data = pd.concat(self.historical_datas.values())
        all_data = all_data.reset_index().rename(columns={'index': 'date'})
        combined_data = pd.concat([df_db, all_data]).drop_duplicates(subset=['date', 'ticker'], keep='last')
        
        combined_data.to_sql('price_action', conn, if_exists='replace', index=False)
        self.df.to_sql('mkt_cap_next_earnings', conn, if_exists='replace', index=False)
        conn.close()

class StockDataAnalyzer:
    def __init__(self):
        self.price_action = None
        self.mkt_cap_data = None
        self.screen_df = None
        
    @staticmethod    
    def minervini_criteria_basic(row):
        criteria = [
            row['Price'] > 5,
            row['Price'] < 300,
            row['Price'] > row['SMA50'],
            row['SMA50'] > row['SMA150'],
            row['SMA150'] > row['SMA200'],
            row['RS Rating'] >= 70,
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
            row['RS Rating'] >= 70,
            row['RS_Rating_6W_Trend'] == 1,
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
        conn = sq.connect('../market_data.db')
        self.price_action = pd.read_sql_query("SELECT * FROM price_action", conn)
        self.mkt_cap_data = pd.read_sql_query("SELECT * FROM mkt_cap_next_earnings", conn)
        conn.close()

    def calculate_metrics(self):
        # Add the 'Price' column representing the latest close for each stock
        self.price_action['Price'] = self.price_action.groupby('ticker')['close'].transform('last')

        # Calculate the 50-day Simple Moving Average (SMA50) for each stock.
        self.price_action['SMA50'] = self.price_action.groupby('ticker')['close'].transform(lambda x: x.rolling(window=50).mean())

        # Calculate the 21-day Exponential Moving Average (EMA21) for each stock.
        self.price_action['EMA21'] = self.price_action.groupby('ticker')['close'].transform(lambda x: x.ewm(span=21, adjust=False).mean())

        # Calculate the 150-day Simple Moving Average (SMA150) for each stock.
        self.price_action['SMA150'] = self.price_action.groupby('ticker')['close'].transform(lambda x: x.rolling(window=150).mean())

        # Calculate the 200-day Simple Moving Average (SMA200) for each stock.
        self.price_action['SMA200'] = self.price_action.groupby('ticker')['close'].transform(lambda x: x.rolling(window=200).mean())

        # Calculate the 90-day Average Volume for each stock.
        self.price_action['Avg_Vol_90d'] = self.price_action.groupby('ticker')['volume'].transform(lambda x: x.rolling(window=90).mean())

        # Calculate the 30-day Average Volume for each stock.
        self.price_action['Avg_Vol_30d'] = self.price_action.groupby('ticker')['volume'].transform(lambda x: x.rolling(window=30).mean())

        # Calculate the 10-day Average Volume for each stock.
        self.price_action['Avg_Vol_10d'] = self.price_action.groupby('ticker')['volume'].transform(lambda x: x.rolling(window=10).mean())

        # Calculate the 52-week high (highest closing price in the last 252 trading days) for each stock.
        self.price_action['52Week-High'] = self.price_action.groupby('ticker')['close'].transform(lambda x: x.rolling(window=252).max())

        # Calculate the 52-week low (lowest closing price in the last 252 trading days) for each stock.
        self.price_action['52Week-Low'] = self.price_action.groupby('ticker')['close'].transform(lambda x: x.rolling(window=252).min())

        # Calculate the price performance over the last 252 trading days
        self.price_action['12M Return'] = self.price_action.groupby('ticker')['close'].pct_change(periods=252)

        # Calculate the RS Rating based on the 12M Return percentile rank
        self.price_action['RS Rating'] = self.price_action['12M Return'].rank(pct=True) * 100

        # Calculate the highest high and lowest low over the last 5 days
        self.price_action['10D_high'] = self.price_action.groupby('ticker')['high'].transform(lambda x: x.rolling(window=10).max())
        self.price_action['10D_low'] = self.price_action.groupby('ticker')['low'].transform(lambda x: x.rolling(window=10).min())

        # Calculate the range as a percentage of the highest high
        self.price_action['10D_range_pct'] = (self.price_action['10D_high'] - self.price_action['10D_low']) / self.price_action['10D_high'] * 100

        # Identify low-risk entries
        self.price_action['Low-Risk_Entry'] = (self.price_action['10D_range_pct'] < 8).astype(int)

        # Calculate the RS Rating trend over the last 6 weeks and SMA200 trend over the last month
        self.price_action['RS_Rating_6W_Trend'] = self.price_action.groupby('ticker')['RS Rating'].transform(lambda x: x.diff(periods=30)).gt(0).astype(int)
        self.price_action['SMA200_1M_Trend'] = self.price_action.groupby('ticker')['SMA200'].transform(lambda x: x.diff(periods=21)).gt(0).astype(int)

        # Calculate criteria for PowerTrend
        self.price_action['lows_above_EMA21'] = self.price_action.groupby('ticker').apply(lambda x: (x['low'] > x['EMA21']).rolling(window=10).sum()).reset_index(level=0, drop=True)
        self.price_action['EMA21_above_SMA50'] = self.price_action.groupby('ticker').apply(lambda x: (x['EMA21'] > x['SMA50']).rolling(window=5).sum()).reset_index(level=0, drop=True)
        self.price_action['SMA50_uptrend'] = self.price_action.groupby('ticker')['SMA50'].transform(lambda x: x.diff().rolling(window=21).sum()).gt(0).astype(int)
        self.price_action['close_above_open'] = (self.price_action['close'] > self.price_action['open']).astype(int)
        self.price_action['high_last_week'] = self.price_action.groupby('ticker')['high'].transform(lambda x: x.rolling(window=5).max())

    def apply_criteria(self):
        # Apply the Minervini criteria to each row
        self.price_action['Minervini_basic'] = self.price_action.apply(self.minervini_criteria_basic, axis=1)
        self.price_action['Minervini_full'] = self.price_action.apply(self.minervini_criteria_full, axis=1)

        # Apply the PowerTrend criteria to each row
        prev_power_trend = 'out'
        power_trends = []

        for _, row in self.price_action.iterrows():
            current_power_trend = self.power_trend(row, prev_power_trend)
            power_trends.append(current_power_trend)
            prev_power_trend = current_power_trend
            
        # Apply the Minervini criteria to each row
        self.price_action['PowerTrend'] = power_trends

    def create_screen_df(self):
        metrics = ['ticker', 'EMA21', 'SMA50', 'SMA150', 'SMA200', 'Avg_Vol_10d', 'Avg_Vol_30d', 'Avg_Vol_90d', '52Week-High', '52Week-Low', 'Minervini_basic', 'Minervini_full', 'PowerTrend', 'Low-Risk_Entry']
        self.screen_df = self.price_action.drop_duplicates(subset='ticker', keep='last')[metrics]
        self.screen_df = pd.merge(self.screen_df, self.mkt_cap_data[['Ticker', 'Mkt Cap', 'Next Earnings Date']], left_on='ticker', right_on='Ticker', how='left')
        self.screen_df.drop('Ticker', axis=1, inplace=True)
        return self.screen_df

    def save_to_excel_and_db(self):
        today = datetime.now().strftime('%Y-%m-%d')
        file_name = f'../output/screen_{today}.xlsx'
        self.screen_df.to_excel(file_name, engine='xlsxwriter')

        current_date = datetime.now().strftime('%d-%m-%Y')
        table_name = f"screen-{current_date}"

        conn = sq.connect('../market_data.db')
        self.screen_df.to_sql(table_name, conn, if_exists='replace', index=False)
        conn.close()

if __name__ == "__main__":
    # Initialize and run StockDataFetcher
    src_path = '../src'
    tv_file = 'america_brazil_2023-08-23_64209.csv'
    tickers_file = 'Copy of NYSE_Stocks'
    start_date = datetime(2023, 8, 22)
    end_date = datetime(2023, 8, 23)
    
    fetcher = StockDataFetcher(src_path, tv_file, tickers_file, start_date, end_date)
    fetcher.load_data()
    fetcher.create_dataframe()
    fetcher.fetch_historical_data()
    fetcher.save_to_db()

    # Initialize and run StockDataAnalyzer
    analyzer = StockDataAnalyzer()
    analyzer.load_data_from_db()
    analyzer.calculate_metrics()
    analyzer.apply_criteria()
    analyzer.create_screen_df()
    analyzer.save_to_excel_and_db()
