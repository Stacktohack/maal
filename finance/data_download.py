import urllib
import pandas as pd
# import matplotlib.pyplot as plt
import scipy.optimize as spo
import os.path
import numpy as np
import gmail
from datetime import datetime
from pytz import timezone
import email_script


def fetch_data(symbol, yahoo_end_date): #Not in course. Adding (mgill)
    """ Downloads .csv files for <symbols> from Yahoo Finance and saves them in 'data' directory. These are later picked up by rese of the program."""

    '''url = "http://ichart.finance.yahoo.com/table.csv?s="+symbol+\
    "&amp;d=1&amp;e=1&amp;f=2016&amp;g=d&amp;a=8&amp;b=7&amp;c=2000&amp;ignore=.csv"
    '''
    time_frame = "m" # d -> daily, w -> weekly, m -> monthly.
    url = "http://real-chart.finance.yahoo.com/table.csv?s="+symbol+\
            "&a=01&b=01&c=2014" + yahoo_end_date + "&g="+time_frame+"+&ignore=.csv"

    urllib.urlretrieve(url, './data/{}.csv'.format(symbol))
    print "DEBUG: Downloading for "+symbol
    print "DEBUG: URL:"+url


def get_data_symbols(symbols, yahoo_end_date):
    # Choose stock symbols to read
    # symbols = ['XLY', 'XLF','XLU','XLP','XLE','XLV','XLB','XLK','XLI']

    for symbol in symbols:
        file_path = './data/{}.csv'.format(symbol)
        if os.path.exists(file_path) == False:
            fetch_data(symbol, yahoo_end_date) #Download csv for symbol loading.


def plot_df(frames):
    print "skip plot"
    """
    #for df in frames:
        df.plot()
    plt.show()
    """

def build_prices(symbols, start_date, end_date):
    print "Building prices dataset for ", symbols
    dates = pd.date_range(start_date, end_date)
    df_base = pd.DataFrame(index=dates)
    for symbol in symbols:
        df_symbol = pd.read_csv('./data/{}.csv'.format(symbol), index_col="Date", parse_dates=True,
                                usecols=['Date', 'Adj Close'], na_values=['nan'])
        df_symbol = df_symbol.rename(columns={'Adj Close': symbol})
        if symbol == 'SPY':
            how = 'inner'
        else:
            how = 'left'
        df_base = df_base.join(df_symbol, how=how)
        df_base.fillna(method='ffill', inplace="TRUE")
        df_base.fillna(method='bfill', inplace='TRUE')
        df_base.dropna()
        #plot_df([df_base])

    # plot_df([df_base])
    df_base.sort_index(ascending=True, inplace=True)
    #print df_base
    #print df_base.ix[0]
    return df_base


def calc_sharp_ratio(alloc_ratio, prices):
    print "Calculating sharp ratio"


def build_portfolio(prices, allocs, start_val):
    # print "Building portfolio for ", allocs
    normed = prices / prices.ix[0]
    alloced = normed * allocs
    position_val = alloced * start_val
    portfolio_val = position_val.sum(axis = 1)
    return portfolio_val, position_val


def get_daily_returns(df):
    daily_returns = df.copy()
    daily_returns[1:] = (df[1:] / df[:-1].values) - 1
    daily_returns.ix[0] = 0
    # plot_df([daily_returns])
    return daily_returns


def portfolio_statistics(portfolio_val):
    #print "Portfolio statistics - "
    daily_returns = get_daily_returns(portfolio_val)
    daily_returns = daily_returns[1:]
    cum_daily_rets = portfolio_val / portfolio_val[0] - 1
    stats = {}
    stats['cum_ret'] = portfolio_val[-1] / portfolio_val[0] - 1
    stats['avg_daily_ret'] = daily_returns.mean()
    stats['std_daily_ret'] = daily_returns.std()
    stats['sharp_ratio'] = stats['avg_daily_ret']/ stats['std_daily_ret']
    # print "portfolio_statistics - ", stats
    return stats


def f(X, prices, start_val):
    # print "X, start_val", X, start_val
    portfolio_val, position_val = build_portfolio(prices, X, start_val)
    stats = portfolio_statistics(portfolio_val)
    Y = stats['sharp_ratio']
    # print "sharp_ratio", Y
    Y *= -1
    return Y


def verify_prices(symbols, start_date, end_date, new_allocs, default_allocs, start_val):
    print "Verifying prices..."
    prices = build_prices(symbols, start_date, end_date)
    portfolio_val_max, position_val = build_portfolio(prices, new_allocs, start_val)
    # plot_df([position_val])
    portfolio_val_default, position_val = build_portfolio(prices, default_allocs, start_val)
    # plot_df([portfolio_val_max, portfolio_val_default])
    return portfolio_val_max, portfolio_val_default


def bbands(price, length=20, numsd=2):
    """ returns average, upper band, and lower band"""
    average = pd.rolling_mean(price.resample("1D", fill_method="ffill"), length)
    # plot_df([average])
    sd = pd.rolling_std(price.resample("1D", fill_method="ffill"), length)
    upper = average + (sd*numsd)
    lower = average - (sd*numsd)
    return  [average, upper, lower]


def optimize_portfolio(symbols, prices, allocs, start_val):
    portfolio_val_default, position_val = build_portfolio(prices, allocs, start_val)
    stats = portfolio_statistics(portfolio_val_default)

    bnds = tuple((0, 1) for x in allocs)
    cons = ({'type': 'eq', 'fun': lambda x: 1 - sum(x)})

    min_result = spo.minimize(f, allocs, args=(prices, start_val), method='SLSQP',
                              bounds=bnds, constraints=cons, options={'disp': True})
    new_allocs = min_result.x

    print "*******min_result**************"
    for i in range(0, len(symbols)):
        if min_result.x[i] > 0.01:
            print symbols[i], min_result.x[i]
    portfolio_val_max, position_val = build_portfolio(prices, min_result.x, start_val)
    # plot_df([portfolio_val_max, portfolio_val_default])

    return new_allocs
    # verify_prices(symbols, '2016-05-01', '2016-06-01', allocs, new_allocs, start_val)


def check_buying_point(symbol, df_base, tolerance):
    print "Checking buying Bollinger cross over"
    price = df_base.iloc[0, 0]
    lower_price = df_base.iloc[0, 3]
    days = 0
    if price > (1-tolerance) * lower_price and price < (1+tolerance) * lower_price:
        prev_price = df_base.iloc[0, 0]
        for i in range(1,100):
            price = df_base.iloc[i,0]
            if price < prev_price:
                days += 1
                prev_price = price
            else:
                break
    else:
        buy_miss = min( abs((1-tolerance) * lower_price - price) , abs((1+tolerance) * lower_price - price) )
        if buy_miss < price * tolerance:
            print "Buy missed by $", buy_miss

    if days >= 1:
        content = "Buying point of {0}, prices rising from {1} days by {2} points ".\
            format(symbol, days, price - prev_price)
        print content
        return content
    else:
        return ""


def check_selling_point(symbol, df_base, tolerance):
    print "Checking Bollinger selling cross over"
    price = df_base.iloc[0, 0]
    upper_price = df_base.iloc[0, 2]
    days = 0
    if price > (1-tolerance) * upper_price and price < (1+tolerance) * upper_price:
        prev_price = df_base.iloc[0, 0]
        for i in range(1,100):
            price = df_base.iloc[i,0]
            if price > prev_price:
                days += 1
                prev_price = price
            else:
                break

    else:
        sell_miss = min(abs((1-tolerance) * upper_price - price), abs((1+tolerance) * upper_price - price))
        if sell_miss < price * tolerance:
            print "Sell missed by $", sell_miss

    if days >= 1:
        content = "Selling point of {0}, prices dropping from {1} days by {2} points ".\
            format(symbol, days, prev_price - price)
        print content
        return content
    else:
        return ""


def plot_bolinger(symbols, start_date, end_date):
    tolerance = .01
    dates = pd.date_range(start_date, end_date)
    df_base = pd.DataFrame(index=dates)
    buy_content = ""
    sell_content = ""
    for symbol in symbols:
        print "Analysing ", symbol
        df_base = pd.DataFrame(index=dates)
        df_symbol = pd.read_csv('./data/{}.csv'.format(symbol), index_col="Date", parse_dates=True,
                                usecols=['Date', 'Adj Close'], na_values=['nan'])
        df_symbol = df_symbol.rename(columns={'Adj Close': symbol})
        df_base = df_base.join(df_symbol, how='inner')
        price_sets = bbands(df_base)
        labels = ['average', 'upper', 'lower']
        a = 0
        for df in price_sets:
            df = df.rename(columns={symbol: symbol + labels[a]})
            a += 1
            df_base = df_base.join(df, how='left')
        plot_df([df_base])
        buy_content = buy_content + check_buying_point(symbol, df_base, tolerance)
        sell_content = sell_content + check_selling_point(symbol, df_base, tolerance)
    email_content = "------BUY-------" + buy_content + "-----SELL------" + sell_content
    return email_content


def clean_up(folder):
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
                # elif os.path.isdir(file_path): shutil.rmtree(file_path)
        except Exception as e:
            print(e)
    print "Cleaned up data folder"
    return True


def main():
    symbols = ['IBM', 'KRX', 'FB', 'GE', 'AAPL', 'NVDA', 'NKE', 'GOOGL', 'TSLA', 'AMZN', 'MSFT', 'TWTR']
    # clean_up('data')
    #symbols = ['SPY', 'GOOGL', 'FB']
    #allocs = [0.3, 0.4, 0.3]
    allocs = np.ones(len(symbols)) * 1/len(symbols)
    allocs = list(allocs)
    start_val = 100

    start_date = '2015-01-01'
    chi = timezone('US/Central')
    chi_time = datetime.now(chi)
    end_date = chi_time.strftime('%Y-%m-%d')
    # end_date = '2016-10-21'
    # prices = build_prices(symbols, start_date, end_date)
    yahoo_end_date = chi_time.strftime('&d=%d&e=%m&f=%Y')

    get_data_symbols(symbols, yahoo_end_date)

    print end_date, yahoo_end_date
    #optimize_portfolio(symbols, prices, allocs, start_val)

    email_content = plot_bolinger(symbols, start_date, end_date)
    print "Email content", email_content
    gmail.send_mail(email_content)
    # email_script.mail_content(email_content)

if __name__ == "__main__":
    main()