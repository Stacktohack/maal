import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mko

from zipline.utils.factory import load_bars_from_yahoo
end = pd.Timestamp.utcnow()
start = end - 2500 * pd.tseries.offsets.BDay()

data = load_bars_from_yahoo(stocks=['IBM', 'GLD', 'XOM', 'AAPL',
                                    'MSFT', 'TLT', 'SHY'],
                            start=start, end=end)

data.loc[:, :, 'price'].plot(figsize=(8,5))
plt.ylabel('price in $')
plt.show()


import zipline
from zipline.api import (history,
                         set_slippage,
                         slippage,
                         set_commission,
                         commission,
                         order_target_percent)

from zipline import TradingAlgorithm


def initialize(context):
    '''
    Called once at the very beginning of a backtest (and live trading).
    Use this method to set up any bookkeeping variables.

    The context object is passed to all the other methods in your algorithm.

    Parameters

    context: An initialized and empty Python dictionary that has been
             augmented so that properties can be accessed using dot
             notation as well as the traditional bracket notation.

    Returns None
    '''
    # Turn off the slippage model
    set_slippage(slippage.FixedSlippage(spread=0.0))
    # Set the commission model (Interactive Brokers Commission)
    set_commission(commission.PerShare(cost=0.01, min_trade_cost=1.0))
    context.tick = 0

def handle_data(context, data):
    '''
    Called when a market event occurs for any of the algorithm's
    securities.

    Parameters

    data: A dictionary keyed by security id containing the current
          state of the securities in the algo's universe.

    context: The same context object from the initialize function.
             Stores the up to date portfolio as well as any state
             variables defined.

    Returns None
    '''
    # Allow history to accumulate 100 days of prices before trading
    # and rebalance every day thereafter.
    context.tick += 1
    if context.tick < 100:
        return
    # Get rolling window of past prices and compute returns
    prices = history(100, '1d', 'price').dropna()
    returns = prices.pct_change().dropna()
    try:
        # Perform Markowitz-style portfolio optimization
        weights, _, _ = mko.optimal_portfolio(returns.T)
        # Rebalance portfolio accordingly
        print "Optimized real weights - ", weights

        for stock, weight in zip(prices.columns, weights):
            print stock, weight
            order_target_percent(stock, weight)
    except ValueError as e:
        # Sometimes this error is thrown
        # ValueError: Rank(A) < p or Rank([P; A; G]) < n
        pass

# Instantinate algorithm
algo = TradingAlgorithm(initialize=initialize,
                        handle_data=handle_data)
# Run algorithm
results = algo.run(data)
results.portfolio_value.plot()