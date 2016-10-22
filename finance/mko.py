
import numpy as np
import matplotlib.pyplot as plt
import cvxopt as opt
import pandas as pd
from cvxopt import blas, solvers
import data_download

np.random.seed(123)

# Turn off progress printing
# solvers.options['show_progress'] = False

## NUMBER OF ASSETS
n_assets = 4

## NUMBER OF OBSERVATIONS
n_obs = 1000

return_vec = np.random.randn(n_assets, n_obs)




plt.plot(return_vec.T, alpha=.4);
plt.xlabel('time')
plt.ylabel('returns')
plt.show()


def rand_weights(n):
    ''' Produces n random weights that sum to 1 '''
    k = np.random.rand(n)
    return k / sum(k)

print rand_weights(n_assets)
print rand_weights(n_assets)


def random_portfolio(returns):
    '''
    Returns the mean and standard deviation of returns for a random portfolio
    '''

    p = np.asmatrix(np.mean(returns, axis=1))
    w = np.asmatrix(rand_weights(returns.shape[0]))
    C = np.asmatrix(np.cov(returns))

    mu = w * p.T
    sigma = np.sqrt(w * C * w.T)

    # This recursion reduces outliers to keep plots pretty

    if sigma > 2:
        return random_portfolio(returns)
    return mu, sigma




n_portfolios = 500
means, stds = np.column_stack([
    random_portfolio(return_vec)
    for _ in xrange(n_portfolios)
])


plt.plot(stds, means, 'o', markersize=5)
plt.xlabel('std')
plt.ylabel('mean')
plt.title('Mean and standard deviation of returns of randomly generated portfolios')
plt.show()


def plot_portfolio_possibilities(prices):
    return_vec = prices.T
    n_portfolios = 500
    means, stds = np.column_stack([
                                      random_portfolio(return_vec)
                                      for _ in xrange(n_portfolios)
                                      ])

    plt.plot(stds, means, 'o', markersize=5)
    plt.xlabel('std')
    plt.ylabel('mean')
    plt.title('Mean and standard deviation of returns of randomly generated portfolios')
    plt.show()


def optimal_portfolio(returns):
    n = len(returns)
    returns = np.asmatrix(returns)

    N = 100
    mus = [10**(5.0 * t/N - 1.0) for t in range(N)]

    # Convert to cvxopt matrices
    S = opt.matrix(np.cov(returns))
    pbar = opt.matrix(np.mean(returns, axis=1))

    # Create constraint matrices
    G = -opt.matrix(np.eye(n))   # negative n x n identity matrix
    h = opt.matrix(0.0, (n ,1))
    A = opt.matrix(1.0, (1, n))
    b = opt.matrix(1.0)

    # Calculate efficient frontier weights using quadratic programming
    portfolios = [solvers.qp(mu*S, -pbar, G, h, A, b)['x']
                  for mu in mus]
    ## CALCULATE RISKS AND RETURNS FOR FRONTIER
    returns = [blas.dot(pbar, x) for x in portfolios]
    risks = [np.sqrt(blas.dot(x, S*x)) for x in portfolios]
    ## CALCULATE THE 2ND DEGREE POLYNOMIAL OF THE FRONTIER CURVE
    m1 = np.polyfit(returns, risks, 2)
    x1 = np.sqrt(m1[2] / m1[0])
    # CALCULATE THE OPTIMAL PORTFOLIO
    wt = solvers.qp(opt.matrix(x1 * S), -pbar, G, h, A, b)['x']
    return np.asarray(wt), returns, risks

weights, returns, risks = optimal_portfolio(return_vec)

plt.plot(stds, means, 'o')
plt.ylabel('mean')
plt.xlabel('std')
plt.plot(risks, returns, 'y-o')

print weights, returns, risks


def mko_analysis(prices, symbols):
    return_vec = data_download.get_daily_returns(prices)
    daily_rets = data_download.get_daily_returns(prices)
    price_vector = prices.as_matrix(columns=daily_rets.columns[0:])
    weights, returns, risks = optimal_portfolio(price_vector.T)

    allocs_mko = list(weights)
    portfolio_val_mko, position_val = data_download.build_portfolio(prices, allocs_mko, 100)

    allocs_default = np.ones(len(symbols)) * 1/len(symbols)
    portfolio_val_default, position_val = data_download.build_portfolio(prices, allocs_default, 100)

    allocs_sharp_ratio_opt = data_download.optimize_portfolio(symbols, prices, allocs_default, 100)
    portfolio_val_sharp_ratio_opt, position_val = data_download.build_portfolio(prices, allocs_sharp_ratio_opt, 100)

    plt.show()
    data_download.plot_df(
        [portfolio_val_mko.to_frame(), portfolio_val_default.to_frame(), portfolio_val_sharp_ratio_opt.to_frame()])

    stats_mko = data_download.portfolio_statistics(portfolio_val_mko)
    stats_def = data_download.portfolio_statistics(portfolio_val_default)
    stats_sharp = data_download.portfolio_statistics(portfolio_val_sharp_ratio_opt)

    print "mko weights = ", weights
    print "allocs_sharp_ratio_opt weights = ", allocs_sharp_ratio_opt

    print "stats_mko = ", stats_mko
    print "stats_def = ", stats_def
    print "stats_sharp = ", stats_sharp

    verify_start_date = '2015-01-01'
    verify_end_date = '2016-09-01'
    portfolio_val_default, portfolio_val_sharp_ratio_opt = data_download.verify_prices(symbols, verify_start_date,
                                                                                       verify_end_date, allocs_default,
                                                                                       allocs_sharp_ratio_opt, 100)
    portfolio_val_mko, portfolio_val_sharp_ratio_opt = data_download.verify_prices(symbols, verify_start_date,
                                                                                   verify_end_date, allocs_mko,
                                                                                   allocs_sharp_ratio_opt, 100)
    print "Verifying prices between - ", verify_start_date, verify_end_date
    stats_mko = data_download.portfolio_statistics(portfolio_val_mko)
    stats_def = data_download.portfolio_statistics(portfolio_val_default)
    stats_sharp = data_download.portfolio_statistics(portfolio_val_sharp_ratio_opt)
    print "stats_mko = ", stats_mko
    print "stats_def = ", stats_def
    print "stats_sharp = ", stats_sharp

    data_download.plot_df(
        [portfolio_val_mko.to_frame(), portfolio_val_default.to_frame(), portfolio_val_sharp_ratio_opt.to_frame()])

    # data_download.plot_df([portfolio_val_mko, portfolio_val_default])


def main():
    symbols = ['GM', 'GOLD', 'GE', 'INTC', 'GOOG', 'SAP', 'MSFT', 'MSI', 'SNE', 'PHG', 'WMT', 'ORCL', 'PFE']
    start_date = '2011-01-01'
    end_date = '2015-01-01'
    data_download.get_data_symbols(symbols)
    prices = data_download.build_prices(symbols, start_date, end_date)
    # plot_portfolio_possibilities(prices)
    mko_analysis(prices, symbols)


if __name__ == "__main__":
    main()