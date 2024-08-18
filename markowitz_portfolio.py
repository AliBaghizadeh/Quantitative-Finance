import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import scipy.optimize as optimization


class PortfolioConstruction:
    """
    A class to implement Markowitz Algorithm on constructing portfolio of different stocks
    """
    def __init__(self, stocks: list[str], Start_date: str, End_date: str, num_trading_days: int, num_portfolios: int):
        """
        stocks: This is the number of stocks we have included in our portfolio, to be downloaded from yahoo finance.
        Start_date: The starting day to downlaod the Stock data from yahoo finance.
        End_date: The last date to download the stock data from yahoo finance.
        num_trading_days: The number trading days per year, normally 252 days, whichis used to calculate daily returns and volatility of the portfolio.
        num_portfolios: How many portfolios (scenarios) we are going to generate.

        """
        self.stocks = stocks
        self.Start_date = Start_date
        self.End_date = End_date
        self.num_trading_days = num_trading_days
        self.num_portfolios = num_portfolios
        self.data = None
        self.log_returns = None
        self.covariance_matrix = None

    def download_stock(self):
        self.data = yf.download(self.stocks, start = self.Start_date, end = self.End_date)["Close"]
        if self.data.empty:
            raise ValueError(f"No data returned for stocks: {self.stocks}. Check the stock tickers and date range.")
        return self.data

    def daily_return(self):
        """
        A fucntion to calculate the logarithm of the return of the stocks.
        The logarithm is used to normalize different stocks for better visualization.
        Output: Logarithm of daily return of stock 
        """
        self.log_returns = np.log(self.data/self.data.shift(1))[1:]
        self.covariance_matrix = self.log_returns.cov() * self.num_trading_days  
        return self.log_returns

    def generate_portfolio(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        A function to generate weights, expected return and volatility of all portfolios for all stocks
        returns: This is the log of daily return
        Output: weights, return and volatility of generated portfolios
        """

        if self.log_returns is None:
            raise ValueError("Log returns are not calculated. Please run the daily_return method first.")
        
        portfolio_wieght = np.random.random(size = (self.num_portfolios,  len(self.stocks)))
        portfolio_wieght /= np.sum(portfolio_wieght, axis = 1)[:, None]

        expected_return = self.log_returns.mean()*self.num_trading_days
        portfolio_return = np.dot(portfolio_wieght, expected_return)

        portfolio_vol = np.sqrt(np.einsum("ij, jk, ik->i", portfolio_wieght, self.covariance_matrix, portfolio_wieght))

        return portfolio_wieght, portfolio_return, portfolio_vol

    def portfolio_statistics(self, weights):
        """
        Calculate the return, variance and sharp ratio of the portfolio
        Weights: The weights of the portfolios
        """
        portfolio_return = np.dot(weights, self.log_returns.mean()*self.num_trading_days)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(self.covariance_matrix, weights)))
        portfolio_sharp_ratio = portfolio_return / portfolio_volatility
        
        return portfolio_return, portfolio_volatility, portfolio_sharp_ratio

    def optimize_portfolio(self, weights):
        """
        A fucntion to optimize the weights according to the constraints and bounds
        weights: The main variable to be minimized
        returns: The additonal variable necessary to run the function
        constrains: The sum of weights for each portfolio should be one
        bounds: The weights should range between zero and one. 
        Output: The statistical information and optimized weights of the optimized portfolio. The weights are saved in the column mamed "x".
        """
        if not isinstance(weights, np.ndarray) or weights.shape[1] != len(self.stocks):
            raise ValueError("Weights must be a numpy array with the correct shape (num_portfolios, num_stocks).")
                
        constraint = ({"type":"eq", "fun":lambda x: np.sum(x)-1})
        bounds = tuple((0,1) for _ in range(len(self.stocks)))

        def maximum_sharp_ratio(weights):
            return -self.portfolio_statistics(weights)[2]

        result = optimization.minimize(fun = maximum_sharp_ratio, 
                                     x0 = weights[0], 
                                     method = "SLSQP",
                                     #args = self.log_returns, 
                                     constraints = constraint, 
                                     bounds = bounds)
        if not result.success:
            raise RuntimeError(f"Optimization failed: {result.message}")
        
        return result

    def print_optimal_portfolio(self, weights):
        """
        The function prints out the optimum weights calculated to maximize the sharp ratio among different given scenarios.
        Weights: The optimized weights calculated using optimize_portfolio method.
        Output: Optimized weights of stocks in the portfolio, and the statistical parameters of the optimized portfolio.
        """
        optimum = self.optimize_portfolio(weights)["x"].round(2)
        expected_return, volatility, sharpe_ratio = self.portfolio_statistics(optimum)
        print("Optimal portfolio: \n", "weigths:", optimum)
        print([f"portfolio: {weight*100:.2f} % x {stock}" for weight, stock in zip(optimum, self.stocks)])
        print(f"Expected return {round(expected_return, 2)}, volatility {round(volatility, 2)} and Sharpe ratio: {round(sharpe_ratio, 2)}")
        
    def show_optimal_portfolio(self, weights, portfolio_rets, portfolio_vols):
        """
        The function plots the expected return versus expected volatility of the Num_portfolios, with color code, the sharp ratio, and 
        the start demonstrates the optimum portfolio, to maximize the sharp ratio. 
        """
        plt.figure(figsize=(10, 6))
        plt.scatter(portfolio_vols, portfolio_rets, c=portfolio_rets / portfolio_vols, marker='o') 
        plt.grid(True)
        plt.xlabel('Expected Volatility')
        plt.ylabel('Expected Return')
        plt.colorbar(label='Sharpe Ratio')
        optimum = self.optimize_portfolio(weights)["x"].round(2)
        plt.plot(self.portfolio_statistics(optimum)[1], 
                self.portfolio_statistics(optimum)[0],
                'g*', markersize=20.0)
        plt.show();

def show_stocks_data(data, title = "Stock Price", xlabel = "Date", ylabel = "Closing Price"):
    """
    A fucntion to plot the closing price of the different stocks downloaded for the analysis in a given time interval
    data: The data is Stock data which can be downloaded from yahoo finance using download_stock() method in portfolio_construction class.
    """
    data.plot(figsize = (14, 14))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show();

if __name__ == "__main__":
    # The seven stocks are chosen to test the algorithm in a time span of 11 years.
    portfolio_1 = PortfolioConstruction(
        stocks = ["META", "GOOGL", "AMZN", "TSLA", "DB", "MS", "UBS"],
        Start_date = "2013-01-01",
        End_date = "2024-01-01",
        num_trading_days = 252,
        num_portfolios = 50000)

    data = portfolio_1.download_stock()  # Download stock data from yahoo finance
    show_stocks_data(data)         # plot the closing date of the stock data
    log_daily_return= portfolio_1.daily_return()    #calculate the logarithm of the daily return of the stock data
    weights, mean_returns, volatility = portfolio_1.generate_portfolio()   #generate portfolios and their corresponding statistical parameters
    optimum = portfolio_1.optimize_portfolio(weights)     #calculate the optimum weights (optimum portfolio) to maximize the sharp ratio
    portfolio_1.print_optimal_portfolio(weights)   #print the information of the optimum portfolio, weights and statistical parameters
    portfolio_1.show_optimal_portfolio(weights, mean_returns, volatility)   #show the sharp ratio plot and the optimum portfolio on the plot

