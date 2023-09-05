# automotive-5year-stock-trading

## Summary
* Analyzed and visualized top 20 major automotive industry companies' stock market (based on Total Marke Capitalization as of June 30, 2023, including US stock market and OTC) from July 1 2018 using <b>yfinance, numpy, pandas and matplotlib.</b>

* Developed and optimized an algorithmic trading strategy, with weighted consideration of:
    - Financial indicators(Exponential Moving Averages(EMA), Moving Average Convergence Divergence (MACD) and Relative Strength Index (RSI))
    - Sentimental analysis using Tiingo News Feed API and Vader analyzer
    - Machine learning predictions using RandomForestRegressor

* Achieved a <b>128.80%</b> return during a 5-year backtesting and realized a <b>18.74%</b> return during a 3-month live trading deployment

![Backtesting](./snapshots/Backtesting.png)
![Live](./snapshots/Live.png)

## Pre-Trading Research Source Data
Over 25,000 data records were extracted from yfinance and can be accessed both locally from .csv files under /source_data or access directly online.

## Data Visualization and Financial Analysis 
Basic visualization of open price, daily volume and total traded $ amount.

Any special values / outliers were identified and inspected.
![Open Price](./snapshots/open_price.png)
![Trade Volume](./snapshots/trade_volume.png)
![Total Traded](./snapshots/total_trade.png)

Furthermore, a better tendency and correlation was visualized using moving averages and scatter matrix.
![Alt text](./snapshots/MA_open.png)
![Alt text](./snapshots/MA_matrix.png)

Return Volatility was identified using histogram, kde graph, box plot.
![Alt text](./snapshots/return_vola_hist.png)
![Alt text](./snapshots/return_vola_kde.png)
![Alt text](./snapshots/return_vola_box.png)

Return Correlation was identified using matrix.
![Alt text](./snapshots/return_cor_matrix.png)
![Alt text](./snapshots/vwagy_mbgyy.png)

A Cumulative Return was generated to show overall performance.
![Alt text](./snapshots/cumu_ret.png)

## Run and edit the jupyter notebook for interactive results

## Find trading algorithm in trading_algo folder