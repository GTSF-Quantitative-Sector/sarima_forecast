import backtrader as bt
import multiprocessing
import pandas as pd
from strategy import SARIMAXStrategy
from analyzers import FinalValue
from data_feed import PandasData
import backtrader.analyzers as btanalyzers

# Load data
gld_data = pd.read_csv('../GLD-Prices.csv',
                       parse_dates=True, index_col='Date')


data_gld = PandasData(dataname=gld_data)

# Initialize Backtrader Cerebro engine
cerebro = bt.Cerebro()

# Add data to Cerebro
cerebro.adddata(data_gld)

# Add strategy to Cerebro
cerebro.optstrategy(
    SARIMAXStrategy,
    forecast_period=[1],
    model_order=[(2, 1, 1)],
    seasonal_order=[(1, 1, 1, 7)],
    threshold=[0.005],          # Adjusted to 0.5% for percentage-based logic
    verbose=[False],            # Set to True for detailed logging
    stop_loss=[0.05, 0.1, 0.15, 0.2],         # 5%, 10%, 15%, 20% stop loss
    take_profit=[0.1, 0.15, 0.2, 0.25],       # 10%, 15%, 20%, 25% take profit
    fit_interval=[5],
)

# Add analyzers to Cerebro
cerebro.addanalyzer(btanalyzers.SharpeRatio, _name='sharpe_ratio')
cerebro.addanalyzer(btanalyzers.DrawDown, _name='drawdown')
cerebro.addanalyzer(FinalValue, _name='final_value')

# Set initial capital
cerebro.broker.setcash(100000.0)

# Set commission
cerebro.broker.setcommission(commission=0.001)

# Run the backtest
num_cpus = multiprocessing.cpu_count()
results = cerebro.run(maxcpus=1)

# Print results
best_value = -(float('inf'))
for strat_list in results:
    for strat in strat_list:
        final = strat.analyzers.getbyname('final_value').get_analysis()
        portfolio_value = final.get('final_value')
        if portfolio_value > best_value:
            best_value = portfolio_value
            best_params = strat.params
            best_sharpe = strat.analyzers.sharpe_ratio.get_analysis()
            best_drawdown = strat.analyzers.drawdown.get_analysis()

print(f"Best Portfolio Value: {best_value:.2f}")
print(
    f"Best Parameters: Stop Loss={best_params.stop_loss}, Take Profit={best_params.take_profit}")
print(f"Sharpe Ratio: {best_sharpe.get('sharperatio', 'N/A')}")
print(f"Max Drawdown: {best_drawdown.max.drawdown}%")
