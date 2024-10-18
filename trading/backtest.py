import backtrader as bt
import multiprocessing
import pandas as pd
from strategy.auto_arima_strategy import AutoArimaStrategy
from analyzers.final_value_analyzer import FinalValue
from utils.data_feed import PandasData


# Load data
gld_data = pd.read_csv('data/GLD-Prices.csv',
                       parse_dates=True, index_col='Date')
data_gld = PandasData(dataname=gld_data)

# Initialize Backtrader Cerebro engine
cerebro = bt.Cerebro()

# Add data to Cerebro
cerebro.adddata(data_gld)

# Add strategy to Cerebro
cerebro.optstrategy(
    AutoArimaStrategy,
    forecast_period=[1],
    seasonal=[True],
    m=[7],
    threshold=[0.005],           # 0.5% threshold
    stop_loss=[0.05, 0.1],       # 5%, 10% stop loss
    take_profit=[0.1, 0.15],     # 10%, 15% take profit
    verbose=[False],
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
results = cerebro.run(maxcpus=num_cpus)

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
