import backtrader as bt


class FinalValue(bt.Analyzer):
    def __init__(self):
        self.final_value = 0.0

    def stop(self):
        self.final_value = self.strategy.broker.getvalue()

    def get_analysis(self):
        return {'final_value': self.final_value}
