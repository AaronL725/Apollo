from typing import Dict
import pandas as pd

class StrategyBase:
    """策略基类"""
    def __init__(self, params: Dict = None):
        self.params = params or {}

        # 添加默认交易数量
        self.default_quantity = 1
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError("子类必须实现calculate_indicators方法")
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError("子类必须实现generate_signals方法")