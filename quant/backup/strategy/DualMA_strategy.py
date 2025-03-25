'''
// 简称: DualMA
// 名称: 双均线交易系统
// 类别: 策略应用
// 类型: 内建应用
'''

from typing import Dict
import pandas as pd
import numpy as np
from module import *
from module.indicators import *
from .base import StrategyBase

class DualMA(StrategyBase):
    """双均线交易系统"""
    def __init__(self, params: Dict = None):
        # 默认参数设置
        default_params = {
            'FastLength': 5,    # 短期指数平均线参数
            'SlowLength': 20,   # 长期指数平均线参数
            'Lots': 1          # 交易手数
        }
        
        # 更新自定义参数
        if params:
            default_params.update(params)
        super().__init__(default_params)
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算技术指标"""
        df = data.copy()
        
        # 计算快速和慢速移动平均线
        df['AvgValue1'] = Average(df['close'], self.params['FastLength'])
        df['AvgValue2'] = Average(df['close'], self.params['SlowLength'])
        
        return df
        
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """生成交易信号矩阵"""
        # 数据验证
        min_length = max(self.params['FastLength'], self.params['SlowLength'])
        if len(data) < min_length:
            return pd.DataFrame()
            
        # 初始化信号矩阵
        signals = pd.DataFrame(index=data.index)
        signals['call'] = np.nan
        
        current_position = 0  # 当前持仓状态: 1多头, -1空头, 0空仓
        
        # 生成交易信号
        for i in range(1, len(data)):
            # 多头信号
            if (current_position != 1 and 
                data['AvgValue1'].iloc[i-1] > data['AvgValue2'].iloc[i-1]):
                signals['call'].iloc[i] = 1
                current_position = 1
                
            # 空头信号    
            elif (current_position != -1 and 
                  data['AvgValue1'].iloc[i-1] < data['AvgValue2'].iloc[i-1]):
                signals['call'].iloc[i] = -1
                current_position = -1
                
        # 在最后一根K线强制平仓
        if current_position == 1:
            signals['call'].iloc[-1] = 0
        elif current_position == -1:
            signals['call'].iloc[-1] = 0
            
        return signals
