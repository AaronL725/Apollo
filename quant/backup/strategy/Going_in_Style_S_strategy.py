'''
// 简称: Going_in_Style_S
// 名称: 价格通道突破, 在价格回调时进行判断，做空
// 类别: 策略应用
// 类型: 内建应用
// 输出:
//------------------------------------------------------------------------
//------------------------------------------------------------------------
// 策略说明:
//            1.计算价格通道
//            2.收盘价减去ATR的一定倍数作为进场价 
//             
// 入场条件:
//            1.上一根Bar创新低
//            2.当前Bar最低价突破上一根Bar收盘价减去ATR的一定倍数
// 出场条件: 
//            1.记录空头进场后的跟踪止损价
//            2.价格向上突破跟踪止损价空头出场
//
//         注: 当前策略仅为做空系统, 如需做多, 请参见CL_Going_in_Style_L
'''

from typing import Dict
import pandas as pd
import numpy as np
from module import *
from module.indicators import *
from .base import StrategyBase

class Going_in_Style_S(StrategyBase):
    """价格通道突破做空策略"""
    def __init__(self, params: Dict = None):
        # 默认参数设置
        default_params = {
            'Length': 10,           # 用于计算ATR和新低价的Bar数
            'Trigger': 0.5,         # 用于计算空头进场价的驱动系数
            'Acceleration': 0.06,   # 抛物线的加速系数
            'FirstBarMultp': 2,     # 用于计算在进场Bar设置止损价的系数
        }
        
        if params:
            default_params.update(params)
        super().__init__(default_params)
        
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算技术指标"""
        df = data.copy()
        
        # 计算ATR
        df['ATR'] = AvgTrueRange(self.params['Length'], df['high'], df['low'], df['close'])
        
        # 计算最低价条件
        df['Condition2'] = df['low'] < Lowest(df['low'].shift(1), self.params['Length'])
        
        # 计算3日真实波幅均值
        df['StopATR'] = TrueRange(df['high'], df['low'], df['close']).rolling(window=3).mean()
        
        return df
        
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """生成交易信号矩阵"""
        # 数据验证
        if len(data) < self.params['Length']:
            return pd.DataFrame()
            
        signals = pd.DataFrame(index=data.index)
        signals['call'] = np.nan
        
        current_position = 0
        position_bars = 0
        stop_price = None
        prev_stop_price = None  # 添加上一个stop_price的记录
        low_value = None
        prev_low_value = None
        af = self.params['Acceleration']
        
        for i in range(1, len(data)):
            # 更新持仓天数
            if current_position != 0:
                position_bars += 1
                
            # 空头入场条件判断
            if (data['Condition2'].iloc[i-1] and  # 上一根Bar创新低
                data['low'].iloc[i] <= data['close'].iloc[i-1] - data['ATR'].iloc[i-1] * self.params['Trigger'] and
                data['vol'].iloc[i] > 0):  # 添加成交量判断
                
                # 空头入场
                if current_position == 0:
                    entry_price = min(data['open'].iloc[i], 
                                    data['close'].iloc[i-1] - data['ATR'].iloc[i-1] * self.params['Trigger'])
                    signals.loc[signals.index[i], 'call'] = -1
                    current_position = -1
                    position_bars = 0
                    
                    # 设置初始止损价和峰值价
                    stop_price = data['high'].iloc[i] + data['StopATR'].iloc[i] * self.params['FirstBarMultp']
                    low_value = data['low'].iloc[i]
                    prev_low_value = low_value
                    af = self.params['Acceleration']
                    
            # 更新跟踪止损价和峰值价
            elif current_position == -1:
                if position_bars == 0:  # 入场当天
                    stop_price = data['high'].iloc[i] + data['StopATR'].iloc[i] * self.params['FirstBarMultp']
                    prev_stop_price = stop_price  # 初始化prev_stop_price
                    low_value = data['low'].iloc[i]
                    prev_low_value = low_value
                    af = self.params['Acceleration']
                elif position_bars > 0:  # 入场后的交易日
                    prev_stop_price = stop_price  # 保存上一个stop_price
                    prev_low_value = low_value
                    if data['low'].iloc[i] < low_value:
                        low_value = data['low'].iloc[i]
                        if low_value < prev_low_value and af < 0.2:
                            af = af + min(self.params['Acceleration'], 0.2 - af)
                    stop_price = stop_price - af * (stop_price - low_value)
                    
                    # 检查止损出场条件 (使用上一根bar的止损价)
                    if data['high'].iloc[i] >= prev_stop_price and data['vol'].iloc[i] > 0:
                        exit_price = max(data['open'].iloc[i], prev_stop_price)  # 确保以不低于止损价的价格出场
                        signals.loc[signals.index[i], 'call'] = 0
                        current_position = 0
                        position_bars = 0
                        stop_price = None
                        prev_stop_price = None
                        low_value = None
                        prev_low_value = None
                        af = self.params['Acceleration']
        
        # 在最后一根K线强制平仓
        if current_position == -1:
            signals.loc[signals.index[-1], 'call'] = 0
            
        return signals
