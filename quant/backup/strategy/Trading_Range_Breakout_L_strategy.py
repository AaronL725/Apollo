'''
// 简称: Trading_Range_Breakout_L
// 名称: 基于初始交易范围突破的思想来建立系统 多
// 类别: 策略应用
// 类型: 内建应用
// 输出:
//------------------------------------------------------------------------
//------------------------------------------------------------------------
// 策略说明:
//              用特定时间周期内的最高价位和最低价位计算交易范围，然后计算真实的波动范围和周期内的ATR对比，如果
//              当前k线的波动范围比n*交易范围的值大，并且真实波动范围比ATR大，这就满足了前两个系统挂单的条件
//             
// 入场条件:
//            1.7周期区间“空隙”之和 >7周期区间高度的2倍
//              当前的k线比交易范围的最高值大, 而且如果当前k线的中间价格高于之前一根k线的最高值
//              做多
//            2.7周期区间“空隙”之和 >7周期区间高度的2倍
//              当前的k线比交易范围的最低值小, 而且如果当前k线的中间价格低于之前一根k线的最低值
//              做空
// 出场条件: 
//             1.初始止损
//             2.跟踪止损（盈利峰值价回落ATR的一定倍数）    
//             3.收盘价创7周期低点，且K线中点低于前K线最低价多头出场
//
//         注: 当前策略仅为做多系统, 如需做空, 请参见CL_Trading_Range_Breakout_S
'''

from typing import Dict
import pandas as pd
import numpy as np
from module import *
from module.indicators import AvgTrueRange, TrueRange, Highest, Lowest
from .base import StrategyBase

class Trading_Range_Breakout_L(StrategyBase):
    """基于初始交易范围突破的思想来建立系统做多策略"""
    def __init__(self, params: Dict = None):
        # 默认参数设置
        default_params = {
            'RangeLen': 7,     # 高低点周期
            'RngPcnt': 200,    # 周期区间高点倍数
            'ATRs': 8,         # 盈利峰值价回落ATR
            'ATRLen': 2,       # 盈利峰值价回落周期
            'Lots': 1          # 交易手数
        }
        
        if params:
            default_params.update(params)
        super().__init__(default_params)
        
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算技术指标"""
        df = data.copy()
        
        # 计算初始指标
        df['RangeH'] = Highest(df['high'].shift(1), self.params['RangeLen'])
        df['RangeL'] = Lowest(df['low'].shift(1), self.params['RangeLen'])
        df['TRange'] = df['RangeH'] - df['RangeL']
        df['ATR'] = AvgTrueRange(self.params['ATRLen'], df['high'], df['low'], df['close'])
        df['ATRMA'] = AvgTrueRange(self.params['RangeLen'], df['high'], df['low'], df['close'])
        
        # 计算NoTrades
        df['NoTrades'] = 0
        for i in range(1, self.params['RangeLen'] + 1):
            mask1 = df['high'].shift(i) <= df['RangeH']
            df.loc[mask1, 'NoTrades'] += df.loc[mask1, 'RangeH'] - df.loc[mask1, 'high'].shift(i)
            
            mask2 = df['low'].shift(i) >= df['RangeL']
            df.loc[mask2, 'NoTrades'] += df.loc[mask2, 'low'].shift(i) - df.loc[mask2, 'RangeL']
        
        # 计算条件指标
        df['TrueRange'] = TrueRange(df['high'], df['low'], df['close'])
        df['MedianPrice'] = (df['high'] + df['low']) * 0.5
        
        df['Condition1'] = df['NoTrades'] >= df['TRange'] * (self.params['RngPcnt'] * 0.01)
        df['Condition2'] = df['TrueRange'] > df['ATRMA'].shift(1)
        df['Condition3'] = (df['close'] > df['RangeH']) & (df['MedianPrice'] > df['high'].shift(1))
        df['Condition4'] = (df['close'] < df['RangeL']) & (df['MedianPrice'] < df['low'].shift(1))
        
        return df
        
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """生成交易信号矩阵"""
        # 数据验证
        min_length = self.params['RangeLen'] + 1
        if len(data) < min_length:
            return pd.DataFrame(index=data.index)
            
        signals = pd.DataFrame(index=data.index)
        signals['call'] = np.nan
        
        current_position = 0
        long_risk = 0
        long_high = 0
        
        for i in range(min_length, len(data)):
            if current_position == 0:  # 空仓
                # 多头入场
                if (data['Condition1'].iloc[i-1] and data['Condition2'].iloc[i-1] and 
                    data['Condition3'].iloc[i-1]):
                    signals.iloc[i] = 1
                    current_position = 1
                    long_risk = data['RangeL'].iloc[i]
                    long_high = data['high'].iloc[i]
                    
            elif current_position == 1:  # 持有多仓
                long_high = max(long_high, data['high'].iloc[i])
                
                # 多头出场条件检查
                # 1. 收盘价创7周期低点，且K线中点低于前K线最低价多头出场
                if data['Condition4'].iloc[i-1]:
                    signals.iloc[i] = 0
                    current_position = 0
                    
                # 2. 跌破初始止损价多头出场
                elif data['low'].iloc[i] <= long_risk:
                    signals.iloc[i] = 0
                    current_position = 0
                    
                # 3. 盈利峰值价回落ATR一定倍数多头出场
                elif (data['low'].iloc[i] <= 
                      long_high - (self.params['ATRs'] * data['ATR'].iloc[i-1])):
                    signals.iloc[i] = 0
                    current_position = 0
        
        # 在最后一根K线强制平仓
        if current_position == 1:
            signals.iloc[-1] = 0
            
        return signals
