'''
// 简称: Trading_Range_Breakout_S
// 名称: 基于初始交易范围突破的思想来建立系统 做空
// 类别: 策略应用
// 类型: 内建应用
// 输出:
//------------------------------------------------------------------------
//------------------------------------------------------------------------
// 策略说明:
//              用特定时间周期内的最高价位和最低价位计算交易范围，然后计算真实的波动范围和周期内的ATR对比，如果
//              当前k线的波动范围比n*交易范围的值小，并且真实波动范围比ATR小，这就满足了前两个系统挂单的条件
//             
// 入场条件:
//            1.7周期区间"空隙"之和 >7周期区间高度的2倍
//              当前的k线比交易范围的最高值大, 而且如果当前k线的中间价格高于之前一根k线的最高值
//              做多
//            2.7周期区间"空隙"之和 >7周期区间高度的2倍
//              当前的k线比交易范围的最低值小, 而且如果当前k线的中间价格低于之前一根k线的最低值
//              做空
// 出场条件: 
//             1.初始止损
//             2.跟踪止损（盈利峰值价回落ATR的一定倍数）    
//             3.收盘价创7周期高点，且K线中点高于前K线最高价空头出场
//
//         注: 当前策略仅为做空系统, 如需做多, 请参见CL_Trading_Range_Breakout_L
'''

from typing import Dict
import pandas as pd
import numpy as np
from module import *
from module.indicators import AvgTrueRange, TrueRange
from strategy.base import StrategyBase

class Trading_Range_Breakout_S(StrategyBase):
    """基于初始交易范围突破的做空策略"""
    
    def __init__(self, params: Dict = None):
        """初始化策略参数"""
        super().__init__(params)
        self.params.update({
            'RangeLen': 7,    # 高低点周期
            'RngPcnt': 200,   # 周期区间高度倍数*100
            'ATRs': 8,        # 盈利峰值价回落ATR
            'ATRLen': 2,      # 盈利峰值价回落周期
            'Lots': 1         # 交易手数
        })
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算技术指标"""
        # 计算7周期高低点
        data['RangeH'] = data['high'].shift(1).rolling(self.params['RangeLen']).max()
        data['RangeL'] = data['low'].shift(1).rolling(self.params['RangeLen']).min()
        data['TRange'] = data['RangeH'] - data['RangeL']
        
        # 计算ATR
        data['ATR'] = AvgTrueRange(self.params['ATRLen'], data['high'], data['low'], data['close'])
        data['ATRMA'] = AvgTrueRange(self.params['RangeLen'], data['high'], data['low'], data['close'])
        
        # 向量化计算NoTrades
        data['NoTrades'] = 0
        range_len = self.params['RangeLen']
        
        # 预计算所有可能的high和low差值
        high_diffs = pd.DataFrame({
            f'high_diff_{i}': data['RangeH'] - data['high'].shift(i)
            for i in range(1, range_len + 1)
        }).clip(lower=0)
        
        low_diffs = pd.DataFrame({
            f'low_diff_{i}': data['low'].shift(i) - data['RangeL']
            for i in range(1, range_len + 1)
        }).clip(lower=0)
        
        # 计算总和
        data['NoTrades'] = high_diffs.sum(axis=1) + low_diffs.sum(axis=1)
        
        return data
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """生成交易信号"""
        signals = pd.DataFrame(index=data.index)
        signals['call'] = np.nan
        
        # 预计算中间价格
        data['mid_price'] = (data['high'] + data['low']) * 0.5
        
        # 预计算交易条件
        data['condition1'] = data['NoTrades'].shift(1) >= data['TRange'].shift(1) * (self.params['RngPcnt'] * 0.01)
        data['condition2'] = TrueRange(data['high'], data['low'], data['close']).shift(1) > data['ATRMA'].shift(2)
        data['condition3'] = (data['close'].shift(1) > data['RangeH'].shift(1)) & (data['mid_price'].shift(1) > data['high'].shift(2))
        data['condition4'] = (data['close'].shift(1) < data['RangeL'].shift(1)) & (data['mid_price'].shift(1) < data['low'].shift(2))
        
        current_position = 0
        short_risk = 0
        short_low = float('inf')
        position_bars = 0
        
        # 使用布尔索引加速信号生成
        valid_rows = data.index[self.params['RangeLen']:]
        
        for i in valid_rows:
            if current_position == 0:
                if (data.at[i, 'condition1'] and data.at[i, 'condition2'] and 
                    data.at[i, 'condition4'] and data.at[i, 'vol'] > 0):
                    signals.at[i, 'call'] = -1
                    current_position = -1
                    short_risk = data.at[i, 'RangeH']
                    short_low = data.at[i, 'low']
                    position_bars = 0
                    
            elif current_position == -1:
                position_bars += 1
                if position_bars > 0:
                    short_low = min(short_low, data.at[i, 'low'])
                    
                    if data.at[i, 'condition3']:
                        signals.at[i, 'call'] = 0
                        current_position = 0
                    elif data.at[i, 'high'] >= short_risk:
                        signals.at[i, 'call'] = 0
                        current_position = 0
                    elif data.at[i, 'high'] >= short_low + (self.params['ATRs'] * data.at[i, 'ATR']):
                        signals.at[i, 'call'] = 0
                        current_position = 0
        
        if current_position == -1:
            signals.iloc[-1]['call'] = 0
        
        print("\n=== 信号矩阵 ===")
        print(signals[signals['call'].notna()])  # 只打印有信号的行
        
        return signals
