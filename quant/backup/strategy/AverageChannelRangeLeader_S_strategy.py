'''
// 简称: AverageChannelRangeLeader_S
// 名称: 基于平移的高低点均值通道与K线中值突破的系统空 
// 类别: 策略应用
// 类型: 内建应用
// 输出:
// ------------------------------------------------------------------------
// ----------------------------------------------------------------------// 
//  策略说明:
//              基于平移的高点和低点均线通道与K线中值突破进行判断
//  系统要素:
//              1. MyRange Leader是个当前K线的中点在之前K线的最高点上, 且当前K线的振幅大于之前K线的振幅的K线
//              2. 计算高点和低点的移动平均线
//  入场条件:
//              1、上根K线为RangeLead，并且上一根收盘价大于N周期前高点的MA，当前无多仓，则开多仓
//              2、上根K线为RangeLead，并且上一根收盘价小于N周期前低点的MA，当前无空仓，则开空仓
// 
//  出场条件:
//              1. 开仓后，5个K线内用中轨止损，5个K线后用外轨止损
// 
//         注:当前策略仅为做空系统, 如需做多, 请参见CL_AverageChannelRangeLeader_L
'''

from typing import Dict
import pandas as pd
import numpy as np
from module import *
from .base import StrategyBase

class AverageChannelRangeLeader_S(StrategyBase):
    """基于平移的高低点均值通道与K线中值突破的做空策略"""
    def __init__(self, params: Dict = None):
        # 默认参数设置
        default_params = {
            'AvgLen': 20,     # 高低点均线计算周期
            'AbsDisp': 5,     # 高低点均线前移周期
            'ExitBar': 5,     # 止损周期参数，该周期以前中轨止损，以后外轨止损
            'Lots': 1         # 交易手数
        }
        super().__init__(params)
        self.params = {**default_params, **(params or {})}

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算技术指标"""
        df = data.copy()
        
        # 计算最小变动价位 (根据品种设置，这里以白银为例)
        min_point = 1  # 白银的最小变动价位是1
        df['min_point'] = min_point
        
        # 计算振幅
        df['MyRange'] = df['high'] - df['low']
        
        # 计算移动平均线
        df['UpperAvg'] = Average(df['high'].shift(self.params['AbsDisp']), self.params['AvgLen'])
        df['LowerAvg'] = Average(df['low'].shift(self.params['AbsDisp']), self.params['AvgLen'])
        
        # 计算K线中价和中价均线
        df['MedianPrice'] = (df['high'] + df['low']) * 0.5
        df['ExitAvg'] = Average(df['MedianPrice'].shift(self.params['AbsDisp']), self.params['AvgLen'])
        
        # 修正RangeLeadS计算逻辑：当前K线中点小于前一根K线低点且当前振幅大于前一根振幅
        df['RangeLeadS'] = (df['MedianPrice'] < df['low'].shift(1)) & (df['MyRange'] > df['MyRange'].shift(1))
        
        return df

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """生成交易信号矩阵"""
        min_length = self.params['AvgLen'] + self.params['AbsDisp']
        if len(data) < min_length:
            raise ValueError('数据长度不足')
            
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0
        signals['quantity'] = 1
        signals['entry_price'] = np.nan
        signals['exit_price'] = np.nan
        signals['position'] = 0
        
        current_position = 0
        position_bars = 0
        
        for i in range(min_length, len(data)):
            if current_position != 0:
                position_bars += 1
                
            # 入场逻辑：空仓时检查入场条件
            if current_position == 0:
                if (data['RangeLeadS'].iloc[i-1] and  # 上一根K线为RangeLeadS
                    data['close'].iloc[i-1] < data['LowerAvg'].iloc[i-1]):  # 上一根收盘价小于下轨
                    entry_price = data['open'].iloc[i]
                    signals.iloc[i] = [-1, self.params['Lots'], np.nan, entry_price, np.nan, 0]
                    current_position = -1
                    position_bars = 0
                    
            # 出场逻辑：持有空仓且不是开仓当根K线
            elif current_position == -1 and position_bars > 0:
                min_point = data['min_point'].iloc[i]
                
                # ExitBar个K线内用中轨止损
                if position_bars <= self.params['ExitBar']:
                    if data['high'].iloc[i] >= data['ExitAvg'].iloc[i]:
                        exit_price = max(data['open'].iloc[i], data['ExitAvg'].iloc[i])
                        signals.iloc[i] = [1, self.params['Lots'], np.nan, np.nan, exit_price, position_bars]
                        current_position = 0
                        position_bars = 0
                
                # ExitBar个K线后用下轨+最小变动价位止损
                else:
                    if data['high'].iloc[i] >= data['LowerAvg'].iloc[i] + min_point:
                        exit_price = max(data['open'].iloc[i], data['LowerAvg'].iloc[i] + min_point)
                        signals.iloc[i] = [1, self.params['Lots'], np.nan, np.nan, exit_price, position_bars]
                        current_position = 0
                        position_bars = 0
        
        # 最后一根K线强制平仓
        if current_position == -1:
            signals.iloc[-1] = [1, self.params['Lots'], np.nan, np.nan, data['close'].iloc[-1], position_bars]
            
        return signals
