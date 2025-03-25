'''
// 简称: AverageChannelRangeLeader_L
// 名称: 基于平移的高低点均值通道与K线中值突破的系统多 
// 类别: 策略应用
// 类型: 内建应用
// 输出:
// -----------------------------------------------------------------------
//     ---------------------------------------------------------------------// 
//     策略说明:
//                 基于平移的高点和低点均线通道与K线中值突破进行判断
//     系统要素:
//                 1. MyRange Leader是个当前K线的中点在之前K线的最高点上, 且当前K线的振幅大于之前K线的振幅的K线
//                 2. 计算高点和低点的移动平均线
//     入场条件:
//                 1、上根K线为RangeLead，并且上一根收盘价大于N周期前高点的MA，当前无多仓，则开多仓
//                 2、上根K线为RangeLead，并且上一根收盘价小于N周期前低点的MA，当前无空仓，则开空仓
//    
//     出场条件:
//                 1. 开仓后，5个K线内用中轨止损，5个K线后用外轨止损
// 
//    注:当前策略仅为做多系统, 如需做空, 请参见CL_AverageChannelRangeLeader_S
'''

from typing import Dict
import pandas as pd
import numpy as np
from module import *
from .base import StrategyBase

class AverageChannelRangeLeader_L(StrategyBase):
    """基于平移的高低点均值通道与K线中值突破的做多策略"""
    def __init__(self, params: Dict = None):
        # 默认参数设置
        default_params = {
            'AvgLen': 20,     # 高低点均线计算周期
            'AbsDisp': 5,     # 高低点均线前移周期
            'ExitBar': 5,     # 止损周期参数，该周期以前中轨止损，以后外轨止损
            'Lots': 1         # 交易手数
        }
        # 更新自定义参数
        if params:
            default_params.update(params)
        super().__init__(default_params)

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算技术指标"""
        df = data.copy()
        
        # 计算振幅
        df['MyRange'] = df['high'] - df['low']
        
        # 计算K线中价
        df['MedianPrice'] = (df['high'] + df['low']) * 0.5
        
        # 计算移动平均线
        df['UpperAvg'] = Average(df['high'].shift(self.params['AbsDisp']), self.params['AvgLen'])
        df['LowerAvg'] = Average(df['low'].shift(self.params['AbsDisp']), self.params['AvgLen'])
        df['ExitAvg'] = Average(df['MedianPrice'].shift(self.params['AbsDisp']), self.params['AvgLen'])
        
        # 修正RangeLeadB计算逻辑：使用 & 替代 and
        df['RangeLeadB'] = (df['MedianPrice'] > df['high'].shift(1)) & (df['MyRange'] > df['MyRange'].shift(1))
        
        return df

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """生成交易信号矩阵"""
        # 数据验证
        min_length = self.params['AvgLen'] + self.params['AbsDisp']
        if len(data) < min_length:
            raise ValueError('数据长度不足')
            
        # 最小变动价位
        minpoint = 1
            
        # 初始化信号矩阵
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0
        signals['quantity'] = 1
        signals['entry_price'] = np.nan
        signals['exit_price'] = np.nan
        signals['position'] = 0
        
        current_position = 0
        position_bars = 0
        
        for i in range(1, len(data)):
            if current_position == 0:
                if (data['RangeLeadB'].iloc[i-1] and 
                    data['close'].iloc[i-1] > data['UpperAvg'].iloc[i-1]):
                    signals.iloc[i] = [1, self.params['Lots'], data['open'].iloc[i], np.nan, 0]
                    current_position = 1
                    position_bars = 0
                    
            elif current_position == 1:
                position_bars += 1
                
                if position_bars <= self.params['ExitBar']:
                    if data['low'].iloc[i] <= data['ExitAvg'].iloc[i]:
                        exit_price = min(data['open'].iloc[i], data['ExitAvg'].iloc[i])
                        signals.iloc[i] = [-1, self.params['Lots'], np.nan, exit_price, position_bars]
                        current_position = 0
                        position_bars = 0
                else:
                    # 使用minpoint调整止损价格
                    if data['low'].iloc[i] <= (data['UpperAvg'].iloc[i] - minpoint):
                        exit_price = min(data['open'].iloc[i], data['UpperAvg'].iloc[i] - minpoint)
                        signals.iloc[i] = [-1, self.params['Lots'], np.nan, exit_price, position_bars]
                        current_position = 0
                        position_bars = 0
        
        if current_position == 1:
            signals.iloc[-1] = [-1, self.params['Lots'], np.nan, data['close'].iloc[-1], position_bars]
            
        return signals
