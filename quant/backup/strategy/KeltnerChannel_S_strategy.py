'''
// 简称: KeltnerChannel_S
// 名称: 基于凯特纳通道的交易系统空
// 类别: 策略应用
// 类型: 内建应用 
// 输出:
// ------------------------------------------------------------------------
// ----------------------------------------------------------------------// 
//  策略说明:
//              基于凯特纳通道的交易系统
// 
//  系统要素:
//              1. 计算关键价格的凯特纳通道
//                 2. 价格突破凯特纳通道后，设定入场触发单
// 
//  入场条件:
//              1、价格突破凯特纳通道后，在当根K线高点之上N倍通道幅度，设定多头触发单，此开仓点将挂单X根k线
//              2、价格突破凯特纳通道后，在当根K线低点之下N倍通道幅度，设定空头触发单，此开仓点将挂单X根k线
// 
//  出场条件:
//              1. 价格下穿轨道中轨时平仓
//                 2. 价格小于N周期低点平仓
// 
//     注：当前策略仅为做空系统, 如需做多, 请参见CL_KeltnerChannel_L
'''

from typing import Dict
import pandas as pd
import numpy as np
from module import *
from .base import StrategyBase

class KeltnerChannel_S(StrategyBase):
    """凯特纳通道做空策略"""
    def __init__(self, params: Dict = None):
        # 默认参数设置
        default_params = {
            'Length': 10,      # 均线参数
            'Constt': 1.2,     # 通道倍数
            'ChanPcnt': 0.5,   # 入场参数
            'sellN': 5,        # 入场触发条件有效K线周期
            'stopN': 4         # 高点止损参数
        }
        # 更新自定义参数
        if params:
            default_params.update(params)
        super().__init__(default_params)
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算策略指标"""
        df = data.copy()
        
        # 计算关键价格和均线
        df['Price'] = df['close']
        df['AvgVal'] = Average(df['Price'], self.params['Length'])
        
        # 计算真实波动均值(ATR)
        df['AvgRange'] = Average(TrueRange(df['high'], df['low'], df['close']), 
                                self.params['Length'])
        
        # 计算通道
        df['KCU'] = df['AvgVal'] + df['AvgRange'] * self.params['Constt']
        df['KCL'] = df['AvgVal'] - df['AvgRange'] * self.params['Constt']
        df['ChanRng'] = (df['KCU'] - df['KCL']) / 2
        
        # 计算下穿下轨信号
        df['CrossDown'] = CrossUnder(df['Price'], df['KCL'])
        
        # 初始化计数器和触发价格
        df['CountS'] = 0
        df['ll'] = np.nan
        
        # 更新计数器和触发价格
        current_count = 0
        for i in range(len(df)):
            if df['CrossDown'].iloc[i]:
                current_count = 0
                df.loc[df.index[i], 'll'] = (df['low'].iloc[i] - 
                                            df['ChanRng'].iloc[i] * self.params['ChanPcnt'])
            else:
                current_count += 1
            df.loc[df.index[i], 'CountS'] = current_count
            
        # 计算止损线
        df['Sstopline'] = Highest(df['high'], self.params['stopN'])
        
        return df
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """生成交易信号矩阵"""
        # 数据验证
        min_length = max(self.params['Length'], self.params['stopN'])
        if len(data) < min_length:
            raise ValueError('数据长度不足')
        
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0
        signals['quantity'] = 1
        signals['entry_price'] = np.nan
        signals['exit_price'] = np.nan
        signals['position'] = 0
        
        current_position = 0
        
        for i in range(1, len(data)):
            # 空仓且满足开仓条件
            if (current_position == 0 and
                data['Price'].iloc[i-1] < data['KCL'].iloc[i-1] and
                data['CountS'].iloc[i] <= self.params['sellN'] and
                data['low'].iloc[i] <= data['ll'].iloc[i-1] and
                data['vol'].iloc[i] > 0):
                
                entry_price = min(data['open'].iloc[i], data['ll'].iloc[i-1])
                signals.iloc[i] = [-1, 1, entry_price, np.nan, -1]
                current_position = -1
            
            # 持有空仓时的平仓条件
            elif current_position == -1:
                # 价格上穿均线
                if (CrossOver(data['close'], data['AvgVal']).iloc[i-1] and
                    data['vol'].iloc[i] > 0):
                    
                    signals.iloc[i] = [1, 1, np.nan, data['open'].iloc[i], 0]
                    current_position = 0
                
                # 价格突破止损线
                elif (data['high'].iloc[i] >= data['Sstopline'].iloc[i-1] and
                      data['vol'].iloc[i] > 0):
                    
                    exit_price = max(data['Sstopline'].iloc[i-1], data['open'].iloc[i])
                    signals.iloc[i] = [1, 1, np.nan, exit_price, 0]
                    current_position = 0
        
        # 在最后一个bar强制平仓
        if current_position == -1:
            last_idx = signals.index[-1]
            signals.loc[last_idx] = [1, 1, np.nan, data['close'].iloc[-1], 0]
        
        return signals 