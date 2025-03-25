'''
// 简称: DisplacedBoll_L 
// 名称: 基于平移布林通道的系统多
// 类别: 策略应用
// 类型: 内建应用
// 输出:
//------------------------------------------------------------------------
//----------------------------------------------------------------------//
// 策略说明:
//             基于平移的boll通道突破系统
//
// 系统要素:
//             1. 平移的boll通道
//
// 入场条件:
//             1、关键价格突破通道上轨，则开多仓
//            2、关键价格突破通道下轨，则开空仓
//
// 出场条件:
//             1、关键价格突破通道上轨，则平空仓
//            2、关键价格突破通道下轨，则平多仓
//
//        注:当前策略仅为做多系统, 如需做空, 请参见CL_DisplacedBoll_S
'''

from typing import Dict
import pandas as pd
import numpy as np
from module import *
from .base import StrategyBase

class DisplacedBoll_L(StrategyBase):
    """基于平移布林通道的做多策略"""
    def __init__(self, params: Dict = None):
        # 默认参数设置
        default_params = {
            'AvgLen': 3,      # boll均线周期参数
            'Disp': 16,       # boll平移参数
            'SDLen': 12,      # boll标准差周期参数
            'SDev': 2,        # boll通道倍数参数
        }
        # 更新自定义参数
        super().__init__(params)
        self.params = {**default_params, **(params or {})}

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算策略指标"""
        df = data.copy()
        
        # 计算关键价格
        df['Price'] = df['close']
        
        # 计算平移boll通道
        df['AvgVal'] = Average(df['Price'], self.params['AvgLen'])
        df['SDmult'] = StandardDev(df['Price'], self.params['SDLen'], 2) * self.params['SDev']
        
        # 计算平移后的通道上下轨
        df['DispTop'] = df['AvgVal'].shift(self.params['Disp']) + df['SDmult']
        df['DispBottom'] = df['AvgVal'].shift(self.params['Disp']) - df['SDmult']
        
        return df

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """生成交易信号矩阵"""
        # 数据验证
        min_length = self.params['Disp'] + max(self.params['AvgLen'], self.params['SDLen'])
        if len(data) < min_length:
            raise ValueError('数据长度不足')
        
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0
        signals['quantity'] = 1
        signals['entry_price'] = np.nan
        signals['exit_price'] = np.nan
        signals['position'] = 0
        
        current_position = 0  # 当前持仓状态
        bars_since_entry = 0  # 入场后的K线计数
        
        for i in range(1, len(data)):
            if current_position == 0:  # 空仓
                if data['high'].iloc[i] >= data['DispTop'].iloc[i-1]:
                    # 开多仓
                    entry_price = max(data['open'].iloc[i], data['DispTop'].iloc[i-1])
                    signals.iloc[i] = [1, 1, entry_price, np.nan, 1]
                    current_position = 1
                    bars_since_entry = 0
                    
            elif current_position == 1:  # 持有多仓
                bars_since_entry += 1
                if bars_since_entry > 0 and data['low'].iloc[i] <= data['DispBottom'].iloc[i-1]:
                    # 平多仓
                    exit_price = min(data['open'].iloc[i], data['DispBottom'].iloc[i-1])
                    signals.iloc[i] = [-1, 1, np.nan, exit_price, 0]
                    current_position = 0
                    bars_since_entry = 0
        
        # 在最后一根K线强制平仓
        if current_position == 1:
            signals.iloc[-1] = [-1, 1, np.nan, data['close'].iloc[-1], 0]
            
        return signals
