'''
// 简称: SupermanSystem_S
// 名称: 基于市场强弱和动量的通道突破系统空 
// 类别: 策略应用
// 类型: 内建应用
// 输出:
//------------------------------------------------------------------------
//----------------------------------------------------------------------//
// 策略说明:
//             本策略是基于市场强弱指标和动量的通道突破系统
//             
// 系统要素:
//             1. 根据N根K线的收盘价相对前一根K线的涨跌计算出市场强弱指标
//             2. 最近9根K线的动量变化趋势
//             3. 最近N根K线的高低点形成的通道
// 入场条件:
//             1. 市场强弱指标为多头，且市场动量由空转多时，突破通道高点做多
//             2. 市场强弱指标为空头，且市场动量由多转空时，突破通道低点做空
// 出场条件: 
//             1. 开多以开仓BAR的最近N根BAR的低点作为止损价
//                开空以开仓BAR的最近N根BAR的高点作为止损价
//             2. 盈利超过止损额的一定倍数止盈
//             3. 出现反向信号止损
//
//         注: 当前策略仅为做空系统, 如需做多, 请参见CL_SupermanSystem_L
'''

from typing import Dict
import pandas as pd
import numpy as np
from module import *
from .base import StrategyBase

class SupermanSystem_S(StrategyBase):
    """市场强弱指标做空策略"""
    def __init__(self, params: Dict = None):
        # 默认参数设置
        default_params = {
            'Length': 5,           # 强弱指标和通道计算的周期值
            'Stop_Len': 5,        # 止损通道的周期值
            'ProfitFactor': 3,    # 止盈相对止损的倍数
            'EntryStrength': 95   # 强弱指标的进场值
        }
        if params:
            default_params.update(params)
        super().__init__(default_params)
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算策略所需的技术指标"""
        df = data.copy()
        
        # 计算收盘价变动
        df['CloseChange'] = df['close'] - df['close'].shift(1)
        
        # 计算市场强弱指标
        def calculate_market_strength(window_data):
            up_closes = window_data[window_data > 0].sum()
            dn_closes = window_data[window_data <= 0].sum()
            sum_change = window_data.sum()
            
            if sum_change >= 0:
                return (sum_change / up_closes * 100) if up_closes != 0 else 0
            else:
                return (sum_change / abs(dn_closes) * 100) if dn_closes != 0 else 0
        
        # 使用rolling window计算MarketStrength
        df['MarketStrength'] = df['CloseChange'].rolling(
            window=self.params['Length']
        ).apply(calculate_market_strength)
        
        # 计算动量指标
        df['Momentum1'] = df['close'] - df['close'].shift(4)
        df['Momentum2'] = df['close'].shift(4) - df['close'].shift(8)
        
        # 计算高低点
        df['HH1'] = HighestFC(df['high'], self.params['Length'])
        df['LL'] = LowestFC(df['low'], self.params['Length'])
        df['HH2'] = HighestFC(df['high'], self.params['Stop_Len'])
        
        return df
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """生成交易信号矩阵"""
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0
        signals['quantity'] = 1
        signals['entry_price'] = np.nan
        signals['exit_price'] = np.nan
        signals['position'] = 0
        
        current_position = 0
        stop_loss = np.nan
        profit_target = np.nan
        entry_price = np.nan
        
        for i in range(8, len(data)):  # 从第9根K线开始（因为需要8根K线的历史数据）
            if current_position == 0:  # 空仓状态
                # 开空仓条件
                if (data['MarketStrength'].iloc[i-1] <= -1 * self.params['EntryStrength'] and
                    data['Momentum1'].iloc[i-1] <= 0 and
                    data['Momentum2'].iloc[i-1] > 0 and
                    data['low'].iloc[i] <= data['LL'].iloc[i-1] and
                    data['vol'].iloc[i] > 0):
                    
                    entry_price = min(data['open'].iloc[i], data['LL'].iloc[i-1])
                    stop_loss = data['HH2'].iloc[i]
                    profit_target = entry_price - (stop_loss - entry_price) * self.params['ProfitFactor']
                    
                    signals.iloc[i] = [-1, 1, entry_price, np.nan, -1]
                    current_position = -1
            
            elif current_position == -1:  # 持有空仓
                if data['vol'].iloc[i] > 0:
                    # 止盈
                    if data['low'].iloc[i] <= profit_target:
                        signals.iloc[i] = [1, 1, np.nan, min(data['open'].iloc[i], profit_target), 0]
                        current_position = 0
                    
                    # 止损
                    elif data['high'].iloc[i] >= stop_loss:
                        signals.iloc[i] = [1, 1, np.nan, max(data['open'].iloc[i], stop_loss), 0]
                        current_position = 0
                    
                    # 反向出场
                    elif (data['MarketStrength'].iloc[i-1] >= self.params['EntryStrength'] and
                          data['Momentum1'].iloc[i-1] > 0 and
                          data['Momentum2'].iloc[i-1] <= 0 and
                          data['high'].iloc[i] >= data['HH1'].iloc[i-1]):
                        signals.iloc[i] = [1, 1, np.nan, max(data['open'].iloc[i], data['HH1'].iloc[i-1]), 0]
                        current_position = 0
        
        return signals 