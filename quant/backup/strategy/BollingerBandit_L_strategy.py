'''
// 简称: BollingerBandit_L
// 名称: 布林强盗_多
// 类别: 策略应用
// 类型: 内建应用
// 输出:
//------------------------------------------------------------------------
/* 
策略说明:
        基于布林通道的突破系统
系统要素:
        1、基于收盘价计算而来的布林通道
        2、基于收盘价计算而来的进场过滤器
        3、自适应出场均线
入场条件:
        1、满足过滤条件，并且价格上破布林通道上轨，开多单
        2、满足过滤条件，并且价格下破布林通道下轨，开空单
出场条件:
        1、持有多单时，自适应出场均线低于布林通道上轨，并且价格下破自适应出场均线，平多单
        2、持有空单时，自适应出场均线高于布林通道下轨，并且价格上破自适应出场均线，平空单
注    意:
        此公式仅做多
'''

from typing import Dict
import pandas as pd
import numpy as np
from module import *
from .base import StrategyBase

class BollingerBandit_L(StrategyBase):
    """布林强盗做多策略"""
    def __init__(self, params: Dict = None):
        # 默认参数设置
        default_params = {
            'bollingerLengths': 50,  # 布林通道参数
            'Offset': 1.25,          # 布林通道参数
            'rocCalcLength': 30,     # 过滤器参数
            'liqLength': 50,         # 自适应出场均线参数
            'Lots': 1                # 交易手数
        }
        # 更新自定义参数
        if params:
            default_params.update(params)
        super().__init__(default_params)

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算策略指标"""
        df = data.copy()
        
        # 布林通道中轨
        df['MidLine'] = Average(df['close'], self.params['bollingerLengths'])
        
        # 计算标准差带
        df['Band'] = StandardDev(df['close'], self.params['bollingerLengths'], 2)
        
        # 布林通道上轨
        df['upBand'] = df['MidLine'] + self.params['Offset'] * df['Band']
        
        # 进场过滤器
        df['rocCalc'] = df['close'] - df['close'].shift(self.params['rocCalcLength'])
        
        # 初始化自适应出场均线参数
        df['liqDays'] = self.params['liqLength']
        
        return df

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """生成交易信号矩阵"""
        # 数据验证
        min_length = max(self.params['bollingerLengths'], self.params['rocCalcLength'])
        if len(data) < min_length:
            raise ValueError('数据长度不足')
            
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0
        signals['quantity'] = 1
        signals['entry_price'] = np.nan
        signals['exit_price'] = np.nan
        signals['position'] = 0
        
        current_position = 0  # 当前持仓状态
        position_bars = 0     # 持仓天数
        
        for i in range(1, len(data)):
            if current_position == 0:  # 无持仓
                # 满足过滤条件，并且价格突破布林通道上轨，开多单
                if (data['rocCalc'].iloc[i-1] > 0 and 
                    data['high'].iloc[i] >= data['upBand'].iloc[i-1]):
                    signals.iloc[i] = [1, self.params['Lots'], 
                                     max(data['open'].iloc[i], data['upBand'].iloc[i-1]),
                                     np.nan, 0]
                    current_position = 1
                    position_bars = 0
                    
            elif current_position == 1:  # 持有多仓
                position_bars += 1
                # 更新自适应出场均线参数
                data.loc[data.index[i], 'liqDays'] = max(data['liqDays'].iloc[i-1] - 1, 10)
                # 计算自适应出场均线
                liq_point = Average(data['close'].iloc[:i+1], int(data['liqDays'].iloc[i]))
                
                # 持有多单时，自适应出场均线低于布林通道上轨，并且价格下破自适应出场均线，平多单
                if (position_bars >= 1 and 
                    liq_point.iloc[-2] < data['upBand'].iloc[i-1] and 
                    data['low'].iloc[i] <= liq_point.iloc[-2]):
                    signals.iloc[i] = [-1, self.params['Lots'], np.nan,
                                     min(data['open'].iloc[i], liq_point.iloc[-2]),
                                     position_bars]
                    current_position = 0
                    
        # 在最后一根K线强制平仓
        if current_position == 1:
            signals.iloc[-1] = [-1, self.params['Lots'], np.nan, data['close'].iloc[-1], position_bars]
            
        return signals
