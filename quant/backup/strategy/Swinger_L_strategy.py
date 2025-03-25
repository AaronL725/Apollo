'''
// 简称: Swinger_L
// 名称: 基于均线与动能的交易系统多 
// 类别: 策略应用
// 类型: 内建应用
//------------------------------------------------------------------------
//----------------------------------------------------------------------//
// 策略说明:
//             本策略基于均线与均线间的动能变化建立的交易系统
//             
// 系统要素:
//             1. 1根长期均线进行趋势判断
//             2. 2根较短均线值之差揭示的动能变化为交易提供基础
// 入场条件:
//             1. 当价格高于长期均线且动能相对之前变强时做多
//             2. 当价格低于长期均线且动能相对之前变弱时做空
// 出场条件: 
//             1. 当动能减弱时, 价格低于ExitStopN根K线低点多头平仓
//             2. 当动能增强时, 价格高于ExitStopN根K线高点空头平仓
//
//         注: 当前策略仅为做多系统, 如需做空, 请参见CL_Swinger_S
'''

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict
import pandas as pd
import numpy as np
from module import *
from module.indicators import AverageFC, PriceOscillator, LowestFC
from strategy.base import StrategyBase

class Swinger_L(StrategyBase):
    """趋势震荡做多策略"""
    def __init__(self, params: Dict = None):
        # 默认参数设置
        default_params = {
            'FastMALength': 5,    # 动能计算中的快均线值
            'SlowMALength': 20,   # 动能计算中的慢均线值
            'TrendMALength': 50,  # 显示趋势的均线值
            'ExitStopN': 3        # 求高低点的bar数值
        }
        # 更新自定义参数
        if params:
            default_params.update(params)
        super().__init__(default_params)
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算策略所需的技术指标"""
        df = data.copy()
        
        # 计算趋势线和均线动能
        df['TrendMA'] = AverageFC(df['close'], self.params['TrendMALength'])
        df['PriceOsci'] = PriceOscillator(
            df['close'], 
            self.params['FastMALength'],
            self.params['SlowMALength']
        )
        
        # 计算出场价格（前N根K线的最低点）
        df['ExitL'] = LowestFC(df['low'], self.params['ExitStopN'])
        
        return df
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """生成交易信号矩阵"""
        # 数据验证
        min_length = max(self.params['TrendMALength'], 
                        self.params['SlowMALength'], 
                        self.params['ExitStopN'])
        if len(data) < min_length:
            raise ValueError('数据长度不足')
        
        signals = pd.DataFrame(index=data.index)
        signals['call'] = np.nan
        
        # 使用.values提高性能
        close = data['close'].values
        open_ = data['open'].values
        low = data['low'].values
        trend_ma = data['TrendMA'].values
        price_osci = data['PriceOsci'].values
        exit_l = data['ExitL'].values
        vol = data['vol'].values
        
        current_position = 0
        
        for i in range(min_length, len(data)):
            # 开多仓条件：
            # 1. 当前无仓位
            # 2. 上根K线收盘价高于趋势线
            # 3. 上根K线动能为负且大于上上根动能
            # 4. 成交量大于0
            if (current_position == 0 and
                close[i-1] > trend_ma[i-1] and
                price_osci[i-1] <= 0 and
                price_osci[i-1] > price_osci[i-2] and
                vol[i] > 0):
                
                signals.iloc[i] = [1]
                current_position = 1
            
            # 平多仓条件：
            # 1. 当前持有多仓
            # 2. 动能减弱（上根动能小于上上根）
            # 3. 最低价跌破前N根K线低点
            # 4. 成交量大于0
            elif (current_position == 1 and
                  price_osci[i-1] < price_osci[i-2] and
                  low[i] <= exit_l[i-1] and
                  vol[i] > 0):
                
                signals.iloc[i] = [0]
                current_position = 0
        
        # 循环结束时强制平仓
        if current_position == 1:
            signals.iloc[i] = [0]
        
        return signals