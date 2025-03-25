'''
// 简称: Swinger_S
// 名称: 基于均线与动能的交易系统空 
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
//         注: 当前策略仅为做空系统, 如需做多, 请参见CL_Swinger_L
'''

from typing import Dict
import pandas as pd
import numpy as np
from module import *
from .base import StrategyBase

class Swinger_S(StrategyBase):
    """趋势震荡做空策略"""
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
        
        # 计算出场价格（前N根K线的最高点）
        df['ExitS'] = HighestFC(df['high'], self.params['ExitStopN'])
        
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
        signals['signal'] = 0
        signals['quantity'] = 1
        signals['entry_price'] = np.nan
        signals['exit_price'] = np.nan
        signals['position'] = 0
        
        # 使用.values提高性能
        close = data['close'].values
        open_ = data['open'].values
        high = data['high'].values
        trend_ma = data['TrendMA'].values
        price_osci = data['PriceOsci'].values
        exit_s = data['ExitS'].values
        vol = data['vol'].values
        
        current_position = 0
        
        for i in range(min_length, len(data)):
            # 开空仓条件：
            # 1. 当前无仓位
            # 2. 上根K线收盘价低于趋势线
            # 3. 上根K线动能为正且小于上上根动能
            # 4. 成交量大于0
            if (current_position == 0 and
                close[i-1] < trend_ma[i-1] and
                price_osci[i-1] >= 0 and
                price_osci[i-1] < price_osci[i-2] and
                vol[i] > 0):
                
                signals.iloc[i] = [-1, 1, open_[i], np.nan, -1]
                current_position = -1
            
            # 平空仓条件：
            # 1. 当前持有空仓
            # 2. 动能增强（上根动能大于上上根）
            # 3. 最高价突破前N根K线高点
            # 4. 成交量大于0
            elif (current_position == -1 and
                  price_osci[i-1] > price_osci[i-2] and
                  high[i] >= exit_s[i-1] and
                  vol[i] > 0):
                
                signals.iloc[i] = [1, 1, np.nan, 
                                 max(open_[i], exit_s[i-1]), 0]
                current_position = 0
        
        return signals 