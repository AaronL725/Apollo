'''
// 简称: Three_EMA_Crossover_System_S
// 名称: 基于指数移动平均线组进行判断 空
// 类别: 策略应用
// 类型: 内建应用
// 输出:
//------------------------------------------------------------------------
//------------------------------------------------------------------------
// 策略说明:
//            1.计算三条指数移动平均线(Avg1, Avg2 , Avg3)；
//            2.通过指数移动平均线的组合来判断趋势
//             
// 入场条件:
//            1.当Avg1向上穿过Avg2并且Avg2大于Avg3时，在下一根k线开盘处买入
//            2.当Avg1向下穿过Avg2并且Avg2小于Avg3时，在下一根k线开盘处卖出
// 出场条件: 
//            1.Avg1上穿Avg2空头出场
//            2.跟踪止损
//
//         注: 当前策略仅为做空系统, 如需做多, 请参见CL_Three_EMA_Crossover_System_L
//----------------------------------------------------------------------//
'''

from typing import Dict
import pandas as pd
import numpy as np
from module import *
from .base import StrategyBase

class Three_EMA_Crossover_System_S(StrategyBase):
    """基于三重指数移动平均线的做空策略"""
    def __init__(self, params: Dict = None):
        # 默认参数设置
        default_params = {
            'AvgLen1': 6,     # 指数移动平均周期1
            'AvgLen2': 12,    # 指数移动平均周期2
            'AvgLen3': 28,    # 指数移动平均周期3
            'RLength': 4,     # 跟踪止损周期
            'Lots': 1         # 交易手数
        }
        
        if params:
            default_params.update(params)
        super().__init__(default_params)
        
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算技术指标"""
        df = data.copy()
        
        # 计算三条指数移动平均线
        df['Avg1'] = df['close'].ewm(span=self.params['AvgLen1'], adjust=False).mean()
        df['Avg2'] = df['close'].ewm(span=self.params['AvgLen2'], adjust=False).mean()
        df['Avg3'] = df['close'].ewm(span=self.params['AvgLen3'], adjust=False).mean()
        
        # 计算K线幅度和平均幅度
        df['MyRange'] = df['high'] - df['low']
        df['RangeS'] = df['MyRange'].rolling(window=self.params['RLength']).mean()
        
        # 计算做空条件：Avg1向下穿过Avg2 (CrossUnder实现)
        df['SellCon1'] = (df['Avg1'] < df['Avg2']) & (df['Avg1'].shift(1) > df['Avg2'].shift(1))
        
        return df
        
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """生成交易信号矩阵"""
        # 数据验证
        min_length = max(self.params['AvgLen1'], self.params['AvgLen2'], 
                        self.params['AvgLen3'], self.params['RLength'])
        if len(data) < min_length:
            raise ValueError('数据长度不足')
            
        signals = pd.DataFrame(index=data.index)
        signals['call'] = np.nan
        
        current_position = 0  # 当前持仓状态
        short_stop_price = 0  # 跟踪止损价
        prev_short_stop_price = 0  # 前一个bar的止损价
        position_bars = 0     # 持仓天数
        
        for i in range(1, len(data)):
            if current_position == 0:  # 当前无仓位
                # 满足做空条件：SellCon1为True且Avg2小于Avg3
                if (data['SellCon1'].iloc[i-1] and 
                    data['Avg2'].iloc[i-1] < data['Avg3'].iloc[i-1] and 
                    data['vol'].iloc[i] > 0):
                    signals.iloc[i] = -1
                    current_position = -1
                    position_bars = 0
                    # 进场后立即设置跟踪止损价
                    short_stop_price = data['high'].iloc[i] + data['RangeS'].iloc[i]
                    prev_short_stop_price = short_stop_price
                    
            elif current_position == -1:  # 当前持有空仓
                position_bars += 1
                
                # 检查出场条件
                # 1. Avg1上穿Avg2
                if (data['Avg1'].iloc[i-1] > data['Avg2'].iloc[i-1] and 
                    data['vol'].iloc[i] > 0):
                    signals.iloc[i] = 0
                    current_position = 0
                    position_bars = 0
                    
                # 2. 突破跟踪止损价 (使用前一bar的止损价)
                elif (data['high'].iloc[i] >= prev_short_stop_price and 
                      data['vol'].iloc[i] > 0):
                    signals.iloc[i] = 0
                    current_position = 0
                    position_bars = 0
                
                # 更新跟踪止损价
                if position_bars > 0:
                    prev_short_stop_price = short_stop_price
                    short_stop_price = short_stop_price - (short_stop_price - data['high'].iloc[i]) / 3
        
        # 在最后一根K线强制平仓
        if current_position == -1:
            signals.iloc[-1] = 0
            
        return signals
