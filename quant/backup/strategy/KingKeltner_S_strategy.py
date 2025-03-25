'''
// 简称: KingKeltner_S
// 名称: 金肯特纳_空
// 类别: 策略应用
// 类型: 内建应用
// 输出:
//------------------------------------------------------------------------
/* 
策略说明:
        基于肯特纳通道的突破系统
系统要素:
        1、基于最高价、最低价、收盘价三者平均值计算而来的三价均线
        2、基于三价均线加减真实波幅计算而来的通道上下轨
入场条件:
        1、三价均线向上，并且价格上破通道上轨，开多单
        2、三价均线向下，并且价格下破通道下轨，开空单
出场条件:
        1、持有多单时，价格下破三价均线，平多单
        2、持有空单时，价格上破三价均线，平空单
注    意:
        此公式仅做空
'''

from typing import Dict
import pandas as pd
import numpy as np
from module import *
from .base import StrategyBase

class KingKeltner_S(StrategyBase):
    """金肯特纳通道做空策略"""
    def __init__(self, params: Dict = None):
        # 默认参数设置
        default_params = {
            'avgLength': 40,     # 三价均线参数
            'atrLength': 40,     # 真实波幅参数
        }
        # 更新自定义参数
        if params:
            default_params.update(params)
        super().__init__(default_params)
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算策略指标"""
        df = data.copy()
        
        # 计算三价均线
        df['TypicalPrice'] = (df['high'] + df['low'] + df['close']) / 3
        df['movAvgVal'] = Average(df['TypicalPrice'], self.params['avgLength'])
        
        # 计算ATR通道
        df['ATR'] = AvgTrueRange(self.params['atrLength'], df['high'], df['low'], df['close'])
        df['dnBand'] = df['movAvgVal'] - df['ATR']
        
        # 出场条件线
        df['liquidPoint'] = df['movAvgVal']
        
        return df
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """生成交易信号矩阵"""
        # 数据验证
        min_length = max(self.params['avgLength'], self.params['atrLength'])
        if len(data) < min_length:
            raise ValueError('数据长度不足')
        
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0
        signals['quantity'] = 1
        signals['entry_price'] = np.nan
        signals['exit_price'] = np.nan
        signals['position'] = 0
        
        current_position = 0  # 当前持仓状态：0表示空仓，-1表示空头
        position_bars = 0     # 持仓天数
        
        # 获取需要的数据列
        mov_avg = data['movAvgVal']
        dn_band = data['dnBand']
        liquid_point = data['liquidPoint']
        high = data['high']
        low = data['low']
        open_ = data['open']
        vol = data['vol']
        
        for i in range(2, len(data)):
            # 更新持仓天数
            if current_position != 0:
                position_bars += 1
            
            # 开空条件：
            # 1. 当前空仓
            # 2. 三价均线向下
            # 3. 最低价下破通道下轨
            # 4. 成交量大于0
            if (current_position == 0 and
                mov_avg.iloc[i-1] < mov_avg.iloc[i-2] and
                low.iloc[i] <= dn_band.iloc[i-1] and
                vol.iloc[i] > 0):
                
                entry_price = min(open_.iloc[i], dn_band.iloc[i-1])
                signals.iloc[i] = [-1, 1, entry_price, np.nan, -1]
                current_position = -1
                position_bars = 0
            
            # 平空条件：
            # 1. 持仓时间大于等于1根K线
            # 2. 最高价上破三价均线
            # 3. 成交量大于0
            elif (current_position == -1 and
                  position_bars >= 1 and
                  high.iloc[i] >= liquid_point.iloc[i-1] and
                  vol.iloc[i] > 0):
                
                exit_price = max(open_.iloc[i], liquid_point.iloc[i-1])
                signals.iloc[i] = [1, 1, np.nan, exit_price, 0]
                current_position = 0
                position_bars = 0
        
        # 在最后一个bar强制平仓
        if current_position == -1:
            last_idx = signals.index[-1]
            signals.loc[last_idx] = [1, 1, np.nan, data['close'].iloc[-1], 0]
        
        return signals