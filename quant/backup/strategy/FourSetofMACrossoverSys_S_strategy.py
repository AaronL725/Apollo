'''
// 简称: FourSetofMACrossoverSys_S
// 名称: 四均线交易系统 空
// 类别: 策略应用
// 类型: 内建应用
// 输出:
// 策略说明:
//                基于4均线系统进行判断交易
//
// 系统要素:
//                (5和20周期均线),(3和10周期均线)构成的两组不同周期的均线组合
//
// 入场条件:    
//                当2组均线均成空头排列时且当前价低于上根BAR最低价入场
//
// 出场条件:    
//                1 小周期空头均线组合成多头排列
//                2 两组多头均线分别多头排列且低于上根BAR最高价出场
'''

from typing import Dict
import pandas as pd
import numpy as np
from module import *
from .base import StrategyBase

class FourSetofMACrossoverSys_S(StrategyBase):
    """四均线交易系统做空策略"""
    def __init__(self, params: Dict = None):
        # 默认参数设置
        default_params = {
            'LEFast': 5,    # 多头入场短均线周期参数
            'LESlow': 20,   # 多头入场长均线周期参数 
            'LXFast': 3,    # 多头出场短均线周期参数
            'LXSlow': 10,   # 多头出场长均线周期参数
            'SEFast': 5,    # 空头入场短均线周期参数
            'SESlow': 20,   # 空头入场长均线周期参数
            'SXFast': 3,    # 空头出场短均线周期参数
            'SXSlow': 10    # 空头出场长均线周期参数
        }
        
        # 更新自定义参数
        if params:
            default_params.update(params)
            
        super().__init__(default_params)
        
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算策略指标"""
        df = data.copy()
        
        # 计算各周期均线
        df['MALEFast'] = Average(df['close'], self.params['LEFast'])  # 多头入场短均线
        df['MALESlow'] = Average(df['close'], self.params['LESlow'])  # 多头入场长均线
        df['MALXFast'] = Average(df['close'], self.params['LXFast'])  # 多头出场短均线
        df['MALXSlow'] = Average(df['close'], self.params['LXSlow'])  # 多头出场长均线
        df['MASEFast'] = Average(df['close'], self.params['SEFast'])  # 空头入场短均线
        df['MASESlow'] = Average(df['close'], self.params['SESlow'])  # 空头入场长均线
        df['MASXFast'] = Average(df['close'], self.params['SXFast'])  # 空头出场短均线
        df['MASXSlow'] = Average(df['close'], self.params['SXSlow'])  # 空头出场长均线
        
        return df
        
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """生成交易信号矩阵"""
        # 数据验证
        min_length = max(self.params['LESlow'], self.params['SESlow'])
        if len(data) < min_length:
            raise ValueError('数据长度不足')
        
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0
        signals['quantity'] = 1
        signals['entry_price'] = np.nan
        signals['exit_price'] = np.nan
        signals['position'] = 0
        
        current_position = 0  # 当前持仓状态
        
        for i in range(100, len(data)):  # 从第100根K线开始交易
            # 获取当前值
            low = data['low'].iloc[i]
            high = data['high'].iloc[i]
            open_ = data['open'].iloc[i]
            vol = data['vol'].iloc[i]
            
            # 获取前一根K线的均线值和价格
            mase_fast_prev = data['MASEFast'].iloc[i-1]
            mase_slow_prev = data['MASESlow'].iloc[i-1]
            masx_fast_prev = data['MASXFast'].iloc[i-1]
            masx_slow_prev = data['MASXSlow'].iloc[i-1]
            male_fast_prev = data['MALEFast'].iloc[i-1]
            male_slow_prev = data['MALESlow'].iloc[i-1]
            low_prev = data['low'].iloc[i-1]
            high_prev = data['high'].iloc[i-1]
            
            # 入场信号
            if (current_position == 0 and
                mase_fast_prev < mase_slow_prev and
                masx_fast_prev < masx_slow_prev and
                low <= low_prev and
                vol > 0):
                
                entry_price = min(open_, low_prev)
                signals.iloc[i] = [-1, 1, entry_price, np.nan, -1]
                current_position = -1
                
            # 出场信号
            elif current_position == -1 and vol > 0:
                # 小周期空头均线组合成多头排列出场
                if masx_fast_prev > masx_slow_prev:
                    signals.iloc[i] = [1, 1, np.nan, open_, 0]
                    current_position = 0
                    
                # 两组均线分别多头排列且高于上根BAR最高价出场
                elif (male_fast_prev > male_slow_prev and
                      masx_fast_prev > masx_slow_prev and
                      high >= high_prev):
                    
                    exit_price = max(open_, high_prev)
                    signals.iloc[i] = [1, 1, np.nan, exit_price, 0]
                    current_position = 0
        
        # 在最后一根K线强制平仓
        if current_position == -1:
            signals.iloc[-1] = [1, 1, np.nan, data['close'].iloc[-1], 0]
            
        return signals