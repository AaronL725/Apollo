'''
// 简称: DynamicBreakOutII_S
// 名称: 动态突破_空
// 类别: 策略应用
// 类型: 内建应用
// 输出:
//------------------------------------------------------------------------
/* 
策略说明:
        基于自适应的布林通道与自适应的唐奇安通道的突破系统
系统要素:
        1、自适应布林通道
        2、自适应唐奇安通道
        3、自适应出场均线
入场条件:
        1、昨日价格大于布林通道上轨，并且当日周期价格大于唐奇安通道上轨，开多单
        2、昨日价格小于布林通道下轨，并且当日周期价格小于唐奇安通道下轨，开空单
出场条件:
        1、持有多单时，价格小于自适应出场均线，平多单
        2、持有空单时，价格大于自适应出场均线，平空单
注    意:
        此公式仅做空
'''

from typing import Dict
import pandas as pd
import numpy as np
from module import *
from .base import StrategyBase

class DynamicBreakOutII_S(StrategyBase):
    """动态突破做空策略"""
    def __init__(self, params: Dict = None):
        # 默认参数设置
        default_params = {
            'ceilingAmt': 60,    # 自适应参数的上限
            'floorAmt': 20,      # 自适应参数的下限
            'bolBandTrig': 2,    # 布林通道参数
            'Lots': 1            # 交易手数
        }
        # 更新自定义参数
        super().__init__(params)
        self.params = {**default_params, **(params or {})}
        
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算策略指标"""
        df = data.copy()
        
        # 当日市场波动
        df['todayVolatility'] = StandardDev(df['close'], 30, 1)
        
        # 昨日市场波动
        df['yesterDayVolatility'] = df['todayVolatility'].shift(1)
        
        # 市场波动的变动率 - 处理除以0和无穷大的情况
        df['deltaVolatility'] = np.where(
            df['todayVolatility'] > 0,
            (df['todayVolatility'] - df['yesterDayVolatility']) / df['todayVolatility'],
            0
        )
        df['deltaVolatility'] = df['deltaVolatility'].fillna(0)  # 填充NaN值
        df['deltaVolatility'] = df['deltaVolatility'].clip(-1, 1)  # 限制变动率在 -100% 到 100% 之间
        
        # 初始化自适应参数
        lookback_days = pd.Series(20, index=df.index, dtype=float)  # 使用float类型
        
        # 计算自适应参数
        for i in range(1, len(df)):
            lookback_days.iloc[i] = lookback_days.iloc[i-1] * (1 + df['deltaVolatility'].iloc[i])
            lookback_days.iloc[i] = int(round(lookback_days.iloc[i]))  # 确保转为整数
            lookback_days.iloc[i] = min(lookback_days.iloc[i], self.params['ceilingAmt'])
            lookback_days.iloc[i] = max(lookback_days.iloc[i], self.params['floorAmt'])
        
        df['lookBackDays'] = lookback_days
        
        # 初始化列以避免NaN问题
        df['MidLine'] = np.nan
        df['Band'] = np.nan
        df['buyPoint'] = np.nan
        df['sellPoint'] = np.nan
        
        # 计算动态指标
        for i in range(len(df)):
            period = int(df['lookBackDays'].iloc[i])
            if i >= period - 1:
                # 自适应布林通道
                df.loc[df.index[i], 'MidLine'] = df['close'].iloc[i-period+1:i+1].mean()
                df.loc[df.index[i], 'Band'] = StandardDev(df['close'].iloc[i-period+1:i+1], period, 2).iloc[-1]
                
                # 自适应唐奇安通道
                df.loc[df.index[i], 'buyPoint'] = df['high'].iloc[i-period+1:i+1].max()
                df.loc[df.index[i], 'sellPoint'] = df['low'].iloc[i-period+1:i+1].min()
        
        # 计算布林带
        df['upBand'] = df['MidLine'] + self.params['bolBandTrig'] * df['Band']
        df['dnBand'] = df['MidLine'] - self.params['bolBandTrig'] * df['Band']
        
        # 自适应出场均线
        df['LiqPoint'] = df['MidLine']
        
        return df
        
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """生成交易信号矩阵"""
        # 数据验证
        if len(data) < self.params['ceilingAmt']:
            raise ValueError('数据长度不足')
        
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0
        signals['quantity'] = self.params['Lots']
        signals['entry_price'] = np.nan
        signals['exit_price'] = np.nan
        signals['position'] = 0
        
        current_position = 0  # 当前持仓状态
        position_bars = 0     # 持仓天数
        
        for i in range(1, len(data)):
            if i < 1:  # 跳过第一根K线
                continue
                
            close = data['close'].iloc[i-1]  # 昨日收盘价
            open_ = data['open'].iloc[i]     # 今日开盘价
            high = data['high'].iloc[i]      # 今日最高价
            low = data['low'].iloc[i]        # 今日最低价
            
            if current_position == 0:  # 空仓
                # 开空仓条件
                if (close < data['dnBand'].iloc[i-1] and 
                    low <= data['sellPoint'].iloc[i-1]):
                    entry_price = min(open_, data['sellPoint'].iloc[i-1])
                    signals.iloc[i] = [-1, self.params['Lots'], entry_price, np.nan, -1]
                    current_position = -1
                    position_bars = 0
                    
            elif current_position == -1:  # 持有空仓
                position_bars += 1
                
                # 平空仓条件1：反向突破
                if (close > data['upBand'].iloc[i-1] and 
                    high >= data['buyPoint'].iloc[i-1]):
                    exit_price = max(open_, data['buyPoint'].iloc[i-1])
                    signals.iloc[i] = [1, self.params['Lots'], np.nan, exit_price, 0]
                    current_position = 0
                    position_bars = 0
                
                # 平空仓条件2：均线平仓
                elif (position_bars >= 1 and 
                      high >= data['LiqPoint'].iloc[i-1]):
                    exit_price = max(open_, data['LiqPoint'].iloc[i-1])
                    signals.iloc[i] = [1, self.params['Lots'], np.nan, exit_price, 0]
                    current_position = 0
                    position_bars = 0
        
        # 在最后一根K线强制平仓
        if current_position == -1:
            signals.iloc[-1] = [1, self.params['Lots'], np.nan, data['close'].iloc[-1], 0]
            
        return signals