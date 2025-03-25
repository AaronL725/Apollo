'''
// 简称: Thermostat_S
// 名称: 恒温器_空
// 类别: 策略应用
// 类型: 内建应用
// 输出:
//------------------------------------------------------------------------
/* 
策略说明:
        通过计算市场的潮汐指数，把市场划分为震荡和趋势两种走势；震荡市中
采用开盘区间突破进场；趋势市中采用布林通道突破进场。
系统要素:
        1、潮汐指数
        2、关键价格
        3、布林通道
        4、真实波幅
        5、出场均线
入场条件:
        1、震荡市中采用开盘区间突破进场
        2、趋势市中采用布林通道突破进场
出场条件:
        1、震荡市时进场单的出场为反手信号和ATR保护性止损
        2、趋势市时进场单的出场为反手信号和均线出场
注    意:
        此公式仅做空
'''

from typing import Dict
import pandas as pd
import numpy as np
from module import *
from module.indicators import TrueRange
from .base import StrategyBase

class Thermostat_S(StrategyBase):
    """恒温器做空策略"""
    def __init__(self, params: Dict = None):
        # 默认参数设置
        default_params = {
            'swingTrendSwitch': 20,    # 潮汐指数小于此值为震荡市，否则为趋势市
            'swingPrcnt1': 0.50,       # 震荡市开仓参数
            'swingPrcnt2': 0.75,       # 震荡市开仓参数
            'atrLength': 10,           # 真实波幅参数
            'bollingerLengths': 50,    # 布林通道参数
            'numStdDevs': 2,           # 布林通道参数
            'trendLiqLength': 50,      # 趋势市时进场单的出场均线参数
            'Lots': 1                  # 交易手数
        }
        if params:
            default_params.update(params)
        super().__init__(default_params)

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算技术指标"""
        df = data.copy()
        
        # 计算潮汐指数用以区分震荡市与趋势市
        df['cmiVal'] = (abs(df['close'] - df['close'].shift(29)) / 
                       (df['high'].rolling(30).max() - df['low'].rolling(30).min())) * 100
        
        df['trendLokBuy'] = df['low'].rolling(3).mean()
        df['trendLokSell'] = df['high'].rolling(3).mean()
        
        # 关键价格
        df['keyOfDay'] = (df['high'] + df['low'] + df['close']) / 3
        
        # 震荡市中收盘价大于关键价格为宜卖市，否则为宜买市
        df['buyEasierDay'] = df['close'].shift(1) <= df['keyOfDay'].shift(1)
        df['sellEasierDay'] = df['close'].shift(1) > df['keyOfDay'].shift(1)
        
        # 计算真实波幅
        df['TR'] = TrueRange(df['high'], df['low'], df['close'])
        df['myATR'] = df['TR'].rolling(window=self.params['atrLength']).mean()
        
        # 计算震荡市的进场价格
        df['swingBuyPt'] = np.where(
            df['buyEasierDay'],
            df['open'] + self.params['swingPrcnt1'] * df['myATR'].shift(1),
            df['open'] + self.params['swingPrcnt2'] * df['myATR'].shift(1)
        )
        
        df['swingSellPt'] = np.where(
            df['sellEasierDay'],
            df['open'] - self.params['swingPrcnt1'] * df['myATR'].shift(1),
            df['open'] - self.params['swingPrcnt2'] * df['myATR'].shift(1)
        )
        
        df['swingBuyPt'] = df[['swingBuyPt', 'trendLokBuy']].max(axis=1)
        df['swingSellPt'] = df[['swingSellPt', 'trendLokSell']].min(axis=1)
        
        # 计算布林通道
        df['MidLine'] = df['close'].rolling(window=self.params['bollingerLengths']).mean()
        df['Band'] = df['close'].rolling(window=self.params['bollingerLengths']).std()
        df['upBand'] = df['MidLine'] + self.params['numStdDevs'] * df['Band']
        df['dnBand'] = df['MidLine'] - self.params['numStdDevs'] * df['Band']
        
        df['trendBuyPt'] = df['upBand']
        df['trendSellPt'] = df['dnBand']
        
        # 计算保护性止损
        df['swingProtStop'] = 3 * df['myATR']
        df['trendProtStop'] = df['close'].rolling(window=self.params['trendLiqLength']).mean()
        
        return df

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """生成交易信号矩阵"""
        signals = pd.DataFrame(index=data.index)
        signals['call'] = np.nan
        signals['EntryPrice'] = np.nan
        
        current_position = 0
        swing_entry = False
        position_bars = 0
        bars_since_exit = 0
        last_entry_price = np.nan
        
        for i in range(1, len(data)):
            if current_position == 0:
                position_bars = 0
                bars_since_exit += 1
            else:
                position_bars += 1
                bars_since_exit = 0
                
            # 震荡市
            if data['cmiVal'].iloc[i-1] < self.params['swingTrendSwitch']:
                # 震荡市入场做空
                if (current_position != -1 and 
                    data['low'].iloc[i] <= data['swingSellPt'].iloc[i]):
                    entry_price = min(data['open'].iloc[i], data['swingSellPt'].iloc[i])
                    signals.loc[signals.index[i], 'call'] = -1
                    signals.loc[signals.index[i], 'EntryPrice'] = entry_price
                    last_entry_price = entry_price
                    current_position = -1
                    swing_entry = True
                    position_bars = 0
                    bars_since_exit = 0
                
                # 震荡市平空    
                elif (current_position == -1 and position_bars >= 1 and
                      data['high'].iloc[i] >= data['swingBuyPt'].iloc[i]):
                    signals.loc[signals.index[i], 'call'] = 0
                    current_position = 0
                    swing_entry = False
                
            # 趋势市
            else:
                if swing_entry:
                    # 震荡市入场的单子在趋势市平空
                    stop_price = last_entry_price + data['swingProtStop'].iloc[i-1]
                    if (current_position == -1 and position_bars >= 1 and
                        data['high'].iloc[i] >= stop_price):
                        signals.loc[signals.index[i], 'call'] = 0
                        current_position = 0
                        swing_entry = False
                        
                else:
                    # 趋势市入场做空
                    if (current_position != -1 and bars_since_exit >= 1 and
                        data['low'].iloc[i] <= data['trendSellPt'].iloc[i-1]):
                        entry_price = min(data['open'].iloc[i], data['trendSellPt'].iloc[i-1])
                        signals.loc[signals.index[i], 'call'] = -1
                        signals.loc[signals.index[i], 'EntryPrice'] = entry_price
                        last_entry_price = entry_price
                        current_position = -1
                        position_bars = 0
                        bars_since_exit = 0
                    
                    # 趋势市平空    
                    elif (current_position == -1 and position_bars >= 1):
                        exit_price = min(data['trendBuyPt'].iloc[i-1],
                                       data['trendProtStop'].iloc[i-1])
                        if data['high'].iloc[i] >= exit_price:
                            signals.loc[signals.index[i], 'call'] = 0
                            current_position = 0

        # 最后一根K线强制平空
        if current_position == -1:
            signals.loc[signals.index[-1], 'call'] = 0
            
        return signals
