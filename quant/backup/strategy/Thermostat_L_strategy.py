'''
// 简称: Thermostat_L
// 名称: 恒温器_多
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
        此公式仅做多
'''

from typing import Dict
import pandas as pd
import numpy as np
from module import *
from module.indicators import AvgTrueRange, Average, AverageFC
from .base import StrategyBase

class Thermostat_L(StrategyBase):
    """恒温器做多策略"""
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
        try:
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
            tr = pd.DataFrame(index=df.index)
            tr['hl'] = df['high'] - df['low']
            tr['hc'] = abs(df['high'] - df['close'].shift(1))
            tr['lc'] = abs(df['low'] - df['close'].shift(1))
            tr['tr'] = tr[['hl', 'hc', 'lc']].max(axis=1)
            df['myATR'] = tr['tr'].rolling(window=self.params['atrLength']).mean()
            
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
            
            df['swingBuyPt'] = pd.Series(df[['swingBuyPt', 'trendLokBuy']].max(axis=1), index=df.index)
            df['swingSellPt'] = pd.Series(df[['swingSellPt', 'trendLokSell']].min(axis=1), index=df.index)
            
            # 计算趋势市的进场价格
            midline = AverageFC(df['close'], self.params['bollingerLengths'])
            if isinstance(midline, (int, float)):
                midline = pd.Series([midline] * len(df), index=df.index)
            df['MidLine'] = midline
            
            df['Band'] = df['close'].rolling(self.params['bollingerLengths']).std() * 2
            df['upBand'] = df['MidLine'] + self.params['numStdDevs'] * df['Band']
            df['dnBand'] = df['MidLine'] - self.params['numStdDevs'] * df['Band']
            
            df['trendBuyPt'] = df['upBand']
            df['trendSellPt'] = df['dnBand']
            
            df['swingProtStop'] = pd.Series(3 * df['myATR'], index=df.index)
            trendstop = Average(df['close'], self.params['trendLiqLength'])
            if isinstance(trendstop, (int, float)):
                trendstop = pd.Series([trendstop] * len(df), index=df.index)
            df['trendProtStop'] = trendstop
            
            return df
            
        except Exception as e:
            raise

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """生成交易信号矩阵"""
        try:
            signals = pd.DataFrame(index=data.index)
            signals['call'] = np.nan
            
            current_position = 0
            swing_entry = False
            position_bars = 0
            entry_price = 0  # 记录入场价格
            
            for i in range(1, len(data)):
                if current_position == 0:  # 无持仓
                    # 震荡市做多: 当价格突破swingBuyPt时开多
                    if data['cmiVal'].iloc[i-1] < self.params['swingTrendSwitch']:
                        if data['high'].iloc[i] >= data['swingBuyPt'].iloc[i]:
                            signals.iloc[i] = 1  # 做多信号
                            current_position = 1
                            swing_entry = True
                            position_bars = 0
                            entry_price = data['swingBuyPt'].iloc[i]  # 记录入场价格
                    # 趋势市做多: 当价格突破布林上轨时开多
                    elif not swing_entry and data['high'].iloc[i] >= data['trendBuyPt'].iloc[i-1]:
                        signals.iloc[i] = 1  # 做多信号
                        current_position = 1
                        position_bars = 0
                        entry_price = data['trendBuyPt'].iloc[i-1]  # 记录入场价格
                        
                elif current_position == 1:  # 持有多仓
                    position_bars += 1
                    # 震荡市平仓: 当价格跌破swingSellPt时平多
                    if data['cmiVal'].iloc[i-1] < self.params['swingTrendSwitch']:
                        if position_bars >= 1 and data['low'].iloc[i] <= data['swingSellPt'].iloc[i]:
                            signals.iloc[i] = 0  # 平多信号
                            current_position = 0
                            swing_entry = False
                            position_bars = 0
                    # 趋势市
                    else:
                        if swing_entry:  # 震荡市进场的多单在趋势市平仓: 当价格跌破ATR止损线时平多
                            if (position_bars >= 1 and 
                                data['low'].iloc[i] <= (entry_price - data['swingProtStop'].iloc[i-1])):
                                signals.iloc[i] = 0  # 平多信号
                                current_position = 0
                                swing_entry = False
                                position_bars = 0
                        else:  # 趋势市平仓: 当价格跌破布林下轨或均线时平多
                            if (position_bars >= 1 and 
                                data['low'].iloc[i] <= max(data['trendSellPt'].iloc[i-1], 
                                                         data['trendProtStop'].iloc[i-1])):
                                signals.iloc[i] = 0  # 平多信号
                                current_position = 0
                                position_bars = 0

            # 最后一根K线强制平多仓
            if current_position == 1:
                signals.iloc[-1] = 0  # 平多信号
                
            return signals
            
        except Exception as e:
            raise
