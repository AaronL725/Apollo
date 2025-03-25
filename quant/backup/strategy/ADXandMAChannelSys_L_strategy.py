'''
// 简称: ADXandMAChannelSys_L
// 名称: 基于ADX及EMA的交易系统多
// 类别: 策略应用
// 类型: 内建应用
// 输出:
// 策略说明:基于ADX及EMA进行判断
// 系统要素:
//                1. 计算30根k线最高价和最低价的EMA价差
//                2. 计算12根k线的ADX
// 入场条件:
//                满足上根K线的收盘价收于EMA30之上,且ADX向上的条件 在EntryBarBAR内该条件成立
//                当前价大于等于BuySetup,做多,当条件满足超过EntryBarBAR后,取消入场
// 出场条件:
//                当前价格下破30根K线最高价的EMA        
'''

from typing import Dict
import pandas as pd
import numpy as np
from module import *
from .base import StrategyBase

class ADXandMAChannelSys_L(StrategyBase):
    """基于ADX及EMA的做多策略"""
    def __init__(self, params: Dict = None):
        # 默认参数设置
        default_params = {
            'DMI_N': 14,      # DMI的N值
            'DMI_M': 30,      # ADX均线周期,DMI的M值
            'AvgLen': 30,     # 最高最低价的EMA周期数
            'EntryBar': 2,    # 保持BuySetup触发BAR数
            'Lots': 1         # 交易手数
        }
        
        if params:
            default_params.update(params)
        super().__init__(default_params)
        
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算技术指标"""
        df = data.copy()
        
        # 调试打印
        print("\n=== DMI计算检查 ===")
        print(f"数据起始时间: {df.index[0]}")
        print(f"数据结束时间: {df.index[-1]}")
        print(f"数据长度: {len(df)}")
        
        # 计算DMI相关指标
        tr = TrueRange(df['high'], df['low'], df['close'])
        print("\nTR前5个值:", tr[:5])
        
        # 修复DMI计算
        plus_dm = pd.Series(0.0, index=df.index)
        minus_dm = pd.Series(0.0, index=df.index)
        
        # 计算DM
        high_diff = df['high'] - df['high'].shift(1)
        low_diff = df['low'].shift(1) - df['low']
        
        # +DM
        plus_dm[((high_diff > low_diff) & (high_diff > 0))] = high_diff
        
        # -DM
        minus_dm[((low_diff > high_diff) & (low_diff > 0))] = low_diff
        
        # 计算平滑值
        tr_ma = pd.Series(tr).ewm(span=self.params['DMI_N'], min_periods=1, adjust=False).mean()
        plus_di = (plus_dm.ewm(span=self.params['DMI_N'], min_periods=1, adjust=False).mean() / tr_ma * 100).fillna(0)
        minus_di = (minus_dm.ewm(span=self.params['DMI_N'], min_periods=1, adjust=False).mean() / tr_ma * 100).fillna(0)
        
        # 计算DX和ADX
        dx = abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, 1) * 100
        adx = dx.ewm(span=self.params['DMI_M'], min_periods=1, adjust=False).mean().fillna(0)
        
        # DMI指标检查
        print("\nDMI指标检查(最后5条):")
        print(f"ADX:\n{adx.tail()}")
        print(f"Plus DI:\n{plus_di.tail()}")
        print(f"Minus DI:\n{minus_di.tail()}")
        
        # 计算EMA通道
        df['UpperMA'] = XAverage(df['high'], self.params['AvgLen'])
        df['LowerMA'] = XAverage(df['low'], self.params['AvgLen'])
        df['ChanSpread'] = (df['UpperMA'] - df['LowerMA']) / 2
        
        # 通道指标检查
        print("\n通道指标检查(最后5条):")
        print(f"Upper MA:\n{df['UpperMA'].tail()}")
        print(f"Lower MA:\n{df['LowerMA'].tail()}")
        print(f"Channel Spread:\n{df['ChanSpread'].tail()}")
        
        # 计算买入条件
        df['ADX'] = adx
        df['BuySetup'] = (df['close'] > df['UpperMA']) & (df['ADX'] > df['ADX'].shift(1))
        
        # 计算买入目标价
        df['BuyTarget'] = np.where(df['BuySetup'], 
                                 df['close'] + df['ChanSpread'], 
                                 np.nan)
        
        # 计算持续满足条件的周期数
        df['MROBS'] = df['BuySetup'].rolling(window=self.params['EntryBar']).sum()
        df['MROBS'] = np.where(df['MROBS'] > self.params['EntryBar'], 0, df['MROBS'])
        
        # 买入条件检查
        print("\n买入条件检查(最后5条):")
        print(f"BuySetup:\n{df['BuySetup'].tail()}")
        print(f"BuyTarget:\n{df['BuyTarget'].tail()}")
        print(f"MROBS:\n{df['MROBS'].tail()}")
        
        return df
        
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """生成交易信号矩阵"""
        # 数据验证
        if len(data) < max(self.params['DMI_N'], self.params['DMI_M'], self.params['AvgLen']):
            return pd.DataFrame()
            
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0
        signals['quantity'] = 1
        signals['entry_price'] = np.nan
        signals['exit_price'] = np.nan
        signals['position'] = 0
        
        current_position = 0
        position_bars = 0
        
        print("\n=== 信号生成检查 ===")
        for i in range(100, len(data)):
            if current_position == 0:  # 空仓
                if (data['MROBS'].iloc[i-1] != 0 and 
                    data['high'].iloc[i] >= data['BuyTarget'].iloc[i-1]):
                    print(f"\n开仓信号触发 - 时间: {data.index[i]}")
                    print(f"MROBS: {data['MROBS'].iloc[i-1]}")
                    print(f"当前最高价: {data['high'].iloc[i]}")
                    print(f"目标买入价: {data['BuyTarget'].iloc[i-1]}")
                    # 开多仓
                    entry_price = max(data['open'].iloc[i], data['BuyTarget'].iloc[i-1])
                    signals.iloc[i] = [1, self.params['Lots'], entry_price, np.nan, 0]
                    current_position = 1
                    position_bars = 0
                    
            elif current_position == 1:  # 持多仓
                position_bars += 1
                if data['low'].iloc[i] <= (data['UpperMA'].iloc[i-1] - 0.01):
                    print(f"\n平仓信号触发 - 时间: {data.index[i]}")
                    print(f"当前最低价: {data['low'].iloc[i]}")
                    print(f"上轨价格: {data['UpperMA'].iloc[i-1]}")
                    # 平多仓
                    exit_price = min(data['open'].iloc[i], data['UpperMA'].iloc[i-1] - 0.01)
                    signals.iloc[i] = [-1, self.params['Lots'], np.nan, exit_price, position_bars]
                    current_position = 0
                    position_bars = 0
        
        # 在最后一根K线强制平仓
        if current_position == 1:
            signals.iloc[-1] = [-1, self.params['Lots'], np.nan, data['close'].iloc[-1], position_bars]
            
        return signals
