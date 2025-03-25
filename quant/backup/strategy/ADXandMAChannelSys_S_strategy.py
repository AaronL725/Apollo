'''
// 简称: ADXandMAChannelSys_S
// 名称: 基于ADX及EMA的交易系统空
// 类别: 策略应用
// 类型: 内建应用
// 输出:
// 策略说明:基于ADX及EMA进行判断
// 系统要素:
//                11. 计算30根k线最高价和最低价的EMA价差
//                2. 计算12根k线的ADX
// 入场条件:
//                 满足上根K线的收盘价收于EMA30之下,且ADX向上的条件,在EntryBarBAR根内该条件成立
//                当前价小于等于SellSetup,做空,当条件满足超过EntryBarBAR后,取消入场
// 出场条件:
//                当前价格上穿30根K线最低价的EMA        
'''

from typing import Dict
import pandas as pd
import numpy as np
from module import *
from .base import StrategyBase

class ADXandMAChannelSys_S(StrategyBase):
    """基于ADX及EMA的做空策略"""
    def __init__(self, params: Dict = None):
        # 默认参数设置
        default_params = {
            'DMI_N': 14,      # DMI的N值
            'DMI_M': 30,      # ADX均线周期,DMI的M值
            'AvgLen': 30,     # 最高最低价的EMA周期数
            'EntryBar': 2,    # 保持BuySetup触发BAR数
            'Lots': 1         # 交易手数
        }
        
        # 更新自定义参数
        if params:
            default_params.update(params)
        super().__init__(default_params)
        
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算技术指标"""
        df = data.copy()
        
        # 计算DMI相关指标
        df['TR'] = TrueRange(df['high'], df['low'], df['close'])
        df['DM_plus'] = np.where(
            (df['high'] - df['high'].shift(1) > df['low'].shift(1) - df['low']) & 
            (df['high'] - df['high'].shift(1) > 0),
            df['high'] - df['high'].shift(1),
            0
        )
        df['DM_minus'] = np.where(
            (df['low'].shift(1) - df['low'] > df['high'] - df['high'].shift(1)) & 
            (df['low'].shift(1) - df['low'] > 0),
            df['low'].shift(1) - df['low'],
            0
        )
        
        # 计算平滑值
        sf = 1/self.params['DMI_N']
        
        # 计算DMI指标
        df['AvgTR'] = df['TR'].rolling(window=self.params['DMI_N']).mean()
        df['AvgDM_plus'] = df['DM_plus'].rolling(window=self.params['DMI_N']).mean()
        df['AvgDM_minus'] = df['DM_minus'].rolling(window=self.params['DMI_N']).mean()
        
        # 计算DI指标
        df['DI_plus'] = 100 * df['AvgDM_plus'] / df['AvgTR']
        df['DI_minus'] = 100 * df['AvgDM_minus'] / df['AvgTR']
        
        # 计算DX和ADX
        df['DX'] = 100 * abs(df['DI_plus'] - df['DI_minus']) / (df['DI_plus'] + df['DI_minus'])
        df['ADX'] = df['DX'].rolling(window=self.params['DMI_M']).mean()
        
        # 计算EMA通道
        df['UpperMA'] = df['high'].ewm(span=self.params['AvgLen'], adjust=False).mean()
        df['LowerMA'] = df['low'].ewm(span=self.params['AvgLen'], adjust=False).mean()
        df['ChanSpread'] = (df['UpperMA'] - df['LowerMA']) / 2
        
        # 计算卖出条件
        df['SellSetup'] = (df['close'] < df['LowerMA']) & (df['ADX'] > df['ADX'].shift(1))
        df['SellTarget'] = np.where(df['SellSetup'], df['close'] - df['ChanSpread'], np.nan)
        
        # 计算持续满足条件的周期数
        df['MROSS'] = df['SellSetup'].rolling(window=self.params['EntryBar']).sum()
        df['MROSS'] = np.where(df['MROSS'] > self.params['EntryBar'], 0, df['MROSS'])
        
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
        
        for i in range(100, len(data)):
            if current_position == 0:  # 空仓
                if (data['MROSS'].iloc[i-1] != 0 and 
                    data['low'].iloc[i] <= data['SellTarget'].iloc[i-1]):
                    # 开空仓
                    entry_price = min(data['open'].iloc[i], data['SellTarget'].iloc[i-1])
                    signals.iloc[i] = [-1, self.params['Lots'], entry_price, np.nan, 0]
                    current_position = -1
                    position_bars = 0
                    
            elif current_position == -1:  # 持有空仓
                position_bars += 1
                if (data['high'].iloc[i] >= data['LowerMA'].iloc[i-1]):
                    # 平空仓
                    exit_price = max(data['open'].iloc[i], data['LowerMA'].iloc[i-1])
                    signals.iloc[i] = [1, self.params['Lots'], np.nan, exit_price, position_bars]
                    current_position = 0
        
        # 在最后一根K线强制平仓
        if current_position == -1:
            signals.iloc[-1] = [1, self.params['Lots'], np.nan, data['close'].iloc[-1], position_bars]
            
        return signals
