'''
// 简称: JailBreakSys_S
// 名称: 基于价格区间突破的交易系统 
// 类别: 策略应用
// 类型: 内建应用
// 输出:
// 策略说明:        基于通道突破的判断
// 系统要素:
//                1. 计算50根k线最低价的区间
//                2. 计算30根k线最高价的区间
//                
// 入场条件:
//                1.价格低于50根K线最低价的区间入场
// 出场条件:
//                1. 当前价格高于30根K线最高价的区间出场
//                2. 当前价格高于入场价一定ATR波动率幅度出场
'''

from typing import Dict
import pandas as pd
import numpy as np
from module import *
from .base import StrategyBase

class JailBreakSys_S(StrategyBase):
    """基于价格区间突破的做空策略"""
    def __init__(self, params: Dict = None):
        # 默认参数设置
        default_params = {
            'Length1': 50,     # 长周期区间参数
            'Length2': 30,     # 短周期区间参数
            'IPS': 4,         # 保护止损波动率参数
            'AtrVal': 10      # 波动率参数
        }
        # 更新自定义参数
        if params:
            default_params.update(params)
        super().__init__(default_params)
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算策略指标"""
        df = data.copy()
        
        # 计算ATR
        df['ATR'] = AvgTrueRange(
            self.params['AtrVal'],
            df['high'],
            df['low'],
            df['close']
        )
        
        # 计算区间参数
        L1 = max(self.params['Length1'], self.params['Length2'])
        L2 = min(self.params['Length1'], self.params['Length2'])
        
        # 计算价格区间
        df['Upperband'] = HighestFC(df['high'], L1)  # 长周期最高价区间
        df['Lowerband'] = LowestFC(df['low'], L1)    # 长周期最低价区间
        df['Exitlong'] = LowestFC(df['low'], L2)     # 短周期最低价区间
        df['Exitshort'] = HighestFC(df['high'], L2)  # 短周期最高价区间
        
        return df
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """生成交易信号矩阵"""
        # 数据验证
        min_length = max(self.params['Length1'], self.params['Length2'])
        if len(data) < min_length:
            raise ValueError('数据长度不足')
        
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0
        signals['quantity'] = 1
        signals['entry_price'] = np.nan
        signals['exit_price'] = np.nan
        signals['position'] = 0
        
        current_position = 0
        protect_stop = None
        
        for i in range(1, len(data)):
            if data['vol'].iloc[i] <= 0:
                continue
                
            # 入场信号
            if (current_position == 0 and 
                data['low'].iloc[i] <= data['Lowerband'].iloc[i-1] and
                data['vol'].iloc[i] > 0):
                
                entry_price = min(data['open'].iloc[i], data['Lowerband'].iloc[i-1])
                signals.iloc[i] = [-1, 1, entry_price, np.nan, -1]
                current_position = -1
                protect_stop = entry_price + self.params['IPS'] * data['ATR'].iloc[i-1]
            
            # 出场信号
            elif current_position == -1:
                # 保护止损
                if (data['high'].iloc[i] >= protect_stop and 
                    protect_stop <= data['Exitshort'].iloc[i-1]):
                    exit_price = max(data['open'].iloc[i], protect_stop)
                    signals.iloc[i] = [1, 1, np.nan, exit_price, 0]
                    current_position = 0
                    protect_stop = None
                
                # 区间突破出场
                elif data['high'].iloc[i] >= data['Exitshort'].iloc[i-1]:
                    exit_price = max(data['open'].iloc[i], data['Exitshort'].iloc[i-1])
                    signals.iloc[i] = [1, 1, np.nan, exit_price, 0]
                    current_position = 0
                    protect_stop = None
        
        # 在最后一根K线强制平仓
        if current_position == -1:  # 如果还持有空仓
            signals.iloc[-1] = [1, 1, np.nan, data['close'].iloc[-1], 0]
        
        return signals 