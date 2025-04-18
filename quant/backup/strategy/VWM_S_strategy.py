'''
// 简称: VolumeWeightedMomentumSys_S
// 名称: 成交量加权动量交易系统 空
// 类别: 策略应用
// 类型: 内建应用
// 输出:
// 策略说明:
//            基于动量系统, 通过交易量加权进行判断
//             
// 系统要素:
//             1. 用UWM下穿零轴判断空头趋势
// 入场条件:
//             1. 价格低于UWM下穿零轴时价格通道,在SetupLen的BAR数目内,做空
//             
// 出场条件: 
//             1. 多头势空单出场

Params 
    Numeric MomLen(5);                                                 //UWM参数
    Numeric AvgLen(20);                                             //UWM参数
    Numeric ATRLen(5);                                                 //ATR参数
    Numeric ATRPcnt(0.5);                                             //入场价格波动率参数
    Numeric SetupLen(5);                                            //条件持续有效K线数
Vars
    Series<Numeric> VWM(0); 
    Series<Numeric> AATR(0);  
    Series<Numeric> SEPrice(0);  
    Series<Bool> BullSetup(False); 
    Series<Bool> BearSetup(False); 
    Series<Numeric> SSetup(0);
'''

from typing import Dict
import pandas as pd
import numpy as np
from module import *
from module.indicators import AvgTrueRange, Momentum, XAverage, CrossOver, CrossUnder
from .base import StrategyBase

class VWM_S(StrategyBase):
    """基于成交量加权动量的做空策略"""
    def __init__(self, params: Dict = None):
        # VWM_S策略的默认参数
        default_params = {
            'MomLen': 5,
            'AvgLen': 20,
            'ATRLen': 5,
            'ATRPcnt': 0.5,
            'SetupLen': 5
        }
        # 如果提供了参数，则更新默认参数
        if params:
            default_params.update(params)
        super().__init__(default_params)
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算策略所需的技术指标"""
        df = data.copy()
        
        # 计算基础指标
        df['Momentum'] = Momentum(df['close'], self.params['MomLen'])
        df['VWM'] = XAverage(df['vol'] * df['Momentum'], self.params['AvgLen'])
        df['AATR'] = AvgTrueRange(self.params['ATRLen'], df['high'], df['low'], df['close'])
        
        # 生成交易信号
        zero_series = pd.Series(0, index=df.index)
        df['BullSetup'] = CrossOver(df['VWM'], zero_series)  # VWM上穿0
        df['BearSetup'] = CrossUnder(df['VWM'], zero_series)  # VWM下穿0
        
        # 计算做空设置
        df['SSetup'] = np.where(df['BearSetup'], 0, np.nan)  # 初始化做空计数器
        df['SEPrice'] = np.where(df['BearSetup'], df['close'], np.nan)  # 记录触发价格
        df['SSetup'] = df['SSetup'].ffill() + 1  # 计数递增
        df['SSetup'] = df['SSetup'].fillna(0)  # 填充空值
        
        return df
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """生成交易信号矩阵"""
        # 数据验证
        if len(data) < self.params['AvgLen']:
            raise ValueError('数据长度不足')
        
        signals = pd.DataFrame(index=data.index)
        signals['position'] = np.nan
        
        # 使用.values提高性能
        low = data['low'].values
        open_ = data['open'].values
        vol = data['vol'].values
        se_price = data['SEPrice'].values
        atr = data['AATR'].values
        s_setup = data['SSetup'].values
        bull_setup = data['BullSetup'].values
        
        current_position = 0
        position_bars = 0
        
        for i in range(self.params['AvgLen'], len(data)):
            if current_position != 0:
                position_bars += 1
            
            # 开空仓条件
            if (current_position == 0 and
                low[i] <= se_price[i-1] - (self.params['ATRPcnt'] * atr[i-1]) and 
                s_setup[i-1] <= self.params['SetupLen'] and
                s_setup[i] >= 1 and
                vol[i] > 0):
                
                signals.loc[i, 'position'] = -1
                current_position = -1
                position_bars = 0
            
            # 平空仓条件
            elif (current_position == -1 and
                  bull_setup[i-1] and
                  vol[i] > 0 and
                  position_bars > 0):
                
                signals.loc[i, 'position'] = 0
                current_position = 0
                position_bars = 0
        
        # 循环结束时强制平仓
        if current_position == -1:
            signals.iloc[-1, signals.columns.get_loc('position')] = 0
        
        return signals