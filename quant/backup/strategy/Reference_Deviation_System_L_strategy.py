'''
// 简称: RelativeStrength_L
// 名称: 相对强弱指标做多策略
// 类别: 策略应用
// 类型: 内建应用
//------------------------------------------------------------------------
//------------------------------------------------------------------------
// 策略说明:
//            1.系统将当前价格和MA之差定义为DRD
//            2.计算RDV: N天DRD的加和除以DRD绝对值的加和
//             
// 入场条件:
//            1.设置ETLong为入市阈值，如果RDV>ETLong,则入场做多
//
// 出场条件: 
//            1.如果RDV下穿0, 多头平仓
//
//         注: 当前策略仅为做多系统
'''

from typing import Dict
import pandas as pd
import numpy as np
from module import *
from .base import StrategyBase

class Reference_Deviation_System_L(StrategyBase):
    """相对强弱指标做多策略"""
    def __init__(self, params: Dict = None):
        # 默认参数设置
        default_params = {
            'ETLong': 5,     # 做多阈值
            'RMALen': 15     # 均值周期
        }
        # 更新自定义参数
        if params:
            default_params.update(params)
        super().__init__(default_params)
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算策略所需的技术指标"""
        df = data.copy()
        
        # 计算移动平均线
        df['RMA'] = AverageFC(df['close'], self.params['RMALen'])
        
        # 计算价格与均线差值
        df['DRD'] = df['close'] - df['RMA']
        
        # 计算N天DRD的加和
        df['NDV'] = df['DRD'].rolling(window=self.params['RMALen']).sum()
        
        # 计算N天DRD绝对值的加和
        df['TDV'] = df['DRD'].abs().rolling(window=self.params['RMALen']).sum()
        
        # 计算相对强弱值
        df['RDV'] = np.where(df['TDV'] > 0, 100 * df['NDV'] / df['TDV'], 0)
        
        # 确保所有指标的前N个值为NaN
        mask = np.arange(len(df)) < self.params['RMALen']
        df.loc[mask, ['RMA', 'DRD', 'NDV', 'TDV', 'RDV']] = np.nan
        
        return df
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """生成交易信号矩阵"""
        # 数据验证
        if len(data) < self.params['RMALen']:
            raise ValueError('数据长度不足')
        
        # 初始化信号矩阵
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0
        signals['quantity'] = 0
        signals['entry_price'] = np.nan
        signals['exit_price'] = np.nan
        signals['position'] = 0
        
        # 使用.values提高性能
        rdv = data['RDV'].values
        open_ = data['open'].values
        vol = data['vol'].values

        current_position = 0
        
        # 从第二个位置开始遍历
        for i in range(1, len(data)):
            # 开多仓条件
            if (current_position == 0 and
                rdv[i-1] > self.params['ETLong'] and  # 使用当前值而不是前一个值
                vol[i] > 0):
                
                signals.iloc[i] = [1, 1, open_[i], np.nan, 1]
                current_position = 1
                
            # 平多仓条件
            elif (current_position == 1 and
                  rdv[i-1] < 0 and  # 使用当前值而不是前一个值
                  vol[i] > 0):
                    
                signals.iloc[i] = [-1, 1, np.nan, open_[i], 0]
                current_position = 0   
        
        # 循环结束后，如果还持有仓位则平仓
        if current_position == 1:
            signals.iloc[-1] = [-1, 1, np.nan, open_[-1], 0]
        
        return signals