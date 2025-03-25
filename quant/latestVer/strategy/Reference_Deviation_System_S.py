'''
// 简称: Reference_Deviation_System_S
// 名称: 基于价格与均线的相关差进行判断 空
// 类别: 策略应用
// 类型: 内建应用
// 输出:
//------------------------------------------------------------------------
//------------------------------------------------------------------------
// 策略说明:
//            1.系统将当前价格和MA之差定义为DRD
//            2.计算RDV: N天DRD的加和除以DRD绝对值的加和
//             
// 入场条件:
//            1.设置ETLong为入市阈值，如果RDV>ETLong,则入场做多
//            2.设置ETShort为入市阈值，如果RDV<ETShort,则入场做空
// 出场条件: 
//            1.如果RDV下穿0, 多头平仓
//            2.如果RDV上穿0, 空头平仓
//
//         注: 当前策略仅为做空系统, 如需做多, 请参见CL_Reference_Deviation_System_L
'''


import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict
import pandas as pd
import numpy as np
from module import *
from strategy.base import StrategyBase


class Reference_Deviation_System_S(StrategyBase):
    """相对强弱指标做空策略"""
    def __init__(self, params: Dict = None):
        # 默认参数设置
        default_params = {
            'ETShort': -5,    # 做空阈值
            'RMALen': 15      # 均值周期
        }
        # 更新自定义参数
        if params:
            default_params.update(params)
        super().__init__(default_params)
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算策略所需的技术指标"""
        df = data.copy()
        
        # 计算均线
        df['RMA'] = AverageFC(df['close'], self.params['RMALen'])
        
        # 计算价格与均线的差值
        df['DRD'] = df['close'] - df['RMA']
        
        # 计算差值的累计
        df['NDV'] = df['DRD'].rolling(window=self.params['RMALen']).sum()
        
        # 计算差值绝对值的累计
        df['TDV'] = df['DRD'].abs().rolling(window=self.params['RMALen']).sum()
        
        # 计算相对强弱指标
        df['RDV'] = np.where(df['TDV'] > 0, 100 * df['NDV'] / df['TDV'], 0)
        
        return df
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """生成交易信号矩阵"""
        # 数据验证
        if len(data) < self.params['RMALen']:
            raise ValueError('数据长度不足')
        
        signals = pd.DataFrame(index=data.index)
        signals['call'] = np.nan
        
        # 使用.values提高性能
        open_ = data['open'].values
        vol = data['vol'].values
        rdv = data['RDV'].values
        
        current_position = 0
        
        for i in range(1, len(data)):
            # 开空仓条件：
            # 1. 当前空仓
            # 2. 相对强弱指标低于阈值
            # 3. 成交量大于0
            if (current_position == 0 and
                rdv[i-1] < self.params['ETShort'] and
                vol[i] > 0):
                
                signals.iloc[i] = [-1]
                current_position = -1
            
            # 平空仓条件：
            # 1. 当前持有空仓
            # 2. 相对强弱指标由负转正
            # 3. 成交量大于0
            elif (current_position == -1 and
                  rdv[i-1] > 0 and
                  vol[i] > 0):
                
                signals.iloc[i] = [0]
                current_position = 0
        
        # 循环结束后，如果还持有空仓则平仓
        if current_position == -1:
            signals.iloc[-1] = [0]
        
        return signals 


#########################主函数#########################
def main():
    """主函数，执行多品种回测"""
    logger = setup_logging()
    
    level = 'day'  # 默认值
    valid_levels = {'min5', 'min15', 'min30', 'min60', 'day'}
    assert level in valid_levels, f"level必须是以下值之一: {valid_levels}"
    
    data_paths = {
        'open': rf'D:\pythonpro\python_test\quant\Data\{level}\open.csv',
        'close': rf'D:\pythonpro\python_test\quant\Data\{level}\close.csv',
        'high': rf'D:\pythonpro\python_test\quant\Data\{level}\high.csv',
        'low': rf'D:\pythonpro\python_test\quant\Data\{level}\low.csv',
        'vol': rf'D:\pythonpro\python_test\quant\Data\{level}\vol.csv'
    }
    
    try:
        data_cache = load_all_data(data_paths, logger, level)
        futures_codes = list(data_cache['open'].columns)
        
        config = {
            'data_paths': data_paths,
            'futures_codes': futures_codes,
            'start_date': '2023-02-28',
            'end_date': '2025-01-10',
            'initial_balance': 20000000.0
        }
        
        data_dict = load_data_vectorized(
            data_cache, 
            config['futures_codes'],
            config['start_date'],
            config['end_date'],
            logger
        )

        # 先计算所有品种的信号
        strategy = Reference_Deviation_System_S()
        signals_dict = {}
        
        for code, data in data_dict.items():
            try:
                # 计算指标和信号
                data_with_indicators = strategy.calculate_indicators(data)
                signals = strategy.generate_signals(data_with_indicators)
                if isinstance(signals, pd.DataFrame) and len(signals) > 0:
                    signals_dict[code] = signals
            except Exception as e:
                logger.error(f"计算{code}信号时出错: {e}")
                continue
        
        # 将信号字典传入回测器
        backtester = Backtester(
            signals_dict=signals_dict,
            data_dict=data_dict,
            config=config,
            logger=logger,
            use_multiprocessing=True
        )
        t_pnl_df = backtester.run_backtest()
        
        plot_combined_pnl(t_pnl_df, logger)
        
    except Exception as e:
        logger.error(f"程序执行过程中发生错误: {e}")


if __name__ == "__main__":
    main() 
