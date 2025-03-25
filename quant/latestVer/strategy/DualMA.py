'''
// 简称: DualMA
// 名称: 双均线交易系统
// 类别: 策略应用
// 类型: 内建应用
'''


import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict
import pandas as pd
import numpy as np
from module import *
from strategy.base import StrategyBase


class DualMA(StrategyBase):
    """双均线交易系统"""
    def __init__(self, params: Dict = None):
        # 默认参数设置
        default_params = {
            'FastLength': 5,    # 短期指数平均线参数
            'SlowLength': 20,   # 长期指数平均线参数
            'Lots': 1          # 交易手数
        }
        
        # 更新自定义参数
        if params:
            default_params.update(params)
        super().__init__(default_params)
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算技术指标"""
        df = data.copy()
        
        # 计算快速和慢速移动平均线
        df['AvgValue1'] = Average(df['close'], self.params['FastLength'])
        df['AvgValue2'] = Average(df['close'], self.params['SlowLength'])
        
        return df
        
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """生成交易信号矩阵"""
        # 数据验证
        min_length = max(self.params['FastLength'], self.params['SlowLength'])
        if len(data) < min_length:
            return pd.DataFrame()
            
        # 初始化信号矩阵
        signals = pd.DataFrame(index=data.index)
        signals['call'] = np.nan
        
        current_position = 0  # 当前持仓状态: 1多头, -1空头, 0空仓
        
        # 生成交易信号
        for i in range(1, len(data)):
            # 多头信号
            if (current_position != 1 and 
                data['AvgValue1'].iloc[i-1] > data['AvgValue2'].iloc[i-1]):
                signals.loc[signals.index[i], 'call'] = 1
                current_position = 1
                
            # 空头信号    
            elif (current_position != -1 and 
                  data['AvgValue1'].iloc[i-1] < data['AvgValue2'].iloc[i-1]):
                signals.loc[signals.index[i], 'call'] = -1
                current_position = -1
                
        # 在最后一根K线强制平仓
        if current_position != 0:
            signals.loc[signals.index[-1], 'call'] = 0
            
        return signals


#########################主函数#########################
def main():
    """主函数，执行多品种回测"""
    logger = setup_logging()
    
    level = 'day'
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
            'start_date': '2016-01-05',
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
        strategy = DualMA()
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
