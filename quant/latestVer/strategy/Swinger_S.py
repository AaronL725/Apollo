'''
// 简称: Swinger_S
// 名称: 基于均线与动能的交易系统空 
// 类别: 策略应用
// 类型: 内建应用
//------------------------------------------------------------------------
//----------------------------------------------------------------------//
// 策略说明:
//             本策略基于均线与均线间的动能变化建立的交易系统
//             
// 系统要素:
//             1. 1根长期均线进行趋势判断
//             2. 2根较短均线值之差揭示的动能变化为交易提供基础
// 入场条件:
//             1. 当价格高于长期均线且动能相对之前变强时做多
//             2. 当价格低于长期均线且动能相对之前变弱时做空
// 出场条件: 
//             1. 当动能减弱时, 价格低于ExitStopN根K线低点多头平仓
//             2. 当动能增强时, 价格高于ExitStopN根K线高点空头平仓
//
//         注: 当前策略仅为做空系统, 如需做多, 请参见CL_Swinger_L
'''

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict
import pandas as pd
import numpy as np
from module import *
from strategy.base import StrategyBase


class Swinger_S(StrategyBase):
    """趋势震荡做空策略"""
    def __init__(self, params: Dict = None):
        # 默认参数设置
        default_params = {
            'FastMALength': 5,    # 动能计算中的快均线值
            'SlowMALength': 20,   # 动能计算中的慢均线值
            'TrendMALength': 50,  # 显示趋势的均线值
            'ExitStopN': 3        # 求高低点的bar数值
        }
        # 更新自定义参数
        if params:
            default_params.update(params)
        super().__init__(default_params)
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算策略所需的技术指标"""
        df = data.copy()
        
        # 计算趋势线和均线动能
        df['TrendMA'] = AverageFC(df['close'], self.params['TrendMALength'])
        df['PriceOsci'] = PriceOscillator(
            df['close'], 
            self.params['FastMALength'],
            self.params['SlowMALength']
        )
        
        # 计算出场价格（前N根K线的最高点）
        df['ExitS'] = HighestFC(df['high'], self.params['ExitStopN'])
        
        return df
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """生成交易信号矩阵"""
        # 数据验证
        min_length = max(self.params['TrendMALength'], 
                        self.params['SlowMALength'], 
                        self.params['ExitStopN'])
        if len(data) < min_length:
            raise ValueError('数据长度不足')
        
        signals = pd.DataFrame(index=data.index)
        signals['call'] = np.nan
        
        # 使用.values提高性能
        close = data['close'].values
        open_ = data['open'].values
        high = data['high'].values
        trend_ma = data['TrendMA'].values
        price_osci = data['PriceOsci'].values
        exit_s = data['ExitS'].values
        vol = data['vol'].values
        
        current_position = 0
        
        for i in range(min_length, len(data)):
            # 开空仓条件：
            # 1. 当前无仓位
            # 2. 上根K线收盘价低于趋势线
            # 3. 上根K线动能为正且小于上上根动能
            # 4. 成交量大于0
            if (current_position == 0 and
                close[i-1] < trend_ma[i-1] and
                price_osci[i-1] >= 0 and
                price_osci[i-1] < price_osci[i-2] and
                vol[i] > 0):
                
                signals.iloc[i] = [-1]
                current_position = -1
            
            # 平空仓条件：
            # 1. 当前持有空仓
            # 2. 动能增强（上根动能大于上上根）
            # 3. 最高价突破前N根K线高点
            # 4. 成交量大于0
            elif (current_position == -1 and
                  price_osci[i-1] > price_osci[i-2] and
                  high[i] >= exit_s[i-1] and
                  vol[i] > 0):
                
                signals.iloc[i] = [0]
                current_position = 0

        # 循环结束时强制平仓
        if current_position == -1:
            signals.iloc[i] = [0]
        
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
        strategy = Swinger_S()
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
