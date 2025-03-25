'''
// 简称: KeltnerChannel_L
// 名称: 基于凯特纳通道的交易系统多
// 类别: 策略应用
// 类型: 内建应用 
// 输出:
// ------------------------------------------------------------------------
// ----------------------------------------------------------------------// 
//  策略说明:
//              基于凯特纳通道的交易系统
// 
//  系统要素:
//              1. 计算关键价格的凯特纳通道
//                 2. 价格突破凯特纳通道后，设定入场触发单
// 
//  入场条件:
//              1、价格突破凯特纳通道后，在当根K线高点之上N倍通道幅度，设定多头触发单，此开仓点将挂单X根k线
//              2、价格突破凯特纳通道后，在当根K线低点之下N倍通道幅度，设定空头触发单，此开仓点将挂单X根k线
// 
//  出场条件:
//              1. 价格下穿轨道中轨时平仓
//                 2. 价格小于N周期低点平仓
// 
//      注：当前策略仅为做多系统, 如需做空, 请参见CL_KeltnerChannel_S
'''


import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict
import pandas as pd
import numpy as np
from module import *
from strategy.base import StrategyBase


class KeltnerChannel_L(StrategyBase):
    """凯特纳通道做多策略"""
    def __init__(self, params: Dict = None):
        # 默认参数设置
        default_params = {
            'Length': 10,      # 均线参数
            'Constt': 1.2,     # 通道倍数
            'ChanPcnt': 0.5,   # 入场参数
            'buyN': 5,         # 入场触发条件有效K线周期
            'stopN': 4         # 低点止损参数
        }
        # 更新自定义参数
        if params:
            default_params.update(params)
        super().__init__(default_params)

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算策略指标"""
        df = data.copy()
        
        # 计算关键价格
        df['Price'] = df['close']
        
        # 计算均线和ATR
        df['AvgVal'] = Average(df['Price'], self.params['Length'])
        df['AvgRange'] = Average(TrueRange(df['high'], df['low'], df['close']), self.params['Length'])
        
        # 计算通道
        df['KCU'] = df['AvgVal'] + df['AvgRange'] * self.params['Constt']
        df['KCL'] = df['AvgVal'] - df['AvgRange'] * self.params['Constt']
        df['ChanRng'] = (df['KCU'] - df['KCL']) / 2
        
        # 初始化计数器和触发条件
        df['CountL'] = 0
        df['SetBar'] = np.nan
        df['hh'] = np.nan
        
        # 处理上穿上轨信号
        current_count = 0
        for i in range(1, len(df)):
            if CrossOver(df['Price'], df['KCU']).iloc[i]:
                df.loc[df.index[i], 'SetBar'] = df['high'].iloc[i]
                current_count = 0
                df.loc[df.index[i], 'hh'] = (df['SetBar'].iloc[i] + 
                                            df['ChanRng'].iloc[i] * self.params['ChanPcnt'])
            else:
                current_count += 1
            df.loc[df.index[i], 'CountL'] = current_count
            
        # 计算止损线
        df['Lstopline'] = Lowest(df['low'], self.params['stopN'])
        
        return df
        
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """生成交易信号矩阵"""
        # 数据验证
        min_length = max(self.params['Length'], self.params['stopN'])
        if len(data) < min_length:
            raise ValueError('数据长度不足')
        
        signals = pd.DataFrame(index=data.index)
        signals['call'] = np.nan
        
        current_position = 0
        
        for i in range(1, len(data)):
            # 开多仓条件
            if (current_position == 0 and
                data['Price'].iloc[i-1] > data['KCU'].iloc[i-1] and
                data['CountL'].iloc[i] <= self.params['buyN'] and
                data['high'].iloc[i] >= data['hh'].iloc[i-1] and
                data['vol'].iloc[i] > 0):
                
                entry_price = max(data['open'].iloc[i], data['hh'].iloc[i-1])
                signals.iloc[i] = [1]
                current_position = 1
            
            # 平多仓条件
            elif current_position == 1:
                # 价格下穿轨道中轨或低于止损线
                if ((CrossUnder(data['close'], data['AvgVal']).iloc[i] or
                     data['low'].iloc[i] <= data['Lstopline'].iloc[i-1]) and
                    data['vol'].iloc[i] > 0):
                    
                    exit_price = min(data['open'].iloc[i], data['Lstopline'].iloc[i-1])
                    signals.iloc[i] = [0]
                    current_position = 0
        
        # 在最后一个bar强制平仓
        if current_position == 1:
            last_idx = signals.index[-1]
            signals.loc[last_idx] = [0]
        
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
        strategy = KeltnerChannel_L()
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
