'''
// 简称: Going_in_Style_L
// 名称: 价格通道突破, 在价格回调时进行判断，做多
// 类别: 策略应用
// 类型: 内建应用
//------------------------------------------------------------------------
//------------------------------------------------------------------------
// 策略说明:
//            1.计算价格通道
//            2.收盘价加上ATR的一定倍数作为进场价
//             
// 入场条件:
//            1.上一根Bar创新高
//            2.当前Bar最高价突破上一根Bar收盘价加上ATR的一定倍数
// 出场条件: 
//            1.记录多头进场后的跟踪止损价
//            2.价格向下突破跟踪止损价多头出场
//
//         注: 当前策略仅为做多系统, 如需做空, 请参见CL_Going_in_Style_S
'''


import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict
import pandas as pd
import numpy as np
from module import *
from strategy.base import StrategyBase


class Going_in_Style_L(StrategyBase):
    """价格通道突破做多策略"""
    def __init__(self, params: Dict = None):
        # 默认参数设置
        default_params = {
            'Length': 10,           # 用于计算ATR和新高价的Bar数
            'Trigger': 0.79,        # 用于计算多头进场价的驱动系数
            'Acceleration': 0.05,   # 抛物线的加速系数
            'FirstBarMultp': 5,     # 用于计算在进场Bar设置止损价的系数
        }
        
        if params:
            default_params.update(params)
        super().__init__(default_params)
        
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算技术指标"""
        df = data.copy()
        
        # 计算ATR
        df['ATR'] = AvgTrueRange(self.params['Length'], df['high'], df['low'], df['close'])
        
        # 计算最高价条件
        df['Condition1'] = df['high'] > Highest(df['high'].shift(1), self.params['Length'])

        # 计算3日真实波幅均值
        df['StopATR'] = TrueRange(df['high'], df['low'], df['close']).rolling(window=3).mean()
        
        return df
        
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """生成交易信号矩阵"""
        # 数据验证
        if len(data) < self.params['Length']:
            return pd.DataFrame()
            
        signals = pd.DataFrame(index=data.index)
        signals['call'] = np.nan
        
        current_position = 0
        position_bars = 0
        stop_price = None
        prev_stop_price = None  # 添加上一个stop_price的记录
        high_value = None
        prev_high_value = None
        af = self.params['Acceleration']
        
        for i in range(1, len(data)):
            # 更新持仓天数
            if current_position != 0:
                position_bars += 1
                
            # 多头入场条件判断
            if (data['Condition1'].iloc[i-1] and  # 上一根Bar创新高
                data['high'].iloc[i] >= data['close'].iloc[i-1] + data['ATR'].iloc[i-1] * self.params['Trigger'] and
                data['vol'].iloc[i] > 0):  # 添加成交量判断
                
                # 多头入场
                if current_position == 0:
                    entry_price = max(data['open'].iloc[i], 
                                    data['close'].iloc[i-1] + data['ATR'].iloc[i-1] * self.params['Trigger'])
                    signals.loc[signals.index[i], 'call'] = 1
                    current_position = 1
                    position_bars = 0
                    
                    # 设置初始止损价和峰值价
                    stop_price = data['low'].iloc[i] - data['StopATR'].iloc[i] * self.params['FirstBarMultp']
                    high_value = data['high'].iloc[i]
                    prev_high_value = high_value
                    af = self.params['Acceleration']
                    
            # 更新跟踪止损价和峰值价
            elif current_position == 1:
                if position_bars == 0:  # 入场当天
                    stop_price = data['low'].iloc[i] - data['StopATR'].iloc[i] * self.params['FirstBarMultp']
                    prev_stop_price = stop_price  # 初始化prev_stop_price
                    high_value = data['high'].iloc[i]
                    prev_high_value = high_value
                    af = self.params['Acceleration']
                elif position_bars > 0:  # 入场后的交易日
                    prev_stop_price = stop_price  # 保存上一个stop_price
                    prev_high_value = high_value
                    if data['high'].iloc[i] > high_value:
                        high_value = data['high'].iloc[i]
                        if high_value > prev_high_value and af < 0.2:
                            af = af + min(self.params['Acceleration'], 0.2 - af)
                    stop_price = stop_price + af * (high_value - stop_price)
                    
                    # 检查止损出场条件 (使用上一根bar的止损价)
                    if data['low'].iloc[i] <= prev_stop_price and data['vol'].iloc[i] > 0:
                        exit_price = min(data['open'].iloc[i], prev_stop_price)  # 确保以不高于止损价的价格出场
                        signals.loc[signals.index[i], 'call'] = 0
                        current_position = 0
                        position_bars = 0
                        stop_price = None
                        prev_stop_price = None
                        high_value = None
                        prev_high_value = None
                        af = self.params['Acceleration']
        
        # 在最后一根K线强制平仓
        if current_position == 1:
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
        strategy = Going_in_Style_L()
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
