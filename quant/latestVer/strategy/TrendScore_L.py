'''
// 简称: TrendScore_L
// 名称: 基于收盘价与之前k线高低进行打分的交易系统多 
// 类别: 策略应用
// 类型: 内建应用
//------------------------------------------------------------------------
//----------------------------------------------------------------------//
// 策略说明:
//             本策略基于当前收盘价与之前k线的高低进行打分, 并通过打分的均值与对应的收盘价均值进行交易
//             
// 系统要素:
//             1. 当当前收盘价格大于之前LookBack根K线内某一根k线的收盘价时记+1分, 否则记-1分, 加总这些分数以获得当前K线的得分
//             2. 对k线的打分计算一条均线
//             3. 对k线的收盘计算一条均线
// 入场条件:
//             1. 当价格高于收盘价均线, 且打分也高于打分均线时的入场做多
//             2. 当价格低于收盘价均线, 且打分也低于打分均线时的入场做空
// 出场条件: 
//             1. 基于ATR的保护性止损
//             2. 基于ATR的跟踪止损
//             3. 基于ATR的盈亏平衡止损
//
//         注: 当前策略仅为做多系统, 如需做空, 请参见CL_TrendScore_S
'''

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict
import pandas as pd
import numpy as np
from module import *
from strategy.base import StrategyBase


class TrendScore_L(StrategyBase):
    """基于收盘价与之前k线高低进行打分的做多策略"""
    def __init__(self, params: Dict = None):
        # 默认参数设置
        default_params = {
            'MALength': 18,              # 均线值
            'ATRLength': 10,             # ATR的值
            'LookBack': 10,              # 用于给当前K线打分的回溯根数
            'ProtectStopATRMulti': 0.5,  # 保护性止损的ATR乘数
            'TrailStopATRMulti': 3,      # 跟踪止损的ATR乘数
            'BreakEvenStopATRMulti': 5,  # 盈亏平衡止损的ATR乘数
            'Lots': 1                    # 交易手数
        }
        
        if params:
            default_params.update(params)
        super().__init__(default_params)
        
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算技术指标"""
        df = data.copy()
        
        # 计算趋势得分
        df['TrendScore'] = 0
        for i in range(len(df)):
            temp = 0
            for j in range(1, self.params['LookBack'] + 1):
                if i - j >= 0:  # 确保索引有效
                    if df['close'].iloc[i] >= df['close'].iloc[i-j]:
                        temp += 1
                    else:
                        temp -= 1
            df.loc[df.index[i], 'TrendScore'] = temp
            
        # 计算均线和ATR
        df['MA'] = Average(df['close'], self.params['MALength'])
        df['TrendScoreMA'] = Average(df['TrendScore'], self.params['MALength'])
        df['ATR'] = AvgTrueRange(self.params['ATRLength'], df['high'], df['low'], df['close'])
        

        # 记录持仓后的最高价
        df['HighAfterEntry'] = df['high'].copy()
        
        return df
        
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """生成交易信号矩阵"""
        # 数据验证
        min_length = max(self.params['MALength'], self.params['ATRLength'], self.params['LookBack'])
        if len(data) < min_length:
            raise ValueError("数据长度不足")
            
        signals = pd.DataFrame(index=data.index)
        signals['call'] = np.nan
        
        current_position = 0
        entry_price = 0
        position_bars = 0
        high_after_entry = 0
        protect_stop = 0
        
        for i in range(1, len(data)):
            if current_position == 0:  # 未持仓
                # 入场条件检查
                if (data['close'].iloc[i-1] >= data['MA'].iloc[i-1] and 
                    data['TrendScore'].iloc[i-1] >= data['TrendScoreMA'].iloc[i-1]):
                    signals.iloc[i] = [1]
                    current_position = 1
                    entry_price = data['open'].iloc[i]
                    high_after_entry = data['high'].iloc[i]
                    protect_stop = (data['low'].iloc[i-1] - 
                                  self.params['ProtectStopATRMulti'] * data['ATR'].iloc[i-1])
                    position_bars = 0
                    
            elif current_position == 1:  # 持有多仓
                position_bars += 1
                high_after_entry = max(high_after_entry, data['high'].iloc[i])
                
                # 计算各种止损价位
                trail_stop = (high_after_entry - 
                            self.params['TrailStopATRMulti'] * data['ATR'].iloc[i-1])
                break_even_stop = entry_price
                
                # 确定出场线
                if high_after_entry >= (entry_price + 
                                      self.params['BreakEvenStopATRMulti'] * data['ATR'].iloc[i-1]):
                    exit_line = max(trail_stop, break_even_stop)
                else:
                    exit_line = max(trail_stop, protect_stop)
                
                # 检查是否触发止损
                if data['low'].iloc[i] <= exit_line:
                    signals.iloc[i] = [0]
                    current_position = 0
                    position_bars = 0
        
        # 在最后一根K线强制平仓
        if current_position == 1:
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
        strategy = TrendScore_L()
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
