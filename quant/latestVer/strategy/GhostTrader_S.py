'''
// 简称: GhostTrader_S
// 名称: 幽灵交易者_空
// 类别: 策略应用
// 类型: 内建应用
// 输出:
//------------------------------------------------------------------------
/* 
策略说明:
        模拟交易产生一次亏损后才启动真实下单交易。
系统要素:
        1、两条指数平均线
        2、RSI指标
        3、唐奇安通道
入场条件:
        1、模拟交易产生一次亏损、短期均线在长期均线之上、RSI低于超买值、创新高，则开多单
        2、模拟交易产生一次亏损、短期均线在长期均线之下、RSI高于超卖值、创新低，则开空单
出场条件:
        1、持有多单时小于唐奇安通道下轨，平多单
        2、持有空单时大于唐奇安通道上轨，平空单
注    意:
        此公式仅做空
'''


import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict
import pandas as pd
import numpy as np
from module import *
from strategy.base import StrategyBase


class GhostTrader_S(StrategyBase):
    """幽灵交易者做空策略"""
    
    def __init__(self, params: Dict = None):
        super().__init__(params)
        self.default_params = {
            'FastLength': 9,     # 短期指数平均线参数
            'SlowLength': 19,    # 长期指数平均线参数
            'Length': 9,         # RSI参数
            'OverSold': 30,      # 超卖
            'OverBought': 70,    # 超买
            'Lots': 1            # 交易手数
        }
        self.params = {**self.default_params, **(params or {})}
        
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算策略指标"""
        # 计算短期和长期指数平均线
        data['AvgValue1'] = XAverage(data['close'], self.params['FastLength'])
        data['AvgValue2'] = XAverage(data['close'], self.params['SlowLength'])
        
        # 计算RSI (严格按照TB的方法)
        length = self.params['Length']
        data['NetChgAvg'] = np.nan
        data['TotChgAvg'] = np.nan
        data['ChgRatio'] = np.nan
        data['RSIValue'] = np.nan
        
        for i in range(length, len(data)):
            if i == length:  # 第一个值的计算
                data.loc[data.index[i], 'NetChgAvg'] = (data['close'].iloc[i] - data['close'].iloc[0]) / length
                data.loc[data.index[i], 'TotChgAvg'] = data['close'].diff().abs().iloc[1:i+1].mean()
            else:  # 后续值使用Wilder平滑
                sf = 1/length
                change = data['close'].iloc[i] - data['close'].iloc[i-1]
                data.loc[data.index[i], 'NetChgAvg'] = (data['NetChgAvg'].iloc[i-1] + 
                    sf * (change - data['NetChgAvg'].iloc[i-1]))
                data.loc[data.index[i], 'TotChgAvg'] = (data['TotChgAvg'].iloc[i-1] + 
                    sf * (abs(change) - data['TotChgAvg'].iloc[i-1]))
            
            # 计算RSI值
            if data['TotChgAvg'].iloc[i] != 0:
                data.loc[data.index[i], 'ChgRatio'] = data['NetChgAvg'].iloc[i] / data['TotChgAvg'].iloc[i]
            else:
                data.loc[data.index[i], 'ChgRatio'] = 0
            
            data.loc[data.index[i], 'RSIValue'] = 50 * (data['ChgRatio'].iloc[i] + 1)
        
        # 计算唐奇安通道
        data['ExitHiBand'] = Highest(data['high'], 20)
        data['ExitLoBand'] = Lowest(data['low'], 20)
        
        return data
        
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """生成交易信号矩阵"""
        if len(data) < max(self.params['FastLength'], self.params['SlowLength'], 20):
            return pd.DataFrame()
            
        signals = pd.DataFrame(index=data.index)
        signals['call'] = np.nan
        
        current_position = 0
        prev_position = 0
        entry_price = None
        profit = 0  # 模拟交易的盈亏
        
        for i in range(1, len(data)):
            prev_position = current_position
            
            # 平空仓条件：持有空单时大于唐奇安通道上轨
            if (current_position == -1 and 
                prev_position == -1 and 
                data['high'].iloc[i] >= data['ExitHiBand'].iloc[i-1]):
                
                exit_price = max(data['open'].iloc[i], data['ExitHiBand'].iloc[i-1])
                profit = entry_price - exit_price if entry_price is not None else 0
                signals.loc[signals.index[i], 'call'] = 0
                current_position = 0
                entry_price = None
                
            # 开空仓条件：
            # 1. 空仓
            # 2. 前一个bar也是空仓
            # 3. 短期均线在长期均线之下
            # 4. RSI高于超卖值
            # 5. 创新低
            elif (current_position == 0 and 
                  prev_position == 0 and
                  data['AvgValue1'].iloc[i-1] < data['AvgValue2'].iloc[i-1] and
                  data['RSIValue'].iloc[i-1] > self.params['OverSold'] and
                  data['low'].iloc[i] <= data['low'].iloc[i-1]):
                
                entry_price = min(data['open'].iloc[i], data['low'].iloc[i-1])
                
                # 只有在模拟交易产生亏损后才实际开仓
                if profit < 0:
                    signals.loc[signals.index[i], 'call'] = -1
                    current_position = -1
                else:
                    current_position = -1  # 仅模拟交易，不发出信号
                
        # 在最后一根K线强制平仓
        if current_position == -1:
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
        strategy = GhostTrader_S()
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
