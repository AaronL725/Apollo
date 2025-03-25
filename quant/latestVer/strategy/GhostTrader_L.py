'''
// 简称: GhostTrader_L
// 名称: 幽灵交易者_多
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
        此公式仅做多
'''


import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict
import pandas as pd
import numpy as np
from module import *
from strategy.base import StrategyBase


class GhostTrader_L(StrategyBase):
    """幽灵交易者做多策略"""
    
    def __init__(self, params: Dict = None):
        super().__init__(params)
        self.default_params = {
            'FastLength': 9,     # 短期指数平均线参数
            'SlowLength': 19,    # 长期指数平均线参数
            'Length': 9,         # RSI参数
            'OverSold': 30,      # 超卖
            'OverBought': 70,    # 超买
            'Lots': 1,          # 交易手数
        }
        self.params = {**self.default_params, **(params or {})}

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算技术指标，完全复制TradeBlazor的计算方法"""
        df = data.copy()
        
        # 计算指数移动平均线
        df['AvgValue1'] = df['close'].ewm(span=self.params['FastLength'], adjust=False).mean()
        df['AvgValue2'] = df['close'].ewm(span=self.params['SlowLength'], adjust=False).mean()
        
        # 计算RSI (使用TradeBlazor的方法)
        df['NetChgAvg'] = 0.0
        df['TotChgAvg'] = 0.0
        df['RSIValue'] = 0.0
        
        length = self.params['Length']
        for i in range(len(df)):
            if i <= length - 1:
                # 初始期间的计算
                if i >= length:  # 确保有足够的数据
                    df.loc[df.index[i], 'NetChgAvg'] = (
                        df['close'].iloc[i] - df['close'].iloc[i-length]
                    ) / length
                    df.loc[df.index[i], 'TotChgAvg'] = (
                        abs(df['close'].diff()).iloc[i-length+1:i+1].mean()
                    )
            else:
                # 后续期间的计算
                sf = 1/length
                change = df['close'].iloc[i] - df['close'].iloc[i-1]
                df.loc[df.index[i], 'NetChgAvg'] = (
                    df['NetChgAvg'].iloc[i-1] + 
                    sf * (change - df['NetChgAvg'].iloc[i-1])
                )
                df.loc[df.index[i], 'TotChgAvg'] = (
                    df['TotChgAvg'].iloc[i-1] + 
                    sf * (abs(change) - df['TotChgAvg'].iloc[i-1])
                )
            
            # 计算RSI值
            if df['TotChgAvg'].iloc[i] != 0:
                chg_ratio = df['NetChgAvg'].iloc[i] / df['TotChgAvg'].iloc[i]
            else:
                chg_ratio = 0
            df.loc[df.index[i], 'RSIValue'] = 50 * (chg_ratio + 1)
        
        # 计算唐奇安通道
        df['ExitHiBand'] = Highest(df['high'], 20)  # 唐奇安通道上轨
        df['ExitLoBand'] = Lowest(df['low'], 20)    # 唐奇安通道下轨
        
        return df

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """生成交易信号矩阵"""
        # 数据验证
        min_length = max(self.params['FastLength'], self.params['SlowLength'], 
                        self.params['Length'], 20)
        if len(data) < min_length:
            return pd.DataFrame()
            
        signals = pd.DataFrame(index=data.index)
        signals['call'] = np.nan  # 信号: 1买入, 0平仓, nan无信号
        
        current_position = 0  # 当前持仓状态
        myProfit = 0         # 模拟交易利润
        myEntryPrice = 0     # 模拟交易开仓价格
        
        for i in range(1, len(data)):
            # 平多仓条件：持有多单时下破唐奇安通道下轨
            if (current_position == 1 and 
                current_position == 1 and
                data['low'].iloc[i] <= data['ExitLoBand'].iloc[i-1]):
                myExitPrice = min(data['open'].iloc[i], data['ExitLoBand'].iloc[i-1])
                signals.loc[signals.index[i], 'call'] = 0  # 使用 .loc
                myProfit = myExitPrice - myEntryPrice
                current_position = 0
                
            # 开多仓条件：模拟交易亏损、短期均线在长期均线之上、RSI低于超买值、创新高
            elif (current_position == 0 and 
                  current_position == 0 and
                  data['AvgValue1'].iloc[i-1] > data['AvgValue2'].iloc[i-1] and
                  data['RSIValue'].iloc[i-1] < self.params['OverBought'] and
                  data['high'].iloc[i] >= data['high'].iloc[i-1]):
                myEntryPrice = max(data['open'].iloc[i], data['high'].iloc[i-1])
                current_position = 1
                if myProfit < 0:
                    signals.loc[signals.index[i], 'call'] = 1  # 使用 .loc
            
        # 在最后一根K线强制平仓
        if current_position == 1:
            signals.loc[signals.index[-1], 'call'] = 0  # 使用 .loc
            
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
        strategy = GhostTrader_L()
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
