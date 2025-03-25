'''
// 简称: NoHurrySystem_L
// 名称: 基于平移通道的交易系统多
// 类别: 策略应用
// 类型: 内建应用
// 输出:
// ------------------------------------------------------------------------
// ----------------------------------------------------------------------// 
//  策略说明:    
//              本策略基于平移后的高低点通道判断入场条件，结合ATR止损
//  系统要素:
//              1. 平移后的高低点通道
//                 2. atr止损
// 
//  入场条件：
//              1.当高点上穿平移通道高点时,开多仓
//              2.当低点下穿平移通道低点时,开空仓
//     
//  出场条件：
//              1.ATR跟踪止盈
//              2.通道止损
// 
//      注:当前策略仅为做多系统, 如需做空, 请参见CL_NoHurrySystem_S
// ----------------------------------------------------------------------// 
'''


import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict
import pandas as pd
import numpy as np
from module import *
from strategy.base import StrategyBase


class NoHurrySystem_L(StrategyBase):
    """基于平移通道的做多策略"""
    def __init__(self, params: Dict = None):
        # 默认参数设置
        default_params = {
            'ChanLength': 20,     # 通道计算周期
            'ChanDelay': 15,      # 通道平移周期
            'TrailingATRs': 3,    # ATR跟踪止损倍数
            'ATRLength': 10,      # ATR计算周期
            'Lots': 1             # 交易数量
        }
        
        # 使用传入的参数更新默认参数
        if params:
            default_params.update(params)
        super().__init__(default_params)
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算技术指标"""
        df = data.copy()
        
        # 最小变动价位, 这里假设为1, 具体应根据实际情况设置
        minpoint = 1  # 假设值
        
        df['UpperChan'] = Highest(df['high'], self.params['ChanLength'])  # UpperChan=N周期高点，默认20
        df['LowerChan'] = Lowest(df['low'], self.params['ChanLength'])  # LowerChan=N周期低点，默认20
        df['ATRVal'] = AvgTrueRange(self.params['ATRLength'], df['high'], df['low'], df['close']) * self.params['TrailingATRs']  # atr均值
        
        return df
       
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """生成交易信号矩阵"""
        # 数据验证
        min_length = self.params['ChanLength'] + self.params['ChanDelay']
        if len(data) < min_length:
            raise ValueError('数据长度不足')

        # 使用传入的data计算指标
        df = self.calculate_indicators(data)

        # 初始化信号矩阵
        signals = pd.DataFrame(index=df.index)
        signals['call'] = np.nan # 信号列

        # 状态变量
        current_position = 0  # 当前持仓状态，1为多头，-1为空头，0为无仓位
        position_bars = 0     # 持仓Bar计数
        PosHigh = 0           # 开仓后最高价
        stopline = None       # 止损线

        # 向量化计算 con 条件
        con = (df['high'] >= df['UpperChan'].shift(self.params['ChanDelay'] + 1)) & (df['high'].shift(1) < df['UpperChan'].shift(self.params['ChanDelay'] + 1))
        minpoint = 1 # 假设值

        open_ = df['open'].values
        high_ = df['high'].values
        low_ = df['low'].values
        UpperChan_shifted = df['UpperChan'].shift(self.params['ChanDelay'] + 1).values
        LowerChan_shifted = df['LowerChan'].shift(self.params['ChanDelay'] + 1).values
        ATRVal_shifted = df['ATRVal'].shift(1).values

        for i in range(min_length, len(df)):
            # 系统入场
            if current_position == 0:
                if con.iloc[i]:
                    signals.iloc[i, signals.columns.get_loc('call')] = 1  # Buy signal
                    current_position = 1
                    PosHigh = high_[i]
                    position_bars = 0

            # 系统出场
            elif current_position == 1:
                position_bars += 1
                PosHigh = max(PosHigh, high_[i])

                stopline = max(PosHigh - ATRVal_shifted[i], LowerChan_shifted[i] - minpoint)
                if low_[i] <= stopline:
                    signals.iloc[i, signals.columns.get_loc('call')] = 0  # Sell signal
                    current_position = 0
                    position_bars = 0
                    stopline = None

        # 在最后一根K线强制平仓
        if current_position == 1:
            signals.iloc[-1, signals.columns.get_loc('call')] = 0 # Sell signal

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
        strategy = NoHurrySystem_L()
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
