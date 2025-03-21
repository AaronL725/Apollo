'''
//  简称: NoHurrySystem_S
//  名称: 基于平移通道的交易系统空
//  类别: 策略应用
//  类型: 内建应用 
//  输出:
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
//     注:当前策略仅为做空系统, 如需做多, 请参见CL_NoHurrySystem_L
'''


import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict
import pandas as pd
import numpy as np
from module import *
from strategy.base import StrategyBase


class NoHurrySystem_S(StrategyBase):
    """基于平移通道的做空策略"""
    def __init__(self, params: Dict = None):
        # // 默认参数设置
        default_params = {
            'ChanLength': 20,     # // 通道计算周期
            'ChanDelay': 15,      # // 通道平移周期
            'TrailingATRs': 3,   # // ATR跟踪止损倍数
            'ATRLength': 10       # // ATR计算周期
        }
        
        # // 更新自定义参数
        if params:
            default_params.update(params)
        super().__init__(default_params)
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算技术指标"""
        df = data.copy()
        
        # // 指标计算
        df['minpoint'] = 1  # // 最小变动价位
        df['UpperChan'] = Highest(df['high'], self.params['ChanLength'])  # // UpperChan=N周期高点，默认20
        df['LowerChan'] = Lowest(df['low'], self.params['ChanLength'])  # // LowerChan=N周期低点，默认20
        df['ATRVal'] = AvgTrueRange(self.params['ATRLength'], df['high'], df['low'], df['close']) * self.params['TrailingATRs']  # // atr均值
        
        return df

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """生成交易信号矩阵"""
        # // 数据验证
        min_length = self.params['ChanLength'] + self.params['ChanDelay']
        if len(data) < min_length:
            raise ValueError('数据长度不足')
        
        signals = pd.DataFrame(index=data.index)
        signals['call'] = np.nan

        current_position = 0  # // 当前持仓状态：0-空仓，-1-空头
        PosLow = None  # // 记录开仓后低点
        stopline = None
        bars_since_entry = 0 # // 记录开仓后的K线数量

        for i in range(self.params['ChanLength'] + self.params['ChanDelay'], len(data)):
            # // 系统入场
            # // 价格向下突破ChanDelay周期前的LowerChan，开空仓
            con = data['low'].iloc[i] <= data['LowerChan'].shift(self.params['ChanDelay'] + 1).iloc[i] and \
                  data['low'].iloc[i-1] > data['LowerChan'].shift(self.params['ChanDelay'] + 1).iloc[i-1]
            
            if current_position == 0:
                if con:
                    signals.loc[data.index[i], 'call'] = -1
                    current_position = -1
                    PosLow = data['low'].iloc[i]
                    bars_since_entry = 0 # // 重置开仓后的K线数量
            
            # // 系统出场
            # // PosLow记录开仓低点
            elif current_position == -1:
                bars_since_entry += 1 # // 更新开仓后的K线数量
                if bars_since_entry == 0:  # //TradeBlazer中此条件恒为false，因为上面一行代码已经+1
                    PosLow = data['low'].iloc[i]
                elif data['low'].iloc[i] < PosLow:
                    PosLow = data['low'].iloc[i]
                
                # // ATR跟踪止损,通道止损
                if bars_since_entry > 0:
                  stopline = min(PosLow + data['ATRVal'].shift(1).iloc[i],
                                data['UpperChan'].shift(self.params['ChanDelay'] + 1).iloc[i] + data['minpoint'].iloc[i])
                
                  if data['high'].iloc[i] >= stopline:
                      signals.loc[data.index[i], 'call'] = 1
                      current_position = 0
                      PosLow = None
                      stopline = None

            # // 在最后一根K线强制平仓
        if current_position == -1:
            signals.loc[data.index[-1], 'call'] = 1

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
        strategy = NoHurrySystem_S()
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
