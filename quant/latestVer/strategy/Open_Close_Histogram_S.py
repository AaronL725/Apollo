'''
// 简称: Open_Close_Histogram_S
// 名称: 基于开收盘价格间的相对关系变化进行判断 空
// 类别: 策略应用
// 类型: 内建应用
// 输出:
//------------------------------------------------------------------------
//----------------------------------------------------------------------//
// 策略说明:
//             本策略计算指数移动平均(10个开盘价和10个收盘价, 然后后者减去前者得到柱状图)，通过柱状图上穿零轴还是下穿零轴来判断上升和下降趋势
//             
// 系统要素:
//             1. 10个开盘价的指数移动平均与10个收盘价的指数移动平均之差若上穿零轴定义为上升趋势，上升趋势定义满足后将上穿K线的最高价加上10周
//                期的ATR的一半作为多头入场触发价，同时将上穿K线的最低价减去10周期的ATR的一半作为多头平仓触发价；
//             2. 10个开盘价的指数移动平均与10个收盘价的指数移动平均之差若下穿零轴定义为下降趋势，下降趋势定义满足后将下穿K线的最低价减去10周
//                期的ATR的一半作为空头入场触发价，同时将下穿K线的最高价加上10周期的ATR的一半作为空头平仓触发价；
// 入场条件:
//             1. 10个开盘价的指数移动平均大于10个收盘价的指数移动平均并且向上突破了多头触发价则进场做多；
//             2. 10个开盘价的指数移动平均小于10个收盘价的指数移动平均并且向下突破了空头触发价则进场做空；
// 出场条件: 
//             1. 跌破多头平仓触发价或者转为下降趋势多头平仓；
//             2. 突破空头平仓触发价或者转为上升趋势空头平仓；
//             
//
//         注: 当前策略仅为做空系统, 如需做多, 请参见CL_Open_Close_Histogram_L
'''


import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict
import pandas as pd
import numpy as np
from module import *
from strategy.base import StrategyBase


class Open_Close_Histogram_S(StrategyBase):
    """基于开收盘价格间的相对关系变化进行判断的做空策略"""
    
    def __init__(self, params: Dict = None):
        # 默认参数设置
        default_params = {
            'OpenLen': 10,     # 用于计算开盘价指数移动平均的周期
            'CloseLen': 10,    # 用于计算收盘价指数移动平均的周期
            'Lots': 1,         # 交易手数
        }
        if params:
            default_params.update(params)
        super().__init__(default_params)

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算技术指标"""
        df = data.copy()
        
        # 计算开盘价和收盘价的指数移动平均
        # XAverage in TradeBlazor is equivalent to EMA
        df['OpenMA'] = df['open'].ewm(span=self.params['OpenLen'], adjust=False).mean()
        df['CloseMA'] = df['close'].ewm(span=self.params['CloseLen'], adjust=False).mean()
        
        # 计算柱状图 (CloseMA - OpenMA, 与TradeBlazor保持一致)
        df['Histogram'] = df['CloseMA'] - df['OpenMA']
        
        # 计算穿越条件 (CrossOver/CrossUnder)
        df['con1'] = (df['Histogram'] > 0) & (df['Histogram'].shift(1) <= 0)  # 上穿
        df['con2'] = (df['Histogram'] < 0) & (df['Histogram'].shift(1) >= 0)  # 下穿
        
        # 计算ATR
        df['ATR10'] = TrueRange(df['high'], df['low'], df['close']).rolling(window=10).mean()
        
        # 计算触发价格 (仅在下穿时更新)
        df.loc[df['con2'], 'SellPrice'] = df['low'] - df['ATR10'] * 0.5
        df.loc[df['con2'], 'ShortExitPrice'] = df['high'] + df['ATR10'] * 0.5
        
        # 向前填充触发价格
        df['SellPrice'] = df['SellPrice'].ffill()
        df['ShortExitPrice'] = df['ShortExitPrice'].ffill()
        
        return df

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """生成交易信号矩阵"""
        # 数据验证
        min_length = max(self.params['OpenLen'], self.params['CloseLen'])
        if len(data) < min_length:
            return pd.DataFrame()
            
        signals = pd.DataFrame(index=data.index)
        signals['call'] = np.nan
        
        current_position = 0
        position_bars = 0
        
        for i in range(1, len(data)):
            # 更新持仓天数
            if current_position != 0:
                position_bars += 1
                
            # 空头入场条件: Histogram[1]<0 And Vol > 0 And low<=SellPrice
            if (current_position == 0 and 
                data['Histogram'].iloc[i-1] < 0 and 
                data['vol'].iloc[i] > 0 and
                data['low'].iloc[i] <= data['SellPrice'].iloc[i]):
                signals.loc[data.index[i], 'call'] = -1
                current_position = -1
                position_bars = 0
                
            # 空头平仓条件1: MarketPosition==-1 And BarsSinceEntry>0 And con1[1] And Vol > 0
            elif (current_position == -1 and 
                  position_bars > 0 and 
                  data['con1'].iloc[i-1] and
                  data['vol'].iloc[i] > 0):
                signals.loc[data.index[i], 'call'] = 0
                current_position = 0
                position_bars = 0
                
            # 空头平仓条件2: MarketPosition==-1 And BarsSinceEntry>0 And High>=ShortExitPrice And Vol > 0
            elif (current_position == -1 and 
                  position_bars > 0 and 
                  data['high'].iloc[i] >= data['ShortExitPrice'].iloc[i] and
                  data['vol'].iloc[i] > 0):
                signals.loc[data.index[i], 'call'] = 0
                current_position = 0
                position_bars = 0
                
        # 在最后一根K线强制平仓
        if current_position == -1:
            signals.loc[data.index[-1], 'call'] = 0
            
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
        strategy = Open_Close_Histogram_S()
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
