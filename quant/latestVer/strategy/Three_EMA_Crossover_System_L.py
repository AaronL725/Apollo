'''
// 简称: Three_EMA_Crossover_System_L
// 名称: 基于指数移动平均线组进行判断 多
// 类别: 策略应用
// 类型: 内建应用
// 输出:
//------------------------------------------------------------------------
//------------------------------------------------------------------------
// 策略说明:
//            1.计算三条指数移动平均线(Avg1, Avg2 , Avg3)；
//            2.通过指数移动平均线的组合来判断趋势
//             
// 入场条件:
//            1.当Avg1向上穿过Avg2并且Avg2大于Avg3时，在下一根k线开盘处买入
//            2.当Avg1向下穿过Avg2并且Avg2小于Avg3时，在下一根k线开盘处卖出
// 出场条件: 
//            1.Avg1下穿Avg2多头出场
//            2.跟踪止损
//
//         注: 当前策略仅为做多系统, 如需做空, 请参见CL_Three_EMA_Crossover_System_S
'''


import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict
import pandas as pd
import numpy as np
from module import *
from strategy.base import StrategyBase


class Three_EMA_Crossover_System_L(StrategyBase):
    """基于三重指数移动平均线的做多策略"""
    
    def __init__(self, params: Dict = None):
        # 默认参数设置
        default_params = {
            'AvgLen1': 6,    # 指数移动平均周期1
            'AvgLen2': 12,   # 指数移动平均周期2
            'AvgLen3': 28,   # 指数移动平均周期3
            'RLength': 4,    # 跟踪止损
            'Lots': 1        # 交易手数
        }
        
        # 更新自定义参数
        if params:
            default_params.update(params)
        super().__init__(default_params)
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算技术指标"""
        df = data.copy()
        
        # 计算三条指数移动平均线
        df['Avg1'] = df['close'].ewm(span=self.params['AvgLen1'], adjust=False).mean()
        df['Avg2'] = df['close'].ewm(span=self.params['AvgLen2'], adjust=False).mean()
        df['Avg3'] = df['close'].ewm(span=self.params['AvgLen3'], adjust=False).mean()
        
        # 计算K线幅度
        df['MyRange'] = df['high'] - df['low']
        df['RangeL'] = df['MyRange'].rolling(window=self.params['RLength']).mean()
        
        # 计算买入条件
        df['BuyCon1'] = (df['Avg1'] > df['Avg2']) & (df['Avg1'].shift(1) <= df['Avg2'].shift(1))
        
        # 计算跟踪止损价
        df['LongStopPrice'] = np.nan
        
        return df
        
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """生成交易信号矩阵"""
        # 数据验证
        min_length = max(self.params['AvgLen1'], self.params['AvgLen2'], 
                        self.params['AvgLen3'], self.params['RLength'])
        if len(data) < min_length:
            raise ValueError("数据长度不足")
        
        # 初始化信号矩阵
        signals = pd.DataFrame(index=data.index)
        signals['call'] = np.nan
        
        current_position = 0  # 当前持仓状态
        position_bars = 0     # 持仓周期计数
        long_stop_price = 0   # 跟踪止损价
        
        for i in range(1, len(data)):
            # 更新持仓信息
            if current_position == 1:
                position_bars += 1
            
            # 入场信号
            if current_position == 0:
                if (data['BuyCon1'].iloc[i-1] and 
                    data['Avg2'].iloc[i-1] > data['Avg3'].iloc[i-1] and
                    data['vol'].iloc[i] > 0):
                    signals.iloc[i] = 1
                    current_position = 1
                    position_bars = 0
                    continue
            
            # 出场信号
            elif current_position == 1:
                # 均线交叉出场
                if (data['Avg1'].iloc[i-1] < data['Avg2'].iloc[i-1] and 
                    position_bars > 0 and
                    data['vol'].iloc[i] > 0):
                    signals.iloc[i] = 0
                    current_position = 0
                    position_bars = 0
                    continue
                
                # 更新跟踪止损价
                if position_bars == 0:
                    long_stop_price = data['low'].iloc[i] - data['RangeL'].iloc[i]
                elif position_bars > 0:
                    long_stop_price = long_stop_price + (data['low'].iloc[i] - long_stop_price) * 0.25
                
                # 跟踪止损出场
                if (position_bars > 0 and 
                    data['low'].iloc[i] <= long_stop_price and 
                    data['vol'].iloc[i] > 0):
                    signals.iloc[i] = 0
                    current_position = 0
                    position_bars = 0
        
        # 在最后一根K线强制平仓
        if current_position == 1:
            signals.iloc[-1] = 0
        
        return signals


#########################主函数#########################
def main():
    """主函数，执行多品种回测"""
    logger = setup_logging()
    
    level = 'min15'  # 默认值
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
        strategy = Three_EMA_Crossover_System_L()
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
