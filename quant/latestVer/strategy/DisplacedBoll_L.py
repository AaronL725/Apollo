'''
// 简称: DisplacedBoll_L 
// 名称: 基于平移布林通道的系统多
// 类别: 策略应用
// 类型: 内建应用
// 输出:
//------------------------------------------------------------------------
//----------------------------------------------------------------------//
// 策略说明:
//             基于平移的boll通道突破系统
//
// 系统要素:
//             1. 平移的boll通道
//
// 入场条件:
//             1、关键价格突破通道上轨，则开多仓
//            2、关键价格突破通道下轨，则开空仓
//
// 出场条件:
//             1、关键价格突破通道上轨，则平空仓
//            2、关键价格突破通道下轨，则平多仓
//
//        注:当前策略仅为做多系统, 如需做空, 请参见CL_DisplacedBoll_S
'''


import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict
import pandas as pd
import numpy as np
from module import *
from strategy.base import StrategyBase


class DisplacedBoll_L(StrategyBase):
    """基于平移布林通道的做多策略"""
    def __init__(self, params: Dict = None):
        # 默认参数设置
        default_params = {
            'AvgLen': 3,      # boll均线周期参数
            'Disp': 16,       # boll平移参数
            'SDLen': 12,      # boll标准差周期参数
            'SDev': 2,        # boll通道倍数参数
        }
        # 更新自定义参数
        super().__init__(params)
        self.params = {**default_params, **(params or {})}

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算策略指标"""
        df = data.copy()
        
        # 计算关键价格
        df['Price'] = df['close']
        
        # 计算平移boll通道
        df['AvgVal'] = Average(df['Price'], self.params['AvgLen'])
        df['SDmult'] = StandardDev(df['Price'], self.params['SDLen'], 2) * self.params['SDev']
        
        # 计算平移后的通道上下轨
        df['DispTop'] = df['AvgVal'].shift(self.params['Disp']) + df['SDmult']
        df['DispBottom'] = df['AvgVal'].shift(self.params['Disp']) - df['SDmult']
        
        return df

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """生成交易信号矩阵"""
        # 数据验证
        min_length = self.params['Disp'] + max(self.params['AvgLen'], self.params['SDLen'])
        if len(data) < min_length:
            raise ValueError('数据长度不足')
        
        signals = pd.DataFrame(index=data.index)
        signals['call'] = np.nan
        
        current_position = 0  # 当前持仓状态
        bars_since_entry = 0  # 入场后的K线计数
        
        for i in range(1, len(data)):
            if current_position == 0:  # 空仓
                if data['high'].iloc[i] >= data['DispTop'].iloc[i-1]:
                    # 开多仓
                    entry_price = max(data['open'].iloc[i], data['DispTop'].iloc[i-1])
                    signals.iloc[i] = [1]
                    current_position = 1
                    bars_since_entry = 0
                    
            elif current_position == 1:  # 持有多仓
                bars_since_entry += 1
                if bars_since_entry > 0 and data['low'].iloc[i] <= data['DispBottom'].iloc[i-1]:
                    # 平多仓
                    exit_price = min(data['open'].iloc[i], data['DispBottom'].iloc[i-1])
                    signals.iloc[i] = [0]
                    current_position = 0
                    bars_since_entry = 0
        
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
        strategy = DisplacedBoll_L()
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
