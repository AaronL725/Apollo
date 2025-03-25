'''
// 简称: JailBreakSys_L
// 名称: 基于价格区间突破的交易系统 
// 类别: 策略应用
// 类型: 内建应用
// 输出:
// 策略说明:        基于通道突破的判断
// 系统要素:
//                1. 计算50根k线最高价的区间
//                2. 计算30根k线最低价的区间
//                
// 入场条件:
//                1.价格高于50根K线最高价的区间入场
// 出场条件:
//                1. 当前价格低于30根K线最低价的区间出场
//                2. 当前价格低于入场价一定ATR波动率幅度出场
'''


import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict
import pandas as pd
import numpy as np
from module import *
from strategy.base import StrategyBase


class JailBreakSys_L(StrategyBase):
    """基于价格区间突破的做多策略"""
    def __init__(self, params: Dict = None):
        # 默认参数设置
        default_params = {
            'Length1': 50,     # 长周期区间参数
            'Length2': 30,     # 短周期区间参数
            'IPS': 4,         # 保护止损波动率参数
            'AtrVal': 10      # 波动率参数
        }
        # 更新自定义参数
        if params:
            default_params.update(params)
        super().__init__(default_params)
        
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算技术指标"""
        df = data.copy()
        
        # 获取参数
        L1 = max(self.params['Length1'], self.params['Length2'])  # 出场周期选择较大的区间参数
        L2 = min(self.params['Length1'], self.params['Length2'])  # 出场周期选择较小的区间参数
        
        # 计算ATR
        df['ATR'] = AvgTrueRange(self.params['AtrVal'], df['high'], df['low'], df['close'])
        
        # 计算价格区间
        df['Upperband'] = HighestFC(df['high'], L1)  # 长周期最高价区间
        df['Lowerband'] = LowestFC(df['low'], L1)    # 长周期最低价区间
        df['Exitlong'] = LowestFC(df['low'], L2)     # 短周期最低价区间
        df['Exitshort'] = HighestFC(df['high'], L2)  # 短周期最高价区间
        
        return df
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """生成交易信号矩阵"""
        # 数据验证
        min_length = max(self.params['Length1'], self.params['Length2'])
        if len(data) < min_length:
            raise ValueError('数据长度不足')
        
        signals = pd.DataFrame(index=data.index)
        signals['call'] = np.nan
        
        current_position = 0  # 当前持仓状态
        protect_stop = None   # 保护性止损价格
        
        for i in range(1, len(data)):
            if current_position == 0:  # 空仓状态
                # 价格大于长周期最高价区间入场做多
                if (data['high'].iloc[i] >= data['Upperband'].iloc[i-1] and 
                    data['vol'].iloc[i] > 0):
                    
                    entry_price = max(data['open'].iloc[i], data['Upperband'].iloc[i-1])
                    signals.iloc[i] = [1]
                    current_position = 1
                    protect_stop = entry_price - self.params['IPS'] * data['ATR'].iloc[i-1]
            
            elif current_position == 1:  # 持有多仓
                # 价格低于入场价以下一定ATR幅度止损
                if (data['low'].iloc[i] <= protect_stop and 
                    protect_stop >= data['Exitlong'].iloc[i-1] and 
                    data['vol'].iloc[i] > 0):
                    
                    exit_price = min(data['open'].iloc[i], protect_stop)
                    signals.iloc[i] = [0]
                    current_position = 0
                    protect_stop = None
                
                # 价格低于短周期最低价区间出场
                elif data['low'].iloc[i] <= data['Exitlong'].iloc[i-1]:
                    exit_price = min(data['open'].iloc[i], data['Exitlong'].iloc[i-1])
                    signals.iloc[i] = [0]
                    current_position = 0
                    protect_stop = None
        
        # 在最后一根K线强制平仓
        if current_position == 1:  # 如果还持有多仓
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
        strategy = JailBreakSys_L()
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
