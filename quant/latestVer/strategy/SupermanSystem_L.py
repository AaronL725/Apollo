'''
// 简称: SupermanSystem_L
// 名称: 基于市场强弱和动量的通道突破系统多 
// 类别: 策略应用
// 类型: 内建应用
// 输出:
//------------------------------------------------------------------------
//----------------------------------------------------------------------//
// 策略说明:
//             本策略是基于市场强弱指标和动量的通道突破系统
//             
// 系统要素:
//             1. 根据N根K线的收盘价相对前一根K线的涨跌计算出市场强弱指标
//             2. 最近9根K线的动量变化趋势
//             3. 最近N根K线的高低点形成的通道
// 入场条件:
//             1. 市场强弱指标为多头，且市场动量由空转多时，突破通道高点做多
//             2. 市场强弱指标为空头，且市场动量由多转空时，突破通道低点做空
// 出场条件: 
//             1. 开多以开仓BAR的最近N根BAR的低点作为止损价
//                开空以开仓BAR的最近N根BAR的高点作为止损价
//             2. 盈利超过止损额的一定倍数止盈
//             3. 出现反向信号止损
//
//         注: 当前策略仅为做多系统, 如需做空, 请参见CL_SupermanSystem_S
'''


import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict
import pandas as pd
import numpy as np
from module import *
from strategy.base import StrategyBase


class SupermanSystem_L(StrategyBase):
    """市场强弱指标做多策略"""
    def __init__(self, params: Dict = None):
        # 默认参数设置
        default_params = {
            'Length': 5,           # 强弱指标和通道计算的周期值
            'Stop_Len': 5,        # 止损通道的周期值
            'ProfitFactor': 3,    # 止盈相对止损的倍数
            'EntryStrength': 95   # 强弱指标的进场值
        }
        if params:
            default_params.update(params)
        super().__init__(default_params)
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算策略所需的技术指标"""
        df = data.copy()
        
        # 计算收盘价变动
        df['CloseChange'] = df['close'] - df['close'].shift(1)
        
        # 计算市场强弱指标
        def calculate_market_strength(changes):
            up_closes = sum(x for x in changes if x > 0)
            dn_closes = sum(x for x in changes if x <= 0)
            sum_change = sum(changes)
            
            if sum_change >= 0:
                return (sum_change / up_closes * 100) if up_closes != 0 else 0
            else:
                return (sum_change / abs(dn_closes) * 100) if dn_closes != 0 else 0
        
        # 使用rolling window计算MarketStrength
        df['MarketStrength'] = df['CloseChange'].rolling(
            window=self.params['Length']
        ).apply(calculate_market_strength)
        
        # 计算动量指标
        df['Momentum1'] = df['close'] - df['close'].shift(4)
        df['Momentum2'] = df['close'].shift(4) - df['close'].shift(8)
        
        # 计算高低点
        df['HH'] = HighestFC(df['high'], self.params['Length'])
        df['LL1'] = LowestFC(df['low'], self.params['Length'])
        df['LL2'] = LowestFC(df['low'], self.params['Stop_Len'])
        
        return df
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """生成交易信号矩阵"""
        signals = pd.DataFrame(index=data.index)
        signals['call'] = np.nan
        
        current_position = 0
        stop_loss = np.nan
        profit_target = np.nan
        entry_price = np.nan
        
        for i in range(8, len(data)):  # 从第9根K线开始（因为需要8根K线的历史数据）
            if current_position == 0:  # 空仓状态
                # 开多仓条件
                if (data['MarketStrength'].iloc[i-1] >= self.params['EntryStrength'] and
                    data['Momentum1'].iloc[i-1] >= 0 and
                    data['Momentum2'].iloc[i-1] < 0 and
                    data['high'].iloc[i] >= data['HH'].iloc[i-1] and
                    data['vol'].iloc[i] > 0):
                    
                    entry_price = max(data['open'].iloc[i], data['HH'].iloc[i-1])
                    stop_loss = data['LL2'].iloc[i]
                    profit_target = entry_price + (entry_price - stop_loss) * self.params['ProfitFactor']
                    
                    signals.iloc[i] = [1]
                    current_position = 1
            
            elif current_position == 1:  # 持有多仓
                if data['vol'].iloc[i] > 0:
                    # 止盈
                    if data['high'].iloc[i] >= profit_target:
                        signals.iloc[i] = [0]
                        current_position = 0
                    
                    # 止损
                    elif data['low'].iloc[i] <= stop_loss:
                        signals.iloc[i] = [0]
                        current_position = 0
                    
                    # 反向出场
                    elif (data['MarketStrength'].iloc[i-1] <= -1 * self.params['EntryStrength'] and
                          data['Momentum1'].iloc[i-1] < 0 and
                          data['Momentum2'].iloc[i-1] >= 0 and
                          data['low'].iloc[i] <= data['LL1'].iloc[i-1]):
                        signals.iloc[i] = [0]
                    current_position = 0
        
        # 循环结束时强制平仓
        if current_position == 1:
            signals.iloc[i] = [0]
        
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
        strategy = SupermanSystem_L()
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
