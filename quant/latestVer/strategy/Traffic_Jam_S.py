'''
// 简称: Traffic_Jam_S
// 名称: 基于DMI中ADX的震荡交易系统空 
// 类别: 策略应用
// 类型: 内建应用
//------------------------------------------------------------------------
//----------------------------------------------------------------------//
// 策略说明:
//             本策略基于DMI指标中的ADX指数判断行情是否为震荡, 然后通过k线形态进行逆势交易的系统
//             
// 系统要素:
//             1. DMI指标中的ADX指数
//             2. ConsecBars根阴线(收盘低于前根即可)或ConsecBars根阳线(收盘高于前根即可)
// 入场条件:
//             当ADX指数低于25且低于ADXLowThanBefore天前的值时
//             1. 如果出现连续ConsecBars根阴线(收盘低于前根即可), 则在下根k线开盘做多
//             2. 如果出现连续ConsecBars根阳线(收盘高于前根即可), 则在下根k线开盘做空
// 出场条件: 
//             1. 基于ATR的保护性止损
//             2. 入场ProactiveStopBars根K线后的主动性平仓
//
//         注: 当前策略仅为做空系统, 如需做多, 请参见CL_Traffic_Jam_L
'''


import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict
import pandas as pd
import numpy as np
from module import *
from strategy.base import StrategyBase


class Traffic_Jam_S(StrategyBase):
    """基于DMI中ADX的震荡交易系统空"""
    def __init__(self, params: Dict = None):
        # 默认参数设置
        default_params = {
            'DMI_N': 14,                    # DMI的N值
            'DMI_M': 6,                     # DMI的M值
            'ADXLevel': 25,                 # ADX低于此值时被认为行情处于震荡中
            'ADXLowThanBefore': 3,          # 入场条件中ADX需要弱于之前值的天数
            'ConsecBars': 3,                # 入场条件中连续阳线或阴线的个数
            'ATRLength': 10,                # ATR值
            'ProtectStopATRMulti': 0.5,     # 保护性止损的ATR乘数
            'ProactiveStopBars': 10,        # 入场后主动平仓的等待根数
            'Lots': 1                       # 交易手数
        }
        if params:
            default_params.update(params)
        super().__init__(default_params)

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算技术指标"""
        df = data.copy()
        n = self.params['DMI_N']
        
        # Calculate True Range
        tr = TrueRange(df['high'], df['low'], df['close'])
        
        # Initialize DMI variables
        df['DMI_plus'] = pd.Series(np.nan, index=df.index)
        df['DMI_minus'] = pd.Series(np.nan, index=df.index)
        df['DMI'] = pd.Series(np.nan, index=df.index)
        df['ADX'] = pd.Series(np.nan, index=df.index)
        df['ADXR'] = pd.Series(np.nan, index=df.index)
        df['Volty'] = pd.Series(np.nan, index=df.index)
        df['avg_plus_dm'] = pd.Series(np.nan, index=df.index)
        df['avg_minus_dm'] = pd.Series(np.nan, index=df.index)
        
        # Smoothing factor
        sf = 1.0 / n
        
        # Initial calculation at bar N
        for i in range(n, len(df)):
            if i == n:
                # Calculate initial sums exactly as TradeBlazor
                sum_plus_dm = 0
                sum_minus_dm = 0
                sum_tr = 0
                
                for j in range(n):
                    upper_move = df['high'].iloc[i-j] - df['high'].iloc[i-j-1]
                    lower_move = df['low'].iloc[i-j-1] - df['low'].iloc[i-j]
                    
                    plus_dm = upper_move if (upper_move > lower_move and upper_move > 0) else 0
                    minus_dm = lower_move if (lower_move > upper_move and lower_move > 0) else 0
                    
                    sum_plus_dm += plus_dm
                    sum_minus_dm += minus_dm
                    sum_tr += tr.iloc[i-j]
                
                df.loc[df.index[i], 'avg_plus_dm'] = sum_plus_dm / n
                df.loc[df.index[i], 'avg_minus_dm'] = sum_minus_dm / n
                df.loc[df.index[i], 'Volty'] = sum_tr / n
                
            elif i > n:
                # Calculate smoothed values exactly as TradeBlazor
                upper_move = df['high'].iloc[i] - df['high'].iloc[i-1]
                lower_move = df['low'].iloc[i-1] - df['low'].iloc[i]
                
                plus_dm = upper_move if (upper_move > lower_move and upper_move > 0) else 0
                minus_dm = lower_move if (lower_move > upper_move and lower_move > 0) else 0
                
                df.loc[df.index[i], 'avg_plus_dm'] = df['avg_plus_dm'].iloc[i-1] + sf * (plus_dm - df['avg_plus_dm'].iloc[i-1])
                df.loc[df.index[i], 'avg_minus_dm'] = df['avg_minus_dm'].iloc[i-1] + sf * (minus_dm - df['avg_minus_dm'].iloc[i-1])
                df.loc[df.index[i], 'Volty'] = df['Volty'].iloc[i-1] + sf * (tr.iloc[i] - df['Volty'].iloc[i-1])
            
            # Calculate DMI values
            if df['Volty'].iloc[i] > 0:
                df.loc[df.index[i], 'DMI_plus'] = 100 * df['avg_plus_dm'].iloc[i] / df['Volty'].iloc[i]
                df.loc[df.index[i], 'DMI_minus'] = 100 * df['avg_minus_dm'].iloc[i] / df['Volty'].iloc[i]
            else:
                df.loc[df.index[i], 'DMI_plus'] = 0
                df.loc[df.index[i], 'DMI_minus'] = 0
            
            # Calculate DX and DMI
            divisor = df['DMI_plus'].iloc[i] + df['DMI_minus'].iloc[i]
            if divisor > 0:
                dx = 100 * abs(df['DMI_plus'].iloc[i] - df['DMI_minus'].iloc[i]) / divisor
            else:
                dx = 0
            df.loc[df.index[i], 'DMI'] = dx
            
            # Calculate ADX exactly as TradeBlazor
            if i > 0:
                if i <= n:
                    df.loc[df.index[i], 'ADX'] = df['DMI'].iloc[:i+1].sum() / (i + 1)
                    df.loc[df.index[i], 'ADXR'] = (df['ADX'].iloc[i] + df['ADX'].iloc[i-1]) * 0.5
                else:
                    df.loc[df.index[i], 'ADX'] = df['ADX'].iloc[i-1] + sf * (dx - df['ADX'].iloc[i-1])
                    df.loc[df.index[i], 'ADXR'] = (df['ADX'].iloc[i] + df['ADX'].iloc[i-self.params['DMI_M']]) * 0.5
        
        # Calculate ATR
        df['ATR'] = AvgTrueRange(self.params['ATRLength'], df['high'], df['low'], df['close'])
        
        # Calculate consecutive up bars count exactly as TradeBlazor
        df['ConsecBarsCount'] = (df['close'] > df['close'].shift(1)).astype(int)
        df['ConsecBarsCount'] = df['ConsecBarsCount'].rolling(window=self.params['ConsecBars']).sum()
        
        # Calculate protection stops
        df['ProtectStopS'] = df['high'].shift(1) + self.params['ProtectStopATRMulti'] * df['ATR'].shift(1)
        
        return df

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """生成交易信号矩阵"""
        min_length = max(self.params['DMI_N'], self.params['ATRLength'])
        if len(data) < min_length:
            return pd.DataFrame()
            
        signals = pd.DataFrame(index=data.index)
        signals['call'] = pd.Series(np.nan, index=data.index)
        
        current_position = 0
        prev_position = 0
        position_bars = 0
        
        for i in range(min_length, len(data)):
            # Entry conditions exactly as TradeBlazor
            adx_condition = (data['ADX'].iloc[i-1] < self.params['ADXLevel'] and 
                           data['ADX'].iloc[i-1] < data['ADX'].iloc[i-self.params['ADXLowThanBefore']-1])
            consec_bars_condition = data['ConsecBarsCount'].iloc[i-1] == self.params['ConsecBars']
            volume_condition = data['vol'].iloc[i] > 0
            
            if current_position != -1 and i > self.params['DMI_N'] and adx_condition and consec_bars_condition and volume_condition:
                signals.loc[signals.index[i], 'call'] = -1
                current_position = -1
                position_bars = 0
            
            elif current_position == -1 and prev_position == -1 and volume_condition:
                position_bars += 1
                
                # Exit conditions exactly as TradeBlazor
                proactive_stop = position_bars >= self.params['ProactiveStopBars']
                protect_stop = data['high'].iloc[i] >= data['ProtectStopS'].iloc[i-1]
                
                if proactive_stop or protect_stop:
                    signals.loc[signals.index[i], 'call'] = 0
                    current_position = 0
                    position_bars = 0
            
            prev_position = current_position
        
        # Force close position at the end
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
        strategy = Traffic_Jam_S()
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
