'''
// 简称: Traffic_Jam_L
// 名称: 基于DMI中ADX的震荡交易系统多 
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
//         注: 当前策略仅为做多系统, 如需做空, 请参见CL_Traffic_Jam_S
'''


import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict
import pandas as pd
import numpy as np
from module import *
from strategy.base import StrategyBase


class Traffic_Jam_L(StrategyBase):
    """基于DMI中ADX的震荡交易系统多"""
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
        
        # Initialize DMI components
        df['DMI_Plus'] = np.nan
        df['DMI_Minus'] = np.nan
        df['ADX'] = np.nan
        df['ADXR'] = np.nan
        df['Volty'] = np.nan
        
        # Calculate True Range
        tr = TrueRange(df['high'], df['low'], df['close'])
        
        # DMI calculation following TradeBlazer logic
        sf = 1/self.params['DMI_N']
        sum_plus_dm = 0
        sum_minus_dm = 0
        sum_tr = 0
        avg_plus_dm = 0
        avg_minus_dm = 0
        s_volty = 0
        
        # Initialize cumulative DMI for ADX calculation
        cumm_dmi = 0
        
        for i in range(len(df)):
            if i == 0:
                continue
            
            # Calculate moves exactly as TradeBlazer
            upper_move = df['high'].iloc[i] - df['high'].iloc[i-1]
            lower_move = df['low'].iloc[i-1] - df['low'].iloc[i]
            
            plus_dm = 0
            minus_dm = 0
            
            if upper_move > lower_move and upper_move > 0:
                plus_dm = upper_move
            elif lower_move > upper_move and lower_move > 0:
                minus_dm = lower_move
            
            if i == self.params['DMI_N']:
                # Initial calculation at DMI_N period - matches TradeBlazer
                for j in range(self.params['DMI_N']):
                    idx = i - j
                    upper_move_init = df['high'].iloc[idx] - df['high'].iloc[idx-1]
                    lower_move_init = df['low'].iloc[idx-1] - df['low'].iloc[idx]
                    
                    plus_dm_init = 0
                    minus_dm_init = 0
                    
                    if upper_move_init > lower_move_init and upper_move_init > 0:
                        plus_dm_init = upper_move_init
                    elif lower_move_init > upper_move_init and lower_move_init > 0:
                        minus_dm_init = lower_move_init
                    
                    sum_plus_dm += plus_dm_init
                    sum_minus_dm += minus_dm_init
                    sum_tr += tr.iloc[idx]
                
                avg_plus_dm = sum_plus_dm / self.params['DMI_N']
                avg_minus_dm = sum_minus_dm / self.params['DMI_N']
                s_volty = sum_tr / self.params['DMI_N']
                
            elif i > self.params['DMI_N']:
                # Subsequent calculations using smoothing - matches TradeBlazer
                avg_plus_dm = avg_plus_dm + sf * (plus_dm - avg_plus_dm)
                avg_minus_dm = avg_minus_dm + sf * (minus_dm - avg_minus_dm)
                s_volty = s_volty + sf * (tr.iloc[i] - s_volty)
            
            if s_volty > 0:
                df.loc[df.index[i], 'DMI_Plus'] = 100 * avg_plus_dm / s_volty
                df.loc[df.index[i], 'DMI_Minus'] = 100 * avg_minus_dm / s_volty
            else:
                df.loc[df.index[i], 'DMI_Plus'] = 0
                df.loc[df.index[i], 'DMI_Minus'] = 0
            
            # Calculate DMI and ADX exactly as TradeBlazer
            divisor = df['DMI_Plus'].iloc[i] + df['DMI_Minus'].iloc[i]
            if divisor > 0:
                dmi = 100 * abs(df['DMI_Plus'].iloc[i] - df['DMI_Minus'].iloc[i]) / divisor
            else:
                dmi = 0
            
            cumm_dmi += dmi
            
            if i > 0:
                if i <= self.params['DMI_N']:
                    df.loc[df.index[i], 'ADX'] = cumm_dmi / i if i > 0 else dmi
                    df.loc[df.index[i], 'ADXR'] = (df['ADX'].iloc[i] + df['ADX'].iloc[i-1]) * 0.5
                else:
                    df.loc[df.index[i], 'ADX'] = df['ADX'].iloc[i-1] + sf * (dmi - df['ADX'].iloc[i-1])
                    df.loc[df.index[i], 'ADXR'] = (df['ADX'].iloc[i] + df['ADX'].iloc[i-self.params['DMI_M']]) * 0.5
        
        # Calculate ATR
        df['ATR'] = AvgTrueRange(self.params['ATRLength'], df['high'], df['low'], df['close'])
        
        # Calculate consecutive down bars exactly as TradeBlazer's CountIf
        df['ConsecBarsCount'] = 0
        for i in range(len(df)):
            if i < self.params['ConsecBars']:
                continue
            count = 0
            for j in range(self.params['ConsecBars']):
                if df['close'].iloc[i-j] < df['close'].iloc[i-j-1]:
                    count += 1
            df.loc[df.index[i], 'ConsecBarsCount'] = count
        
        # Calculate protection stop level
        df['ProtectStopL'] = df['low'] - self.params['ProtectStopATRMulti'] * df['ATR']
        
        return df

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """生成交易信号矩阵"""
        min_length = max(self.params['DMI_N'], self.params['ATRLength'])
        if len(data) < min_length:
            return pd.DataFrame()
        
        signals = pd.DataFrame(index=data.index)
        signals['call'] = np.nan
        
        current_position = 0
        position_bars = 0
        
        for i in range(min_length, len(data)):
            # Entry conditions - exactly matching TradeBlazer logic
            if current_position == 0:
                adx_condition = (data['ADX'].iloc[i-1] < self.params['ADXLevel'] and 
                               data['ADX'].iloc[i-1] < data['ADX'].iloc[i-1-self.params['ADXLowThanBefore']])
                consec_bars_condition = data['ConsecBarsCount'].iloc[i-1] == self.params['ConsecBars']
                volume_condition = data['vol'].iloc[i] > 0
                
                if adx_condition and consec_bars_condition and volume_condition:
                    signals.loc[data.index[i], 'call'] = 1
                    current_position = 1
                    position_bars = 0
                    
            # Exit conditions - exactly matching TradeBlazer logic
            elif current_position == 1:
                position_bars += 1
                volume_condition = data['vol'].iloc[i] > 0
                
                if volume_condition:
                    # ProactiveStopBars exit
                    if position_bars >= self.params['ProactiveStopBars']:
                        signals.loc[data.index[i], 'call'] = 0
                        current_position = 0
                        position_bars = 0
                    
                    # Protection stop loss
                    elif data['low'].iloc[i] <= data['ProtectStopL'].iloc[i-1]:
                        signals.loc[data.index[i], 'call'] = 0
                        current_position = 0
                        position_bars = 0
        
        # Force close position at the end
        if current_position == 1:
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
        strategy = Traffic_Jam_L()
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
