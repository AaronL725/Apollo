'''
// 简称: DynamicBreakOutII_L
// 名称: 动态突破_多
// 类别: 策略应用
// 类型: 内建应用
// 输出:
//------------------------------------------------------------------------
/* 
策略说明:
        基于自适应的布林通道与自适应的唐奇安通道的突破系统
系统要素:
        1、自适应布林通道
        2、自适应唐奇安通道
        3、自适应出场均线
入场条件:
        1、昨日价格大于布林通道上轨，并且当日周期价格大于唐奇安通道上轨，开多单
        2、昨日价格小于布林通道下轨，并且当日周期价格小于唐奇安通道下轨，开空单
出场条件:
        1、持有多单时，价格小于自适应出场均线，平多单
        2、持有空单时，价格大于自适应出场均线，平空单
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


class DynamicBreakOutII_L(StrategyBase):
    """动态突破做多策略"""
    def __init__(self, params: Dict = None):
        # 默认参数设置
        default_params = {
            'ceilingAmt': 60,    # 自适应参数的上限
            'floorAmt': 20,      # 自适应参数的下限
            'bolBandTrig': 2,    # 布林通道参数
            'Lots': 1            # 交易手数
        }
        # 更新自定义参数
        if params:
            default_params.update(params)
        super().__init__(default_params)

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算技术指标"""
        df = data.copy()
        
        # 计算当日市场波动
        df['todayVolatility'] = StandardDev(df['close'], 30, 1)
        
        # 计算昨日市场波动
        df['yesterDayVolatility'] = df['todayVolatility'].shift(1)
        
        # 计算市场波动的变动率 - 处理除以0和无穷大的情况
        df['deltaVolatility'] = np.where(
            df['todayVolatility'] > 0,
            (df['todayVolatility'] - df['yesterDayVolatility']) / df['todayVolatility'],
            0
        )
        df['deltaVolatility'] = df['deltaVolatility'].fillna(0)  # 填充NaN值
        df['deltaVolatility'] = df['deltaVolatility'].clip(-1, 1)  # 限制变动率在 -100% 到 100% 之间
        
        # 初始化自适应参数
        lookback_days = pd.Series(20, index=df.index, dtype=float)  # 使用float类型
        
        # 计算自适应参数
        for i in range(1, len(df)):
            lookback_days.iloc[i] = lookback_days.iloc[i-1] * (1 + df['deltaVolatility'].iloc[i])
            lookback_days.iloc[i] = int(round(lookback_days.iloc[i]))  # 确保转为整数
            lookback_days.iloc[i] = min(lookback_days.iloc[i], self.params['ceilingAmt'])
            lookback_days.iloc[i] = max(lookback_days.iloc[i], self.params['floorAmt'])
        
        df['lookBackDays'] = lookback_days
        
        # 初始化列以避免NaN问题
        df['MidLine'] = np.nan
        df['Band'] = np.nan
        df['buyPoint'] = np.nan
        df['sellPoint'] = np.nan
        
        # 计算动态指标
        for i in range(len(df)):
            period = int(df['lookBackDays'].iloc[i])
            if i >= period - 1:
                df.loc[df.index[i], 'MidLine'] = Average(df['close'].iloc[max(0, i-period+1):i+1], period).iloc[-1]
                df.loc[df.index[i], 'Band'] = StandardDev(df['close'].iloc[max(0, i-period+1):i+1], period, 2).iloc[-1]
                
                # 计算自适应唐奇安通道
                df.loc[df.index[i], 'buyPoint'] = HighestFC(df['high'].iloc[max(0, i-period+1):i+1], period).iloc[-1]
                df.loc[df.index[i], 'sellPoint'] = LowestFC(df['low'].iloc[max(0, i-period+1):i+1], period).iloc[-1]
        
        # 计算布林带
        df['upBand'] = df['MidLine'] + self.params['bolBandTrig'] * df['Band']
        df['dnBand'] = df['MidLine'] - self.params['bolBandTrig'] * df['Band']
        
        # 自适应出场均线
        df['LiqPoint'] = df['MidLine']
        
        return df

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """生成交易信号矩阵"""
        # 数据验证
        if len(data) < self.params['floorAmt']:
            raise ValueError('数据长度不足')
        
        signals = pd.DataFrame(index=data.index)
        signals['call'] = np.nan
        
        current_position = 0
        
        for i in range(1, len(data)):
            close = data['close'].iloc[i-1]  # 昨日收盘价
            up_band = data['upBand'].iloc[i-1]  # 昨日上轨
            dn_band = data['dnBand'].iloc[i-1]  # 昨日下轨
            buy_point = data['buyPoint'].iloc[i-1]  # 昨日买入点
            sell_point = data['sellPoint'].iloc[i-1]  # 昨日卖出点
            liq_point = data['LiqPoint'].iloc[i-1]  # 昨日清算点
            
            high = data['high'].iloc[i]  # 当日最高价
            low = data['low'].iloc[i]  # 当日最低价
            open_ = data['open'].iloc[i]  # 当日开盘价
            
            # 开多仓条件
            if (current_position == 0 and 
                close > up_band and 
                high >= buy_point):
                
                entry_price = max(open_, buy_point)
                signals.iloc[i] = [1]
                current_position = 1
            
            # 平多仓条件1：价格跌破自适应出场均线
            elif (current_position == 1 and 
                  low <= liq_point):
                
                exit_price = min(open_, liq_point)
                signals.iloc[i] = [0]
                current_position = 0
            
            # 平多仓条件2：价格跌破布林带下轨和唐奇安通道下轨
            elif (current_position == 1 and 
                  close < dn_band and 
                  low <= sell_point):
                
                exit_price = min(open_, sell_point)
                signals.iloc[i] = [0]
                current_position = 0
        
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
        strategy = DynamicBreakOutII_L()
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
