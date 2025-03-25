'''
可开关多进程版本
'''

from typing import Dict, Any
import pandas as pd
import numpy as np
import logging
from strategy.base import StrategyBase
from .config import FUTURES_PARAMS
from multiprocessing import Pool


class CurrentPosition:
    """当前持仓状态"""
    def __init__(self):

        self.direction = 0  # 持仓方向：0空仓，1多头，-1空头
        self.quantity = 0   # 持仓手数
        self.entry_price = 0  # 开仓价格

class Backtester:
    """回测器类，支持多品种并行回测"""
    def __init__(self, signals_dict: Dict[str, pd.DataFrame], 
                 data_dict: Dict[str, pd.DataFrame], 
                 config: Dict[str, Any], 
                 logger: logging.Logger,
                 use_multiprocessing: bool = True):
        self.signals_dict = signals_dict
        self.data_dict = data_dict
        self.config = config
        self.logger = logger
        self.use_multiprocessing = use_multiprocessing

    @staticmethod
    def _process_single_futures(args) -> pd.Series:
        """单品种回测处理函数"""
        code, data, signals = args
        try:
            if isinstance(signals, pd.DataFrame) and len(signals) > 0:
                contract_multiplier = FUTURES_PARAMS[code]['contract_multiplier']
                
                position = signals['call'].ffill().fillna(0)
                position_change = position.diff()
                
                close_arr = data['close'].values
                open_arr = data['open'].values
                pos_arr = position.values
                
                holding_pnl = pos_arr[:-1] * np.diff(close_arr) * contract_multiplier
                exit_mask = position_change.shift() != 0
                exit_pnl = pos_arr[:-1] * (open_arr[1:] - close_arr[:-1]) * contract_multiplier
                
                pnl = pd.Series(0.0, index=data.index)
                mask_1 = ~exit_mask.iloc[1:]
                mask_2 = exit_mask.iloc[1:]
                
                pnl.loc[pnl.index[1:][mask_1]] = holding_pnl[mask_1]
                pnl.loc[pnl.index[1:][mask_2]] = exit_pnl[mask_2]
                
                return pnl
            
            return pd.Series(0.0, index=data.index)
            
        except Exception as e:
            return pd.Series(0.0, index=data.index)

    def run_backtest(self) -> pd.DataFrame:
        """执行回测"""
        try:
            process_args = [(code, self.data_dict[code], self.signals_dict[code]) 
                          for code in self.signals_dict.keys()]
            
            if self.use_multiprocessing:
                self.logger.info("开始并行多品种回测")
                with Pool() as pool:
                    all_pnl = pool.map(self._process_single_futures, process_args)
            else:
                self.logger.info("开始单进程多品种回测")
                all_pnl = [self._process_single_futures(args) for args in process_args]
            
            combined_pnl = pd.concat([pnl for pnl in all_pnl if not pnl.empty], axis=1).sum(axis=1)
            t_pnl_df = pd.DataFrame(combined_pnl, columns=['pnl'])
            
            self.logger.info("回测完成")
            return t_pnl_df
            
        except Exception as e:
            self.logger.error(f"回测执行错误: {e}")
            raise
