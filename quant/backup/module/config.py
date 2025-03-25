from dataclasses import dataclass
from typing import Dict, List
from datetime import datetime
import os
import sys
import logging

def setup_logging(log_file: str = os.path.join("logs", "trading_strategy.log")) -> logging.Logger:
    """设置日志记录器"""
    # 确保logs目录存在

    log_dir = os.path.dirname(log_file)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # 配置日志格式
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger()
    logger.info("日志系统初始化完成")
    return logger

'''
#从akshare获取期货合约参数
def get_futures_params():
    """从akshare获取期货合约参数"""
    try:
        futures_fees_info_df = ak.futures_fees_info()
        futures_params = {}
        
        # 按品种分组并只取每组第一条记录
        for symbol, group in futures_fees_info_df.groupby('品种代码'):
            futures_params[symbol] = {
                'contract_multiplier': float(group.iloc[0]['合约乘数'])
            }
            
        return futures_params
    except Exception as e:
        raise RuntimeError(f"获取期货合约参数失败: {e}")

# 替换原有的FUTURES_PARAMS常量
FUTURES_PARAMS = get_futures_params()
'''

FUTURES_PARAMS = {
    'AP': {'contract_multiplier': 10.0},
    'CF': {'contract_multiplier': 5.0},
    'CJ': {'contract_multiplier': 5.0},
    'FG': {'contract_multiplier': 20.0},
    'MA': {'contract_multiplier': 10.0},
    'OI': {'contract_multiplier': 10.0},
    'PF': {'contract_multiplier': 5.0},
    'PK': {'contract_multiplier': 5.0},
    'RM': {'contract_multiplier': 10.0},
    'SA': {'contract_multiplier': 20.0},
    'SF': {'contract_multiplier': 5.0},
    'SM': {'contract_multiplier': 5.0},
    'SR': {'contract_multiplier': 10.0},
    'TA': {'contract_multiplier': 5.0},
    'UR': {'contract_multiplier': 20.0},
    'a': {'contract_multiplier': 10.0},
    'ag': {'contract_multiplier': 15.0},
    'al': {'contract_multiplier': 5.0},
    'au': {'contract_multiplier': 1000.0},
    'b': {'contract_multiplier': 10.0},
    'bu': {'contract_multiplier': 10.0},
    'c': {'contract_multiplier': 10.0},
    'cs': {'contract_multiplier': 10.0},
    'cu': {'contract_multiplier': 5.0},
    'eb': {'contract_multiplier': 5.0},
    'eg': {'contract_multiplier': 10.0},
    'fu': {'contract_multiplier': 10.0},
    'hc': {'contract_multiplier': 10.0},
    'i': {'contract_multiplier': 100.0},
    'j': {'contract_multiplier': 100.0},
    'jd': {'contract_multiplier': 10.0},
    'jm': {'contract_multiplier': 60.0},
    'l': {'contract_multiplier': 5.0},
    'lh': {'contract_multiplier': 16.0},
    'lu': {'contract_multiplier': 10.0},
    'm': {'contract_multiplier': 10.0},
    'ni': {'contract_multiplier': 1.0},
    'nr': {'contract_multiplier': 10.0},
    'p': {'contract_multiplier': 10.0},
    'pb': {'contract_multiplier': 5.0},
    'pg': {'contract_multiplier': 20.0},
    'pp': {'contract_multiplier': 5.0},
    'rb': {'contract_multiplier': 10.0},
    'rr': {'contract_multiplier': 10.0},
    'ru': {'contract_multiplier': 10.0},
    'sc': {'contract_multiplier': 1000.0},
    'sn': {'contract_multiplier': 1.0},
    'sp': {'contract_multiplier': 10.0},
    'ss': {'contract_multiplier': 5.0},
    'v': {'contract_multiplier': 5.0},
    'y': {'contract_multiplier': 10.0},
    'zn': {'contract_multiplier': 5.0}
}

@dataclass
class TradingConfig:
    def __init__(
        self,
        data_paths: Dict[str, str],
        futures_codes: List[str],
        strategy_weights: Dict[str, float],
        start_date: str,
        end_date: str,
        initial_balance: float,
    ):
        self.data_paths = data_paths
        self.futures_codes = futures_codes
        self.strategy_weights = strategy_weights
        self.start_date = start_date
        self.end_date = end_date
        self.initial_balance = initial_balance
        
        # 验证策略权重
        if abs(sum(self.strategy_weights.values()) - 1.0) > 1e-6:
            raise ValueError("策略权重之和必须等于1")
        
        self._validate_parameters()
    
    def _validate_parameters(self):
        try:
            datetime.strptime(self.start_date, '%Y-%m-%d')
            datetime.strptime(self.end_date, '%Y-%m-%d')
        except ValueError:
            raise ValueError("日期格式无效，应为'YYYY-MM-DD'")
        
        if self.initial_balance <= 0:
            raise ValueError("初始资金必须大于0")
