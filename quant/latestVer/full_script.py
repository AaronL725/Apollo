import os
import sys
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any
from dataclasses import dataclass
from datetime import datetime
import matplotlib.pyplot as plt
import pickle
from multiprocessing import Pool
# import akshare as ak


##############数据加载####################
def load_single_file(args):
    """单个文件加载函数"""
    data_type, path, dtype = args
    # 使用内存映射读取大文件
    df = pd.read_csv(
        path,
        dtype=dtype,
        index_col=0,
        parse_dates=True,
        memory_map=True,  # 使用内存映射
        cache_dates=True
    )
    return data_type, df

def load_all_data(data_paths: Dict[str, str], logger: logging.Logger, level: str) -> Dict[str, pd.DataFrame]:
    """一次性加载所有数据文件，支持缓存和内存映射"""
    try:
        logger.info("开始加载所有数据文件")
        cache_dir = "data_cache"
        os.makedirs(cache_dir, exist_ok=True)
        data_cache = {}
        
        # 检查缓存 - 修改缓存文件名以包含级别
        cache_valid = True
        for data_type, path in data_paths.items():
            cache_path = os.path.join(cache_dir, f"{data_type}_{level}_cache.pkl")
            if not os.path.exists(cache_path) or os.path.getmtime(path) > os.path.getmtime(cache_path):
                cache_valid = False
                break
        
        # 如果缓存有效，直接从缓存加载
        if cache_valid:
            logger.info("从缓存加载数据")
            for data_type in data_paths.keys():
                cache_path = os.path.join(cache_dir, f"{data_type}_{level}_cache.pkl")
                with open(cache_path, 'rb') as f:
                    data_cache[data_type] = pickle.load(f)
            logger.info("缓存数据加载完成")
            return data_cache
        
        # 定义数据类型字典
        dtypes = {
            'open': np.float32,
            'close': np.float32,
            'high': np.float32,
            'low': np.float32,
            'vol': np.float32
        }
        
        # 准备并行加载参数
        load_args = []
        for data_type, path in data_paths.items():
            # 首先读取少量数据来获取列名
            sample_df = pd.read_csv(path, nrows=5)
            columns = sample_df.columns
            dtype_dict = {col: dtypes[data_type] for col in columns if col != sample_df.index.name}
            load_args.append((data_type, path, dtype_dict))
        
        # 使用进程池并行加载
        with Pool() as pool:
            results = pool.map(load_single_file, load_args)
        
        # 整理结果
        for data_type, df in results:
            data_cache[data_type] = df
            # 保存到缓存 - 修改缓存文件名以包含级别
            cache_path = os.path.join(cache_dir, f"{data_type}_{level}_cache.pkl")
            with open(cache_path, 'wb') as f:
                pickle.dump(df, f)
            logger.info(f"已加载并缓存 {data_type} 数据")
        
        return data_cache
        
    except Exception as e:
        logger.error(f"加载数据文件时发生错误: {e}")
        raise

def load_data(data_cache: Dict[str, pd.DataFrame], futures_code: str, logger: logging.Logger) -> pd.DataFrame:
    """从缓存中提取特定期货品种的数据"""
    try:
        # 使用更高效的方式检查列是否存在
        if futures_code not in set(data_cache['open'].columns):
            logger.warning(f"期货品种 {futures_code} 在数据中不存在，跳过加载")
            return pd.DataFrame()
        
        # 使用向量化操作检查是否全为空值
        open_data = data_cache['open'][futures_code]
        if open_data.isna().all():
            logger.warning(f"期货品种 {futures_code} 的开盘价数据全为空值，跳过加载")
            return pd.DataFrame()
        
        # 使用布尔索引代替loc，提高性能
        valid_mask = ~open_data.isna()
        
        # 一次性创建数据框
        data = pd.DataFrame({
            'open': data_cache['open'].loc[valid_mask, futures_code],
            'close': data_cache['close'].loc[valid_mask, futures_code],
            'high': data_cache['high'].loc[valid_mask, futures_code],
            'low': data_cache['low'].loc[valid_mask, futures_code],
            'vol': data_cache['vol'].loc[valid_mask, futures_code],
        })
        
        logger.info(f"{futures_code} 数据加载完成，共有 {len(data)} 条记录")
        return data
        
    except Exception as e:
        logger.error(f"处理 {futures_code} 数据时发生错误: {e}")
        raise

def load_data_vectorized(data_cache: Dict[str, pd.DataFrame], futures_codes: List[str], 
                        start_date: str, end_date: str, logger: logging.Logger) -> Dict[str, pd.DataFrame]:
    """
    向量化加载多个品种的数据
    """
    try:
        # 预先创建日期掩码以避免重复计算
        date_mask = (data_cache['open'].index >= start_date) & (data_cache['open'].index <= end_date)
        
        # 一次性获取所有需要的数据，使用numpy操作代替pandas
        open_data = data_cache['open'].loc[date_mask, futures_codes].values
        close_data = data_cache['close'].loc[date_mask, futures_codes].values
        high_data = data_cache['high'].loc[date_mask, futures_codes].values
        low_data = data_cache['low'].loc[date_mask, futures_codes].values
        vol_data = data_cache['vol'].loc[date_mask, futures_codes].values
        
        # 获取日期索引
        dates = data_cache['open'].loc[date_mask].index
        
        # 使用numpy的nansum检查非空数据
        valid_codes_mask = ~np.isnan(open_data).all(axis=0)
        valid_futures = np.array(futures_codes)[valid_codes_mask]
        
        # 预分配字典空间
        data_dict = {}
        
        # 使用numpy的列视图避免复制
        valid_data = open_data[:, valid_codes_mask]
        for i, code in enumerate(valid_futures):
            # 创建有效数据掩码
            valid_mask = ~np.isnan(valid_data[:, i])
            
            if np.any(valid_mask):
                # 使用布尔索引一次性创建数据框
                df_data = {
                    'open': open_data[valid_mask, i],
                    'close': close_data[valid_mask, i],
                    'high': high_data[valid_mask, i],
                    'low': low_data[valid_mask, i],
                    'vol': vol_data[valid_mask, i]
                }
                
                data_dict[code] = pd.DataFrame(
                    df_data,
                    index=dates[valid_mask]
                )
                
                logger.info(f"{code} 数据加载完成，共有 {len(data_dict[code])} 条记录")
            
        return data_dict
        
    except Exception as e:
        logger.error(f"向量化加载数据时发生错误: {e}")
        raise



####################配置####################
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
        


####################技术指标####################
def Momentum(series: pd.Series, length: int) -> pd.Series:
    """
    计算价格动量。
    
    参数:
        series (pd.Series): 价格序列
        length (int): 计算周期
        
    返回:
        pd.Series: 动量值序列
    """
    return series - series.shift(length)


def XAverage(series: pd.Series, length: int) -> pd.Series:
    """
    计算指数加权移动平均。
    
    参数:
        series (pd.Series): 输入序列
        length (int): 计算周期
        
    返回:
        pd.Series: 指数移动平均序列
    """
    return series.ewm(span=length, adjust=False).mean()


def AvgTrueRange(length: int, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """
    计算平均真实波幅(ATR)
    
    参数:
        length (int): ATR计算周期
        high (pd.Series): 最高价序列
        low (pd.Series): 最低价序列
        close (pd.Series): 收盘价序列
        
    返回:
        pd.Series: ATR值序列
    """
    tr = TrueRange(high, low, close)
    return Average(tr, length)


def CrossOver(series1: pd.Series, series2: pd.Series) -> pd.Series:
    return (series1.shift(1) <= series2.shift(1)) & (series1 > series2)


def CrossUnder(series1: pd.Series, series2: pd.Series) -> pd.Series:
    return (series1.shift(1) >= series2.shift(1)) & (series1 < series2)


def TrueRange(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """
    计算真实波幅(True Range)
    
    参数:
        high (pd.Series): 最高价序列
        low (pd.Series): 最低价序列
        close (pd.Series): 收盘价序列
    """
    # 第一根K线直接用high-low
    tr = high - low
    
    # 其他K线用TrueHigh - TrueLow
    true_high = TrueHigh(high, close)
    true_low = TrueLow(low, close)
    tr[1:] = true_high[1:] - true_low[1:]
    
    return tr


def TrueHigh(high: pd.Series, close: pd.Series) -> pd.Series:
    """
    计算真实高点
    
    参数:
        high (pd.Series): 最高价序列
        close (pd.Series): 收盘价序列
        
    返回:
        pd.Series: 真实高点序列
    """
    prev_close = close.shift(1)
    return pd.Series(np.where(high >= prev_close, high, prev_close), index=high.index)


def TrueLow(low: pd.Series, close: pd.Series) -> pd.Series:
    """
    计算真实低点
    
    参数:
        low (pd.Series): 最低价序列
        close (pd.Series): 收盘价序列
        
    返回:
        pd.Series: 真实低点序列
    """
    prev_close = close.shift(1)
    return pd.Series(np.where(low <= prev_close, low, prev_close), index=low.index)


def Highest(series: pd.Series, length: int) -> pd.Series:
    """
    因为懒得再到每个策略里改了，所以Highest和HighestFC都通过向量化计算

    计算指定周期内的最高值
    
    参数:
        price (pd.Series): 价格序列
        length (int): 计算周期
        
    返回:
        pd.Series: 最高值序列
    """
    return series.rolling(window=length, min_periods=1).max()


def Lowest(series: pd.Series, length: int) -> pd.Series:
    """
    因为懒得再到每个策略里改了，所以Lowest和LowestFC都通过向量化计算

    计算指定周期内的最低值
    
    参数:
        price (pd.Series): 价格序列
        length (int): 计算周期
        
    返回:
        pd.Series: 最低值序列
    """
    return series.rolling(window=length, min_periods=1).min()


def HighestFC(series: pd.Series, length: int) -> pd.Series:
    """
    计算过去N个周期的最高值(Fast Calculation版本)
    
    参数:
        series (pd.Series): 输入数据
        length (int): 计算周期
    """
    return series.rolling(window=length, min_periods=1).max()


def LowestFC(series: pd.Series, length: int) -> pd.Series:
    """
    计算过去N个周期的最低值(Fast Calculation版本)
    
    参数:
        series (pd.Series): 输入数据
        length (int): 计算周期
    """
    return series.rolling(window=length, min_periods=1).min()


def AverageFC(series: pd.Series, length: int) -> pd.Series:
    """
    计算快速算术平均值(Fast Calculation版本)
    
    参数:
        series (pd.Series): 输入序列
        length (int): 计算周期
        
    返回:
        pd.Series: 移动平均序列
    """
    return series.rolling(window=length, min_periods=1).mean()


def Average(price: pd.Series, length: int) -> pd.Series:
    """
    计算简单移动平均
    
    参数:
        price (pd.Series): 输入序列
        length (int): 计算周期
        
    返回:
        pd.Series: 移动平均序列
    """
    return Summation(price, length) / length


def PriceOscillator(price: pd.Series, FastLength: int, SlowLength: int) -> pd.Series:
    """
    计算价格震荡指标
    
    参数:
        price (pd.Series): 价格序列
        FastLength (int): 快速移动平均的计算周期
        SlowLength (int): 慢速移动平均的计算周期
        
    返回:
        pd.Series: 快速移动平均与慢速移动平均的差值
    """
    # 计算快速移动平均
    fast_ma = price.rolling(window=FastLength, min_periods=1).mean()
    # 计算慢速移动平均
    slow_ma = price.rolling(window=SlowLength, min_periods=1).mean()
    # 返回差值
    return fast_ma - slow_ma


def Summation(price: pd.Series, length: int) -> pd.Series:
    """
    计算指定周期内数值的总和
    
    参数:
        price (pd.Series): 输入序列(价格、函数或公式)
        length (int): 计算周期
        
    返回:
        pd.Series: 最近length个周期的总和
    """
    return price.rolling(window=length, min_periods=1).sum()


def VariancePS(price: pd.Series, length: int, data_type: int = 1) -> pd.Series:
    """
    计算估计方差（向量化版本）
    
    参数:
        price (pd.Series): 价格序列
        length (int): 计算周期
        data_type (int): 1-总体方差, 2-样本方差
        
    返回:
        pd.Series: 方差序列
    """
    # 初始化结果序列
    result = pd.Series(0.0, index=price.index)
    
    # 计算移动平均
    mean = Average(price, length)
    
    # 创建移位矩阵，每行包含length个历史值
    shifted_matrix = pd.concat([price.shift(i) for i in range(length)], axis=1)
    
    # 计算每个时间点的均值矩阵（广播mean到每一列）
    mean_matrix = mean.values.reshape(-1, 1).repeat(length, axis=1)
    
    # 计算差值的平方和
    squared_diff = (shifted_matrix.sub(mean_matrix)) ** 2
    sum_squared_diff = squared_diff.sum(axis=1)
    
    # 根据data_type选择除数
    divisor = length if data_type == 1 else length - 1
    
    # 只在有足够数据的位置计算方差
    valid_mask = price.index >= price.index[length - 1]
    result[valid_mask] = sum_squared_diff[valid_mask] / divisor
    
    return result


def StandardDev(price: pd.Series, length: int, data_type: int = 1) -> pd.Series:
    """
    计算标准差
    
    参数:
        price (pd.Series): 价格序列
        length (int): 计算周期
        data_type (int): 1-总体标准差, 2-样本标准差
        
    返回:
        pd.Series: 标准差序列
    """
    var_ps = VariancePS(price, length, data_type)
    return pd.Series(np.where(var_ps > 0, np.sqrt(var_ps), 0), index=price.index)


def Cum(price: pd.Series) -> pd.Series:
    """
    计算累计值
    
    参数:
        price (pd.Series): 输入序列
        
    返回:
        pd.Series: 累计值序列
    """
    return price.cumsum()


def NthCon(condition: pd.Series, n: int) -> pd.Series:
    """
    计算第N个满足条件的Bar距当前的Bar数目
    
    参数:
        condition (pd.Series): 条件序列(布尔值)
        n (int): 向前查找第n个满足条件的bar
        
    返回:
        pd.Series: 距离序列
    """
    bar_nums = pd.Series(0, index=condition.index)
    pre_con_index = pd.Series(0, index=condition.index)
    
    # 计算基础距离
    for i in range(len(condition)):
        if condition.iloc[i]:
            bar_nums.iloc[i] = 0
            pre_con_index.iloc[i] = bar_nums.iloc[i-1] + 1 if i > 0 else 1
        else:
            bar_nums.iloc[i] = bar_nums.iloc[i-1] + 1 if i > 0 else 1
            pre_con_index.iloc[i] = bar_nums.iloc[i]
    
    # 计算第N个满足条件的距离
    result = bar_nums.copy()
    for i in range(len(condition)):
        re_bars = result.iloc[i]
        for j in range(2, n + 1):
            if i - int(re_bars) >= 0:  # 确保索引有效
                re_bars += pre_con_index.iloc[i - int(re_bars)]
        result.iloc[i] = re_bars
        
    return result


def CountIf(condition: pd.Series, length: int) -> pd.Series:
    """
    计算最近N个周期内条件满足的次数(Fast Calculation版本)
    
    参数:
        condition (pd.Series): 条件序列(布尔值)
        length (int): 计算周期
    """
    return condition.rolling(window=length, min_periods=1).sum()



####################策略基类####################
class StrategyBase:
    """策略基类"""
    def __init__(self, params: Dict = None):
        self.params = params or {}

        # 添加默认交易数量
        self.default_quantity = 1
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError("子类必须实现calculate_indicators方法")
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError("子类必须实现generate_signals方法")



####################策略####################
class Swinger_L(StrategyBase):
    """趋势震荡做多策略"""
    def __init__(self, params: Dict = None):
        # 默认参数设置
        default_params = {
            'FastMALength': 5,    # 动能计算中的快均线值
            'SlowMALength': 20,   # 动能计算中的慢均线值
            'TrendMALength': 50,  # 显示趋势的均线值
            'ExitStopN': 3        # 求高低点的bar数值
        }
        # 更新自定义参数
        if params:
            default_params.update(params)
        super().__init__(default_params)
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算策略所需的技术指标"""
        df = data.copy()
        
        # 计算趋势线和均线动能
        df['TrendMA'] = AverageFC(df['close'], self.params['TrendMALength'])
        df['PriceOsci'] = PriceOscillator(
            df['close'], 
            self.params['FastMALength'],
            self.params['SlowMALength']
        )
        
        # 计算出场价格（前N根K线的最低点）
        df['ExitL'] = LowestFC(df['low'], self.params['ExitStopN'])
        
        return df
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """生成交易信号矩阵"""
        # 数据验证
        min_length = max(self.params['TrendMALength'], 
                        self.params['SlowMALength'], 
                        self.params['ExitStopN'])
        if len(data) < min_length:
            raise ValueError('数据长度不足')
        
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0
        signals['quantity'] = 1
        signals['entry_price'] = np.nan
        signals['exit_price'] = np.nan
        signals['position'] = 0
        signals['call'] = np.nan
        
        # 使用.values提高性能
        close = data['close'].values
        open_ = data['open'].values
        low = data['low'].values
        trend_ma = data['TrendMA'].values

        price_osci = data['PriceOsci'].values
        exit_l = data['ExitL'].values
        vol = data['vol'].values
        
        current_position = 0
        
        for i in range(min_length, len(data)):
            # 开多仓条件：
            # 1. 当前无仓位
            # 2. 上根K线收盘价高于趋势线
            # 3. 上根K线动能为负且大于上上根动能
            # 4. 成交量大于0
            if (current_position == 0 and
                close[i-1] > trend_ma[i-1] and
                price_osci[i-1] <= 0 and
                price_osci[i-1] > price_osci[i-2] and
                vol[i] > 0):
                
                signals.iloc[i] = [1, 1, open_[i], np.nan, 1, 1]
                current_position = 1
            
            # 平多仓条件：
            # 1. 当前持有多仓
            # 2. 动能减弱（上根动能小于上上根）
            # 3. 最低价跌破前N根K线低点
            # 4. 成交量大于0
            elif (current_position == 1 and
                  price_osci[i-1] < price_osci[i-2] and
                  low[i] <= exit_l[i-1] and
                  vol[i] > 0):
                
                signals.iloc[i] = [-1, 1, np.nan, 
                                 min(open_[i], exit_l[i-1]), 0, 0]
                current_position = 0
        
        return signals


####################回测器####################
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



####################可视化####################
def calculate_max_drawdown(cumulative_pnl: np.ndarray) -> float:
    """
    计算最大回撤
    
    参数:
        cumulative_pnl: 已经计算好的累计收益序列
    """
    if len(cumulative_pnl) == 0:
        return 0.0
    
    running_max = np.maximum.accumulate(cumulative_pnl)
    drawdowns = running_max - cumulative_pnl
    return float(np.max(drawdowns))

def calculate_sharpe_ratio(pnl: np.ndarray, risk_free_rate: float = 0.02, 
                          trading_days: int = 252) -> float:
    """
    计算夏普比率
    
    参数:
        pnl (np.ndarray): 每根k线的pnl序列
        risk_free_rate (float): 无风险利率
        trading_days (int): 年交易日数
    """
    if len(pnl) < 2:
        return 0.0
    
    excess_returns = pnl - (risk_free_rate / trading_days)
    mean_excess_returns = np.mean(excess_returns)
    std_excess_returns = np.std(excess_returns, ddof=1)
    
    if std_excess_returns == 0:
        return 0.0
        
    return float(np.sqrt(trading_days) * mean_excess_returns / std_excess_returns)



def plot_combined_pnl(t_pnl_df: pd.DataFrame, logger: logging.Logger):
    """绘制合并盈亏曲线"""
    try:
        if len(t_pnl_df) == 0:
            logger.info("没有交易盈亏数据可绘制")
            return
            
        # 计算一次累计收益并储存
        cumulative_pnl = t_pnl_df['pnl'].cumsum()
        
        # 绘图
        plt.figure(figsize=(12, 6))
        plt.plot(cumulative_pnl.index, cumulative_pnl.values, label="Combined PnL", color='blue')
        plt.xlabel("Date")
        plt.ylabel("Profit and Loss")
        plt.title("Combined Trading PnL Curve")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()
        
        # 直接使用累计收益序列计算统计指标
        total_pnl = float(cumulative_pnl.iloc[-1])
        max_drawdown = calculate_max_drawdown(cumulative_pnl.values)  # 直接传入累计收益
        sharpe_ratio = calculate_sharpe_ratio(t_pnl_df['pnl'].values)
        
        print("\n=== 合并回测统计 ===")
        print(f"总盈亏: {total_pnl:,.2f}")
        print(f"最大回撤: {max_drawdown:,.2f}")
        print(f"夏普比率: {sharpe_ratio:.2f}")
        
        logger.info("合并盈亏曲线绘制完成")
        
    except Exception as e:
        logger.error(f"绘制合并盈亏曲线时出错: {e}")
        raise

####################主函数####################
def main():
    """主函数，执行多品种回测"""
    logger = setup_logging()
    
    level = 'min5'  # 默认值
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
        strategy = Swinger_L()
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
