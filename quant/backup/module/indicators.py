import pandas as pd
import numpy as np

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
