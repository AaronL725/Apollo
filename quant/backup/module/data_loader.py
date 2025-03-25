import pandas as pd
import os
import logging
from typing import Dict, List
import numpy as np
import pickle
from multiprocessing import Pool

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
