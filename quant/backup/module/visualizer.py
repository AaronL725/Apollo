import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging

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
