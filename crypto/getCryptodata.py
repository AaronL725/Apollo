import os
import datetime
import pytz
import pandas as pd
import ccxt  # 加密货币交易所统一API库
import click  # 命令行工具库

from dateutil.relativedelta import relativedelta
from pathlib import Path

# 初始化Binance交易所连接配置
exchange = ccxt.binance(
    {
        "enableRateLimit": True,  # 必须开启！防止被交易所封IP
        "timeout": 15000,         # 超时设为15秒
        "proxies": {
            "http": "http://127.0.0.1:10809",
            "https": "http://127.0.0.1:10809"
        }
        # 其他常用代理端口:
        # Clash: 7890
        # V2rayN: 10809
        # Shadowsocks: 1080
    }
)

def download(symbol: str, start=None, end=None, timeframe="1d", save_dir="."):
    """下载指定交易对的历史K线数据
    
    Args:
        symbol: 交易对符号（例：BTC/USDT）
        start: 开始时间（默认5年前）
        end: 结束时间（默认当前时间）
        timeframe: K线周期（默认1天）
        save_dir: 数据保存目录
    """
    # 处理时间范围（带时区）
    if end is None:
        end = datetime.datetime.now(pytz.UTC)  # 默认结束时间为当前UTC时间
    else:
        # 如果输入的是字符串，先转换为datetime
        if isinstance(end, str):
            end = datetime.datetime.strptime(end, '%Y-%m-%d %H:%M')
        end = end.replace(tzinfo=pytz.UTC)

    if start is None:
        start = end - relativedelta(years=5)
    else:
        if isinstance(start, str):
            start = datetime.datetime.strptime(start, '%Y-%m-%d %H:%M')
        start = start.replace(tzinfo=pytz.UTC)

    max_limit = 1000  # 单次请求最大数据条数（交易所限制）
    since = start.timestamp()  # 转换为时间戳
    end_time = int(end.timestamp() * 1e3)  # 转换为毫秒时间戳

    # 创建保存目录（如果不存在）
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # 生成绝对路径文件名（替换交易对中的斜杠）
    absolute_path = os.path.join(
        save_dir, f"{symbol.replace('/', '-')}_{timeframe}.csv"
    )

    # 分页获取历史数据
    ohlcvs = []
    while True:
        # 带重试机制的API请求
        new_ohlcvs = exchange.fetch_ohlcv(
            symbol, 
            since=int(since * 1e3),  # 转换为毫秒
            timeframe=timeframe, 
            limit=max_limit, 
            params={"endTime": end_time}  # 添加结束时间参数
        )
        
        # 无新数据时停止循环
        if len(new_ohlcvs) == 0:
            break
            
        ohlcvs += new_ohlcvs  # 合并数据
        
        # 更新下次请求的起始时间（最后一条数据的时间戳+1秒）
        since = ohlcvs[-1][0]/1e3 + 1
        # 打印下载进度（覆盖式输出）
        print(f"下载进度 [{symbol}]: {datetime.datetime.fromtimestamp(ohlcvs[-1][0]/1e3, tz=pytz.UTC)}\r", end="")
    print()  # 换行

    # 数据清洗与保存
    data = pd.DataFrame(
        ohlcvs, columns=["timestamp_ms", "open", "high", "low", "close", "volume"]
    )
    data.drop_duplicates(inplace=True)  # 去重
    # 转换时间戳为UTC时区的datetime索引
    data.set_index(
        pd.DatetimeIndex(pd.to_datetime(data["timestamp_ms"], unit="ms", utc=True)),
        inplace=True,
    )
    data.index.name = "datetime"
    del data["timestamp_ms"]  # 删除原始时间戳列
    data.to_csv(absolute_path)  # 保存CSV文件
    print(f"数据已保存至：{absolute_path}")

@click.command()
@click.option("--symbol", required=True, help="交易对符号（例：BTC/USDT）")
@click.option("--start", type=click.DateTime(), help="开始时间（格式：YYYY-MM-DD）")
@click.option("--end", type=click.DateTime(), help="结束时间（格式：YYYY-MM-DD）")
@click.option("--timeframe", default="1d", help="K线周期（1m,5m,15m,1h,1d等）")
@click.option("--save-dir", default=".", help="数据保存目录")
def main(symbol, start, end, timeframe, save_dir):
    """
    从Binance下载OHLCV数据的命令行工具
    
    使用示例：
        python getCryptodata.py --symbol=SOL/USDT --start=2023-01-01 --end=2025-01-01 --timeframe=5m --save-dir=./Data
        
    可用的timeframe选项:
        1m, 3m, 5m, 15m, 30m      分钟
        1h, 2h, 4h, 6h, 8h, 12h   小时
        1d, 3d                     天
        1w                         周
        1M                         月
    """
    download(
        symbol=symbol, 
        start=start,
        end=end,
        timeframe=timeframe,
        save_dir=save_dir
    )

if __name__ == "__main__":
    utc = pytz.UTC
    download(
        symbol="ETH/USDT",
        start='2024-08-01 00:00',  
        end='2025-02-21 03:00',   
        timeframe="5m",
        save_dir="./Data"
    )