import requests
import pandas as pd
from datetime import datetime, timedelta
from fake_useragent import UserAgent
import time
import random

# 构造日期范围
def generate_dates(start_date, end_date):
    current_date = start_date
    dates = []
    while current_date <= end_date:
        dates.append(current_date.strftime("%Y%m%d"))
        current_date += timedelta(days=1)
    return dates

# 获取每个日期的 .dat 文件
def fetch_data_for_date(date):
    url = f"https://www.shfe.com.cn/data/tradedata/future/dailydata/{date}dailystock.dat"
    
    # 使用 fake_useragent 库生成一个随机 User-Agent
    ua = UserAgent()
    headers = {
        'User-Agent': ua.random
    }
    
    try:
        response = requests.get(url, headers=headers)  # 添加 headers 到请求中
        if response.status_code == 200:
            return response.json(), True
        elif response.status_code == 404:
            return None, False  # 非交易日
        else:
            print(f"未能获取数据，状态码：{response.status_code} 日期：{date}")
            return None, False
    except requests.RequestException as e:
        print(f"请求失败：{e} 日期：{date}")
        return None, False

# 解析数据字典并提取需要的字段
def parse_data(data, date):
    parsed_data = []
    
    # 确保 data 包含 'o_cursor' 字段并且是列表
    if isinstance(data, dict) and 'o_cursor' in data and isinstance(data['o_cursor'], list):
        for entry in data['o_cursor']:
            # 清理 $$ 后的内容
            regname = str(entry.get("REGNAME", "")).split('$$')[0]
            
            # 如果 REGNAME 为空，则跳过此行
            if not regname.strip():
                continue

            varname = str(entry.get("VARNAME", "")).split('$$')[0]
            whabbrname = str(entry.get("WHABBRNAME", "")).split('$$')[0]
            
            # 如果 WHABBRNAME 中包含 "合计"，则跳过此行
            if "合计" in whabbrname:
                continue
            
            parsed_data.append([  
                date,               # 日期
                varname,            # 商品 (VARNAME)
                regname,            # 地区 (REGNAME)
                whabbrname,         # 仓库 (WHABBRNAME)
                entry.get("WRTWGHTS", 0),    # 期货 (WRTWGHTS)
                entry.get("WRTCHANGE", 0)    # 增减 (WRTCHANGE)
            ])
    else:
        print(f"警告：返回的数据不包含 'o_cursor' 或其数据格式不正确：{data}")
    
    return parsed_data

# 合并所有日期的数据到一个DataFrame
def merge_data(dates):
    all_data = []
    for date in dates:
        print(f"正在爬取 {date} 数据...")
        data, is_valid = fetch_data_for_date(date)
        
        if is_valid and data:
            parsed_data = parse_data(data, date)
            all_data.extend(parsed_data)
        else:
            # 如果是非交易日，插入非交易日标记
            all_data.append([date, "非交易日", "", "", "", ""])  # 根据实际列数调整空列数
            
        # 在每次请求后添加随机延迟
        delay = random.uniform(1, 3)  # 随机延迟1到3秒之间
        print(f"等待 {delay:.2f} 秒...")
        time.sleep(delay)
    
    # 将所有数据转换为 DataFrame
    columns = ['日期', '商品', '地区', '仓库', '期货', '增减']
    df = pd.DataFrame(all_data, columns=columns)
    
    return df

# 保存数据到 Excel 文件
def save_to_excel(df, filename="tradedata.xlsx"):
    df.to_excel(filename, index=False)
    print(f"数据已保存到 {filename}")


def main():
    # 设置开始和结束日期
    start_date = datetime(2024, 12, 25)  # 起始日期
    end_date = datetime(2025, 1, 5)   # 结束日期
    dates = generate_dates(start_date, end_date)
    
    # 获取并合并数据
    df = merge_data(dates)
    
    # 保存到 Excel
    save_to_excel(df)

# 执行爬虫
if __name__ == "__main__":
    main()
