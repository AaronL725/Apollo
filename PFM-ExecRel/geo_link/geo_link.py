import pandas as pd
import plotly.express as px
import numpy as np
import csv
import os

# 读取基金管理人地理位置数据并提取经纬度信息
def extract_fund_locations():
    # 构建CSV文件的绝对路径（位于脚本的上级目录）
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(os.path.dirname(current_dir), 'manager_data.csv')
    
    # 使用CSV模块直接读取文件，避免pandas可能的解析问题
    fund_data = []
    
    with open(csv_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        for row in reader:
            if len(row) >= 18:  # 检查数据行是否包含足够的列数
                # 获取第2列的基金管理人名称
                fund_name = row[1]
                
                # 获取第18列的地理坐标信息
                geo_str = row[17] if len(row) > 17 else ""
                
                if isinstance(geo_str, str) and ',' in geo_str:
                    try:
                        # 解析"纬度,经度"格式的坐标字符串
                        lat_str, lng_str = geo_str.replace('"', '').split(',')
                        lat = float(lat_str)
                        lng = float(lng_str)
                        
                        # 将有效数据添加到结果列表
                        fund_data.append({
                            'name': fund_name,
                            'latitude': lat,
                            'longitude': lng
                        })
                    except:
                        # 忽略无法解析的坐标数据
                        continue
    
    # 将列表转换为DataFrame格式便于后续处理
    return pd.DataFrame(fund_data)

# 生成基金管理人分布的交互式地图可视化
def create_scatter_plot(df):
    # 创建基于Mapbox的散点地图
    fig = px.scatter_mapbox(
        df,
        lon='longitude',    # 指定经度列
        lat='latitude',     # 指定纬度列
        hover_name='name',  # 鼠标悬停时显示的信息
        title='私募基金管理人地理分布',
        zoom=2.7,           # 初始缩放级别
        height=1080,        # 图表高度（像素）
        width=1920          # 图表宽度（像素）
    )
    
    # 配置地图样式和显示范围
    fig.update_layout(
        mapbox_style="carto-positron",  # 选择清晰的地图底图样式
        mapbox=dict(
            center={"lat": 38, "lon": 105},  # 设置地图中心点位置（中国中部）
            # 设置地图边界范围，确保完整显示中国全境
            bounds={"west": 65, "east": 150, "south": 15, "north": 60}
        ),
        margin={"r":0,"t":50,"l":0,"b":0},  # 设置图表边距
        autosize=True  # 允许图表自适应容器大小
    )
    
    # 配置交互功能和显示选项
    config = {
        'displayModeBar': True,   # 显示交互工具栏
        'scrollZoom': True,       # 允许滚轮缩放
        'displaylogo': False,     # 不显示Plotly logo
        'responsive': True        # 启用响应式布局
    }
    
    # 将可视化结果保存为HTML文件（与脚本位于同一目录）
    html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fund_distribution.html')
    fig.write_html(html_path, config=config, full_html=True, include_plotlyjs='cdn')
    
    # 在浏览器中展示地图
    fig.show(config=config)

# 程序入口函数
def main():
    # 提取并加载基金管理人位置数据
    fund_df = extract_fund_locations()
    
    # 数据清洗 - 筛选出位于中国地理范围内的数据点
    fund_df = fund_df[
        (fund_df['latitude'] > 18) & (fund_df['latitude'] < 45) &  # 中国纬度范围（南至北）
        (fund_df['longitude'] > 73) & (fund_df['longitude'] < 135)  # 中国经度范围（西至东）
    ]
    
    # 生成并显示地图可视化
    create_scatter_plot(fund_df)
    
    print(f"共处理 {len(fund_df)} 条基金管理人位置信息")

if __name__ == "__main__":
    main()
