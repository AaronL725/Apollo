import pandas as pd
import json
import re
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.font_manager import FontProperties
matplotlib.use('Agg')
from io import BytesIO
import base64
import webbrowser
import os

def extract_executives_info(row):
    """使用简单字符串搜索提取高管信息"""
    executives = []
    company_name = row[1]  # 基金管理人名称
    
    try:
        # 获取字符串
        json_str = row[-1]
        
        # 按段落分割数据
        segments = json_str.split('}, {')
        
        # 初始化变量
        current_exec = None
        
        for segment in segments:
            # 修复段落格式以进行检查
            if not segment.startswith('{'):
                segment = '{' + segment
            if not segment.endswith('}'):
                segment = segment + '}'
                
            # 检查这个段落是否包含姓名
            if '"姓名' in segment:
                # 如果已有高管信息，保存它
                if current_exec is not None and 'work_records' in current_exec:
                    executives.append({
                        "姓名": current_exec.get('name', '未知'),
                        "职务": current_exec.get('position', '未知'),
                        "原任职公司": company_name,
                        "资格获取方式": current_exec.get('qualification', '未知'),
                        "是否有基金从业资格": current_exec.get('has_qualification', '未知'),
                        "工作履历": current_exec.get('work_records', [])
                    })
                
                # 提取姓名和职务
                name_match = re.search(r'"姓名\'": \'"([^\']+)\'"', segment)
                position_match = re.search(r'"职务\'": \'"([^\']+)\'"', segment)
                
                current_exec = {
                    'name': name_match.group(1) if name_match else '未知',
                    'position': position_match.group(1) if position_match else '未知',
                    'work_records': []
                }
                
            # 检查这个段落是否包含资格信息
            elif '"资格获取方式' in segment and current_exec is not None:
                qual_match = re.search(r'"资格获取方式\'": \'"([^\']+)\'"', segment)
                has_qual_match = re.search(r'"是否有基金从业资格\'": \'"([^\']+)\'"', segment)
                
                if qual_match:
                    current_exec['qualification'] = qual_match.group(1)
                if has_qual_match:
                    current_exec['has_qualification'] = has_qual_match.group(1)
                    
            # 检查这个段落是否包含工作履历
            elif '"工作履历' in segment and current_exec is not None:
                # 提取所有工作记录
                work_entries = re.findall(r'{([^{}]+)}', segment)
                
                for entry in work_entries:
                    time_match = re.search(r'"时间\'": \'"([^\']+)\'"', entry)
                    role_match = re.search(r'"职务\'": \'"([^\']+)\'"', entry)
                    company_match = re.search(r'"任职单位\'": \'"(.*?)\'"', entry)
                    dept_match = re.search(r'"任职部门\'": \'"(.*?)\'"', entry)
                    
                    if time_match and role_match and company_match:
                        work_record = {
                            "时间": time_match.group(1),
                            "职务": role_match.group(1),
                            "任职单位": company_match.group(1),
                            "任职部门": dept_match.group(1) if dept_match else ""
                        }
                        current_exec['work_records'].append(work_record)
        
        # 保存最后一个高管
        if current_exec is not None and 'work_records' in current_exec:
            executives.append({
                "姓名": current_exec.get('name', '未知'),
                "职务": current_exec.get('position', '未知'),
                "原任职公司": company_name,
                "资格获取方式": current_exec.get('qualification', '未知'),
                "是否有基金从业资格": current_exec.get('has_qualification', '未知'),
                "工作履历": current_exec.get('work_records', [])
            })
            
    except Exception as e:
        print(f"解析错误 ({company_name}): {str(e)}")
        import traceback
        traceback.print_exc()
    
    return executives

def count_companies(all_executives):
    """统计每家公司曾经有多少高管待过"""
    company_counts = {}
    
    for exec_info in all_executives:
        if "工作履历" not in exec_info:
            continue
            
        for work in exec_info["工作履历"]:
            company = work.get("任职单位", "")
            if company:
                company_counts[company] = company_counts.get(company, 0) + 1
    
    # 排序
    sorted_companies = sorted(company_counts.items(), key=lambda x: x[1], reverse=True)
    return sorted_companies

def plot_companies(company_data, top_n=20):
    """生成横向柱状图，并解决中文显示问题"""
    # 限制显示前N家公司
    company_data = company_data[:top_n]
    
    companies = [item[0] for item in company_data]
    counts = [item[1] for item in company_data]
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei']  # 优先使用的中文字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    
    # 创建图表
    plt.figure(figsize=(12, 8))
    
    # 如果公司名太长，截断显示
    shortened_companies = []
    for company in companies:
        if len(company) > 15:
            shortened_companies.append(company[:15] + '...')
        else:
            shortened_companies.append(company)
    
    bars = plt.barh(shortened_companies, counts, color='purple')
    
    # 在柱子末端添加数值
    for bar in bars:
        width = bar.get_width()
        plt.text(width, bar.get_y() + bar.get_height()/2, 
                 f'{int(width)}', ha='left', va='center')
    
    plt.xlabel('高管人数')
    plt.ylabel('公司名称')
    plt.title('各公司高管任职人数统计')
    plt.tight_layout()
    
    # 将图表转换为base64编码用于HTML嵌入
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=300)
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    return img_str

def generate_html(company_data):
    """生成只包含详细数据表格的HTML网页"""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>公司高管统计详细数据</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }
            .container { max-width: 1200px; margin: 0 auto; background-color: white; padding: 20px; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
            h1 { color: #333; text-align: center; }
            table { width: 100%; border-collapse: collapse; margin-top: 20px; }
            th, td { padding: 10px; text-align: left; border-bottom: 1px solid #ddd; word-break: break-word; }
            th { background-color: #f2f2f2; }
            tr:hover { background-color: #f5f5f5; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>公司高管任职人数统计</h1>
            <table>
                <tr>
                    <th>排名</th>
                    <th style="width: 70%;">公司名称</th>
                    <th>高管人数</th>
                </tr>
    """
    
    for i, (company, count) in enumerate(company_data):
        html += f"""
                <tr>
                    <td>{i+1}</td>
                    <td>{company}</td>
                    <td>{count}</td>
                </tr>
        """
    
    html += """
            </table>
        </div>
    </body>
    </html>
    """
    
    return html

def main():
    # 读取CSV文件
    csv_path = 'manager_data.csv'
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    all_executives = []
    
    for line in lines:
        # 找到最后一个方括号开始的位置
        json_start = line.find('[{')
        if json_start > 0:
            # 提取基本信息
            base_info = line[:json_start].split(',')
            company_name = base_info[1] if len(base_info) > 1 else ""
            
            # 提取JSON部分
            json_str = line[json_start:]
            try:
                # 提取高管信息
                executives = extract_executives_info([None, company_name, None, None, json_str])
                all_executives.extend(executives)
            except Exception as e:
                print(f"解析错误，跳过该行: {e}")
    
    # 打印找到的高管数量
    print(f"成功提取了 {len(all_executives)} 名高管的信息")
    
    # 统计各公司高管人数
    company_data = count_companies(all_executives)
    
    # 生成HTML (不再需要生成图表)
    html_content = generate_html(company_data)
    
    # 保存并打开HTML文件
    html_path = os.path.join('company_stay', 'company_executives_statistics.html')
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    # 在浏览器中打开HTML
    webbrowser.open('file://' + os.path.realpath(html_path))

if __name__ == "__main__":
    main()
