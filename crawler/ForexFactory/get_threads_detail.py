import undetected_chromedriver as uc
import time
import random
import pyautogui
import os
import re
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse
import atexit
import html2text
from bs4 import BeautifulSoup
import datetime
import psutil

class ForexFactoryCrawler:
    def __init__(self):
        """
        初始化爬虫类
        """
        self.driver = None
        self.base_url = "https://www.forexfactory.com/thread/38542-pivot-point-with-money-management-strategy"
        # 获取脚本所在目录
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        # 设置输出目录为脚本所在目录下的unextracted_files文件夹
        self.output_dir = os.path.join(self.script_dir, "unextracted_files")
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        # 初始化HTML转Markdown转换器
        self.h2t = html2text.HTML2Text()
        self.h2t.ignore_links = False
        self.h2t.ignore_images = False
        self.h2t.body_width = 0  # 不限制行宽
        # 最大页码
        self.max_page = None

    def start_browser(self):
        """
        启动浏览器
        """
        # 创建Chrome浏览器实例，添加选项以避免某些问题
        options = uc.ChromeOptions()
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--no-sandbox")
        
        self.driver = uc.Chrome(options=options)
        self.driver.maximize_window()
        time.sleep(2)  # 等待浏览器完全启动

    def visit_url(self, url):
        """
        模拟人类访问URL
        """
        # 直接使用selenium导航到URL，避免使用pyautogui输入长URL
        try:
            self.driver.get(url)
            # 随机等待3-5秒确保页面加载完成
            time.sleep(random.uniform(3, 5))
        except Exception as e:
            print(f"导航到URL时出错: {e}")
            # 如果直接导航失败，尝试备用方法
            self.driver.get('about:blank')
            time.sleep(random.uniform(1, 2))
            self.driver.execute_script(f"window.location.href = '{url}';")
            time.sleep(random.uniform(3, 5))

    def get_page_content(self, page_num):
        """
        获取指定页面的内容
        """
        # 构建带页码的URL
        parsed_url = list(urlparse(self.base_url))
        query_dict = parse_qs(parsed_url[4])
        query_dict['page'] = [str(page_num)]
        parsed_url[4] = urlencode(query_dict, doseq=True)
        url = urlunparse(parsed_url)
        
        # 打印调试信息
        print(f"解析后的URL: {parsed_url}")
        print(f"构建的完整URL: {url}")
        
        # 访问URL
        self.visit_url(url)
        
        # 获取页面内容
        content = self.driver.page_source
        return content

    def convert_to_markdown(self, html_content):
        """
        将HTML内容转换为Markdown格式
        """
        # 使用BeautifulSoup清理HTML
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # 移除不需要的元素
        for element in soup.find_all(['script', 'style', 'iframe']):
            element.decompose()
            
        # 转换为Markdown
        markdown = self.h2t.handle(str(soup))
        return markdown

    def detect_max_page(self, html_content):
        """
        从HTML内容中检测最大页码
        """
        # 使用BeautifulSoup解析HTML
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # 查找带有"Last Page"标题的链接
        last_page_link = soup.find('a', title='Last Page')
        if last_page_link:
            # 提取链接文本，通常就是页码
            max_page = int(last_page_link.text.strip())
            print(f"检测到最大页码: {max_page}")
            return max_page
        
        print("未检测到最大页码，将使用指定的结束页码")
        return None

    def crawl_pages(self, start_page, end_page):
        """
        爬取指定范围的页面
        """
        try:
            self.start_browser()
            all_content = []
            
            # 先爬取第一页，检测最大页码和提取线程标题
            print(f"正在爬取第 {start_page} 页并检测最大页码...")
            first_page_content = self.get_page_content(start_page)
            all_content.append(first_page_content)
            
            # 检测最大页码
            self.max_page = self.detect_max_page(first_page_content)
            
            # 提取线程标题
            thread_title = self.extract_thread_title(first_page_content)
            
            # 调整结束页码
            if self.max_page and end_page > self.max_page:
                print(f"调整结束页码: {end_page} -> {self.max_page}")
                end_page = self.max_page
            
            # 爬取剩余页面
            for page in range(start_page + 1, end_page + 1):
                print(f"正在爬取第 {page} 页...")
                content = self.get_page_content(page)
                all_content.append(content)
                
                # 随机等待1-2秒再访问下一页
                if page < end_page:
                    time.sleep(random.uniform(1, 2))
            
            # 合并所有内容
            combined_html = '\n'.join(all_content)
            
            # 转换为Markdown
            markdown_content = self.convert_to_markdown(combined_html)
            
            # 清理Markdown内容
            cleaned_markdown = self.clean_markdown(markdown_content)
            
            # 获取当前时间，格式化为年月日时分秒
            current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 保存Markdown文件，使用线程标题和时间戳作为文件名
            filename = f"{thread_title}_{current_time}.md" if thread_title else f"thread_content_{current_time}.md"
            output_file = os.path.join(self.output_dir, filename)
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(cleaned_markdown)
            
            print(f"爬取完成！内容已保存到: {output_file}")
            
        finally:
            # 在函数结束时直接关闭浏览器进程，而不是使用driver的方法
            if self.driver:
                try:
                    # 尝试直接关闭浏览器窗口
                    self.driver.close()
                except:
                    pass
                
                try:
                    # 使用操作系统级别的方法终止Chrome进程
                    for proc in psutil.process_iter():
                        try:
                            # 检查是否是Chrome进程
                            if "chrome" in proc.name().lower():
                                # 检查命令行参数，确认是我们启动的Chrome
                                cmdline = " ".join(proc.cmdline()).lower()
                                if "--remote-debugging-port" in cmdline:
                                    proc.terminate()
                        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                            pass
                except:
                    # 如果psutil不可用或出错，忽略
                    pass
                
                # 将driver设为None，避免后续引用
                self.driver = None

    def extract_thread_title(self, html_content):
        """
        从base_url中提取线程标题
        """
        # 直接从base_url中提取标题
        match = re.search(r'/thread/\d+-(.+?)(?:\?|$)', self.base_url)
        if match:
            thread_title = match.group(1)
            print(f"从URL中提取到标题: {thread_title}")
            return thread_title
        
        print("未能从URL中提取到线程标题")
        return None

    def clean_markdown(self, markdown_content):
        """
        清理Markdown内容，移除导航菜单等不需要的内容
        """
        # 移除包含导航菜单元素的行
        patterns = [
            r"^.*\[ Home \].*$",
            r"^.*\[ Forums \].*$",
            r"^.*\[ Trades \].*$", 
            r"^.*\[ News \].*$",
            r"^.*\[ Calendar \].*$",
            r"^.*\[ Market \].*$",
            r"^.*\[ Brokers \].*$",
            r"^.*\[\]\(/login\).*$",
            r"^.*Menu \[ \]\(/\).*$",
            r"^.*\[ Login \].*$",
            r"^.*\[ Create Account \].*$",
            r"^.*\[ \]\(/\).*$",
            r"^.*\[\]\(javascript:void.*\).*$",
            r"^.*\[Print Thread\].*$",
            r"^.*\[Cancel\].*$",
            r"^.*\[Exit Attachments\].*$",
            r"^.*\[Last Post\].*$",
            r"^.*\[First Page\].*$",
            r"^.*\[Last Page\].*$",
            r"^.*\[Quote\].*$",
            r"^.*\[Reply to Thread\].*$",
            r"^.*\[Subscribe\].*$",
            r"^.*\[Page \d+\].*$",
            r"^.*\d+ trader.*viewing now.*$",
            r"^.*Top of Page.*$",
            r"^.*\[Facebook\].*$",
            r"^.*\[X\].*$",
            r"^.*\*\*About FF\*\*.*$",
            r"^.*\*\*FF Products\*\*.*$",
            r"^.*\*\*FF Website\*\*.*$",
            r"^.*\*\*Follow FF\*\*.*$",
            r"^.*FF Sister Sites:.*$",
            r"^.*\[Metals Mine\].*$",
            r"^.*\[Energy EXCH\].*$",
            r"^.*\[Crypto Craft\].*$",
            r"^.*Forex Factory.*$",
            r"^.*\[Terms of Service\].*$",
            r"^.*\[Mission\].*$",
            r"^.*\[Products\].*$",
            r"^.*\[User Guide\].*$",
            r"^.*\[Media Kit\].*$",
            r"^.*\[Blog\].*$",
            r"^.*\[Contact\].*$",
            r"^.*\[Trade Explorer\].*$",
            r"^.*\[Homepage\].*$",
            r"^.*\[Search\].*$",
            r"^.*\[Traders\].*$",
            r"^.*\[Report a Bug\].*$",
            r"^.*Options.*$",
            r"^.*Bookmark Thread.*$",
            r"^.*First Unread.*$",
            r"^.*Similar Threads.*$",
            r"^.*\[Login\].*$",
            r"^.*\[Create Account\].*$",
            r"^.*User/Email.*$",
            r"^.*\(/timezone.*$"
        ]
        
        # 逐个应用模式删除匹配的行
        cleaned_content = markdown_content
        for pattern in patterns:
            cleaned_content = re.sub(pattern, '', cleaned_content, flags=re.MULTILINE)
        
        # 移除连续的空行
        cleaned_content = re.sub(r'\n{3,}', '\n\n', cleaned_content)
        
        return cleaned_content

if __name__ == "__main__":
    crawler = ForexFactoryCrawler()
    # 示例：爬取第1页到第5页
    crawler.crawl_pages(1, 500)
