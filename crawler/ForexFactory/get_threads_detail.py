import undetected_chromedriver as uc
import time
import random
import os
import re
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse
import html2text
from bs4 import BeautifulSoup
import datetime
import psutil

class ForexFactoryCrawler:
    def __init__(self):
        # 初始化爬虫对象，设置基础URL和输出目录
        self.driver = None
        self.base_url = "https://www.forexfactory.com/thread/38542-pivot-point-with-money-management-strategy"
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.output_dir = os.path.join(self.script_dir, "unextracted_files")
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        # 设置HTML转Markdown配置
        self.h2t = html2text.HTML2Text()
        self.h2t.ignore_links = False
        self.h2t.ignore_images = False
        self.h2t.body_width = 0  # 不限制行宽
        self.max_page = None

    def start_browser(self):
        # 启动无头Chrome浏览器
        options = uc.ChromeOptions()
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--no-sandbox")
        
        self.driver = uc.Chrome(options=options)
        self.driver.maximize_window()
        time.sleep(2)

    def visit_url(self, url):
        # 访问指定URL，模拟人类行为
        try:
            self.driver.get(url)
            time.sleep(random.uniform(3, 5))
        except Exception as e:
            print(f"导航到URL时出错: {e}")
            self.driver.get('about:blank')
            time.sleep(random.uniform(1, 2))
            self.driver.execute_script(f"window.location.href = '{url}';")
            time.sleep(random.uniform(3, 5))

    def get_page_content(self, page_num):
        # 获取指定页码的页面内容
        parsed_url = list(urlparse(self.base_url))
        query_dict = parse_qs(parsed_url[4])
        query_dict['page'] = [str(page_num)]
        parsed_url[4] = urlencode(query_dict, doseq=True)
        url = urlunparse(parsed_url)
        
        print(f"解析后的URL: {parsed_url}")
        print(f"构建的完整URL: {url}")
        
        self.visit_url(url)
        
        content = self.driver.page_source
        return content

    def convert_to_markdown(self, html_content):
        # 将HTML内容转换为Markdown格式
        soup = BeautifulSoup(html_content, 'html.parser')
        
        for element in soup.find_all(['script', 'style', 'iframe']):
            element.decompose()
            
        markdown = self.h2t.handle(str(soup))
        return markdown

    def detect_max_page(self, html_content):
        # 从页面内容中检测最大页码
        soup = BeautifulSoup(html_content, 'html.parser')
        
        last_page_link = soup.find('a', title='Last Page')
        if last_page_link:
            max_page = int(last_page_link.text.strip())
            print(f"检测到最大页码: {max_page}")
            return max_page
        
        print("未检测到最大页码，将使用指定的结束页码")
        return None

    def crawl_pages(self, start_page, end_page):
        # 爬取指定范围内的所有页面
        try:
            self.start_browser()
            all_content = []
            
            print(f"正在爬取第 {start_page} 页并检测最大页码...")
            first_page_content = self.get_page_content(start_page)
            all_content.append(first_page_content)
            
            self.max_page = self.detect_max_page(first_page_content)
            
            thread_title = self.extract_thread_title(first_page_content)
            
            if self.max_page and end_page > self.max_page:
                print(f"调整结束页码: {end_page} -> {self.max_page}")
                end_page = self.max_page
            
            for page in range(start_page + 1, end_page + 1):
                print(f"正在爬取第 {page} 页...")
                content = self.get_page_content(page)
                all_content.append(content)
                
                if page < end_page:
                    time.sleep(random.uniform(1, 2))
            
            combined_html = '\n'.join(all_content)
            
            markdown_content = self.convert_to_markdown(combined_html)
            
            cleaned_markdown = self.clean_markdown(markdown_content)
            
            current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            filename = f"{thread_title}_{current_time}.md" if thread_title else f"thread_content_{current_time}.md"
            output_file = os.path.join(self.output_dir, filename)
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(cleaned_markdown)
            
            print(f"爬取完成！内容已保存到: {output_file}")
            
        finally:
            # 确保浏览器进程被彻底关闭
            if self.driver:
                try:
                    self.driver.close()
                except:
                    pass
                
                try:
                    for proc in psutil.process_iter():
                        try:
                            if "chrome" in proc.name().lower():
                                cmdline = " ".join(proc.cmdline()).lower()
                                if "--remote-debugging-port" in cmdline:
                                    proc.terminate()
                        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                            pass
                except:
                    pass
                
                self.driver = None

    def extract_thread_title(self, html_content):
        # 从URL中提取线程标题
        match = re.search(r'/thread/\d+-(.+?)(?:\?|$)', self.base_url)
        if match:
            thread_title = match.group(1)
            print(f"从URL中提取到标题: {thread_title}")
            return thread_title
        
        print("未能从URL中提取到线程标题")
        return None

    def clean_markdown(self, markdown_content):
        # 清理Markdown内容，移除网站导航、菜单等无关内容
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
        
        cleaned_content = markdown_content
        for pattern in patterns:
            cleaned_content = re.sub(pattern, '', cleaned_content, flags=re.MULTILINE)
        
        cleaned_content = re.sub(r'\n{3,}', '\n\n', cleaned_content)
        
        return cleaned_content

if __name__ == "__main__":
    # 程序入口点，创建爬虫实例并开始爬取指定页面范围
    crawler = ForexFactoryCrawler()
    crawler.crawl_pages(1, 500)
