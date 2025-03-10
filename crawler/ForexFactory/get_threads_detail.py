import asyncio
import os
from datetime import datetime
from pathlib import Path
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, BrowserConfig, DefaultMarkdownGenerator
from crawl4ai.content_filter_strategy import RelevantContentFilter
import re

class ForexPostFilter(RelevantContentFilter):
    """
    # 论坛帖子内容过滤器
    """
    POST_PATTERN = r'Post #(\d+)'
    TIME_PATTERN = r'(?:Edited |)\d{1,2}:\d{2}(?:am|pm)'
    USER_PATTERN = r'\[\s*([^\]]+)\].*?Status:'
    CONTENT_PATTERN = r'Status:.*?\n(.*?)(?:\[\d+\s*\]|\Z)'
    
    def filter_content(self, html: str, min_word_threshold=None) -> list:
        posts = re.findall(r'Post #\d+.*?(?=Post #\d+|$)', html, re.DOTALL)
        cleaned_posts = []
        
        for post in posts:
            post_info = {
                'number': self._extract_pattern(post, self.POST_PATTERN, group=1),
                'time': self._extract_pattern(post, self.TIME_PATTERN),
                'author': self._extract_pattern(post, self.USER_PATTERN, group=1),
                'content': self._clean_content(post)
            }
            
            if post_info['content']:
                cleaned_posts.append(self._format_post(post_info))
        
        return cleaned_posts if cleaned_posts else [html]
    
    def _extract_pattern(self, text: str, pattern: str, group=0) -> str:
        match = re.search(pattern, text, re.DOTALL)
        return match.group(group).strip() if match else 'Unknown'
    
    def _clean_content(self, post: str) -> str:
        content_match = re.search(self.CONTENT_PATTERN, post, re.DOTALL)
        if not content_match:
            return ''
        content = content_match.group(1).strip()
        content = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', content)
        return re.sub(r'\n{3,}', '\n\n', content)
    
    def _format_post(self, post_info: dict) -> str:
        return (f"Post #{post_info['number']}\n"
                f"Author: {post_info['author']}\n"
                f"Time: {post_info['time']}\n\n"
                f"{post_info['content']}\n")

def clean_markdown_content(content: str) -> str:
    """清理markdown内容"""
    # 删除开头到第一个 Post 之前的所有内容
    content = re.sub(r'^[\s\S]*?(?=  \* \[Post)', '', content, count=1)
    
    # 删除包含 "traders/trader viewing now" 的行及其后面所有内容
    content = re.sub(r'^\s*\d+\s*traders?\s*viewing\s*now[\s\S]*$', '', content, flags=re.MULTILINE)
    
    # 删除包含 "**Reply to Thread**" 的行向上三行及其后面所有内容
    content = re.sub(r'(?:.*\n){3}.*\*\*Reply to Thread\*\*.*\n[\s\S]*$', '', content)
    
    cleaners = [
        # 清除导航菜单
        (r'\[ \]\(https://www\.forexfactory\.com/thread/</>\)[\s\S]*?Menu \[ \]', '', re.MULTILINE),
        
        # 清除包含 Posts] 的行
        (r'.*\s+Posts\].*\n', '', re.MULTILINE),
        
        # 清除包含两个空格和[Quote]的行
        (r'  \* \[Quote\].*\n', '', re.MULTILINE),
        
        # 清除包含日期时间格式的行
        (r'.*[A-Z][a-z]{2}\s+\d{1,2},\s+\d{4}\s+\d{1,2}:\d{2}(?:am|pm).*\n', '', re.MULTILINE),
        
        # 清除包含特定URL格式的行
        (r'\(https://www\.forexfactory\.com/thread/</thread/[^>]+>\s*"[^"]+"\)', '', re.MULTILINE),
        
        # 清除包含特定格式的行
        (r'.*  \* .*\]\(https://www\.forexfactory\.com/thread/<.*\n', '', re.MULTILINE),
        
        # 清除附件图片相关的四行内容
        (r'Attached Image.*\n\[\!\[Click to Enlarge\n(?:Name: [^\n]*\n)(?:Size: [^\n]*\n)', '', re.MULTILINE),
        
        # 清除Attached Image/Images及其后带感叹号的行
        (r'Attached Images?\n[^\n]*![^\n]*\n', '', re.MULTILINE),
        
        # 清除包含 > Disliked 的行
        (r'>\s*Disliked\n', '', re.MULTILINE),
        
        # 清除包含 > Ignored 的行
        (r'>\s*Ignored\n', '', re.MULTILINE),
        
        # 清除Name:行及其后面包含/attachment/image/的行
        (r'Name: [^\n]*\n.*?/attachment/image/[^\n]*\n', '', re.MULTILINE),
        
        # 清除Options部分
        (r'## Options[\s\S]*?## Similar Threads', '## Similar Threads', re.MULTILINE),
        
        # 清除页面导航部分
        (r'\* \[Page \d+\].*?\[2232\].*?Last Page"\)', '', re.MULTILINE),
        
        # 清除底部菜单和版权信息
        (r'FF Sister Sites:[\s\S]*?Forex Factory®.*$', '', re.MULTILINE),
        
        # 清除javascript链接
        (r'\(https://www\.forexfactory\.com/thread/<javascript:.*?\)', '', re.MULTILINE),
        
        # 清除类似 [0 ];>) [0 ];>) 这样的行
        (r'^\s*\[\d+\s*\];>\)\s*\[\d+\s*\];>\)\s*$', '', re.MULTILINE),
        
        # 清除多余空行
        (r'\n{3,}', '\n\n', 0),
        
        # 清除空链接
        (r'\[\]\([^)]+\)', '', 0),
        
        # 清除用户/邮箱密码行
        (r'User/Email:Password:.*?\n', '', 0),
        
        # 清除时区信息行
        (r'\* \[ \d+:\d+[ap]m \].*?\n', '', 0),
        
        # 清除附件相关内容
        (r'Attachments:.*?Cancel.*?\n', '', re.MULTILINE),
    ]
    
    for pattern, replacement, flags in cleaners:
        content = re.sub(pattern, replacement, content, flags=flags)
    
    return content.strip()

def extract_title_from_url(thread_url: str) -> str:
    """
    从线程URL中提取标题
    
    Args:
        thread_url: 帖子URL，如 "https://www.forexfactory.com/thread/588764-pivot-trading"
        
    Returns:
        str: 提取的标题，如 "pivot-trading"
    """
    # 提取URL的最后一部分
    url_parts = thread_url.rstrip('/').split('/')
    last_part = url_parts[-1]
    
    # 查找数字-文本格式
    match = re.search(r'\d+-(.+)$', last_part)
    if match:
        # 提取数字后面的部分
        title = match.group(1)
        # 将连字符替换为空格并首字母大写
        title = title.replace('-', ' ').title()
        return title
    
    # 如果没有匹配到预期格式，返回原始的最后部分
    return last_part

def extract_max_page(content: str) -> int:
    """
    从原始内容中提取最大页数
    
    Args:
        content: 原始markdown内容
        
    Returns:
        int: 最大页数，如果未找到则返回1
    """
    # 查找 "Last Page" 前面的 ?page= 后的数字
    match = re.search(r'\?page=(\d+)[^"]*"Last Page"', content)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            pass
    return 1  # 默认为1页

async def get_threads_detail(thread_url: str, start_page: int = 1, max_pages: int = 3500, max_concurrent: int = 50) -> str:
    """
    # 异步获取论坛帖子详情
    
    Args:
        thread_url: 帖子URL，如 "https://www.forexfactory.com/thread/588764-pivot-trading"
        start_page: 开始爬取的页码，默认为1
        max_pages: 最多爬取的页数，默认为3500
        max_concurrent: 最大并行任务数，默认为50
        
    Returns:
        str: 清理后的markdown内容
    """
    # 确保URL格式正确
    base_url = thread_url.split('?')[0]  # 移除可能存在的查询参数
    
    # 从URL中提取标题
    thread_title = extract_title_from_url(thread_url)
    
    browser_config = BrowserConfig(
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Connection": "keep-alive",
        }
    )

    run_config = CrawlerRunConfig(
        markdown_generator=DefaultMarkdownGenerator(content_filter=ForexPostFilter()),
        word_count_threshold=5,
        wait_for="body",
        page_timeout=15000,
        simulate_user=True,
        magic=True,
        excluded_tags=["nav", "footer", "header"],
        exclude_external_links=True,
        js_code="""
            await new Promise(r => setTimeout(r, 5000));
            document.querySelectorAll('button, [role="button"]').forEach(btn => btn.click());
            for (let i = 0; i < 5; i++) {
                window.scrollTo(0, document.body.scrollHeight * i / 4);
                await new Promise(r => setTimeout(r, 1000));
            }
        """
    )

    # 创建保存目录
    output_dir = Path(os.path.dirname(os.path.abspath(__file__))) / "unextracted_files"
    output_dir.mkdir(exist_ok=True)
    
    # 提取线程ID用于文件命名
    thread_id = re.search(r'/thread/(\d+)', thread_url)
    thread_id = thread_id.group(1) if thread_id else "unknown"
    
    combined_content = ""
    actual_max_pages = None  # 初始化为 None，表示还未获取最大页数
    
    try:
        crawler = AsyncWebCrawler(config=browser_config)
        await crawler.start()
        
        try:
            all_results = {}
            last_page_content = None
            found_duplicate = False
            
            # 先抓取起始页，获取最大页码
            start_url = f"{base_url}?page={start_page}"
            start_result = await crawler.arun(url=start_url, config=run_config, session_id=f"page_{start_page}")
            
            if start_result.success:
                raw_content = str(start_result.markdown_v2.raw_markdown)
                actual_max_pages = extract_max_page(raw_content)
                print(f"Found max pages: {actual_max_pages}")  # 调试输出
                
                # 计算从start_page到最大页数的实际页数
                remaining_pages = actual_max_pages + 1 - start_page
                # 更新最大爬取页数
                max_pages = min(max_pages, remaining_pages)
                print(f"Starting from page {start_page}, remaining pages: {remaining_pages}")
                print(f"Adjusted max pages to: {max_pages}")  # 调试输出
                
                # 保存起始页内容
                content = clean_markdown_content(str(start_result.markdown_v2.fit_markdown or start_result.markdown_v2.raw_markdown))
                all_results[start_page] = content
                last_page_content = content
                
                # 如果还有更多页面需要爬取
                if max_pages > 1:
                    # 计算结束页码
                    end_page = start_page + max_pages
                    
                    # 分批并行爬取剩余页面
                    for batch_start in range(start_page + 1, end_page, max_concurrent):
                        if found_duplicate:
                            break
                            
                        batch_end = min(batch_start + max_concurrent, end_page)
                        current_batch = []
                        tasks = []
                        
                        # 创建当前批次的任务
                        for page_num in range(batch_start, batch_end):
                            current_batch.append((page_num, f"{base_url}?page={page_num}"))
                        
                        print(f"Processing batch: {batch_start} to {batch_end-1}")  # 调试输出
                        
                        # 创建并执行任务
                        for page_num, url in current_batch:
                            session_id = f"page_{page_num}"
                            task = asyncio.create_task(crawler.arun(url=url, config=run_config, session_id=session_id))
                            tasks.append((page_num, task))
                        
                        # 等待当前批次的所有任务完成
                        for page_num, task in tasks:
                            try:
                                result = await task
                                if result.success:
                                    current_content = clean_markdown_content(str(result.markdown_v2.fit_markdown or result.markdown_v2.raw_markdown))
                                    
                                    # 检查是否与上一页内容相同
                                    if current_content == last_page_content:
                                        print(f"Found duplicate content at page {page_num}")  # 调试输出
                                        found_duplicate = True
                                        break
                                    
                                    all_results[page_num] = current_content
                                    last_page_content = current_content
                                    print(f"Successfully processed page {page_num}")  # 调试输出
                            except Exception as e:
                                print(f"Error processing page {page_num}: {e}")
                        
                        if found_duplicate:
                            break
            
            # 按页码顺序处理结果
            all_content = []
            for page_num in sorted(all_results.keys()):
                all_content.append(all_results[page_num])
            
            # 合并所有内容并保存
            if all_content:
                # 将标题添加到合并内容的顶部
                combined_content = f"# {thread_title}\n\n" + "\n\n---\n\n".join(all_content)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = output_dir / f"thread_{thread_id}_{timestamp}.md"
                
                output_file.write_text(combined_content, encoding="utf-8")
            else:
                pass
                
        finally:
            await asyncio.sleep(1)  # 给予额外时间确保所有任务都完成
            await crawler.close()
            
    except Exception as e:
        print(f"Crawler error: {e}")
    
    return combined_content


async def main():
    # 示例用法
    thread_url = "https://www.forexfactory.com/thread/588764-pivot-trading"
    content = await get_threads_detail(thread_url, start_page=1, max_pages=2)

if __name__ == "__main__":
    asyncio.run(main()) 
