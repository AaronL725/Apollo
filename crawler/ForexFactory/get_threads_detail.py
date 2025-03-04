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
    cleaners = [
        # 需要 DOTALL 标志的正则
        (r'^.*?(?=Post #\d+)', '', re.DOTALL),  
        (r'Menu.*?(?=Post #)|About FF.*', '', re.DOTALL),
        
        # 需要 MULTILINE 标志的正则 - 修改这一行，使用更精确的正则表达式捕获完整的帖子编号
        (r'^Post #([\d,]+).*?"Post Permalink"\)\s*$', lambda m: f'  * Post #{m.group(1)}', re.MULTILINE),
        
        # 不需要特殊标志的正则
        (r'\[(.*?)\]\(https?://[^\)]+\)', r'\1', 0),
        (r'\[\]\([^\)]+\)', '', 0),
        (r'\d+\s*;>\)\s*\d+\s*;>\)|\[\d+\s*\]|\[\s*\d+\s*\]', '', 0),
        (r'!\[.*?\]\(.*?\)|!\[\]\(https?://[^\)]+\)', '', 0),
        (r'Attached File\(s\).*?(?=\n|$)', '', 0),
        (r'Status:.*?\|', '|', 0),
        (r'\n\s*\n\s*\n', '\n\n', 0),
        (r'> (?:Disliked|Ignored)\n', '', 0),
        (r'  \* Quote\n  \* (?:First Post: |).*?\n\n  \*\s*\n  \* (?:\| |)Joined.*?Posts\n', '\n', 0)
    ]
    
    for pattern, replacement, flags in cleaners:
        content = re.sub(pattern, replacement, content, flags=flags)
    
    # 添加新的清洗规则：删除 "Reply to Thread" 行及其上面三行和下面所有内容
    reply_thread_pattern = r'(?:.*\n){3}  \*  \*\*Reply to Thread\*\*.*\n[\s\S]*$'
    content = re.sub(reply_thread_pattern, '', content)
    
    return content.strip()

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
    
    try:
        # 创建爬虫实例
        crawler = AsyncWebCrawler(config=browser_config)
        await crawler.start()
        
        try:
            all_results = {}  # 用字典存储结果，键为页码
            
            # 生成所有页面的URL
            urls = []
            for page_num in range(start_page, start_page + max_pages):
                urls.append((page_num, f"{base_url}?page={page_num}"))
            
            # 分批并行爬取
            for i in range(0, len(urls), max_concurrent):
                batch = urls[i:i + max_concurrent]
                tasks = []
                
                for page_num, url in batch:
                    # 为每个并行任务创建唯一的session_id
                    session_id = f"page_{page_num}"
                    task = asyncio.create_task(crawler.arun(url=url, config=run_config, session_id=session_id))
                    tasks.append((page_num, task))
                
                # 等待当前批次的所有任务完成
                for page_num, task in tasks:
                    try:
                        result = await task
                        if result.success:
                            content = clean_markdown_content(str(result.markdown_v2.fit_markdown or result.markdown_v2.raw_markdown))
                            all_results[page_num] = content
                        else:
                            pass
                    except Exception as e:
                        pass
            
            # 按页码顺序处理结果
            all_content = []
            last_page_content = ""
            
            for page_num in range(start_page, start_page + max_pages):
                if page_num not in all_results:
                    continue
                    
                content = all_results[page_num]
                
                # 检查是否与上一页内容相同
                if content == last_page_content:
                    continue
                
                all_content.append(content)
                last_page_content = content
            
            # 合并所有内容并保存
            if all_content:
                combined_content = "\n\n---\n\n".join(all_content)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = output_dir / f"thread_{thread_id}_{timestamp}.md"
                
                output_file.write_text(combined_content, encoding="utf-8")
            else:
                pass
                
        finally:
            # 确保爬虫实例被正确关闭
            await crawler.close()
            
    except Exception as e:
        pass
    
    return combined_content


'''
async def main():
    # 示例用法
    thread_url = "https://www.forexfactory.com/thread/588764-pivot-trading"
    content = await get_threads_detail(thread_url, start_page=2230, max_pages=5)

if __name__ == "__main__":
    asyncio.run(main()) 
'''
