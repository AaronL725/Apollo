import asyncio
import json
import sys
from bs4 import BeautifulSoup
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode, BrowserConfig

async def get_threads_link(start_page=1, end_page=1, max_concurrent=10):
    """
    # 获取ForexFactory论坛指定页码范围内的线程链接
    # 参数:
    #   start_page: 起始页码
    #   end_page: 结束页码
    #   max_concurrent: 最大并发爬取页面数
    # 返回:
    #   JSON字符串，包含所有页面的线程链接
    """
    async def extract_thread_links(url, crawler, config):
        """
        # 从论坛页面提取线程链接
        """
        # 爬取页面内容
        result = await crawler.arun(url=url, config=config)
        
        if not result.success:
            return {}
        
        html_content = result.html if hasattr(result, 'html') else result.cleaned_html
        
        # 解析HTML并提取链接
        soup = BeautifulSoup(html_content, 'html.parser')
        all_links = soup.find_all('a', href=True)
        
        # 过滤符合条件的线程链接
        thread_links = {}
        link_count = 0
        max_links = 40  # 每页最多获取40个链接
        
        for link in all_links:
            if link_count >= max_links:
                break
                
            href = link.get('href')
            title = link.text.strip()
            
            # 跳过无标题链接
            if not title:
                continue
            
            # 处理相对URL
            if href and not href.startswith('http'):
                href = f"https://www.forexfactory.com{href}"
            
            # 筛选线程链接（包含/thread/但不包含/thread/post/）
            if (href and 
                href.startswith("https://www.forexfactory.com/thread/") and 
                not href.startswith("https://www.forexfactory.com/thread/post/")):
                
                # 避免重复链接
                if href not in thread_links.values():
                    thread_links[title] = href
                    link_count += 1
        
        return thread_links

    # 存储所有页面的结果
    all_threads = []
    
    # 创建浏览器配置
    browser_config = BrowserConfig(
        headless=True,
        extra_args=["--disable-gpu", "--disable-dev-shm-usage", "--no-sandbox"],
    )
    
    # 创建爬虫实例
    crawler = AsyncWebCrawler(config=browser_config)
    await crawler.start()
    
    try:
        # 准备所有页面的URL
        urls = []
        for page in range(start_page, end_page + 1):
            forum_url = f"https://www.forexfactory.com/forum/71-trading-systems?sort=replycount&order=desc&page={page}"
            urls.append(forum_url)
        
        # 分批并发爬取
        for i in range(0, len(urls), max_concurrent):
            batch = urls[i:i + max_concurrent]
            tasks = []
            
            for j, url in enumerate(batch):
                # 为每个并发任务创建唯一的会话ID
                session_id = f"page_session_{i + j}"
                config = CrawlerRunConfig(
                    cache_mode=CacheMode.BYPASS,
                    session_id=session_id
                )
                
                # 创建爬取任务
                task = extract_thread_links(url, crawler, config)
                tasks.append(task)
            
            # 并发执行所有任务
            results = await asyncio.gather(*tasks)
            
            # 处理结果
            for url, thread_links in zip(batch, results):
                if thread_links:
                    # 只保留title和link
                    page_threads = [{"title": title, "link": url} for title, url in thread_links.items()]
                    all_threads.extend(page_threads)
    
    finally:
        # 关闭爬虫实例
        await crawler.close()
    
    # 转换为JSON格式
    return json.dumps(all_threads, ensure_ascii=False, indent=2)



'''

# 命令行调用示例
if __name__ == "__main__":
    # 从命令行参数获取页码范围
    start_page = 1
    end_page = 3
    max_concurrent = 15
    
    if len(sys.argv) > 1:
        try:
            start_page = int(sys.argv[1])
            if start_page < 1:
                start_page = 1
        except ValueError:
            pass
    
    if len(sys.argv) > 2:
        try:
            end_page = int(sys.argv[2])
            if end_page < start_page:
                end_page = start_page
        except ValueError:
            end_page = start_page
    
    if len(sys.argv) > 3:
        try:
            max_concurrent = int(sys.argv[3])
            if max_concurrent < 1:
                max_concurrent = 1
        except ValueError:
            pass
    
    # 运行函数并打印结果
    async def run():
        json_result = await get_threads_link(start_page, end_page, max_concurrent)
        print(json_result)
    
    asyncio.run(run())
    
'''



# 输出格式
'''
[
  {
    "title": "Trading Made Simple",
    "link": "https://www.forexfactory.com/thread/291622-trading-made-simple"
  },
  {
    "title": "Supernova GBP/JPY Mini Trend Catcher",
    "link": "https://www.forexfactory.com/thread/44216-supernova-gbpjpy-mini-trend-catcher"
  },
  {
    "title": "Come Surfing Fx With Me System",
    "link": "https://www.forexfactory.com/thread/527711-come-surfing-fx-with-me-system"
  },
  {
    "title": "Roadmap - A Way To Read Markets",
    "link": "https://www.forexfactory.com/thread/993524-roadmap-a-way-to-read-markets"
  }
]
'''
