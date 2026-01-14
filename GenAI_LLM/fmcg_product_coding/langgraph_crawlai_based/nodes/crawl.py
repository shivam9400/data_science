from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from crawl4ai.content_filter_strategy import BM25ContentFilter
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator

async def crawl_node(state):
    idx = state.get("current_url_index", 0)
    if idx >= len(state['found_urls']):
        return {"error": "All URLs exhausted"}
    
    if not state.get('found_urls'):
        return {"error": "No URLs found"}

    target_url = state['found_urls'][idx]
    bm25_filter = BM25ContentFilter(user_query=state['raw_input'], bm25_threshold=1.0)
    md_generator = DefaultMarkdownGenerator(content_filter=bm25_filter)

    # 1. Light Browser Configuration
    browser_cfg = BrowserConfig(
        headless=True,            # operate without opening a browser
        enable_stealth=True,      # injects scripts to hide "WebDriver" flags
        user_agent_mode="random", # Switches identities so you aren't flagged as one "bot"
        use_persistent_context=True
        #text_mode=True,          # 'text_mode' disables images and heavy media at the browser level
        #light_mode=True
    )

    # 2. Performance-Focused Run Configuration
    run_cfg = CrawlerRunConfig(
        magic=True,               # Crucial: Simulates human scrolling/timing in the background
        # markdown_generator=md_generator,
        target_elements=["main", "article", ".product-detail", ".item-info"],
        wait_until="networkidle",
        delay_before_return_html=2.0,
        # ADDED JS_CODE: This clicks common "Read More" or "Ingredients" buttons
        js_code=[
            "document.querySelectorAll('a[data-action=\"a-expander-external\"]').forEach(b => b.click());",
            "document.querySelector('#importantInformation')?.scrollIntoView();"
        ],
        word_count_threshold=10,
        excluded_selector="#sp_detail, #vse-related-videos, #nav-belt", # Kill ads specifically
        excluded_tags=["nav", "footer", 
                       "header", "aside",
                       "script", "style"],   # Remove common web "noise" like headers, footers, and ads
        
        wait_for_images=False,    # Skip waiting for images to load (saves significant time)
        wait_for_timeout=30000
        )

    async with AsyncWebCrawler(config=browser_cfg) as crawler:
        # Crawl only—don't run an LLM strategy here for maximum speed
        result = await crawler.arun(
            url=target_url,
            config=run_cfg
        )
        content = result.markdown.fit_markdown if len(result.markdown.fit_markdown) > 200 else result.markdown.raw_markdown

        return {
            "scraped_content": content,
            "current_url_index": idx + 1
        }