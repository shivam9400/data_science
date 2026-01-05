from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode

async def crawl_node(state):
    if not state.get('found_urls'):
        return {"error": "No URLs found"}

    # 1. Light Browser Configuration
    browser_cfg = BrowserConfig(
        headless=True,
        # 'text_mode' disables images and heavy media at the browser level
        text_mode=True,
        light_mode=True
    )

    # 2. Performance-Focused Run Configuration
    run_cfg = CrawlerRunConfig(
        # Skip waiting for images to load (saves significant time)
        wait_for_images=False,
        # Remove common web "noise" like headers, footers, and ads
        excluded_tags=["nav", "footer", "header", "aside"],
        # Use cache if we've seen this URL before
        cache_mode=CacheMode.ENABLED,
        # Only extract the main content area
        word_count_threshold=10
    )

    async with AsyncWebCrawler(config=browser_cfg) as crawler:
        # Crawl only—don't run an LLM strategy here for maximum speed
        result = await crawler.arun(
            url=state['found_urls'][0],
            config=run_cfg
        )
        
        # Use fit_markdown: It's a pruned, smaller version of the text
        # that is much faster for your 'classify_node' to process later.
        return {
            "scraped_content": result.markdown.fit_markdown or result.markdown,
            "success": result.success
        }