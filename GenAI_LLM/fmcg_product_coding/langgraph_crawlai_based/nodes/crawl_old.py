''' high-performance "Reader" using Crawl4AI '''

from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, LLMConfig
from crawl4ai.extraction_strategy import LLMExtractionStrategy
from schema import FMCGProduct

async def crawl_node_(state):
    if not state['found_urls']:
        return {"error": "No URLs found"}
    
    async with AsyncWebCrawler() as crawler:
        # We crawl the first relevant URL found
        result = await crawler.arun(url=state['found_urls'][0])
        return {"scraped_content": result.markdown}
    
async def crawl_node(state):
    # Configure strategy for local Ollama
    llm_strategy = LLMExtractionStrategy(
        llm_config=LLMConfig(
            provider="ollama/llama3.2", # Use the prefix 'ollama/'
            base_url="http://localhost:11434" # Note: no /v1 for LiteLLM/Crawl4AI usually
        ),
        schema=FMCGProduct.model_json_schema(),
        instruction="Extract brand, category, and size from this text."
    )

    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(
            url=state['found_urls'][0],
            config=CrawlerRunConfig(extraction_strategy=llm_strategy)
        )
        return {"scraped_content": result.markdown, "extracted_data": result.extracted_content}