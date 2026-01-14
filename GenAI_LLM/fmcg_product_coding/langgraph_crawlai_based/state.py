''' tracks the "memory" of the agent as it moves through the graph. '''
from typing import TypedDict, List, Optional
from schema import FMCGProduct

class FMCGState(TypedDict):
    raw_input: str
    found_urls: List[str]
    current_url_index: int
    scraped_content: str
    final_output: Optional[FMCGProduct]
    error: Optional[str]
    thought: Optional[str]