'''  Defines the "Target" of your coding here. '''

from pydantic import BaseModel, Field

class FMCGProduct(BaseModel):
    brand: str = Field(description="The brand name, e.g., Coca-Cola")
    category: str = Field(description="High-level category, e.g., Beverages")
    confidence: float = Field(description="Score from 0 to 1 on accuracy")