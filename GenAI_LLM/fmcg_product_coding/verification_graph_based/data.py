"""
Synthetic input data generator
"""

def get_sample_product():
    """
    Returns a single synthetic FMCG product record
    """
    return {
        "description": "Parle-G original glucose biscuits, 250g pack",
        "predicted": {
            "category": "Biscuits",
            "brand": "Parle",
            "age_group": "All"
        },
        "image_url": "mock_image.jpg",
        "web_text": "Parle-G is a popular Indian brand of glucose biscuits manufactured by Parle Products."
    }
