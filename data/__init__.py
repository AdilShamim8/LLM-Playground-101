"""
LLM Playground Data Package.

NOTE: Submodules are NOT eagerly imported here to avoid pulling in heavy
optional dependencies (warcio, trafilatura, fasttext) when only the
tokenizer or cleaners are needed.

Import directly from submodules instead:
    from data.tokenization.bpe_tokenizer import BPEConfig, ByteLevelBPETokenizer
    from data.cleaning.refined_web import RefinedWebCleaner, RefinedWebConfig
    from data.crawling.web_crawler import WebCrawler, CrawlConfig

The __all__ list below documents what's available, but imports are lazy.
"""

__all__ = [
    # Crawling
    "WebCrawler",
    "CrawlConfig",
    "CommonCrawlProcessor",
    "CommonCrawlConfig",
    # Cleaning
    "TextQualityFilter",
    "QualityConfig",
    "ContentFilter",
    "RefinedWebCleaner",
    "RefinedWebConfig",
    "DolmaTextCleaner",
    "DolmaConfig",
    "FineWebCleaner",
    "FineWebConfig",
    # Tokenization
    "ByteLevelBPETokenizer",
    "BPEConfig",
]


def __getattr__(name: str):
    """Lazy imports — only import the submodule when the name is accessed.

    This avoids importing warcio/trafilatura/fasttext when you only need
    the BPE tokenizer or a cleaner.
    """
    # Map names to their source modules
    _import_map = {
        # Crawling
        "WebCrawler": "data.crawling.web_crawler",
        "CrawlConfig": "data.crawling.web_crawler",
        "CommonCrawlProcessor": "data.crawling.common_crawl",
        "CommonCrawlConfig": "data.crawling.common_crawl",
        # Cleaning
        "TextQualityFilter": "data.cleaning.quality_filters",
        "QualityConfig": "data.cleaning.quality_filters",
        "ContentFilter": "data.cleaning.quality_filters",
        "RefinedWebCleaner": "data.cleaning.refined_web",
        "RefinedWebConfig": "data.cleaning.refined_web",
        "DolmaTextCleaner": "data.cleaning.dolma_cleaner",
        "DolmaConfig": "data.cleaning.dolma_cleaner",
        "FineWebCleaner": "data.cleaning.fineweb_cleaner",
        "FineWebConfig": "data.cleaning.fineweb_cleaner",
        # Tokenization
        "ByteLevelBPETokenizer": "data.tokenization.bpe_tokenizer",
        "BPEConfig": "data.tokenization.bpe_tokenizer",
    }

    if name in _import_map:
        import importlib

        module = importlib.import_module(_import_map[name])
        return getattr(module, name)

    raise AttributeError(f"module 'data' has no attribute '{name}'")
