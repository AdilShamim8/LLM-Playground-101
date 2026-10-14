"""Web crawling subpackage."""

from data.crawling.common_crawl import (
    CommonCrawlConfig,
    CommonCrawlProcessor,
    WARCRecord,
)
from data.crawling.utils import (
    deduplicate_urls,
    extract_domain,
    is_valid_url,
    merge_crawl_results,
    normalize_url,
)
from data.crawling.web_crawler import CrawlConfig, CrawledPage, WebCrawler

__all__ = [
    "WebCrawler",
    "CrawlConfig",
    "CrawledPage",
    "CommonCrawlProcessor",
    "CommonCrawlConfig",
    "WARCRecord",
    "normalize_url",
    "is_valid_url",
    "extract_domain",
    "merge_crawl_results",
    "deduplicate_urls",
]
