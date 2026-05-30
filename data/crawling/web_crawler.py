"""
Production-level web crawler for data collection.
Implements polite crawling with rate limiting, robots.txt respect,
deduplication, and async processing.
"""

import asyncio
import hashlib
import re
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import aiohttp
import trafilatura
from bs4 import BeautifulSoup
from loguru import logger


@dataclass
class CrawlConfig:
    max_depth: int = 3
    max_pages: int = 5000
    delay: float = 1.0
    timeout: int = 30
    max_concurrent: int = 10
    user_agent: str = (
        "LLMPlaygroundBot/1.0 "
        "(Educational Research; "
        "+https://github.com/llm-playground)"
    )
    respect_robots: bool = True
    allowed_domains: list = field(default_factory=list)
    excluded_extensions: list = field(default_factory=lambda: [
        '.pdf', '.jpg', '.jpeg', '.png', '.gif',
        '.zip', '.tar', '.gz', '.mp3', '.mp4'
    ])


@dataclass
class CrawledPage:
    url: str
    title: str
    text: str
    html: str
    depth: int
    timestamp: float
    content_hash: str
    links: list[str]
    metadata: dict


class RobotsTxtCache:
    """Cache for robots.txt files to avoid repeated fetching."""

    def __init__(self):
        self._cache: dict[str, RobotFileParser] = {}
        self._lock = asyncio.Lock()

    async def can_fetch(
        self,
        url: str,
        user_agent: str,
        session: aiohttp.ClientSession
    ) -> bool:
        domain = urlparse(url).netloc
        async with self._lock:
            if domain not in self._cache:
                rp = RobotFileParser()
                robots_url = f"{urlparse(url).scheme}://{domain}/robots.txt"
                try:
                    async with session.get(
                        robots_url, timeout=aiohttp.ClientTimeout(total=10)
                    ) as resp:
                        if resp.status == 200:
                            content = await resp.text()
                            rp.parse(content.splitlines())
                        else:
                            rp.allow_all = True
                except Exception:
                    rp.allow_all = True
                self._cache[domain] = rp
        return self._cache[domain].can_fetch(user_agent, url)


class WebCrawler:
    """
    Async web crawler with politeness policies,
    deduplication, and structured text extraction.
    """

    def __init__(self, config: CrawlConfig):
        self.config = config
        self.visited_urls: set[str] = set()
        self.content_hashes: set[str] = set()
        self.robots_cache = RobotsTxtCache()
        self.pages: list[CrawledPage] = []
        self._semaphore = asyncio.Semaphore(config.max_concurrent)
        self._domain_last_access: dict[str, float] = {}

    def _normalize_url(self, url: str) -> str:
        parsed = urlparse(url)
        normalized = parsed._replace(fragment='').geturl()
        return normalized.rstrip('/')

    def _is_valid_url(self, url: str) -> bool:
        try:
            parsed = urlparse(url)
            if parsed.scheme not in ('http', 'https'):
                return False
            if any(
                url.lower().endswith(ext)
                for ext in self.config.excluded_extensions
            ):
                return False
            if (
                self.config.allowed_domains
                and parsed.netloc not in self.config.allowed_domains
            ):
                return False
            return True
        except Exception:
            return False

    def _extract_text(self, html: str, url: str) -> tuple[str, str]:
        """Extract clean text using trafilatura with BS4 fallback."""
        text = trafilatura.extract(
            html,
            include_comments=False,
            include_tables=True,
            no_fallback=False
        )
        if not text:
            soup = BeautifulSoup(html, 'html.parser')
            for tag in soup(['script', 'style', 'nav', 'footer']):
                tag.decompose()
            text = soup.get_text(separator=' ', strip=True)
            text = re.sub(r'\s+', ' ', text).strip()

        soup = BeautifulSoup(html, 'html.parser')
        title = soup.title.string if soup.title else ''
        return text or '', title or ''

    def _extract_links(self, html: str, base_url: str) -> list[str]:
        soup = BeautifulSoup(html, 'html.parser')
        links = []
        for anchor in soup.find_all('a', href=True):
            href = anchor['href']
            full_url = urljoin(base_url, href)
            normalized = self._normalize_url(full_url)
            if self._is_valid_url(normalized):
                links.append(normalized)
        return list(set(links))

    async def _respect_rate_limit(self, domain: str):
        now = time.time()
        last = self._domain_last_access.get(domain, 0)
        elapsed = now - last
        if elapsed < self.config.delay:
            await asyncio.sleep(self.config.delay - elapsed)
        self._domain_last_access[domain] = time.time()

    async def _fetch_page(
        self,
        url: str,
        depth: int,
        session: aiohttp.ClientSession
    ) -> Optional[CrawledPage]:
        async with self._semaphore:
            domain = urlparse(url).netloc

            if self.config.respect_robots:
                can_fetch = await self.robots_cache.can_fetch(
                    url, self.config.user_agent, session
                )
                if not can_fetch:
                    logger.debug(f"Robots.txt disallows: {url}")
                    return None

            await self._respect_rate_limit(domain)

            try:
                headers = {'User-Agent': self.config.user_agent}
                timeout = aiohttp.ClientTimeout(total=self.config.timeout)

                async with session.get(
                    url, headers=headers, timeout=timeout
                ) as response:
                    if response.status != 200:
                        return None

                    content_type = response.headers.get(
                        'content-type', ''
                    ).lower()
                    if 'text/html' not in content_type:
                        return None

                    html = await response.text(errors='replace')
                    text, title = self._extract_text(html, url)

                    if not text or len(text) < 100:
                        return None

                    content_hash = hashlib.sha256(
                        text.encode()
                    ).hexdigest()
                    if content_hash in self.content_hashes:
                        logger.debug(f"Duplicate content: {url}")
                        return None

                    self.content_hashes.add(content_hash)
                    links = self._extract_links(html, url)

                    page = CrawledPage(
                        url=url,
                        title=title,
                        text=text,
                        html=html,
                        depth=depth,
                        timestamp=time.time(),
                        content_hash=content_hash,
                        links=links,
                        metadata={
                            'content_type': content_type,
                            'status_code': response.status,
                            'size': len(html)
                        }
                    )
                    logger.info(
                        f"Crawled [{depth}]: {url} "
                        f"({len(text)} chars)"
                    )
                    return page

            except asyncio.TimeoutError:
                logger.warning(f"Timeout: {url}")
            except Exception as e:
                logger.error(f"Error crawling {url}: {e}")
            return None

    async def crawl(self, seed_urls: list[str]) -> list[CrawledPage]:
        """BFS crawl starting from seed URLs."""
        queue = deque(
            [(self._normalize_url(u), 0) for u in seed_urls]
        )
        connector = aiohttp.TCPConnector(
            limit=self.config.max_concurrent,
            ssl=False
        )

        async with aiohttp.ClientSession(connector=connector) as session:
            while queue and len(self.pages) < self.config.max_pages:
                batch = []
                while queue and len(batch) < self.config.max_concurrent:
                    url, depth = queue.popleft()
                    if url in self.visited_urls:
                        continue
                    if depth > self.config.max_depth:
                        continue
                    self.visited_urls.add(url)
                    batch.append((url, depth))

                if not batch:
                    break

                tasks = [
                    self._fetch_page(url, depth, session)
                    for url, depth in batch
                ]
                results = await asyncio.gather(*tasks, return_exceptions=True)

                for result in results:
                    if isinstance(result, CrawledPage):
                        self.pages.append(result)
                        if result.depth < self.config.max_depth:
                            for link in result.links:
                                if link not in self.visited_urls:
                                    queue.append(
                                        (link, result.depth + 1)
                                    )

        logger.info(
            f"Crawl complete. Pages: {len(self.pages)}, "
            f"Visited: {len(self.visited_urls)}"
        )
        return self.pages

    def save_results(self, output_path: str):
        import json
        import os
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            for page in self.pages:
                record = {
                    'url': page.url,
                    'title': page.title,
                    'text': page.text,
                    'depth': page.depth,
                    'timestamp': page.timestamp,
                    'content_hash': page.content_hash,
                    'metadata': page.metadata
                }
                f.write(json.dumps(record) + '\n')
        logger.info(f"Saved {len(self.pages)} pages to {output_path}")