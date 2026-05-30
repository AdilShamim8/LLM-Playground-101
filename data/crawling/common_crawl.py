"""
Common Crawl data downloader and processor.
Fetches WARC files from Common Crawl and extracts text content.
"""

import gzip
import io
import json
import os
from dataclasses import dataclass
from typing import Iterator

import requests
import warcio
from loguru import logger
from warcio.archiveiterator import ArchiveIterator


@dataclass
class CommonCrawlConfig:
    base_url: str = "https://data.commoncrawl.org"
    crawl_id: str = "CC-MAIN-2023-50"
    num_segments: int = 5
    max_pages_per_segment: int = 10000
    output_dir: str = "./data/raw/common_crawl"
    languages: list = None

    def __post_init__(self):
        if self.languages is None:
            self.languages = ["en"]


@dataclass
class WARCRecord:
    url: str
    text: str
    language: str
    timestamp: str
    content_hash: str
    warc_file: str


class CommonCrawlProcessor:
    """
    Downloads and processes Common Crawl WARC files.
    Handles wet files (pre-extracted text) and warc files.
    """

    def __init__(self, config: CommonCrawlConfig):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'LLMPlayground/1.0 Research'
        })
        os.makedirs(config.output_dir, exist_ok=True)

    def get_crawl_index(self) -> list[str]:
        """Fetch list of WARC segment paths."""
        index_url = (
            f"{self.config.base_url}/crawl-data/"
            f"{self.config.crawl_id}/wet.paths.gz"
        )
        logger.info(f"Fetching index from {index_url}")

        response = self.session.get(index_url, stream=True)
        response.raise_for_status()

        compressed = io.BytesIO(response.content)
        with gzip.open(compressed, 'rt', encoding='utf-8') as f:
            paths = [line.strip() for line in f if line.strip()]

        logger.info(f"Found {len(paths)} WET segments")
        return paths[:self.config.num_segments]

    def stream_wet_file(
        self, path: str
    ) -> Iterator[WARCRecord]:
        """Stream records from a WET file."""
        url = f"{self.config.base_url}/{path}"
        logger.info(f"Streaming WET: {url}")

        response = self.session.get(url, stream=True)
        response.raise_for_status()

        count = 0
        stream = io.BytesIO(response.content)

        try:
            for record in ArchiveIterator(stream):
                if record.rec_type != 'conversion':
                    continue
                if count >= self.config.max_pages_per_segment:
                    break

                target_uri = record.rec_headers.get_header(
                    'WARC-Target-URI'
                )
                timestamp = record.rec_headers.get_header(
                    'WARC-Date'
                )
                language = record.rec_headers.get_header(
                    'WARC-Identified-Content-Language', 'unknown'
                )

                if not any(
                    lang in language
                    for lang in self.config.languages
                ):
                    continue

                try:
                    content = record.content_stream().read()
                    text = content.decode('utf-8', errors='replace')
                    text = text.strip()
                except Exception as e:
                    logger.warning(
                        f"Decode error for {target_uri}: {e}"
                    )
                    continue

                if len(text) < 200:
                    continue

                import hashlib
                content_hash = hashlib.sha256(
                    text.encode()
                ).hexdigest()

                yield WARCRecord(
                    url=target_uri or '',
                    text=text,
                    language=language,
                    timestamp=timestamp or '',
                    content_hash=content_hash,
                    warc_file=path
                )
                count += 1

        except Exception as e:
            logger.error(f"Error streaming {path}: {e}")

    def process_and_save(self, output_file: str):
        """Download segments and save extracted records."""
        paths = self.get_crawl_index()
        total = 0
        seen_hashes: set[str] = set()

        with open(output_file, 'w', encoding='utf-8') as out:
            for path in paths:
                try:
                    for record in self.stream_wet_file(path):
                        if record.content_hash in seen_hashes:
                            continue
                        seen_hashes.add(record.content_hash)

                        out.write(json.dumps({
                            'url': record.url,
                            'text': record.text,
                            'language': record.language,
                            'timestamp': record.timestamp,
                            'content_hash': record.content_hash,
                            'source': 'common_crawl',
                            'warc_file': record.warc_file
                        }) + '\n')
                        total += 1

                        if total % 10000 == 0:
                            logger.info(
                                f"Processed {total} records"
                            )
                except Exception as e:
                    logger.error(f"Failed segment {path}: {e}")
                    continue

        logger.info(
            f"Done. Total records: {total}, "
            f"Output: {output_file}"
        )
        return total

    def fetch_cdx_records(
        self,
        domain: str,
        limit: int = 1000
    ) -> list[dict]:
        """
        Use CDX API to find URLs from a specific domain
        in Common Crawl.
        """
        cdx_url = "http://index.commoncrawl.org/"
        cdx_url += f"{self.config.crawl_id}-index"

        params = {
            'url': f'{domain}/*',
            'output': 'json',
            'limit': limit,
            'filter': 'status:200'
        }

        records = []
        response = self.session.get(cdx_url, params=params)

        for line in response.text.splitlines():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

        logger.info(
            f"Found {len(records)} CDX records for {domain}"
        )
        return records