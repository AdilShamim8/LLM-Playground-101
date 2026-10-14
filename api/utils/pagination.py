"""
Pagination utilities for list endpoints.

WHY THIS EXISTS:
List endpoints that return all records at once will break at scale:
- 100K users → 100K records in one response → OOM, timeout
- No way for client to request "page 2"

This module provides a standard pagination pattern that all list
endpoints should use.
"""

from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar("T")

DEFAULT_PAGE_SIZE = 20
MAX_PAGE_SIZE = 100


@dataclass
class PaginatedResponse(Generic[T]):
    """Standard paginated response envelope.

    All list endpoints should return this structure so the client
    always knows how to get the next page.
    """

    items: list[T]
    total: int
    page: int
    page_size: int
    total_pages: int
    has_next: bool
    has_prev: bool

    def to_dict(self) -> dict:
        """Convert to dict for JSON response."""
        return {
            "items": self.items,
            "pagination": {
                "total": self.total,
                "page": self.page,
                "page_size": self.page_size,
                "total_pages": self.total_pages,
                "has_next": self.has_next,
                "has_prev": self.has_prev,
            },
        }


def paginate(
    items: list[T],
    page: int = 1,
    page_size: int = DEFAULT_PAGE_SIZE,
) -> PaginatedResponse[T]:
    """Paginate a list of items (in-memory pagination).

    Args:
        items: Full list of items (already loaded in memory)
        page: 1-indexed page number
        page_size: Items per page (capped at MAX_PAGE_SIZE)

    Returns:
        PaginatedResponse with the current page's items and metadata.
    """
    page = max(1, page)
    page_size = max(1, min(page_size, MAX_PAGE_SIZE))

    total = len(items)
    total_pages = (total + page_size - 1) // page_size if total > 0 else 1
    page = min(page, total_pages)

    start = (page - 1) * page_size
    end = start + page_size
    page_items = items[start:end]

    return PaginatedResponse(
        items=page_items,
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
        has_next=page < total_pages,
        has_prev=page > 1,
    )


def paginate_query(query, page: int = 1, page_size: int = DEFAULT_PAGE_SIZE):
    """Paginate a SQLAlchemy query (database-level pagination).

    More efficient than paginate() for large datasets because it
    only fetches the current page from the database using LIMIT/OFFSET.

    Args:
        query: SQLAlchemy query object
        page: 1-indexed page number
        page_size: Items per page

    Returns:
        PaginatedResponse with items and pagination metadata.
    """
    page = max(1, page)
    page_size = max(1, min(page_size, MAX_PAGE_SIZE))

    total = query.count()
    total_pages = (total + page_size - 1) // page_size if total > 0 else 1
    page = min(page, total_pages)

    offset = (page - 1) * page_size
    items = query.offset(offset).limit(page_size).all()

    return PaginatedResponse(
        items=items,
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
        has_next=page < total_pages,
        has_prev=page > 1,
    )
