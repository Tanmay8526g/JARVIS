"""Web search and page-reading tools."""
from __future__ import annotations
from urllib.parse import urlparse
import requests
from bs4 import BeautifulSoup
from ddgs import DDGS
from ..config import TIMEOUT, USER_AGENT, short

def web_search(query: str, max_results: int = 5) -> str:
    """Search DuckDuckGo and return result links."""
    try:
        limit = max(1, min(int(max_results), 10))
        rows = [{"title": r.get("title", ""), "snippet": r.get("body", ""), "url": r.get("href", "")} for r in DDGS(timeout=TIMEOUT).text(str(query), max_results=limit)]
        return short(rows) if rows else "No search results were found."
    except Exception as error:
        return f"Web search failed: {error}"

def read_webpage(url: str) -> str:
    """Read bounded text from an HTTP(S) page."""
    try:
        if urlparse(str(url)).scheme not in {"http", "https"}:
            return "Only http and https URLs are allowed."
        response = requests.get(str(url), headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
            tag.decompose()
        return short(" ".join(soup.get_text(" ", strip=True).split()))
    except Exception as error:
        return f"Webpage read failed: {error}"
