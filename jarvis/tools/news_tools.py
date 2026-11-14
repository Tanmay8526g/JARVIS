"""Public RSS news tool."""
import json
from urllib.parse import quote_plus
import feedparser
import requests
from ..config import TIMEOUT, USER_AGENT, short

def get_news(topic: str = "", max_items: int = 5) -> str:
    """Read RSS headlines with a source and link for every item."""
    try:
        limit = max(1, min(int(max_items), 10))
        urls = [f"https://news.google.com/rss/search?q={quote_plus(str(topic))}&hl=en-IN&gl=IN&ceid=IN:en"] if topic else ["https://feeds.bbci.co.uk/news/world/rss.xml", "https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml"]
        rows = []
        for url in urls:
            response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT)
            response.raise_for_status()
            feed = feedparser.parse(response.content)
            for entry in feed.entries[:limit]:
                source = entry.get("source", {})
                if isinstance(source, dict):
                    source = source.get("title", "")
                rows.append({"headline": entry.get("title", ""), "source": source or feed.feed.get("title", "RSS"), "published": entry.get("published", entry.get("updated", "")), "link": entry.get("link", "")})
        return json.dumps(rows[:limit], ensure_ascii=False) if rows else "No news items were found."
    except Exception as error: return f"News lookup failed: {error}"
