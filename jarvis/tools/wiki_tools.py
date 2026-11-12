"""Wikipedia tool."""
from urllib.parse import quote_plus
import requests
from ..config import TIMEOUT, USER_AGENT, short

def wikipedia_summary(topic: str) -> str:
    """Return a bounded Wikipedia summary and link."""
    try:
        response = requests.get(f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote_plus(str(topic).strip().replace(' ', '_'))}", headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT)
        response.raise_for_status()
        data = response.json()
        return short({"title": data.get("title"), "summary": data.get("extract"), "url": data.get("content_urls", {}).get("desktop", {}).get("page"), "source": "Wikipedia"})
    except Exception as error: return f"Wikipedia lookup failed: {error}"
