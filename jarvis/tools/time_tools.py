"""Local time tools."""
from datetime import datetime

def get_current_datetime() -> str:
    """Return local weekday, date, time, and timezone."""
    return datetime.now().astimezone().strftime("%A, %d %B %Y, %I:%M:%S %p (%Z)")
