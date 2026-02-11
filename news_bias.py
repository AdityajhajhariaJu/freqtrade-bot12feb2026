import re
import json
from pathlib import Path
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from urllib.request import urlopen
from xml.etree import ElementTree as ET

RSS_URL = "https://www.binance.com/en/support/announcement/rss"
STATE_PATH = Path("/opt/multi-strat-engine/reports/news_bias_state.json")

POS_KEYWORDS = [
    "listing", "launch", "partnership", "support", "integration",
    "airdrop", "staking", "upgrade", "reward", "new trading pair"
]
NEG_KEYWORDS = [
    "delist", "suspend", "hack", "exploit", "outage", "lawsuit",
    "security", "incident", "maintenance", "regulatory", "investigation"
]


def _fetch_rss():
    with urlopen(RSS_URL, timeout=10) as r:
        return r.read()


def _parse_items(xml_bytes):
    root = ET.fromstring(xml_bytes)
    channel = root.find("channel")
    if channel is None:
        return []
    items = []
    for it in channel.findall("item"):
        title = (it.findtext("title") or "").strip()
        pub = (it.findtext("pubDate") or "").strip()
        items.append({"title": title, "pubDate": pub})
    return items


def _score_title(title: str, pos_bias: float, neg_bias: float):
    t = title.lower()
    pos = any(k in t for k in POS_KEYWORDS)
    neg = any(k in t for k in NEG_KEYWORDS)
    if neg and not pos:
        return -neg_bias
    if pos and not neg:
        return pos_bias
    if neg and pos:
        return -neg_bias  # negative wins
    return 0.0


def _extract_assets(title: str, bases: list[str]):
    found = set()
    up = title.upper()
    for b in bases:
        if re.search(rf"\b{re.escape(b)}\b", up):
            found.add(b)
    return found


def get_news_bias(pairs, pos_bias=0.03, neg_bias=0.04, half_life_hours=2.0):
    bases = [p.replace("USDT", "").replace("/USDT", "").upper() for p in pairs]
    now = datetime.now(timezone.utc)
    bias = {p: 0.0 for p in pairs}
    try:
        items = _parse_items(_fetch_rss())
    except Exception:
        return bias

    for it in items[:80]:
        title = it.get("title", "")
        pub = it.get("pubDate", "")
        try:
            dt = parsedate_to_datetime(pub).astimezone(timezone.utc)
        except Exception:
            dt = now
        age_hours = max(0.0, (now - dt).total_seconds() / 3600.0)
        decay = 0.5 ** (age_hours / half_life_hours) if half_life_hours else 1.0
        score = _score_title(title, pos_bias, neg_bias)
        if score == 0:
            continue
        assets = _extract_assets(title, bases)
        if not assets:
            continue
        for b in assets:
            pair = f"{b}USDT"
            if pair in bias:
                bias[pair] += score * decay

    # clamp and persist state
    for p in bias:
        if bias[p] > pos_bias:
            bias[p] = pos_bias
        if bias[p] < -neg_bias:
            bias[p] = -neg_bias

    try:
        STATE_PATH.write_text(json.dumps({"ts": now.isoformat(), "bias": bias}, indent=2))
    except Exception:
        pass
    return bias
