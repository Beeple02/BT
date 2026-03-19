"""
data_utils.py — Candle normalization and data helpers. No Flask.
"""
from datetime import datetime, timedelta, timezone


def norm_candles(candles, days=None):
    """Sort oldest-first, fix Atlas date formats, enforce days cutoff."""
    cutoff = None
    if days:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=int(days))).strftime("%Y-%m-%d")
    out = []
    for cv in candles:
        d = cv.get("date") or cv.get("timestamp") or cv.get("ts") or ""
        if isinstance(d, (int, float)):
            try: d = datetime.fromtimestamp(d, tz=timezone.utc).strftime("%Y-%m-%d")
            except: d = ""
        d = str(d) if d else ""
        if d and len(d)==8 and d[2]=="-" and d[5]=="-":
            p = d.split("-"); d = f"20{p[2]}-{p[1]}-{p[0]}"
        if d and len(d)>10 and "T" in d:
            d = d[:10]
        if cutoff and d and d < cutoff: continue
        out.append({**cv, "date": d})
    out.sort(key=lambda x: x.get("date",""))
    return out


def build_candles_from_history(pts, days=None):
    """Convert Atlas price_history [{id,price,timestamp,volume}] → daily OHLCV candles."""
    seen_ids = set(); unique = []
    for pt in (pts or []):
        pid = pt.get("id")
        if pid is not None:
            if pid in seen_ids: continue
            seen_ids.add(pid)
        unique.append(pt)

    by_day = {}
    for pt in unique:
        ts_raw = pt.get("timestamp") or pt.get("ts") or ""
        if not ts_raw: continue
        try:
            if isinstance(ts_raw, (int, float)):
                dt = datetime.fromtimestamp(ts_raw, tz=timezone.utc)
            else:
                dt = datetime.fromisoformat(str(ts_raw).replace("Z", "+00:00"))
            ds = dt.strftime("%Y-%m-%d")
        except: continue
        px = float(pt.get("price") or 0); vol = int(pt.get("volume") or 0)
        if not px: continue
        if ds not in by_day:
            by_day[ds] = {"date":ds,"open":px,"high":px,"low":px,"close":px,"volume":vol}
        else:
            c = by_day[ds]; c["high"]=max(c["high"],px); c["low"]=min(c["low"],px)
            c["close"]=px; c["volume"]+=vol

    candles = sorted(by_day.values(), key=lambda x: x["date"])
    if days and candles:
        cutoff = (datetime.now(timezone.utc)-timedelta(days=int(days))).strftime("%Y-%m-%d")
        trimmed = [c for c in candles if c["date"]>=cutoff]
        if trimmed: candles = trimmed
    return candles


def fetch_candles(ticker, days, atlas_get):
    """Unified candle fetch: price_history for TSE, ohlcv for NER."""
    if ticker.upper().startswith("TSE:"):
        _, raw = atlas_get(f"/history/{ticker}", params={"days":365,"limit":5000}, ttl=120)
        pts = raw.get("data",[]) if isinstance(raw,dict) else raw
        return build_candles_from_history(pts, days)
    else:
        _, raw = atlas_get(f"/analytics/ohlcv/{ticker}", params={"days":days}, ttl=120)
        return norm_candles(raw.get("candles",[]), days=days) if isinstance(raw,dict) else []


def fetch_live_price(ticker, atlas_get):
    """Get live price for any ticker. Returns float or None."""
    s, d = atlas_get(f"/securities/{ticker}", ttl=60)
    if s==200 and d.get("market_price") is not None:
        return float(d["market_price"])
    s2, d2 = atlas_get(f"/price/{ticker}", ttl=15)
    if s2==200 and d2.get("market_price") is not None:
        return float(d2["market_price"])
    return None
