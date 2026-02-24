#!/usr/bin/env python3
"""
Bloomberg Terminal — NER Exchange
Architecture: Flask proxy backend + split HTML page files
Run: python app.py → http://localhost:5000
"""
import os, time, math, statistics
import requests
from flask import Flask, jsonify, request, render_template

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_TMPL_DIR  = os.path.join(_BASE_DIR, "templates")
_STAT_DIR  = os.path.join(_BASE_DIR, "static")
print(f"[NER Terminal] Base dir : {_BASE_DIR}")
print(f"[NER Terminal] Templates: {_TMPL_DIR}  exists={os.path.isdir(_TMPL_DIR)}")
print(f"[NER Terminal] Static   : {_STAT_DIR}  exists={os.path.isdir(_STAT_DIR)}")
app = Flask(
    __name__,
    template_folder=_TMPL_DIR,
    static_folder=_STAT_DIR,
)

NER_BASE = "http://150.230.117.88:8082"
API_KEY  = os.environ.get("NER_API_KEY", "ner_l7nBYB_pFwRvVPcW2rum-UeI9qrJh2BWekgG__BDeYk")
AUTH_H   = {"Content-Type": "application/json", "X-API-Key": API_KEY}
PUB_H    = {"Content-Type": "application/json"}
_cache: dict = {}

# ── Cache helpers ─────────────────────────────────────────────────────────────
# Only caches 200 responses — errors are never stored so stale 404s can't persist.
# Auth endpoints (orders, funds, transactions) use ttl=0 to always bypass cache
# and hit NER live — these are user-specific real-time state.
def cached_get(path, params=None, auth=False, ttl=15):
    if ttl == 0:
        # Live/uncached path — always hit NER directly
        try:
            r = requests.get(f"{NER_BASE}{path}", headers=AUTH_H if auth else PUB_H,
                             params=params, timeout=10)
            return r.status_code, r.json()
        except Exception as ex:
            return 503, {"detail": str(ex)}
    key = path + str(params) + str(auth)
    e = _cache.get(key)
    if e and time.time() - e["ts"] < ttl:
        return e["s"], e["d"]
    try:
        r = requests.get(f"{NER_BASE}{path}", headers=AUTH_H if auth else PUB_H,
                         params=params, timeout=10)
        result = (r.status_code, r.json())
    except Exception as ex:
        result = (503, {"detail": str(ex)})
    if result[0] == 200:
        _cache[key] = {"ts": time.time(), "s": result[0], "d": result[1]}
    return result

# Bulk orderbook cache — one call, filter by ticker in Python
_ob_cache = {"ts": 0, "data": None}
_OB_TTL = 4  # seconds

def get_all_orderbooks():
    """Single /orderbook call that returns all tickers. Shared across all routes."""
    global _ob_cache
    if time.time() - _ob_cache["ts"] < _OB_TTL and _ob_cache["data"] is not None:
        return _ob_cache["data"]
    try:
        r = requests.get(f"{NER_BASE}/orderbook", headers=PUB_H, timeout=10)
        if r.status_code == 200:
            data = r.json()
            _ob_cache = {"ts": time.time(), "data": data}
            return data
    except Exception:
        pass
    return _ob_cache["data"]  # stale is better than nothing

def get_ticker_orderbook(ticker):
    """Get single-ticker orderbook from the bulk cache."""
    all_ob = get_all_orderbooks()
    if not isinstance(all_ob, list):
        return all_ob  # might be a dict already (single ticker response)
    return next((o for o in all_ob if o.get("ticker") == ticker), None)

# Bulk shareholders cache — one call covers all tickers
# Per-ticker shareholders cache — NER requires ticker param, no bulk endpoint
_sh_ticker_cache: dict = {}   # ticker → {"ts": float, "data": list}
_SH_TICKER_TTL = 90           # seconds

def get_ticker_shareholders(ticker):
    """Fetch shareholders for one ticker, with 90s cache per ticker."""
    e = _sh_ticker_cache.get(ticker)
    if e and time.time() - e["ts"] < _SH_TICKER_TTL:
        return e["data"]
    try:
        # Try with X-API-Key for better compatibility
        r = requests.get(f"{NER_BASE}/shareholders", headers=AUTH_H,
                         params={"ticker": ticker}, timeout=10)
        if r.status_code == 200:
            data = r.json()
            if isinstance(data, list):
                _sh_ticker_cache[ticker] = {"ts": time.time(), "data": data}
                return data
            # NER returned non-list (e.g. {"detail":...}) — log and skip
            print(f"[shareholders/{ticker}] unexpected response: {str(data)[:120]}")
        else:
            print(f"[shareholders/{ticker}] HTTP {r.status_code}: {r.text[:120]}")
    except Exception as ex:
        print(f"[shareholders/{ticker}] exception: {ex}")
    # Return stale data if available rather than nothing
    return _sh_ticker_cache.get(ticker, {}).get("data", [])

def ner_post(path, payload):
    try:
        r = requests.post(f"{NER_BASE}{path}", headers=AUTH_H, json=payload, timeout=10)
        return r.status_code, r.json()
    except Exception as e:
        return 503, {"detail": str(e)}

# ── Pass-through proxies ──────────────────────────────────────────────────────
@app.route("/api/securities")
def securities():
    s, d = cached_get("/securities", ttl=20); return jsonify(d), s

@app.route("/api/market_price/<ticker>")
def market_price(ticker):
    s, d = cached_get(f"/market_price/{ticker}", ttl=5); return jsonify(d), s

@app.route("/api/shareholders")
def shareholders():
    tk = request.args.get("ticker","")
    if tk:
        d = get_ticker_shareholders(tk)
        return jsonify(d), 200
    # No ticker — return empty (NER requires ticker param)
    return jsonify([]), 200

@app.route("/api/orderbook")
def orderbook():
    t = request.args.get("ticker")
    all_ob = get_all_orderbooks()
    if all_ob is None:
        return jsonify({"detail": "Orderbook unavailable"}), 503
    if t:
        ob = next((o for o in all_ob if isinstance(all_ob, list) and o.get("ticker") == t), all_ob)
        return jsonify(ob), 200
    return jsonify(all_ob), 200

@app.route("/api/analytics/price_history/<ticker>")
def price_history(ticker):
    s, d = cached_get(f"/analytics/price_history/{ticker}",
                      params={"days": request.args.get("days", 30)}, ttl=60)
    return jsonify(d), s

@app.route("/api/analytics/ohlcv/<ticker>")
def ohlcv(ticker):
    s, d = cached_get(f"/analytics/ohlcv/{ticker}",
                      params={"days": request.args.get("days", 30)}, ttl=60)
    return jsonify(d), s

@app.route("/api/portfolio")
def portfolio():
    s, d = cached_get("/portfolio", auth=True, ttl=0); return jsonify(d), s  # always live

# ── Trading ───────────────────────────────────────────────────────────────────
@app.route("/api/orders/buy_limit",   methods=["POST"])
def buy_limit():   s,d=ner_post("/orders/buy_limit",   request.json); return jsonify(d),s
@app.route("/api/orders/sell_limit",  methods=["POST"])
def sell_limit():  s,d=ner_post("/orders/sell_limit",  request.json); return jsonify(d),s
@app.route("/api/orders/buy_market",  methods=["POST"])
def buy_market():  s,d=ner_post("/orders/buy_market",  request.json); return jsonify(d),s
@app.route("/api/orders/sell_market", methods=["POST"])
def sell_market(): s,d=ner_post("/orders/sell_market", request.json); return jsonify(d),s

# ── Analytics math helpers ────────────────────────────────────────────────────
def _sma(s, n):
    return [None if i < n-1 else round(sum(s[i-n+1:i+1])/n, 6) for i in range(len(s))]

def _ema(s, n):
    k = 2/(n+1); r = [s[0]]
    for v in s[1:]: r.append(round(v*k + r[-1]*(1-k), 6))
    return r

def _rsi(s, p=14):
    result = [None]*p
    for i in range(p, len(s)):
        g = [max(s[j]-s[j-1], 0) for j in range(i-p+1, i+1)]
        l = [max(s[j-1]-s[j], 0) for j in range(i-p+1, i+1)]
        ag, al = sum(g)/p, sum(l)/p
        result.append(round(100.0 if al==0 else 100-100/(1+ag/al), 2))
    return result

def _atr(highs, lows, closes, p=14):
    trs = []
    for i in range(1, len(closes)):
        trs.append(max(highs[i]-lows[i], abs(highs[i]-closes[i-1]), abs(lows[i]-closes[i-1])))
    atr = [None]
    for i in range(len(trs)):
        if i < p-1: atr.append(None)
        else: atr.append(round(sum(trs[max(0,i-p+1):i+1])/p, 6))
    return atr

def _bollinger(closes, p=20, k=2):
    upper, mid, lower = [], [], []
    for i in range(len(closes)):
        if i < p-1: upper.append(None); mid.append(None); lower.append(None)
        else:
            window = closes[i-p+1:i+1]
            m = sum(window)/p
            sd = (sum((x-m)**2 for x in window)/p)**0.5
            mid.append(round(m,6)); upper.append(round(m+k*sd,6)); lower.append(round(m-k*sd,6))
    return upper, mid, lower

def _macd(closes, fast=12, slow=26, signal=9):
    f_ema = _ema(closes, fast); s_ema = _ema(closes, slow)
    macd_line = [round(f-s, 6) for f,s in zip(f_ema, s_ema)]
    sig_line  = _ema(macd_line, signal)
    histogram = [round(m-s, 6) for m,s in zip(macd_line, sig_line)]
    return macd_line, sig_line, histogram

def _vwap_series(highs, lows, closes, volumes):
    typical = [(h+l+c)/3 for h,l,c in zip(highs,lows,closes)]
    cum_tv = 0; cum_v = 0; result = []
    for tp, v in zip(typical, volumes):
        cum_tv += tp*v; cum_v += v
        result.append(round(cum_tv/cum_v, 6) if cum_v else None)
    return result

def _ann_vol(closes):
    if len(closes) < 2: return 0
    rets = [(closes[i]-closes[i-1])/closes[i-1] for i in range(1,len(closes))]
    if len(rets) < 2: return 0
    return round(statistics.stdev(rets) * (252**0.5) * 100, 2)

def _max_drawdown(series):
    peak = series[0]; max_dd = 0
    for v in series:
        if v > peak: peak = v
        dd = (peak - v) / peak if peak > 0 else 0
        if dd > max_dd: max_dd = dd
    return round(max_dd * 100, 2)

def _sharpe(closes, rf=0):
    if len(closes) < 2: return 0
    rets = [(closes[i]-closes[i-1])/closes[i-1] for i in range(1,len(closes))]
    if len(rets) < 2: return 0
    avg = sum(rets)/len(rets)
    std = statistics.stdev(rets)
    return round((avg - rf/252) / std * (252**0.5), 3) if std > 0 else 0

# ── TICKER STATS (fixed + enhanced) ──────────────────────────────────────────

def _mean_reversion(closes):
    """Z-score of current price vs SMA20 — high absolute value = mean reversion opportunity."""
    if len(closes) < 20: return None
    sma = sum(closes[-20:]) / 20
    std = (sum((c - sma)**2 for c in closes[-20:]) / 20) ** 0.5
    if std == 0: return None
    return round((closes[-1] - sma) / std, 2)

def _downside_vol(closes):
    """Annualized downside deviation (Sortino denominator)."""
    if len(closes) < 2: return None
    rets = [(closes[i]-closes[i-1])/closes[i-1] for i in range(1, len(closes))]
    neg = [r for r in rets if r < 0]
    if not neg: return 0.0
    return round((sum(r**2 for r in neg)/len(neg))**0.5 * (252**0.5) * 100, 2)

@app.route("/api/ticker_stats/<ticker>")
def ticker_stats(ticker):
    days = int(request.args.get("days", 60))
    s, raw = cached_get(f"/analytics/ohlcv/{ticker}", params={"days": days}, ttl=60)
    if s != 200: return jsonify({"detail": raw.get("detail","API error"), "status": s}), s
    candles = raw.get("candles", [])
    if not candles: return jsonify({"detail": "No candle data available"}), 404

    closes  = [c["close"]  for c in candles]
    highs   = [c["high"]   for c in candles]
    lows    = [c["low"]    for c in candles]
    volumes = [c["volume"] for c in candles]

    # MACD
    ml, sl2, hist = _macd(closes)
    # Bollinger
    bb_up, bb_mid, bb_lo = _bollinger(closes)
    # ATR
    atr = _atr(highs, lows, closes)
    # VWAP
    vwap = _vwap_series(highs, lows, closes, volumes)
    # Vol MA
    vol_ma20 = _sma(volumes, 20)
    # Rolling vol (20-day)
    rolling_vol = []
    for i in range(len(closes)):
        if i < 20: rolling_vol.append(None)
        else:
            window = closes[i-20:i+1]
            rets = [(window[j]-window[j-1])/window[j-1] for j in range(1,len(window))]
            rolling_vol.append(round(statistics.stdev(rets)*100, 4) if len(rets)>1 else None)

    # Breakout detection
    last_close = closes[-1]
    period_high = max(highs)
    period_low  = min(lows)
    near_high = last_close >= period_high * 0.98
    near_low  = last_close <= period_low  * 1.02

    # Price structure
    chg_pct = round((closes[-1]-closes[0])/closes[0]*100, 2) if closes else 0

    return jsonify({
        "ticker":   ticker,
        "candles":  candles,
        "sma20":    _sma(closes, 20),
        "sma50":    _sma(closes, 50),
        "sma100":   _sma(closes, 100),
        "sma200":   _sma(closes, 200),
        "ema12":    _ema(closes, 12),
        "ema26":    _ema(closes, 26),
        "vwap":     vwap,
        "rsi14":    _rsi(closes, 14),
        "macd_line":    ml,
        "macd_signal":  sl2,
        "macd_hist":    hist,
        "bb_upper": bb_up,
        "bb_mid":   bb_mid,
        "bb_lower": bb_lo,
        "atr14":    atr,
        "vol_ma20": vol_ma20,
        "rolling_vol": rolling_vol,
        "volatility_ann_pct": _ann_vol(closes),
        "sharpe":   _sharpe(closes),
        "max_drawdown_pct": _max_drawdown(closes),
        "price_chg_pct": chg_pct,
        "high_period": period_high,
        "low_period":  period_low,
        "avg_volume":  round(sum(volumes)/len(volumes)) if volumes else 0,
        "near_high": near_high,
        "near_low":  near_low,
        "vol_spike": round(volumes[-1] / (sum(volumes[:-1])/max(len(volumes)-1,1)), 2) if len(volumes) > 1 and sum(volumes[:-1]) > 0 else None,
        "mean_reversion_score": _mean_reversion(closes),
        "downside_vol": _downside_vol(closes),
    })

# ── COMPARE (fixed) ───────────────────────────────────────────────────────────
@app.route("/api/compare")
def compare():
    tickers = [t.strip() for t in request.args.get("tickers","").split(",") if t.strip()]
    days    = int(request.args.get("days", 30))
    result  = {}
    for t in tickers:
        s, raw = cached_get(f"/analytics/price_history/{t}", params={"days": days}, ttl=60)
        if s != 200 or not raw or not isinstance(raw, list) or len(raw) == 0:
            continue
        base = raw[0]["price"]
        if base == 0: continue
        result[t] = [{"ts": p["timestamp"][:10], "norm": round((p["price"]/base - 1)*100, 4)} for p in raw]
    return jsonify(result)

# ── MARKET BREADTH ────────────────────────────────────────────────────────────
@app.route("/api/market_breadth")
def market_breadth():
    days = int(request.args.get("days", 7))
    s, secs = cached_get("/securities", ttl=20)
    if s != 200: return jsonify({"detail": "Cannot fetch securities"}), s

    results = []
    for sec in secs:
        ticker = sec["ticker"]
        os2, raw = cached_get(f"/analytics/ohlcv/{ticker}", params={"days": days}, ttl=180)
        if os2 != 200 or not raw.get("candles"): continue
        cs = raw["candles"]
        closes  = [c["close"]  for c in cs]
        volumes = [c["volume"] for c in cs]
        if len(closes) < 2: continue
        chg = (closes[-1]-closes[0])/closes[0]*100
        vol_total = sum(volumes)
        ann_v = _ann_vol(closes)
        prd_hi = max(closes)
        prd_hi_pct = round((closes[-1] - prd_hi) / prd_hi * 100, 2) if prd_hi else None
        avg_vol = sum(volumes[:-1]) / max(len(volumes)-1, 1) if len(volumes) > 1 else volumes[0]
        vol_spike = round(volumes[-1] / avg_vol, 2) if avg_vol > 0 else None
        results.append({
            "ticker":    ticker,
            "name":      sec.get("full_name",""),
            "chg_pct":   round(chg, 2),
            "last_price": closes[-1],
            "total_shares": sec.get("total_shares", 0),
            "market_cap": round(closes[-1] * sec.get("total_shares", 0), 2),
            "volume":    vol_total,
            "volatility": ann_v,
            "sharpe":    _sharpe(closes),
            "max_dd":    _max_drawdown(closes),
            "prd_hi_pct": prd_hi_pct,
            "vol_spike":  vol_spike,
            "frozen":     sec.get("frozen", False),
        })

    if not results: return jsonify({"detail": "No data"}), 503

    ups   = [r for r in results if r["chg_pct"] > 0]
    downs = [r for r in results if r["chg_pct"] < 0]
    flat  = [r for r in results if r["chg_pct"] == 0]
    chgs  = [r["chg_pct"] for r in results]
    vols  = [r["volatility"] for r in results if r["volatility"] > 0]

    # Equal-weight index
    ew_return = round(sum(chgs)/len(chgs), 4) if chgs else 0
    # Vol dispersion
    vol_disp  = round(statistics.stdev(vols), 2) if len(vols) > 1 else 0

    return jsonify({
        "securities": results,
        "summary": {
            "total": len(results),
            "advancing": len(ups),
            "declining": len(downs),
            "unchanged": len(flat),
            "adv_dec_ratio": round(len(ups)/max(len(downs),1), 2),
            "avg_return": round(sum(chgs)/len(chgs), 2) if chgs else 0,
            "median_return": round(statistics.median(chgs), 2) if chgs else 0,
            "ew_index_return": ew_return,
            "vol_dispersion": vol_disp,
            "top_gainer": max(results, key=lambda x: x["chg_pct"])["ticker"] if results else None,
            "top_loser":  min(results, key=lambda x: x["chg_pct"])["ticker"] if results else None,
        }
    })

# ── GLOBAL ORDERBOOK AGGREGATOR ───────────────────────────────────────────────
@app.route("/api/market_orderbook")
def market_orderbook():
    s_secs, secs = cached_get("/securities", ttl=20)
    if s_secs != 200: return jsonify({"detail": "Cannot fetch securities"}), s_secs
    all_ob = get_all_orderbooks(); s_ob = 200 if all_ob else 503
    if s_ob != 200 or not isinstance(all_ob, list):
        return jsonify({"detail": "Cannot fetch orderbook"}), s_ob

    sec_map = {s["ticker"]: s for s in secs}
    books = []
    total_bid = 0; total_ask = 0

    for ob in all_ob:
        ticker  = ob.get("ticker","")
        bids    = ob.get("bids", [])
        asks    = ob.get("asks", [])
        bid_qty = sum(b["quantity"] for b in bids)
        ask_qty = sum(a["quantity"] for a in asks)
        total_depth = bid_qty + ask_qty
        total_bid += bid_qty; total_ask += ask_qty

        shares = sec_map.get(ticker, {}).get("total_shares", 1) or 1
        spread = None
        if ob.get("best_ask") and ob.get("best_bid"):
            spread = round((ob["best_ask"] - ob["best_bid"]) / ob["best_bid"] * 100, 4)

        books.append({
            "ticker":      ticker,
            "bid_qty":     bid_qty,
            "ask_qty":     ask_qty,
            "total_depth": total_depth,
            "imbalance":   round((bid_qty - ask_qty) / max(bid_qty + ask_qty, 1) * 100, 2),
            "liquidity_score": round(total_depth / shares * 100, 4),
            "spread_pct":  spread,
            "best_bid":    ob.get("best_bid"),
            "best_ask":    ob.get("best_ask"),
            "mid":         ob.get("mid"),
        })

    books.sort(key=lambda x: x["total_depth"], reverse=True)
    total = total_bid + total_ask

    return jsonify({
        "books": books,
        "global": {
            "total_bid_depth":  total_bid,
            "total_ask_depth":  total_ask,
            "total_depth":      total,
            "global_imbalance": round((total_bid - total_ask)/max(total,1)*100, 2),
            "top5_deepest":     [b["ticker"] for b in books[:5]],
            "most_illiquid":    [b["ticker"] for b in sorted(books, key=lambda x: x["total_depth"])[:5]],
        }
    })

# ── EXCHANGE ANALYTICS ────────────────────────────────────────────────────────
@app.route("/api/exchange_analytics")
def exchange_analytics():
    days = int(request.args.get("days", 7))
    s_secs, secs = cached_get("/securities", ttl=20)
    if s_secs != 200: return jsonify({"detail": "Cannot fetch securities"}), s_secs

    ticker_data = []
    total_mcap = 0
    all_vols = []; all_sharpes = []; all_chgs = []

    for sec in secs:
        t = sec["ticker"]
        os2, raw = cached_get(f"/analytics/ohlcv/{t}", params={"days": days}, ttl=180)
        if os2 != 200 or not raw.get("candles"): continue
        cs = raw["candles"]
        closes  = [c["close"]  for c in cs]
        volumes = [c["volume"] for c in cs]
        if len(closes) < 2: continue

        mcap = closes[-1] * sec.get("total_shares", 0)
        total_mcap += mcap
        chg  = (closes[-1]-closes[0])/closes[0]*100
        vol  = _ann_vol(closes)
        sh   = _sharpe(closes)
        all_vols.append(vol); all_sharpes.append(sh); all_chgs.append(chg)

        ticker_data.append({
            "ticker":  t, "name": sec.get("full_name",""),
            "mcap":    round(mcap, 2),
            "volume":  sum(volumes),
            "chg_pct": round(chg, 2),
            "volatility": vol, "sharpe": sh,
            "max_dd":  _max_drawdown(closes),
            "last":    closes[-1],
            "total_shares": sec.get("total_shares", 0),
        })

    ticker_data.sort(key=lambda x: x["mcap"], reverse=True)

    # Capital flow: volume-weighted
    total_vol = sum(t["volume"] for t in ticker_data) or 1
    for t in ticker_data:
        t["vol_share_pct"] = round(t["volume"]/total_vol*100, 2)
        t["mcap_share_pct"] = round(t["mcap"]/max(total_mcap,1)*100, 2)

    # Exchange-level indices
    vix_proxy = round(sum(all_vols)/len(all_vols), 2) if all_vols else 0
    avg_sharpe = round(sum(all_sharpes)/len(all_sharpes), 3) if all_sharpes else 0

    # HHI of market cap concentration
    mcap_shares = [t["mcap"]/max(total_mcap,1)*100 for t in ticker_data]
    hhi = round(sum(s**2 for s in mcap_shares), 1)

    return jsonify({
        "tickers": ticker_data,
        "exchange": {
            "total_market_cap":  round(total_mcap, 2),
            "total_volume":      sum(t["volume"] for t in ticker_data),
            "num_securities":    len(ticker_data),
            "volatility_index":  vix_proxy,
            "avg_sharpe":        avg_sharpe,
            "concentration_hhi": hhi,
            "avg_return":        round(sum(all_chgs)/len(all_chgs), 2) if all_chgs else 0,
            "top_by_mcap":       [t["ticker"] for t in ticker_data[:5]],
            "top_by_volume":     [t["ticker"] for t in sorted(ticker_data, key=lambda x: x["volume"], reverse=True)[:5]],
        }
    })

# ── LIQUIDITY LAB ─────────────────────────────────────────────────────────────
@app.route("/api/liquidity/<ticker>")
def liquidity_lab(ticker):
    # Use bulk orderbook cache — no per-ticker call
    ob = get_ticker_orderbook(ticker)
    if ob is None:
        return jsonify({"detail": "Orderbook unavailable"}), 503
    s_sec, secs_data = cached_get("/securities", ttl=20)
    sec = next((s for s in (secs_data if isinstance(secs_data, list) else []) if s["ticker"]==ticker), {})
    total_shares = sec.get("total_shares", 1) or 1
    s_ob = 200  # we got data from bulk cache

    bids = sorted(ob.get("bids",[]), key=lambda x: x["price"], reverse=True)
    asks = sorted(ob.get("asks",[]), key=lambda x: x["price"])
    best_bid = ob.get("best_bid"); best_ask = ob.get("best_ask"); mid = ob.get("mid")

    # Cumulative depth
    cum_bid_depth = []; cum_ask_depth = []
    cum = 0
    for b in bids:
        cum += b["quantity"]; cum_bid_depth.append({"price": b["price"], "cum_qty": cum, "qty": b["quantity"]})
    cum = 0
    for a in asks:
        cum += a["quantity"]; cum_ask_depth.append({"price": a["price"], "cum_qty": cum, "qty": a["quantity"]})

    # Slippage simulation for various order sizes
    def sim_slippage(side, qty):
        book = asks if side=="buy" else list(reversed(bids))
        filled = 0; cost = 0
        for level in book:
            take = min(level["quantity"], qty - filled)
            cost += take * level["price"]; filled += take
            if filled >= qty: break
        if filled == 0: return None
        avg_px = cost/filled
        ref = best_ask if side=="buy" else best_bid
        if not ref: return None
        slip = (avg_px - ref)/ref*100 if side=="buy" else (ref - avg_px)/ref*100
        return {"qty": qty, "filled": filled, "avg_px": round(avg_px,4), "slippage_pct": round(slip,4), "cost": round(cost,2)}

    total_shares_float = float(total_shares)
    sim_sizes = [10, 50, 100, 500, 1000, int(total_shares_float*0.001), int(total_shares_float*0.005)]
    sim_sizes = sorted(set(s for s in sim_sizes if s > 0))

    slippage_buy  = [r for r in (sim_slippage("buy",  s) for s in sim_sizes) if r]
    slippage_sell = [r for r in (sim_slippage("sell", s) for s in sim_sizes) if r]

    # Depth imbalance
    total_bid_qty = sum(b["quantity"] for b in bids)
    total_ask_qty = sum(a["quantity"] for a in asks)
    imbalance = round((total_bid_qty-total_ask_qty)/max(total_bid_qty+total_ask_qty,1)*100, 2)

    # Spread
    spread = round((best_ask-best_bid)/best_bid*100,4) if best_ask and best_bid else None
    # Liquidity score
    liq_score = round((total_bid_qty+total_ask_qty)/total_shares*100, 4)

    # Fake wall detection: orders >3x avg size at or within 2% of mid
    avg_bid_sz = total_bid_qty/max(len(bids),1)
    avg_ask_sz = total_ask_qty/max(len(asks),1)
    walls = []
    if mid:
        for b in bids:
            if b["price"] >= mid*0.98 and b["quantity"] > avg_bid_sz*3:
                walls.append({"side":"bid","price":b["price"],"qty":b["quantity"],"multiple":round(b["quantity"]/avg_bid_sz,1)})
        for a in asks:
            if a["price"] <= mid*1.02 and a["quantity"] > avg_ask_sz*3:
                walls.append({"side":"ask","price":a["price"],"qty":a["quantity"],"multiple":round(a["quantity"]/avg_ask_sz,1)})

    return jsonify({
        "ticker": ticker, "mid": mid, "best_bid": best_bid, "best_ask": best_ask,
        "spread_pct": spread, "imbalance_pct": imbalance, "liquidity_score": liq_score,
        "total_bid_qty": total_bid_qty, "total_ask_qty": total_ask_qty,
        "cum_bid_depth": cum_bid_depth, "cum_ask_depth": cum_ask_depth,
        "slippage_buy": slippage_buy, "slippage_sell": slippage_sell,
        "walls": walls, "total_shares": total_shares,
        "bids": bids, "asks": asks,
    })

# ── HOLDER INTELLIGENCE ───────────────────────────────────────────────────────
@app.route("/api/holder_intel")
def holder_intel():
    """Exchange-wide shareholder intelligence."""
    s_secs, secs = cached_get("/securities", ttl=20)
    if s_secs != 200: return jsonify({"detail": "Cannot fetch securities"}), s_secs

    all_data = []
    for sec in secs:
        t = sec["ticker"]
        holders = get_ticker_shareholders(t)
        if not holders: continue
        total_qty = sum(h["quantity"] for h in holders)
        if total_qty == 0: continue

        # HHI
        shares = [h["quantity"]/total_qty for h in holders]
        hhi    = round(sum(s**2 for s in shares)*10000, 1)
        # Concentration
        top1  = round(shares[0]*100,2) if shares else 0
        top3  = round(sum(shares[:3])*100,2) if len(shares)>=3 else round(sum(shares)*100,2)
        top5  = round(sum(shares[:5])*100,2) if len(shares)>=5 else round(sum(shares)*100,2)
        top10 = round(sum(shares[:10])*100,2)
        # Whale (anyone >10%)
        whales = [h for h in holders if h["quantity"]/total_qty > 0.10]

        all_data.append({
            "ticker":     t,
            "name":       sec.get("full_name",""),
            "num_holders": len(holders),
            "total_shares_held": total_qty,
            "hhi":        hhi,
            "top1_pct":   top1,
            "top3_pct":   top3,
            "top5_pct":   top5,
            "top10_pct":  top10,
            "whale_count": len(whales),
            "holders":    holders[:20],
        })

    # Cross-exchange whale tracking
    whale_map = {}
    for d in all_data:
        t = d["ticker"]
        total = d["total_shares_held"]
        for h in d["holders"]:
            uid = h["user_id"]
            pct = round(h["quantity"]/total*100, 2)
            if pct >= 5:
                if uid not in whale_map: whale_map[uid] = []
                whale_map[uid].append({"ticker": t, "qty": h["quantity"], "pct": pct})

    whales = [{"user_id": uid, "positions": pos, "num_positions": len(pos)}
              for uid, pos in whale_map.items()]
    whales.sort(key=lambda x: x["num_positions"], reverse=True)

    return jsonify({
        "securities": all_data,
        "whales": whales,
        "summary": {
            "most_concentrated":   max(all_data, key=lambda x: x["hhi"])["ticker"] if all_data else None,
            "most_distributed":    min(all_data, key=lambda x: x["hhi"])["ticker"] if all_data else None,
            "most_whales":         max(all_data, key=lambda x: x["whale_count"])["ticker"] if all_data else None,
            "avg_hhi":             round(sum(d["hhi"] for d in all_data)/len(all_data),1) if all_data else 0,
            "avg_holders":         round(sum(d["num_holders"] for d in all_data)/len(all_data),1) if all_data else 0,
        }
    })

@app.route("/api/holder_intel/<ticker>")
def holder_intel_ticker(ticker):
    holders = get_ticker_shareholders(ticker)
    if not isinstance(holders, list) or not holders:
        return jsonify({"detail": "No holders found for ticker"}), 404
    _, sec_list = cached_get("/securities", ttl=20)
    sec = next((s for s in (sec_list if isinstance(sec_list,list) else []) if s["ticker"]==ticker), {})
    total_shares = sec.get("total_shares", 0)
    total_held   = sum(h["quantity"] for h in holders) if holders else 0

    if not holders: return jsonify({"holders":[],"stats":{},"ticker":ticker})

    shares_norm = [h["quantity"]/total_held for h in holders]
    hhi = round(sum(s**2 for s in shares_norm)*10000, 1)

    # Histogram buckets
    buckets = {"<0.1%":0,"0.1-1%":0,"1-5%":0,"5-10%":0,">10%":0}
    for h in holders:
        p = h["quantity"]/total_held*100
        if p < 0.1:   buckets["<0.1%"]  += 1
        elif p < 1:   buckets["0.1-1%"] += 1
        elif p < 5:   buckets["1-5%"]   += 1
        elif p < 10:  buckets["5-10%"]  += 1
        else:         buckets[">10%"]   += 1

    return jsonify({
        "ticker":  ticker,
        "holders": holders,
        "total_shares": total_shares,
        "total_held":   total_held,
        "float_pct":    round(total_held/total_shares*100,2) if total_shares else None,
        "stats": {
            "num_holders": len(holders),
            "hhi":         hhi,
            "top1_pct":    round(shares_norm[0]*100,2),
            "top3_pct":    round(sum(shares_norm[:3])*100,2),
            "top5_pct":    round(sum(shares_norm[:5])*100,2),
            "whale_count": sum(1 for s in shares_norm if s>0.10),
            "gini":        round(sum(abs(a-b) for a in shares_norm for b in shares_norm)/(2*len(shares_norm)**2),4) if len(shares_norm)>1 else 0,
        },
        "histogram": buckets,
    })

# ── BACKTEST ──────────────────────────────────────────────────────────────────
@app.route("/api/backtest", methods=["POST"])
def backtest():
    body=request.json or {}
    ticker=body.get("ticker",""); days=min(int(body.get("days",90)),365)
    strategy=body.get("strategy","buy_hold"); params=body.get("params",{})
    status,raw=cached_get(f"/analytics/ohlcv/{ticker}",params={"days":days},ttl=60)
    if status!=200: return jsonify(raw),status
    candles=raw.get("candles",[])
    if len(candles)<3: return jsonify({"detail":"Not enough data"}),400
    closes=[c["close"] for c in candles]; dates=[c["date"] for c in candles]
    highs=[c["high"] for c in candles]; lows=[c["low"] for c in candles]

    signals=[0]*len(closes)
    if strategy=="sma_cross":
        f2=_sma(closes,int(params.get("fast",5))); sl=_sma(closes,int(params.get("slow",20)))
        for i in range(1,len(closes)):
            if f2[i] and sl[i] and f2[i-1] and sl[i-1]:
                if f2[i]>sl[i] and f2[i-1]<=sl[i-1]: signals[i]=1
                elif f2[i]<sl[i] and f2[i-1]>=sl[i-1]: signals[i]=-1
    elif strategy=="rsi":
        p=int(params.get("rsi_period",14)); ob=float(params.get("overbought",70)); os_=float(params.get("oversold",30))
        rsi=_rsi(closes,p)
        for i in range(1,len(closes)):
            if rsi[i] and rsi[i-1]:
                if rsi[i]<os_ and rsi[i-1]>=os_: signals[i]=1
                elif rsi[i]>ob and rsi[i-1]<=ob: signals[i]=-1
    elif strategy=="macd":
        ml,sl2,_=_macd(closes,int(params.get("fast",12)),int(params.get("slow",26)),int(params.get("signal",9)))
        for i in range(1,len(closes)):
            if ml[i]>sl2[i] and ml[i-1]<=sl2[i-1]: signals[i]=1
            elif ml[i]<sl2[i] and ml[i-1]>=sl2[i-1]: signals[i]=-1
    elif strategy=="bb_reversal":
        bb_u,bb_m,bb_l=_bollinger(closes)
        for i in range(1,len(closes)):
            if bb_l[i] and closes[i]<bb_l[i] and closes[i-1]>=bb_l[i-1]: signals[i]=1
            elif bb_u[i] and closes[i]>bb_u[i] and closes[i-1]<=bb_u[i-1]: signals[i]=-1
    elif strategy=="buy_hold": signals[0]=1

    sl_pct  = float(params.get("stop_loss_pct",  0)) / 100
    tp_pct  = float(params.get("take_profit_pct",0)) / 100
    cash=10000.0; shares=0; equity=[]; trades=[]; pos=False; entry_px=0
    for i,(sig,price) in enumerate(zip(signals,closes)):
        # Stop loss / take profit
        if pos and entry_px > 0:
            if sl_pct > 0 and price <= entry_px*(1-sl_pct):
                cash+=shares*price*0.995; trades.append({"date":dates[i],"action":"STOP","price":price,"qty":shares}); shares=0; pos=False
            elif tp_pct > 0 and price >= entry_px*(1+tp_pct):
                cash+=shares*price*0.995; trades.append({"date":dates[i],"action":"TP","price":price,"qty":shares}); shares=0; pos=False
        if sig==1 and not pos and cash>price:
            qty=int(cash/price); cost=qty*price*1.005
            if cost<=cash: cash-=cost; shares=qty; pos=True; entry_px=price; trades.append({"date":dates[i],"action":"BUY","price":price,"qty":qty})
        elif sig==-1 and pos and shares>0:
            cash+=shares*price*0.995; trades.append({"date":dates[i],"action":"SELL","price":price,"qty":shares}); shares=0; pos=False; entry_px=0
        equity.append(round(cash+shares*price,4))

    end_eq=equity[-1] if equity else 10000
    total_ret=(end_eq-10000)/10000*100; bh_ret=(closes[-1]-closes[0])/closes[0]*100 if closes else 0
    max_dd=_max_drawdown(equity); sharpe_val=_sharpe(equity)
    win_trades=[t for t in zip(trades[::2],trades[1::2]) if t[1]["price"]>t[0]["price"]]
    calmar = round(abs(total_ret/max_dd),2) if max_dd > 0 else 0

    return jsonify({"ticker":ticker,"strategy":strategy,"days":days,"candles":candles,
        "signals":signals,"equity_curve":equity,"dates":dates,"trades":trades,
        "metrics":{"total_return_pct":round(total_ret,2),"benchmark_return_pct":round(bh_ret,2),
            "max_drawdown_pct":max_dd,"sharpe_ratio":sharpe_val,"calmar_ratio":calmar,
            "num_trades":len(trades),"win_rate":round(len(win_trades)/max(len(trades)//2,1)*100,1),
            "final_equity":round(end_eq,2)}})

# ── PAGES ─────────────────────────────────────────────────────────────────────
@app.route("/")
def index():           return render_template("terminal.html")
@app.route("/page/<name>")
def page(name):
    allowed = ["market","ticker","portfolio","orders","backtest","compare","watchlist","heatmap","exchange","liquidity","holders","screener","alerts","fundamentals","transactions","news"]
    if name not in allowed: return "Not found", 404
    return render_template(f"pages/{name}.html")


# ══════════════════════════════════════════════════════════════════════════════
# VERSION 4 ENDPOINTS — Rolling correlation, pairs, Monte Carlo, VaR, fundamentals
# ══════════════════════════════════════════════════════════════════════════════

# ── FUNDAMENTALS (uses /securities/{ticker}/stats) ────────────────────────────
@app.route("/api/fundamentals/<ticker>")
def fundamentals(ticker):
    s, d = cached_get(f"/securities/{ticker}/stats", ttl=180)
    if s != 200: return jsonify({"detail": d.get("detail","Not found")}), s
    s2, sec = cached_get(f"/securities/{ticker}", ttl=60)
    sec = sec if s2 == 200 else {}
    return jsonify({
        "ticker":       ticker,
        "eps":          d.get("eps"),
        "pe_ratio":     d.get("pe_ratio"),
        "pb_ratio":     d.get("pb_ratio"),
        "roa_percent":  d.get("roa_percent"),
        "book_value":   d.get("book_value"),
        "net_profit":   d.get("net_profit"),
        "market_price": d.get("market_price"),
        "total_shares": d.get("total_shares"),
        "market_cap":   round(d.get("market_price",0)*d.get("total_shares",0),2) if d.get("market_price") and d.get("total_shares") else None,
        "shareholder_count": sec.get("shareholder_count"),
    })

# ── TRANSACTION HISTORY ────────────────────────────────────────────────────────
@app.route("/api/transactions")
def transactions():
    limit  = request.args.get("limit", 500)
    offset = request.args.get("offset", 0)
    ttype  = request.args.get("type", "")
    params = {"limit": limit, "offset": offset}
    if ttype: params["type"] = ttype
    s, d = cached_get("/transactions", params=params, auth=True, ttl=0)  # always live
    return jsonify(d), s

# ── OPEN ORDERS ────────────────────────────────────────────────────────────────
@app.route("/api/orders_open")
def orders_open():
    s, d = cached_get("/orders", auth=True, ttl=0)  # always live
    return jsonify(d), s

@app.route("/api/orders_open/<order_id>", methods=["DELETE"])
def cancel_order(order_id):
    import requests as req
    url = f"{NER_BASE}/orders/{order_id}"
    r = req.delete(url, headers=AUTH_H, timeout=10)
    try: return jsonify(r.json()), r.status_code
    except: return jsonify({"detail":"Error"}), r.status_code

# ── FUNDS ─────────────────────────────────────────────────────────────────────
@app.route("/api/funds")
def funds():
    s, d = cached_get("/funds", auth=True, ttl=0)  # always live
    return jsonify(d), s

# ── ROLLING CORRELATION + PAIRS SPREAD ────────────────────────────────────────
@app.route("/api/rolling_correlation")
def rolling_correlation():
    t1 = request.args.get("t1",""); t2 = request.args.get("t2","")
    days   = int(request.args.get("days", 60))
    window = int(request.args.get("window", 14))
    if not t1 or not t2: return jsonify({"detail":"t1 and t2 required"}), 400
    _, r1 = cached_get(f"/analytics/ohlcv/{t1}", params={"days":days}, ttl=60)
    _, r2 = cached_get(f"/analytics/ohlcv/{t2}", params={"days":days}, ttl=60)
    c1 = [c["close"] for c in r1.get("candles",[])]
    c2 = [c["close"] for c in r2.get("candles",[])]
    dates = [c["date"] for c in r1.get("candles",[])]
    n = min(len(c1), len(c2)); c1=c1[-n:]; c2=c2[-n:]; dates=dates[-n:]
    if n < window+2: return jsonify({"detail":"Not enough data","dates":[],"corr":[],"spread":[],"spread_z":[]}), 200
    # Daily returns
    r1d = [(c1[i]-c1[i-1])/c1[i-1] for i in range(1,n)]
    r2d = [(c2[i]-c2[i-1])/c2[i-1] for i in range(1,n)]
    # Rolling Pearson correlation
    corrs = [None]  # offset by 1 for dates alignment
    for i in range(len(r1d)):
        if i < window-1: corrs.append(None); continue
        a = r1d[i-window+1:i+1]; b = r2d[i-window+1:i+1]
        ma = sum(a)/window; mb = sum(b)/window
        num = sum((a[j]-ma)*(b[j]-mb) for j in range(window))
        da  = (sum((x-ma)**2 for x in a))**0.5
        db  = (sum((x-mb)**2 for x in b))**0.5
        corrs.append(round(num/da/db,4) if da*db>0 else None)
    # Normalized spread (pairs trading signal)
    base1=c1[0]; base2=c2[0]
    norm1 = [c/base1*100 for c in c1]
    norm2 = [c/base2*100 for c in c2]
    spread = [round(norm1[i]-norm2[i],4) for i in range(n)]
    sp_vals = [s for s in spread if s is not None]
    sp_mean = sum(sp_vals)/len(sp_vals) if sp_vals else 0
    sp_std  = (sum((s-sp_mean)**2 for s in sp_vals)/len(sp_vals))**0.5 if len(sp_vals)>1 else 1
    spread_z = [round((s-sp_mean)/sp_std,4) if sp_std>0 else 0 for s in spread]
    # Current Pearson (full period)
    full_corr = corrs[-1] if corrs else None
    # Static full-period pearson
    if len(r1d) >= 2:
        am=sum(r1d)/len(r1d); bm=sum(r2d)/len(r2d)
        num=sum((r1d[i]-am)*(r2d[i]-bm) for i in range(len(r1d)))
        da=(sum((x-am)**2 for x in r1d))**0.5; db=(sum((x-bm)**2 for x in r2d))**0.5
        full_corr = round(num/da/db,4) if da*db>0 else None
    return jsonify({
        "dates": dates, "corr": corrs, "spread": spread, "spread_z": spread_z,
        "norm1": [round(v,4) for v in norm1], "norm2": [round(v,4) for v in norm2],
        "sp_mean": round(sp_mean,4), "sp_std": round(sp_std,4),
        "full_corr": full_corr, "t1": t1, "t2": t2,
        "current_spread_z": spread_z[-1] if spread_z else None,
    })

# ── EXCHANGE CORRELATION MATRIX (all-pairs for convergence index) ─────────────
@app.route("/api/correlation_matrix")
def correlation_matrix():
    days = int(request.args.get("days", 30))
    _, secs_data = cached_get("/securities", ttl=20)
    tickers = [s["ticker"] for s in (secs_data if isinstance(secs_data,list) else [])][:20]  # cap at 20
    # Fetch closes for all
    closes_map = {}
    for t in tickers:
        _, raw = cached_get(f"/analytics/ohlcv/{t}", params={"days":days}, ttl=180)
        cs = [c["close"] for c in raw.get("candles",[])]
        if len(cs) >= 5: closes_map[t] = cs
    valid = list(closes_map.keys())
    # Align lengths
    min_len = min(len(v) for v in closes_map.values()) if closes_map else 0
    if min_len < 3: return jsonify({"tickers":[],"matrix":[],"avg_corr":0})
    rets = {}
    for t in valid:
        c = closes_map[t][-min_len:]
        r = [(c[i]-c[i-1])/c[i-1] for i in range(1,len(c))]
        rets[t] = r
    # Build matrix
    matrix = []
    all_corrs = []
    for t1 in valid:
        row = []
        for t2 in valid:
            if t1==t2: row.append(1.0); continue
            a=rets[t1]; b=rets[t2]
            n=min(len(a),len(b)); a=a[:n]; b=b[:n]
            am=sum(a)/n; bm=sum(b)/n
            num=sum((a[i]-am)*(b[i]-bm) for i in range(n))
            da=(sum((x-am)**2 for x in a))**0.5; db=(sum((x-bm)**2 for x in b))**0.5
            c_val = round(num/da/db,4) if da*db>0 else 0
            row.append(c_val); all_corrs.append(abs(c_val))
        matrix.append(row)
    avg_corr = round(sum(all_corrs)/len(all_corrs),4) if all_corrs else 0
    return jsonify({"tickers":valid,"matrix":matrix,"avg_corr":avg_corr})

# ── MONTE CARLO ───────────────────────────────────────────────────────────────
@app.route("/api/monte_carlo", methods=["POST"])
def monte_carlo():
    import random, math
    body = request.json or {}
    ticker = body.get("ticker",""); days = int(body.get("days",60))
    n_sims = min(int(body.get("sims",1000)),2000); horizon = int(body.get("horizon",30))
    _, raw = cached_get(f"/analytics/ohlcv/{ticker}", params={"days":days}, ttl=60)
    cs = [c["close"] for c in raw.get("candles",[])]
    if len(cs) < 5: return jsonify({"detail":"Not enough data"}), 404
    rets = [(cs[i]-cs[i-1])/cs[i-1] for i in range(1,len(cs))]
    mu  = sum(rets)/len(rets)
    std = (sum((r-mu)**2 for r in rets)/len(rets))**0.5
    start = cs[-1]
    # Run simulations
    sim_ends = []
    paths_sample = []  # store 50 paths for chart
    random.seed(42)
    for sim in range(n_sims):
        price = start; path = [start]
        for _ in range(horizon):
            # Box-Muller normal
            u1=random.random(); u2=random.random()
            z = math.sqrt(-2*math.log(max(u1,1e-10)))*math.cos(2*math.pi*u2)
            price *= (1 + mu + std*z)
            price = max(price, 0.0001)
            path.append(round(price,4))
        sim_ends.append(round(path[-1],4))
        if sim < 50: paths_sample.append(path)
    sim_ends.sort()
    n = len(sim_ends)
    pct5  = sim_ends[int(n*0.05)]
    pct25 = sim_ends[int(n*0.25)]
    pct50 = sim_ends[int(n*0.50)]
    pct75 = sim_ends[int(n*0.75)]
    pct95 = sim_ends[int(n*0.95)]
    # Drawdown distribution from paths_sample
    max_dds = []
    for path in paths_sample:
        peak=path[0]; dd=0
        for p in path:
            if p>peak: peak=p
            dd = max(dd,(peak-p)/peak if peak>0 else 0)
        max_dds.append(round(dd*100,2))
    prob_profit = round(sum(1 for e in sim_ends if e>start)/n*100,1)
    prob_dd20   = round(sum(1 for d in max_dds if d>20)/len(max_dds)*100,1) if max_dds else 0
    return jsonify({
        "ticker":ticker,"start":start,"horizon":horizon,"sims":n_sims,
        "p5":pct5,"p25":pct25,"p50":pct50,"p75":pct75,"p95":pct95,
        "prob_profit":prob_profit,"prob_dd20":prob_dd20,
        "paths":paths_sample,"sim_ends":sim_ends[::max(1,n//200)],  # sampled for histogram
        "mu_daily":round(mu*100,4),"std_daily":round(std*100,4),
    })

# ── PARAMETER SENSITIVITY HEATMAP ─────────────────────────────────────────────
@app.route("/api/param_sensitivity", methods=["POST"])
def param_sensitivity():
    body = request.json or {}
    ticker = body.get("ticker",""); days = int(body.get("days",90))
    strategy = body.get("strategy","sma_cross")
    _, raw = cached_get(f"/analytics/ohlcv/{ticker}", params={"days":days}, ttl=180)
    cs = [c["close"] for c in raw.get("candles",[])]
    if len(cs) < 30: return jsonify({"detail":"Not enough data"}), 404
    def run_sma(fast,slow):
        if fast>=slow: return None
        sma_f=[None if i<fast-1 else sum(cs[i-fast+1:i+1])/fast for i in range(len(cs))]
        sma_s=[None if i<slow-1 else sum(cs[i-slow+1:i+1])/slow for i in range(len(cs))]
        pos=0; equity=cs[0]; trades=0
        for i in range(1,len(cs)):
            if sma_f[i] and sma_s[i] and sma_f[i-1] and sma_s[i-1]:
                if sma_f[i]>sma_s[i] and sma_f[i-1]<=sma_s[i-1] and pos==0:
                    pos=equity/cs[i]; trades+=1
                elif sma_f[i]<sma_s[i] and sma_f[i-1]>=sma_s[i-1] and pos>0:
                    equity=pos*cs[i]; pos=0; trades+=1
        if pos>0: equity=pos*cs[-1]
        ret=round((equity-cs[0])/cs[0]*100,2)
        return ret
    def run_rsi(period,ob,os_):
        # RSI strategy: buy when crosses up from oversold, sell when crosses down from overbought
        def rsi(closes,p):
            r=[None]*p
            for i in range(p,len(closes)):
                g=[max(closes[j]-closes[j-1],0) for j in range(i-p+1,i+1)]
                l=[max(closes[j-1]-closes[j],0) for j in range(i-p+1,i+1)]
                ag,al=sum(g)/p,sum(l)/p
                r.append(100.0 if al==0 else round(100-100/(1+ag/al),2))
            return r
        rv=rsi(cs,period); pos=0; equity=cs[0]; trades=0
        for i in range(1,len(rv)):
            if rv[i] and rv[i-1]:
                if rv[i]>os_ and rv[i-1]<=os_ and pos==0:
                    pos=equity/cs[i]; trades+=1
                elif rv[i]<ob and rv[i-1]>=ob and pos>0:
                    equity=pos*cs[i]; pos=0; trades+=1
        if pos>0: equity=pos*cs[-1]
        return round((equity-cs[0])/cs[0]*100,2)
    results = {"strategy":strategy,"ticker":ticker,"rows":[],"cols":[]}
    if strategy=="sma_cross":
        fast_range=[3,5,7,10,14,20]; slow_range=[10,14,20,30,50]
        results["rows"]=fast_range; results["cols"]=slow_range
        results["row_label"]="Fast SMA"; results["col_label"]="Slow SMA"
        for fast in fast_range:
            row=[]
            for slow in slow_range:
                row.append(run_sma(fast,slow))
            results["rows_data"]=results.get("rows_data",[])+[row]
    elif strategy=="rsi":
        periods=[7,10,14,20,25]; ob_levels=[65,70,75,80]
        results["rows"]=periods; results["cols"]=ob_levels
        results["row_label"]="RSI Period"; results["col_label"]="Overbought Level"
        for p in periods:
            row=[]
            for ob in ob_levels:
                row.append(run_rsi(p,ob,100-ob))
            results["rows_data"]=results.get("rows_data",[])+[row]
    results["bh_return"]=round((cs[-1]-cs[0])/cs[0]*100,2)
    return jsonify(results)

# ── PORTFOLIO VaR + METRICS ───────────────────────────────────────────────────
@app.route("/api/portfolio_analytics")
def portfolio_analytics():
    days = int(request.args.get("days", 60))
    # Get portfolio
    s, pf = cached_get("/portfolio", auth=True, ttl=10)
    if s != 200: return jsonify({"detail":"Portfolio unavailable"}), s
    holdings = pf.get("holdings", [])
    if not holdings: return jsonify({"detail":"No holdings"}), 200
    # Fetch price histories
    hist_data = {}
    for h in holdings:
        _, raw = cached_get(f"/analytics/ohlcv/{h['ticker']}", params={"days":days}, ttl=180)
        cs = [c["close"] for c in raw.get("candles",[])]
        if cs: hist_data[h["ticker"]] = cs
    if not hist_data: return jsonify({"detail":"No price data"}), 200
    # Build portfolio return series (weighted daily returns)
    min_len = min(len(v) for v in hist_data.values())
    total_mktval = sum(h["market_value"] for h in holdings if h["ticker"] in hist_data)
    if total_mktval <= 0: return jsonify({"detail":"Zero market value"}), 200
    weights = {h["ticker"]: h["market_value"]/total_mktval for h in holdings if h["ticker"] in hist_data}
    # Daily portfolio returns
    port_rets = []
    for i in range(1, min_len):
        day_ret = 0
        for t, w in weights.items():
            cs = hist_data[t]
            n = len(cs)
            idx_curr = n - min_len + i; idx_prev = idx_curr - 1
            if idx_curr < n and idx_prev >= 0 and cs[idx_prev] > 0:
                day_ret += w * (cs[idx_curr]-cs[idx_prev])/cs[idx_prev]
        port_rets.append(round(day_ret,6))
    if len(port_rets) < 5: return jsonify({"detail":"Insufficient return data"}), 200
    # VaR (Historical Simulation)
    sorted_rets = sorted(port_rets)
    n = len(sorted_rets)
    var95 = abs(sorted_rets[int(n*0.05)])
    var99 = abs(sorted_rets[int(n*0.01)]) if n >= 100 else abs(sorted_rets[0])
    cvar95 = abs(sum(sorted_rets[:max(1,int(n*0.05))])/max(1,int(n*0.05)))
    # Beta to EW index
    ew_rets = []
    for i in range(1, min_len):
        day=0; cnt=0
        for t in hist_data:
            cs=hist_data[t]; idx=len(cs)-min_len+i
            if idx>0 and idx<len(cs) and cs[idx-1]>0:
                day+=(cs[idx]-cs[idx-1])/cs[idx-1]; cnt+=1
        if cnt: ew_rets.append(day/cnt)
    beta = None
    if len(ew_rets) >= 5 and len(port_rets) >= 5:
        n2=min(len(ew_rets),len(port_rets))
        pr=port_rets[:n2]; er=ew_rets[:n2]
        pm=sum(pr)/n2; em=sum(er)/n2
        cov=sum((pr[i]-pm)*(er[i]-em) for i in range(n2))/n2
        var_mkt=sum((e-em)**2 for e in er)/n2
        beta = round(cov/var_mkt,3) if var_mkt>0 else None
    # Sortino
    neg_rets=[r for r in port_rets if r<0]
    downside_std=(sum(r**2 for r in neg_rets)/len(neg_rets))**0.5 if neg_rets else 0
    avg_ret=sum(port_rets)/len(port_rets)
    sortino=round(avg_ret/downside_std*(252**0.5),3) if downside_std>0 else None
    # Sharpe
    std_ret=(sum((r-avg_ret)**2 for r in port_rets)/len(port_rets))**0.5
    sharpe=round(avg_ret/std_ret*(252**0.5),3) if std_ret>0 else None
    # Per-holding attribution
    attribution=[]
    for h in holdings:
        if h["ticker"] not in hist_data: continue
        cs=hist_data[h["ticker"]]
        total_ret=round((cs[-1]-cs[0])/cs[0]*100,2) if cs[0]>0 else 0
        contribution=round(weights.get(h["ticker"],0)*total_ret,3)
        attribution.append({"ticker":h["ticker"],"weight":round(weights.get(h["ticker"],0)*100,2),"return":total_ret,"contribution":contribution})
    attribution.sort(key=lambda x:x["contribution"],reverse=True)
    # Kelly sizing per holding
    kelly=[]
    for h in holdings:
        if h["ticker"] not in hist_data: continue
        cs=hist_data[h["ticker"]]
        if len(cs)<2: continue
        rets=[(cs[i]-cs[i-1])/cs[i-1] for i in range(1,len(cs))]
        mu2=sum(rets)/len(rets); var2=sum((r-mu2)**2 for r in rets)/len(rets)
        k=round(mu2/var2*100,1) if var2>0 else 0
        kelly.append({"ticker":h["ticker"],"kelly_pct":max(0,min(k,100)),"current_pct":round(weights.get(h["ticker"],0)*100,2)})
    return jsonify({
        "var95_pct":round(var95*100,3),"var99_pct":round(var99*100,3),
        "cvar95_pct":round(cvar95*100,3),
        "beta":beta,"sortino":sortino,"sharpe":sharpe,
        "port_rets":port_rets,"ew_rets":ew_rets,
        "attribution":attribution,"kelly":kelly,
        "days":days,"n_obs":len(port_rets),
    })


# ── DEBUG — check raw NER shareholders response ───────────────────────────────
@app.route("/api/debug/shareholders/<ticker>")
def debug_shareholders(ticker):
    """Exposes raw NER /shareholders response for debugging."""
    try:
        r = requests.get(f"{NER_BASE}/shareholders", headers=AUTH_H,
                         params={"ticker": ticker}, timeout=10)
        return jsonify({
            "status": r.status_code,
            "is_list": isinstance(r.json(), list),
            "count": len(r.json()) if isinstance(r.json(), list) else None,
            "sample": r.json()[:3] if isinstance(r.json(), list) else r.json(),
            "cache_keys": list(_sh_ticker_cache.keys()),
        }), 200
    except Exception as ex:
        return jsonify({"error": str(ex)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n  Bloomberg Terminal — NER Exchange")
    print(f"  http://localhost:{port}\n")
    app.run(host="0.0.0.0", port=port, debug=True)