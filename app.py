#!/usr/bin/env python3
"""
Bloomberg Terminal — NER Exchange
Architecture: Flask proxy backend + split HTML page files
Modules: quant_utils.py, data_utils.py, enterprise_store.py, enterprise_api.py
"""
import os, time, math, statistics, json, queue, threading, uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from flask import Flask, jsonify, request, render_template, Response, stream_with_context

# ── Local modules ─────────────────────────────────────────────────────────────
import quant_utils as Q   # pure math — sma/ema/rsi/sharpe/bt_run etc.
import data_utils   as D  # candle normalization, fetch helpers
import enterprise_store as ES  # portfolio persistence

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_TMPL_DIR  = os.path.join(_BASE_DIR, "templates")
_STAT_DIR  = os.path.join(_BASE_DIR, "static")
print(f"[NER Terminal] Base dir : {_BASE_DIR}")
print(f"[NER Terminal] Templates: {_TMPL_DIR}  exists={os.path.isdir(_TMPL_DIR)}")
print(f"[NER Terminal] Static   : {_STAT_DIR}  exists={os.path.isdir(_STAT_DIR)}")

app = Flask(__name__, template_folder=_TMPL_DIR, static_folder=_STAT_DIR)

NER_BASE   = "http://150.230.117.88:8082"
TSE_BASE   = "https://market.installe.us"
ATLAS_BASE = os.environ.get("ATLAS_URL", "https://atlas-production-1438.up.railway.app").rstrip("/")
ATLAS_KEY  = os.environ.get("ATLAS_API_KEY") or ""
ATLAS_H    = {"X-Atlas-Key": ATLAS_KEY, "Content-Type": "application/json"}
TSE_KEY    = os.environ.get("TSE_API_KEY") or ""
TSE_H      = {"X-API-Key": TSE_KEY}

_DELISTED = {"RNHC", "CGF", "RNC-B", "RDS"}

def _is_active(ticker: str) -> bool:
    bare = ticker[4:] if ticker.startswith("TSE:") else ticker
    return bare not in _DELISTED and ticker not in _DELISTED

_HOLDER_NAMES: dict = {}
for _k, _v in os.environ.items():
    if _k.startswith("HOLDER_NAME_"):
        _suffix = _k[len("HOLDER_NAME_"):].lower()
        _HOLDER_NAMES[_suffix] = _v.strip()

def _resolve_holder(user_id: str) -> str:
    if not user_id: return "—"
    uid_lower = user_id.lower()
    return (_HOLDER_NAMES.get(uid_lower) or _HOLDER_NAMES.get(uid_lower[-8:]) or f"…{user_id[-8:]}")

_tse_cache: dict = {}

def tse_get(path, params=None, ttl=60):
    key = path + str(params or {})
    e = _tse_cache.get(key)
    if e and time.time() - e["ts"] < ttl: return e["s"], e["d"]
    try:
        r = _session.get(f"{TSE_BASE}{path}", headers=TSE_H, params=params, timeout=8)
        if r.status_code == 200:
            d = r.json(); _tse_cache[key] = {"ts": time.time(), "s": 200, "d": d}; return 200, d
        return r.status_code, {}
    except Exception as ex:
        print(f"[TSE] {path} error: {ex}")
        if e: return e["s"], e["d"]
        return 503, {}

_session = requests.Session()
_session.mount("http://", requests.adapters.HTTPAdapter(pool_connections=4, pool_maxsize=8, max_retries=0))
# NER_API_KEY is the PLATFORM key — identifies our terminal to NER on every request.
# It is NEVER used to view any user's portfolio. All user-facing authenticated
# endpoints require the user to supply their own Discord User ID + passcode.
NER_PLATFORM_KEY = os.environ.get("NER_API_KEY") or ""
API_KEY          = NER_PLATFORM_KEY  # alias kept for any remaining public-route uses
AUTH_H           = {"Content-Type": "application/json", "X-API-Key": NER_PLATFORM_KEY}
PUB_H            = {"Content-Type": "application/json"}

def _passcode_headers(user_id: str, passcode: str) -> dict:
    """Build headers for a user request via the passcode flow.
    Platform key (X-API-Key) identifies us as the partner site.
    User ID + passcode identify which user is trading."""
    return {
        "Content-Type":   "application/json",
        "X-API-Key":      NER_PLATFORM_KEY,
        "X-NER-User-Id":  str(user_id),
        "X-NER-Passcode": str(passcode),
    }

def _get_user_auth(req) -> dict | None:
    """Extract user auth from request headers.
    Returns passcode headers if user_id + passcode present, else None.
    None means: no user logged in — caller should return 401."""
    user_id  = req.headers.get("X-NER-User-Id") or req.args.get("ner_user_id")
    passcode = req.headers.get("X-NER-Passcode") or req.args.get("ner_passcode")
    if user_id and passcode:
        return _passcode_headers(user_id, passcode)
    return None  # no user credentials — do not fall back to platform key
_cache: dict = {}

_sse_queues: set = set()
_sse_lock = threading.Lock()
_sse_state: dict = {}

def sse_broadcast(event_type: str, data: dict):
    payload = f"event: {event_type}\ndata: {json.dumps(data)}\n\n"
    dead = set()
    with _sse_lock:
        for q in _sse_queues:
            try: q.put_nowait(payload)
            except queue.Full: dead.add(q)
        for q in dead: _sse_queues.discard(q)

def cached_get(path, params=None, auth=False, ttl=15):
    if ttl == 0:
        try:
            r = _session.get(f"{NER_BASE}{path}", headers=AUTH_H if auth else PUB_H, params=params, timeout=8)
            if r.status_code == 429: return 429, {"detail": "Rate limited by NER"}
            return r.status_code, r.json()
        except Exception as ex: return 503, {"detail": str(ex)}
    key = path + str(params) + str(auth)
    e = _cache.get(key)
    if e and time.time() - e["ts"] < ttl: return e["s"], e["d"]
    try:
        r = _session.get(f"{NER_BASE}{path}", headers=AUTH_H if auth else PUB_H, params=params, timeout=8)
        if r.status_code == 429:
            if e: return e["s"], e["d"]
            return 429, {"detail": "Rate limited by NER"}
        result = (r.status_code, r.json())
    except Exception as ex: result = (503, {"detail": str(ex)})
    if result[0] == 200: _cache[key] = {"ts": time.time(), "s": result[0], "d": result[1]}
    return result

_atlas_cache: dict = {}
_atlas_cache_lock = threading.Lock()
_ATLAS_ONLY_PREFIXES = ("/market/","/analytics/","/history/","/ohlcv/","/orderbook","/securities","/derived","/shareholders/","/price/","/holder_intel/")

def atlas_get(path, params=None, ttl=30):
    key = path + str(params or {})
    with _atlas_cache_lock:
        e = _atlas_cache.get(key)
        if e and time.time() - e["ts"] < ttl: return e["s"], e["d"]
        stale = e
    try:
        r = _session.get(f"{ATLAS_BASE}{path}", headers=ATLAS_H, params=params, timeout=6)
        try: d = r.json()
        except: d = {}
        if r.status_code == 200:
            with _atlas_cache_lock: _atlas_cache[key] = {"ts": time.time(), "s": 200, "d": d}
            return 200, d
        print(f"[atlas] {path} → HTTP {r.status_code}")
        if stale: return stale["s"], stale["d"]
        atlas_only = any(path.startswith(p) for p in _ATLAS_ONLY_PREFIXES)
        if atlas_only: return r.status_code, d
    except Exception as ex:
        print(f"[atlas] {path} exception: {ex}")
        if e: return e["s"], e["d"]
        atlas_only = any(path.startswith(p) for p in _ATLAS_ONLY_PREFIXES)
        if atlas_only: return 503, {"detail": str(ex)}
    return cached_get(path, params=params, ttl=ttl)

_ob_cache = {"ts": 0, "data": None}
_OB_TTL = 10

def get_all_orderbooks():
    global _ob_cache
    if time.time() - _ob_cache["ts"] < _OB_TTL and _ob_cache["data"] is not None: return _ob_cache["data"]
    try:
        r = _session.get(f"{ATLAS_BASE}/orderbook", headers=ATLAS_H, timeout=10)
        if r.status_code == 200:
            data = r.json(); _ob_cache = {"ts": time.time(), "data": data}; return data
    except: pass
    return _ob_cache["data"]

def get_ticker_orderbook(ticker):
    all_ob = get_all_orderbooks()
    if not isinstance(all_ob, list): return all_ob
    return next((o for o in all_ob if o.get("ticker") == ticker), None)

_sh_ticker_cache: dict = {}
_SH_TICKER_TTL = 600

def get_ticker_shareholders(ticker):
    e = _sh_ticker_cache.get(ticker)
    if e and time.time() - e["ts"] < _SH_TICKER_TTL: return e["data"]
    try:
        s_atl, data_atl = atlas_get(f"/shareholders/{ticker}", ttl=600)
        if s_atl == 200 and isinstance(data_atl, list):
            _sh_ticker_cache[ticker] = {"ts": time.time(), "data": data_atl}; return data_atl
        r = _session.get(f"{NER_BASE}/shareholders", headers=AUTH_H, params={"ticker": ticker}, timeout=6)
        if r.status_code == 200:
            data = r.json()
            if isinstance(data, list):
                _sh_ticker_cache[ticker] = {"ts": time.time(), "data": data}; return data
            print(f"[shareholders/{ticker}] unexpected: {str(data)[:120]}")
        else: print(f"[shareholders/{ticker}] HTTP {r.status_code}: {r.text[:120]}")
    except Exception as ex: print(f"[shareholders/{ticker}] exception: {ex}")
    return _sh_ticker_cache.get(ticker, {}).get("data", [])

def ner_post(path, payload, _retries=2, auth_headers=None):
    """POST to NER. auth_headers overrides AUTH_H — use _passcode_headers() for client passcode flow."""
    hdrs = auth_headers or AUTH_H
    for attempt in range(_retries + 1):
        try:
            r = _session.post(f"{NER_BASE}{path}", headers=hdrs, json=payload, timeout=(4, 10))
            if r.status_code == 429 and attempt < _retries:
                retry_after = int(r.headers.get("Retry-After", 2))
                print(f"[ner_post] 429 on {path} — retrying in {retry_after}s")
                time.sleep(min(retry_after, 5)); continue
            return r.status_code, r.json()
        except Exception as e:
            if attempt < _retries: time.sleep(1); continue
            return 503, {"detail": str(e)}
    return 429, {"detail": "Order rate limited by NER"}

def _tse_stock_to_sec(stk):
    px = stk.get("current_price")
    try: px = float(px) if px else None
    except: px = None
    return {"ticker":stk.get("symbol",""),"full_name":stk.get("company_name",stk.get("symbol","")),"market_price":px,"change_pct":None,"frozen":stk.get("status")=="halted","total_shares":stk.get("total_shares",0),"sector":stk.get("sector",""),"exchange":"TSE","_tse_id":stk.get("stock_id","")}


# ── Alias old private names → quant_utils (keeps existing routes working) ─────
_sma  = Q.sma;  _ema  = Q.ema;  _rsi  = Q.rsi;  _atr  = Q.atr
_bollinger = Q.bollinger; _macd = Q.macd; _vwap_series = Q.vwap_series
_ann_vol   = Q.ann_vol;   _max_drawdown = Q.max_drawdown; _sharpe = Q.sharpe
_mean_reversion = Q.mean_reversion_score; _downside_vol = Q.downside_vol
_bt_run    = Q.bt_run;    _bt_metrics   = Q.bt_metrics
_norm_candles = D.norm_candles; _build_candles_from_history = D.build_candles_from_history

# ─── TSE market data proxy routes ──────────────────────────────────────────

@app.route("/api/tse/stock/<symbol>")
def tse_stock(symbol):
    """Full TSE stock detail (sector, volume, high/low 24h, market_cap)."""
    s, d = tse_get(f"/api/v1/stocks/{symbol}", ttl=15)
    return jsonify(d), s

@app.route("/api/tse/candles/<symbol>")
def tse_candles(symbol):
    """OHLCV candles from TSE. Supports interval=30s|1m|5m|15m|1h and limit."""
    interval = request.args.get("interval", "1h")
    limit    = request.args.get("limit", "300")
    start    = request.args.get("start")
    end      = request.args.get("end")
    params   = {"interval": interval, "limit": int(limit)}
    if start: params["start"] = int(start)
    if end:   params["end"]   = int(end)
    s, d = tse_get(f"/api/v1/market/{symbol}/candles", params=params, ttl=10)
    return jsonify(d), s

@app.route("/api/tse/orderbook/<symbol>")
def tse_orderbook(symbol):
    """TSE order book for a symbol. depth param supported."""
    depth = request.args.get("depth", "20")
    s, d = tse_get(f"/api/v1/market/{symbol}/orderbook",
                   params={"depth": int(depth)}, ttl=5)
    return jsonify(d), s

@app.route("/api/tse/trades/<symbol>")
def tse_trades(symbol):
    """Recent TSE trades for a symbol."""
    limit = request.args.get("limit", "50")
    s, d = tse_get(f"/api/v1/market/{symbol}/trades",
                   params={"limit": int(limit)}, ttl=5)
    return jsonify(d), s

@app.route("/api/tse/market/<symbol>")
def tse_market(symbol):
    """TSE market summary for a symbol (bid/ask, volume, price)."""
    s, d = tse_get(f"/api/v1/market/{symbol}", ttl=10)
    return jsonify(d), s

@app.route("/api/tse/orders")
def tse_orders_open():
    """TSE open orders for the authenticated user. Requires X-TSE-User-Key header."""
    hdrs = _tse_user_headers(request)
    try:
        r = _session.get(f"{TSE_BASE}/api/v1/orders", headers=hdrs,
                         params={"status": "open", "limit": 100}, timeout=8)
        d = r.json() if r.content else {}
        orders = d if isinstance(d, list) else d.get("orders", d.get("open_orders", []))
        return jsonify({"orders": orders}), r.status_code
    except Exception as e:
        return jsonify({"orders": [], "detail": str(e)}), 503

@app.route("/api/tse/orders/history")
def tse_orders_history():
    """TSE filled/cancelled order history for the authenticated user. Requires X-TSE-User-Key."""
    hdrs = _tse_user_headers(request)
    limit = request.args.get("limit", 200)
    def _fetch(status):
        try:
            r = _session.get(f"{TSE_BASE}/api/v1/orders", headers=hdrs,
                             params={"status": status, "limit": limit}, timeout=8)
            d = r.json() if r.content else {}
            return d if isinstance(d, list) else d.get("orders", d.get("history", []))
        except Exception:
            return []
    orders = _fetch("filled") + _fetch("partial") + _fetch("cancelled")
    orders.sort(key=lambda o: o.get("updated_ts") or o.get("updated_at") or o.get("created_ts") or o.get("created_at") or "", reverse=True)
    return jsonify({"orders": orders}), 200

@app.route("/api/tse/orders/<order_id>", methods=["DELETE"])
def tse_cancel_order(order_id):
    """Cancel a TSE order by ID. Requires X-TSE-User-Key header."""
    hdrs = _tse_user_headers(request)
    try:
        r = _session.delete(f"{TSE_BASE}/api/v1/orders/{order_id}", headers=hdrs, timeout=(4, 8))
        try: return jsonify(r.json()), r.status_code
        except: return jsonify({"detail": "Cancelled"}), r.status_code
    except Exception as e:
        return jsonify({"detail": str(e)}), 503

@app.route("/api/tse/account/portfolio")
def tse_account_portfolio():
    """TSE account portfolio (holdings + cash). Requires X-TSE-User-Key header."""
    hdrs = _tse_user_headers(request)
    try:
        r = _session.get(f"{TSE_BASE}/api/v1/account/portfolio", headers=hdrs, timeout=(4, 10))
        try: return jsonify(r.json()), r.status_code
        except: return jsonify({"detail": "Bad response from TSE"}), 502
    except Exception as e:
        return jsonify({"detail": str(e)}), 503

@app.route("/api/market_price/<ticker>")
def market_price(ticker):
    s, d = atlas_get(f"/price/{ticker}", ttl=15); return jsonify({"market_price": d.get("market_price")} if s==200 else d), s

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

@app.route("/api/orderbook/<path:ticker>")
def orderbook_ticker(ticker):
    """Single-ticker orderbook — proxies to Atlas /orderbook/{ticker}.
    Handles both NER tickers (BB) and TSE tickers (TSE:ECO) since Atlas aggregates both."""
    s, d = atlas_get(f"/orderbook/{ticker}", ttl=5)
    if s != 200:
        return jsonify({"detail": "Orderbook unavailable", "status": s}), s
    return jsonify(d), 200

@app.route("/api/analytics/price_history/<ticker>")
def price_history(ticker):
    s, d = atlas_get(f"/history/{ticker}",
                     params={"days": request.args.get("days", 30)}, ttl=120)
    if s == 200 and isinstance(d, dict):
        d = d.get("data", [])
    return jsonify(d), s

@app.route("/api/history/<path:ticker>")
def trade_history(ticker):
    """Raw trade history from Atlas — used by TSE ticker page for recent prints."""
    limit = int(request.args.get("limit", 50))
    s, d = atlas_get(f"/history/{ticker}", params={"limit": limit}, ttl=15)
    return jsonify(d), s

@app.route("/api/analytics/ohlcv/<path:ticker>")
def ohlcv(ticker):
    days = request.args.get("days", 30)
    s, d = atlas_get(f"/analytics/ohlcv/{ticker}", params={"days": days}, ttl=120)
    if s != 200 or not isinstance(d, dict):
        return jsonify({"ticker": ticker, "candles": [], "detail": "No data"}), 200
    # Debug: log raw candle shape for TSE tickers so we can see the date format
    if ticker.startswith("TSE:"):
        raw_candles = d.get("candles", [])
        sample = raw_candles[:2] if raw_candles else []
        print(f"[ohlcv debug] {ticker}: {len(raw_candles)} candles, keys={list(sample[0].keys()) if sample else []}, sample={sample}", flush=True)
    # Normalise candle dates
    d["candles"] = _norm_candles(d.get("candles", []))
    if ticker.startswith("TSE:"):
        normed = d["candles"]
        print(f"[ohlcv debug] {ticker}: after norm, {len(normed)} candles, first_date={normed[0].get('date') if normed else None}", flush=True)
    return jsonify(d), 200

@app.route("/api/portfolio")
def portfolio():
    auth_hdrs = _get_user_auth(request)
    if not auth_hdrs:
        return jsonify({"detail": "Login required — set your Discord ID and passcode in the terminal."}), 401
    try:
        r = _session.get(f"{NER_BASE}/portfolio", headers=auth_hdrs, timeout=8)
        return jsonify(r.json()), r.status_code
    except Exception as e:
        return jsonify({"detail": str(e)}), 503

# ── Trading ───────────────────────────────────────────────────────────────────
def _tse_user_headers(req=None):
    """Return TSE headers using the user's personal API key if provided, else platform key."""
    user_key = (req.headers.get("X-TSE-User-Key") or "").strip() if req else ""
    if user_key:
        return {"X-API-Key": user_key, "Content-Type": "application/json"}
    return dict(TSE_H)

def tse_post(path, payload, req=None):
    """POST to TSE exchange API."""
    try:
        hdrs = _tse_user_headers(req)
        r = _session.post(f"{TSE_BASE}{path}", headers=hdrs, json=payload, timeout=(4, 10))
        try:
            d = r.json()
        except Exception:
            d = {}
        return r.status_code, d
    except Exception as e:
        return 503, {"detail": str(e)}

def _build_tse_order(payload, side, order_type):
    """Translate terminal payload to TSE OrderCreate schema."""
    import uuid as _uuid
    ticker = str(payload.get("ticker", ""))
    symbol = ticker.split(":", 1)[-1] if ":" in ticker else ticker
    tse_payload = {
        "instrument_type": "stock",
        "symbol":          symbol,
        "side":            side,
        "order_type":      order_type,
        "quantity":        payload.get("quantity"),
        "idempotency_key": str(_uuid.uuid4()),
        "time_in_force":   "GTC",
    }
    if order_type == "limit":
        tse_payload["limit_price"] = payload.get("limit_price")
    return tse_payload

def _is_tse(payload):
    return str(payload.get("ticker","")).startswith("TSE:")

def _route_order(ner_path, side, order_type, payload, req=None):
    """Route to TSE exchange for TSE tickers, NER for everything else.
    req: Flask request object — passcode auth required for NER orders."""
    if _is_tse(payload):
        return tse_post("/api/v1/orders", _build_tse_order(payload, side, order_type), req=req)
    auth_hdrs = _get_user_auth(req) if req else None
    if not auth_hdrs:
        return 401, {"detail": "Login required — set your Discord ID and passcode in the terminal."}
    return ner_post(ner_path, payload, auth_headers=auth_hdrs)


# ── NER PASSCODE SESSION ──────────────────────────────────────────────────────
# Lets users set their Discord ID + passcode once; the terminal stores it
# in the browser session via localStorage on the client side.
# The server just validates the credentials work by calling /portfolio.

@app.route("/api/ner_auth/validate", methods=["POST"])
def ner_auth_validate():
    """Validate a user_id + passcode pair against the NER API.
    Returns 200 + portfolio summary if valid, or the NER error if not."""
    body     = request.get_json(silent=True) or {}
    user_id  = str(body.get("user_id","")).strip()
    passcode = str(body.get("passcode","")).strip()
    if not user_id or not passcode:
        return jsonify({"detail": "user_id and passcode required"}), 400
    hdrs = _passcode_headers(user_id, passcode)
    try:
        r = _session.get(f"{NER_BASE}/portfolio", headers=hdrs, timeout=8)
        return jsonify(r.json()), r.status_code
    except Exception as e:
        return jsonify({"detail": str(e)}), 503

@app.route("/api/orders/buy_limit",   methods=["POST"])
def buy_limit():
    s,d=_route_order("/orders/buy_limit",  "buy",  "limit",  request.json, request)
    return jsonify(d),s

@app.route("/api/orders/sell_limit",  methods=["POST"])
def sell_limit():
    s,d=_route_order("/orders/sell_limit", "sell", "limit",  request.json, request)
    return jsonify(d),s

@app.route("/api/orders/buy_market",  methods=["POST"])
def buy_market():
    s,d=_route_order("/orders/buy_market", "buy",  "market", request.json, request)
    return jsonify(d),s

@app.route("/api/orders/sell_market", methods=["POST"])
def sell_market():
    s,d=_route_order("/orders/sell_market","sell", "market", request.json, request)
    return jsonify(d),s

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

@app.route("/api/ticker_stats/<path:ticker>")
def ticker_stats(ticker):
    days = int(request.args.get("days", 60))
    s, raw = atlas_get(f"/analytics/ohlcv/{ticker}", params={"days": days}, ttl=60)
    if s != 200 or not isinstance(raw, dict):
        return jsonify({"detail": "No data from Atlas", "status": s}), 404
    candles = _norm_candles(raw.get("candles", []), days=days)
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
        s, raw = atlas_get(f"/history/{t}", params={"days": days}, ttl=60)
        if s != 200 or not isinstance(raw, dict): continue
        pts = raw.get("data", [])
        if not isinstance(pts, list) or len(pts) == 0: continue
        # Sort oldest-first (Atlas returns newest-first)
        pts_sorted = sorted(pts, key=lambda p: p.get("timestamp",""))
        base = pts_sorted[0].get("price")
        if not base or base == 0: continue
        result[t] = [{"ts": p["timestamp"][:10], "norm": round((p["price"]/base - 1)*100, 4)} for p in pts_sorted]
    return jsonify(result)

# ── MARKET BREADTH ────────────────────────────────────────────────────────────

def _parallel_ohlcv(tickers, days, ttl=600):
    """Fetch /analytics/ohlcv for multiple tickers in parallel. Returns {ticker: raw_dict}."""
    results = {}
    def fetch(t):
        s, d = atlas_get(f"/analytics/ohlcv/{t}", params={"days": days}, ttl=ttl)
        return t, (d if s == 200 and isinstance(d, dict) else None)
    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = {ex.submit(fetch, t): t for t in tickers}
        for f in as_completed(futures):
            t, d = f.result()
            results[t] = d
    return results

@app.route("/api/securities")
def securities():
    """List of all securities with metadata. Used by terminal init, market page, and many others."""
    s, d = atlas_get("/securities", ttl=60)
    if s != 200:
        s, d = cached_get("/securities", ttl=60)
    if isinstance(d, list):
        d = [sec for sec in d if _is_active(sec.get("ticker",""))]
    return jsonify(d), s

@app.route("/api/securities/<path:ticker>")
def security_detail(ticker):
    """Single security detail."""
    s, d = atlas_get(f"/securities/{ticker}", ttl=60)
    if s != 200:
        s, d = cached_get(f"/securities/{ticker}", ttl=60)
    return jsonify(d), s

@app.route("/api/market_breadth")
def market_breadth():
    """Parallel-fetch version — all OHLCV fired concurrently, no serial loops."""
    days = int(request.args.get("days", 7))

    # Fetch securities meta + breadth + orderbook in parallel
    with ThreadPoolExecutor(max_workers=3) as ex:
        f_secs  = ex.submit(atlas_get, "/securities",     None,           120)
        f_bread = ex.submit(atlas_get, "/market/breadth", {"days": days}, 30)
        f_ob    = ex.submit(atlas_get, "/orderbook",      None,           10)
        _, secs_raw = f_secs.result()
        s_br, d_br  = f_bread.result()
        _, ob_all   = f_ob.result()

    sec_meta = {}
    if isinstance(secs_raw, list):
        for s in secs_raw:
            if _is_active(s["ticker"]):
                sec_meta[s["ticker"]] = s

    all_tickers = list(sec_meta.keys())

    # Fire all OHLCV calls in parallel (cache hits after first load = ~0ms)
    ohlcv_30  = _parallel_ohlcv(all_tickers, 30,   ttl=600)
    ohlcv_day = _parallel_ohlcv(all_tickers, days, ttl=600) if days != 30 else ohlcv_30

    raw_secs = []
    if s_br == 200 and isinstance(d_br, dict) and d_br.get("securities"):
        raw_secs = d_br["securities"] if isinstance(d_br["securities"], list) else []

    enriched = []
    tickers_seen = set()

    for item in raw_secs:
        ticker = item.get("ticker", "")
        if not ticker or not _is_active(ticker): continue
        tickers_seen.add(ticker)
        meta    = sec_meta.get(ticker, {})
        derived = meta.get("derived", {}) or {}
        lp      = meta.get("market_price") or item.get("last_price") or 0

        vol       = item.get("volatility") or derived.get("volatility_7d")
        shrp      = item.get("sharpe")
        chg_pct   = item.get("chg_pct")
        prd_hi    = item.get("hi52")
        prd_lo    = item.get("lo52")
        vol_spike = item.get("vol_spike")
        vlday     = []

        o30 = ohlcv_30.get(ticker)
        if o30 and isinstance(o30, dict):
            cs30 = _norm_candles(o30.get("candles", []), days=30)
            if len(cs30) >= 2:
                cl30 = [c["close"] for c in cs30]
                if vol  is None: vol  = _ann_vol(cl30)
                if shrp is None: shrp = _sharpe(cl30)

        oday = ohlcv_day.get(ticker)
        if oday and isinstance(oday, dict):
            csday = _norm_candles(oday.get("candles", []), days=days)
            if len(csday) >= 2:
                clday = [c["close"]  for c in csday]
                vlday = [c["volume"] for c in csday]
                if chg_pct is None:
                    chg_pct = round((clday[-1]-clday[0])/clday[0]*100, 2) if clday[0] else 0
                if prd_hi    is None: prd_hi    = round(max(clday), 4)
                if prd_lo    is None: prd_lo    = round(min(clday), 4)
                if vol_spike is None and len(vlday) > 1:
                    avg_v     = sum(vlday[:-1])/max(len(vlday)-1, 1)
                    vol_spike = round(vlday[-1]/avg_v, 2) if avg_v > 0 else None

        enriched.append({
            "ticker":     ticker,
            "name":       meta.get("full_name", ticker),
            "exchange":   "TSE" if ticker.startswith("TSE:") else "NER",
            "last_price": lp,
            "chg_pct":    chg_pct if chg_pct is not None else 0,
            "volatility": vol,
            "sharpe":     shrp,
            "market_cap": item.get("market_cap") or round(lp*(meta.get("total_shares") or 0), 2),
            "volume":     sum(vlday) if vlday else 0,
            "hi52":       prd_hi,
            "lo52":       prd_lo,
            "vol_spike":  vol_spike,
            "frozen":     bool(meta.get("frozen", False)),
        })

    # Any tickers in sec_meta not returned by breadth
    for ticker, meta in sec_meta.items():
        if ticker in tickers_seen: continue
        lp   = meta.get("market_price") or 0
        oday = ohlcv_day.get(ticker)
        csday, clday, vlday = [], [], []
        if oday and isinstance(oday, dict):
            csday = _norm_candles(oday.get("candles", []), days=days)
        if len(csday) >= 2:
            clday     = [c["close"]  for c in csday]
            vlday     = [c["volume"] for c in csday]
            chg_pct   = round((clday[-1]-clday[0])/clday[0]*100, 2) if clday[0] else 0
            avg_v     = sum(vlday[:-1])/max(len(vlday)-1,1) if len(vlday)>1 else (vlday[0] or 1)
            enriched.append({
                "ticker": ticker, "name": meta.get("full_name", ticker),
                "exchange": "TSE" if ticker.startswith("TSE:") else "NER",
                "last_price": lp, "chg_pct": chg_pct,
                "volatility": _ann_vol(clday), "sharpe": _sharpe(clday),
                "market_cap": round(lp*(meta.get("total_shares") or 0), 2),
                "volume": sum(vlday), "hi52": max(clday), "lo52": min(clday),
                "vol_spike": round(vlday[-1]/avg_v,2) if avg_v>0 else None,
                "frozen": bool(meta.get("frozen", False)),
            })
        else:
            enriched.append({
                "ticker": ticker, "name": meta.get("full_name", ticker),
                "exchange": "TSE" if ticker.startswith("TSE:") else "NER",
                "last_price": lp, "chg_pct": 0, "volatility": None, "sharpe": None,
                "market_cap": round(lp*(meta.get("total_shares") or 0), 2),
                "volume": 0, "hi52": None, "lo52": None, "vol_spike": None,
                "frozen": bool(meta.get("frozen", False)),
            })

    if not enriched:
        return jsonify({"detail": "No market data available"}), 503

    # Global orderbook imbalance
    global_bid = 0; global_ask = 0
    if isinstance(ob_all, list):
        for ob in ob_all:
            global_bid += ob.get("bid_depth", 0) or 0
            global_ask += ob.get("ask_depth", 0) or 0
    total_depth  = global_bid + global_ask
    global_imbal = round((global_bid-global_ask)/max(total_depth,1)*100, 2)

    chgs = [e["chg_pct"] for e in enriched]
    ups  = [e for e in enriched if e["chg_pct"] > 0]
    dns  = [e for e in enriched if e["chg_pct"] < 0]
    flt  = [e for e in enriched if e["chg_pct"] == 0]
    vols = [e["volatility"] for e in enriched
            if isinstance(e.get("volatility"), (int,float)) and e["volatility"] > 0]

    return jsonify({
        "securities": enriched,
        "summary": {
            "total":            len(enriched),
            "advancing":        len(ups),
            "declining":        len(dns),
            "unchanged":        len(flt),
            "adv_dec_ratio":    round(len(ups)/max(len(dns),1), 2),
            "avg_return":       round(sum(chgs)/len(chgs), 2) if chgs else 0,
            "ew_index_return":  round(sum(chgs)/len(chgs), 4) if chgs else 0,
            "vol_dispersion":   round(statistics.stdev(vols), 2) if len(vols)>1 else 0,
            "global_imbalance": global_imbal,
            "total_depth":      total_depth,
        }
    }), 200


# ── GLOBAL ORDERBOOK AGGREGATOR ───────────────────────────────────────────────
@app.route("/api/market_orderbook")
def market_orderbook():
    s_secs, secs = atlas_get("/securities", ttl=60)
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
    """Exchange analytics — all OHLCV fetched in parallel."""
    days = int(request.args.get("days", 7))
    s_secs, secs = atlas_get("/securities", ttl=120)
    if s_secs != 200: return jsonify({"detail": "Cannot fetch securities"}), s_secs

    ner_secs = [s for s in secs if not s["ticker"].startswith("TSE:") and _is_active(s["ticker"])]
    tickers  = [s["ticker"] for s in ner_secs]

    # Parallel OHLCV fetch
    ohlcv = _parallel_ohlcv(tickers, days, ttl=600)

    ticker_data = []
    total_mcap  = 0
    all_vols, all_sharpes, all_chgs = [], [], []

    for sec in ner_secs:
        t   = sec["ticker"]
        raw = ohlcv.get(t)
        if not raw or not raw.get("candles"): continue
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
            "ticker":  t, "name": sec.get("full_name", ""),
            "mcap":    round(mcap, 2),
            "volume":  sum(volumes),
            "chg_pct": round(chg, 2),
            "volatility": vol, "sharpe": sh,
            "max_dd":  _max_drawdown(closes),
            "last":    closes[-1],
            "total_shares": sec.get("total_shares", 0),
        })

    ticker_data.sort(key=lambda x: x["mcap"], reverse=True)
    total_vol = sum(t["volume"] for t in ticker_data) or 1
    for t in ticker_data:
        t["vol_share_pct"]  = round(t["volume"]/total_vol*100, 2)
        t["mcap_share_pct"] = round(t["mcap"]/max(total_mcap,1)*100, 2)

    vix_proxy  = round(sum(all_vols)/len(all_vols), 2)     if all_vols    else 0
    avg_sharpe = round(sum(all_sharpes)/len(all_sharpes),3) if all_sharpes else 0
    mcap_shares = [t["mcap"]/max(total_mcap,1)*100 for t in ticker_data]
    hhi         = round(sum(s**2 for s in mcap_shares), 1)

    # Correlation matrix (fast — already have closes in memory)
    corr_matrix = {}
    closes_map  = {}
    for sec in ner_secs:
        t   = sec["ticker"]
        raw = ohlcv.get(t)
        if raw and raw.get("candles"):
            closes_map[t] = [c["close"] for c in raw["candles"]]

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
    s_sec, secs_data = atlas_get("/securities", ttl=60)
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
    """Exchange-wide shareholder intelligence — parallel fetches with semaphore throttle."""
    s_secs, secs = atlas_get("/securities", ttl=120)
    if s_secs != 200: return jsonify({"detail": "Cannot fetch securities"}), s_secs

    # Semaphore limits concurrency to 4 simultaneous shareholder calls (avoids 429)
    sem = threading.Semaphore(4)

    def fetch_one(sec):
        t = sec["ticker"]
        with sem:
            holders = get_ticker_shareholders(t)
        if not holders: return None
        total_qty = sum(h["quantity"] for h in holders)
        if total_qty == 0: return None
        shares = [h["quantity"]/total_qty for h in holders]
        hhi    = round(sum(s**2 for s in shares)*10000, 1)
        top1  = round(shares[0]*100,2) if shares else 0
        top3  = round(sum(shares[:3])*100,2) if len(shares)>=3 else round(sum(shares)*100,2)
        top5  = round(sum(shares[:5])*100,2) if len(shares)>=5 else round(sum(shares)*100,2)
        top10 = round(sum(shares[:10])*100,2)
        whales = [h for h in holders if h["quantity"]/total_qty > 0.10]
        return {
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
            "holders":    [{**h, "display_name": _resolve_holder(h.get("user_id",""))} for h in holders[:20]],
        }

    all_data = []
    with ThreadPoolExecutor(max_workers=4) as ex:
        futures = {ex.submit(fetch_one, sec): sec for sec in secs}
        for f in as_completed(futures):
            result = f.result()
            if result: all_data.append(result)

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

    whales = [{"user_id": uid, "display_name": _resolve_holder(uid), "positions": pos, "num_positions": len(pos)}
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
    s_atl, d_atl = atlas_get(f"/holder_intel/{ticker}", ttl=60)
    if s_atl == 200: return jsonify(d_atl), 200
    holders = get_ticker_shareholders(ticker)
    if not isinstance(holders, list) or not holders:
        return jsonify({"detail": "No holders found for ticker"}), 404
    _, sec_list = atlas_get("/securities", ttl=60)
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

# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════
# BACKTEST ENGINE v2  —  clean rewrite
# Core invariant: buy_hold equity curve MUST track price * (10000/price[0])
# ══════════════════════════════════════════════════════════════════════════════

def _bt_run(closes, dates, signals, sl_pct=0.0, tp_pct=0.0):
    """
    Pure simulation. Returns (equity_curve, trades).
    signals[i] in {-1, 0, 1}  (1=buy, -1=sell/short-exit)
    No fractional shares, commission = 0.5% per side, deducted from proceeds/cost.
    Key fix: qty = floor((cash * 0.995) / price)  so commission never exceeds residual cash.
    """
    COMMISSION = 0.005          # 0.5 % per trade
    cash   = 10_000.0
    shares = 0
    pos    = False
    entry  = 0.0
    equity = []
    trades = []

    for i, price in enumerate(closes):
        if price is None or price <= 0:
            equity.append(round(cash + shares * (closes[i-1] if i else 0), 4))
            continue

        sig = signals[i]

        # ── Stop-loss / take-profit ──────────────────────────────────────────
        if pos and entry > 0:
            hit_sl = sl_pct > 0 and price <= entry * (1 - sl_pct)
            hit_tp = tp_pct > 0 and price >= entry * (1 + tp_pct)
            if hit_sl or hit_tp:
                proceeds     = shares * price * (1 - COMMISSION)
                cash        += proceeds
                action       = "STOP" if hit_sl else "TP"
                trades.append({"date": dates[i], "action": action,
                                "price": round(price, 4), "qty": shares,
                                "value": round(proceeds, 2)})
                shares = 0; pos = False; entry = 0.0

        # ── Entry ────────────────────────────────────────────────────────────
        if sig == 1 and not pos:
            # Buy as many shares as cash allows AFTER commission
            qty = int(cash / (price * (1 + COMMISSION)))
            if qty > 0:
                cost    = qty * price * (1 + COMMISSION)
                cash   -= cost
                shares  = qty
                pos     = True
                entry   = price
                trades.append({"date": dates[i], "action": "BUY",
                                "price": round(price, 4), "qty": qty,
                                "value": round(cost, 2)})

        # ── Exit ─────────────────────────────────────────────────────────────
        elif sig == -1 and pos and shares > 0:
            proceeds     = shares * price * (1 - COMMISSION)
            cash        += proceeds
            trades.append({"date": dates[i], "action": "SELL",
                            "price": round(price, 4), "qty": shares,
                            "value": round(proceeds, 2)})
            shares = 0; pos = False; entry = 0.0

        equity.append(round(cash + shares * price, 4))

    return equity, trades


def _bt_metrics(equity, closes, trades):
    start_eq = equity[0] if equity else 10_000
    end_eq   = equity[-1] if equity else 10_000
    total_ret  = round((end_eq - 10_000) / 10_000 * 100, 2)
    bh_ret     = round((closes[-1] - closes[0]) / closes[0] * 100, 2) if len(closes) >= 2 and closes[0] else 0
    max_dd     = _max_drawdown(equity)
    sharpe_val = _sharpe(equity)
    calmar     = round(abs(total_ret / max_dd), 3) if max_dd > 0 else 0

    # Win rate: pair BUY→SELL/STOP/TP
    buys   = [t for t in trades if t["action"] == "BUY"]
    exits  = [t for t in trades if t["action"] in ("SELL", "STOP", "TP")]
    pairs  = list(zip(buys, exits))
    wins   = sum(1 for b, e in pairs if e["price"] > b["price"])
    win_rt = round(wins / len(pairs) * 100, 1) if pairs else 0.0

    return {
        "total_return_pct":    total_ret,
        "benchmark_return_pct": bh_ret,
        "alpha_pct":           round(total_ret - bh_ret, 2),
        "max_drawdown_pct":    max_dd,
        "sharpe_ratio":        sharpe_val,
        "calmar_ratio":        calmar,
        "num_trades":          len(trades),
        "win_rate":            win_rt,
        "final_equity":        round(end_eq, 2),
    }


@app.route("/api/backtest", methods=["POST"])
def backtest():
    body     = request.json or {}
    ticker   = body.get("ticker", "").upper()
    days     = min(int(body.get("days", 90)), 365)
    strategy = body.get("strategy", "buy_hold")
    params   = body.get("params", {})

    if not ticker:
        return jsonify({"detail": "ticker required"}), 400

    # TSE tickers: Atlas /analytics/ohlcv returns empty candles.
    # Use /history/{ticker} (price_history) instead — same source as market page.
    if ticker.startswith("TSE:"):
        s2, raw2 = atlas_get(f"/history/{ticker}", params={"days": 365, "limit": 5000}, ttl=60)
        if s2 != 200:
            return jsonify(raw2), s2
        pts = raw2.get("data", []) if isinstance(raw2, dict) else raw2
        candles = _build_candles_from_history(pts, days)
    else:
        status, raw = atlas_get(f"/analytics/ohlcv/{ticker}", params={"days": days}, ttl=60)
        if status != 200:
            return jsonify(raw), status
        candles = _norm_candles(raw.get("candles", []), days=days)

    if len(candles) < 3:
        return jsonify({"detail": f"Not enough data ({len(candles)} candles)"}), 400

    closes = [float(c["close"]) for c in candles]
    dates  = [c["date"]         for c in candles]
    highs  = [float(c.get("high",  c["close"])) for c in candles]
    lows   = [float(c.get("low",   c["close"])) for c in candles]

    # ── Signal generation ────────────────────────────────────────────────────
    signals = [0] * len(closes)

    if strategy == "buy_hold":
        # One buy on day 0, never sell — equity tracks price exactly
        signals[0] = 1

    elif strategy == "sma_cross":
        fast = int(params.get("fast", 5))
        slow = int(params.get("slow", 20))
        f_ma = _sma(closes, fast)
        s_ma = _sma(closes, slow)
        for i in range(1, len(closes)):
            if f_ma[i] and s_ma[i] and f_ma[i-1] and s_ma[i-1]:
                if f_ma[i] > s_ma[i] and f_ma[i-1] <= s_ma[i-1]:  signals[i] =  1
                elif f_ma[i] < s_ma[i] and f_ma[i-1] >= s_ma[i-1]: signals[i] = -1

    elif strategy == "rsi":
        p   = int(params.get("rsi_period", 14))
        ob  = float(params.get("overbought", 70))
        os_ = float(params.get("oversold",   30))
        rsi = _rsi(closes, p)
        for i in range(1, len(closes)):
            if rsi[i] and rsi[i-1]:
                if   rsi[i] < os_ and rsi[i-1] >= os_: signals[i] =  1
                elif rsi[i] > ob  and rsi[i-1] <= ob:  signals[i] = -1

    elif strategy == "macd":
        ml, sl2, _ = _macd(closes,
                            int(params.get("fast", 12)),
                            int(params.get("slow", 26)),
                            int(params.get("signal", 9)))
        for i in range(1, len(closes)):
            if ml[i] and sl2[i] and ml[i-1] and sl2[i-1]:
                if   ml[i] > sl2[i] and ml[i-1] <= sl2[i-1]: signals[i] =  1
                elif ml[i] < sl2[i] and ml[i-1] >= sl2[i-1]: signals[i] = -1

    elif strategy == "bb_reversal":
        bb_u, _, bb_l = _bollinger(closes)
        for i in range(1, len(closes)):
            if bb_l[i] and closes[i] < bb_l[i] and closes[i-1] >= bb_l[i-1]:  signals[i] =  1
            elif bb_u[i] and closes[i] > bb_u[i] and closes[i-1] <= bb_u[i-1]: signals[i] = -1

    # ── Simulate ─────────────────────────────────────────────────────────────
    sl_pct = float(params.get("stop_loss_pct",   0)) / 100
    tp_pct = float(params.get("take_profit_pct", 0)) / 100
    equity, trades = _bt_run(closes, dates, signals, sl_pct=sl_pct, tp_pct=tp_pct)
    metrics        = _bt_metrics(equity, closes, trades)

    # Buy-and-hold reference curve (always correct regardless of strategy)
    bh_curve = [round(10_000 * p / closes[0], 2) for p in closes]

    return jsonify({
        "ticker":      ticker,
        "strategy":    strategy,
        "days":        days,
        "dates":       dates,
        "closes":      closes,
        "equity_curve": equity,
        "bh_curve":    bh_curve,
        "signals":     signals,
        "trades":      trades,
        "metrics":     metrics,
    })


# ══════════════════════════════════════════════════════════════════════════════
# SESSION PERSISTENCE — algo jobs & watchlist
# ══════════════════════════════════════════════════════════════════════════════
import json as _json_mod

_SESSION_FILE = os.path.join(os.path.dirname(__file__), "session_state.json")

def _load_session():
    try:
        if os.path.exists(_SESSION_FILE):
            with open(_SESSION_FILE, "r") as f:
                return _json_mod.load(f)
    except Exception:
        pass
    return {"algos": [], "watchlist": []}

def _save_session(data):
    try:
        with open(_SESSION_FILE, "w") as f:
            _json_mod.dump(data, f)
    except Exception as ex:
        print(f"[session] save error: {ex}")

@app.route("/api/session/algos", methods=["GET"])
def session_algos_get():
    """Return persisted algo jobs (serializable fields only, no tick functions)."""
    s = _load_session()
    return jsonify(s.get("algos", [])), 200

@app.route("/api/session/algos", methods=["POST"])
def session_algos_post():
    """Save/replace the full algo jobs list."""
    try:
        jobs = request.get_json(silent=True) or []
        s = _load_session()
        s["algos"] = jobs
        _save_session(s)
        return jsonify({"ok": True, "count": len(jobs)}), 200
    except Exception as ex:
        return jsonify({"ok": False, "detail": str(ex)}), 500

@app.route("/api/session/algos/<job_id>", methods=["DELETE"])
def session_algos_delete(job_id):
    """Remove a single job by id."""
    s = _load_session()
    before = len(s.get("algos", []))
    s["algos"] = [j for j in s.get("algos", []) if j.get("id") != job_id]
    _save_session(s)
    return jsonify({"ok": True, "removed": before - len(s["algos"])}), 200

@app.route("/api/session/watchlist", methods=["GET"])
def session_watchlist_get():
    s = _load_session()
    return jsonify(s.get("watchlist", [])), 200

@app.route("/api/session/watchlist", methods=["POST"])
def session_watchlist_post():
    try:
        wl = request.get_json(silent=True) or []
        s = _load_session()
        s["watchlist"] = wl
        _save_session(s)
        return jsonify({"ok": True}), 200
    except Exception as ex:
        return jsonify({"ok": False, "detail": str(ex)}), 500

# ── WEBHOOK RECEIVER (NER → Flask → SSE clients) ─────────────────────────────
@app.route("/webhook/ner", methods=["POST"])
def webhook_ner():
    """
    Receives NER market_update webhooks.
    Payload: {event, ticker, market_price, frozen, orderbook, updated_at}
    Stores latest state and broadcasts to all SSE clients.
    """
    try:
        data = request.get_json(silent=True) or {}
        ticker = data.get("ticker")
        if not ticker:
            return jsonify({"ok": False, "detail": "No ticker"}), 400
        # Silently drop updates for delisted securities
        if not _is_active(ticker):
            return jsonify({"ok": True, "detail": "delisted — ignored"}), 200

        # Update SSE state cache
        _sse_state[ticker] = {
            "ticker":       ticker,
            "market_price": data.get("market_price"),
            "frozen":       data.get("frozen", False),
            "orderbook":    data.get("orderbook", {}),
            "updated_at":   data.get("updated_at", ""),
        }

        # Broadcast to all SSE subscribers
        sse_broadcast("market_update", _sse_state[ticker])
        return jsonify({"ok": True}), 200
    except Exception as ex:
        return jsonify({"ok": False, "detail": str(ex)}), 500


# ── SSE STREAM (Flask → browser) ──────────────────────────────────────────────
@app.route("/api/stream")
def sse_stream():
    """
    Server-Sent Events endpoint.
    Immediately sends current state for all known tickers, then streams live updates.
    Clients: EventSource('/api/stream')
    """
    def generate():
        q = queue.Queue(maxsize=200)
        with _sse_lock:
            _sse_queues.add(q)

        # Send current state snapshot to new subscriber
        try:
            if _sse_state:
                snapshot = {"tickers": list(_sse_state.values())}
                yield f"event: snapshot\ndata: {json.dumps(snapshot)}\n\n"

            # Heartbeat comment every 25s to keep connection alive through proxies
            last_hb = time.time()
            while True:
                try:
                    msg = q.get(timeout=25)
                    yield msg
                    last_hb = time.time()
                except queue.Empty:
                    # Send SSE comment as heartbeat (ignored by EventSource)
                    yield f": heartbeat {int(time.time())}\n\n"
                    last_hb = time.time()
        except GeneratorExit:
            pass
        finally:
            with _sse_lock:
                _sse_queues.discard(q)

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # disable nginx buffering
            "Connection": "keep-alive",
        }
    )


# ── SSE STATE ENDPOINT (for initial load without waiting for webhook) ──────────
@app.route("/api/stream/state")
def sse_state():
    """Returns current SSE state snapshot (for clients that missed the snapshot event)."""
    return jsonify({"tickers": list(_sse_state.values())}), 200





# ── PAGES ─────────────────────────────────────────────────────────────────────
@app.route("/api/atlas-proxy")
def atlas_proxy():
    path   = request.args.get("path", "/status")
    params = {k: v for k, v in request.args.items() if k != "path"}
    s, d   = atlas_get(path, params=params or None, ttl=10)
    return jsonify(d), s

@app.route("/")
def index(): return render_template("terminal.html")

@app.route("/enterprise")
def enterprise():
    return "Enterprise edition not available in this build.", 404

@app.route("/enterprise/page/<pg>")
def enterprise_page(pg):
    return "Enterprise edition not available in this build.", 404

@app.route("/page/<n>")
def page(n):
    allowed = ["market","ticker","portfolio","orders","backtest","compare","watchlist",
               "heatmap","exchange","liquidity","holders","screener","alerts",
               "fundamentals","transactions","news","debug"]
    if n not in allowed: return "Not found", 404
    return render_template(f"pages/{n}.html")


# ── ENTERPRISE API — Portfolio CRUD + Analytics ────────────────────────────────
# All persistence via enterprise_store.py, math via quant_utils.py

@app.route("/api/enterprise/portfolios", methods=["GET"])
def ent_portfolios_get():
    return jsonify(ES.get_all_portfolios()), 200

@app.route("/api/enterprise/portfolios", methods=["POST"])
def ent_portfolios_post():
    body = request.get_json(silent=True) or {}
    return jsonify(ES.upsert_portfolio(body)), 200

@app.route("/api/enterprise/portfolios/<pf_id>", methods=["GET"])
def ent_portfolio_get(pf_id):
    pf = ES.get_portfolio(pf_id)
    if not pf: return jsonify({"detail": "Not found"}), 404
    return jsonify(pf), 200

@app.route("/api/enterprise/portfolios/<pf_id>", methods=["PATCH"])
def ent_portfolio_patch(pf_id):
    body = request.get_json(silent=True) or {}
    pf = ES.update_portfolio_meta(pf_id, body)
    if not pf: return jsonify({"detail": "Not found"}), 404
    return jsonify(pf), 200

@app.route("/api/enterprise/portfolios/<pf_id>", methods=["DELETE"])
def ent_portfolios_delete(pf_id):
    removed = ES.delete_portfolio(pf_id)
    return jsonify({"ok": True, "removed": removed}), 200

@app.route("/api/enterprise/portfolios/<pf_id>/positions", methods=["POST"])
def ent_add_position(pf_id):
    body = request.get_json(silent=True) or {}
    pos  = ES.add_position(pf_id, body)
    if pos is None: return jsonify({"detail": "Portfolio not found"}), 404
    return jsonify(pos), 200

@app.route("/api/enterprise/portfolios/<pf_id>/positions/<pos_id>", methods=["DELETE"])
def ent_remove_position(pf_id, pos_id):
    removed = ES.remove_position(pf_id, pos_id)
    return jsonify({"ok": True, "removed": removed}), 200

@app.route("/api/enterprise/portfolios/<pf_id>/positions/<pos_id>", methods=["PATCH"])
def ent_update_position(pf_id, pos_id):
    body = request.get_json(silent=True) or {}
    pos  = ES.update_position(pf_id, pos_id, body)
    if pos is None: return jsonify({"detail": "Not found"}), 404
    return jsonify(pos), 200

@app.route("/api/enterprise/portfolios/<pf_id>/import", methods=["POST"])
def ent_import_portfolio(pf_id):
    """
    Parse and import a NER portfolio table.
    Accepts either:
      - body.raw_text: the raw ASCII table pasted from NER terminal
      - body.positions: pre-parsed [{ticker, qty, entry_price}]
    Returns {imported: N, skipped: N, positions: [...]}
    """
    import re
    body = request.get_json(silent=True) or {}
    pf   = ES.get_portfolio(pf_id)
    if not pf: return jsonify({"detail": "Portfolio not found"}), 404

    positions_to_add = []

    if body.get("positions"):
        # Pre-parsed from frontend
        positions_to_add = body["positions"]
    elif body.get("raw_text"):
        # Parse the ASCII table
        # Format: | BB  | 35 | $ 31.31 | $ 35.28 | $ 139.04 |
        raw = body["raw_text"]
        for line in raw.splitlines():
            line = line.strip()
            if not line.startswith("|"): continue
            parts = [p.strip() for p in line.split("|") if p.strip()]
            if len(parts) < 2: continue
            ticker = parts[0].upper().strip()
            # Skip header rows
            if ticker in ("TICKER", "---", "") or ticker.startswith("-"): continue
            try:
                qty = float(parts[1].replace(",","").strip())
                # Avg cost: strip $ and commas
                avg_cost_str = parts[2].replace("$","").replace(",","").strip() if len(parts)>2 else "0"
                avg_cost = float(avg_cost_str)
                if qty <= 0 or avg_cost <= 0: continue
                positions_to_add.append({
                    "ticker":      ticker,
                    "qty":         qty,
                    "entry_price": avg_cost,
                    "entry_date":  "",
                    "notes":       "Imported from NER terminal",
                    "type":        "long"
                })
            except (ValueError, IndexError):
                continue

    if not positions_to_add:
        return jsonify({"detail": "No valid positions parsed", "imported": 0, "skipped": 0}), 400

    # Skip tickers already in portfolio to avoid duplicates
    existing_tickers = {p["ticker"] for p in pf.get("positions", [])}
    to_add   = [p for p in positions_to_add if p["ticker"] not in existing_tickers]
    skipped  = len(positions_to_add) - len(to_add)

    added = ES.bulk_add_positions(pf_id, to_add)
    return jsonify({
        "imported":  len(added),
        "skipped":   skipped,
        "positions": added
    }), 200


@app.route("/api/enterprise/portfolios/<pf_id>/positions/<pos_id>/close", methods=["POST"])
def ent_close_position(pf_id, pos_id):
    """
    Close all or part of a position.
    body: {close_price, close_qty, close_date, notes}
    Records realised P&L in cash_log, reduces qty (removes if fully closed).
    """
    from datetime import datetime as _dt2, timezone as _tz2
    body        = request.get_json(silent=True) or {}
    close_price = float(body.get("close_price") or 0)
    close_date  = body.get("close_date") or _dt2.now(_tz2.utc).strftime("%Y-%m-%d")
    close_notes = body.get("notes") or ""

    pf = ES.get_portfolio(pf_id)
    if not pf: return jsonify({"detail": "Portfolio not found"}), 404
    pos = next((p for p in pf.get("positions", []) if p["id"] == pos_id), None)
    if not pos: return jsonify({"detail": "Position not found"}), 404

    entry_price = float(pos.get("entry_price") or 0)
    open_qty    = float(pos.get("qty") or 0)
    close_qty   = float(body.get("close_qty") or open_qty)  # default: full close
    close_qty   = min(close_qty, open_qty)  # can't close more than held

    realised_pnl   = (close_price - entry_price) * close_qty
    proceeds       = close_price * close_qty
    remaining_qty  = open_qty - close_qty

    # Add proceeds to cash
    note = f"Close {pos['ticker']} {int(close_qty)}@{close_price:.4f} | PnL {'+' if realised_pnl>=0 else ''}{realised_pnl:.2f}"
    if close_notes: note += f" | {close_notes}"
    ES.adjust_cash(pf_id, proceeds, note)

    # Record close event in position history
    if "closes" not in pos: pos["closes"] = []
    pos["closes"].append({
        "close_price": close_price,
        "close_qty":   close_qty,
        "close_date":  close_date,
        "realised_pnl": round(realised_pnl, 2),
        "proceeds":    round(proceeds, 2),
        "notes":       close_notes,
        "ts":          _dt2.now(_tz2.utc).isoformat()
    })

    if remaining_qty <= 0:
        # Fully closed — remove position
        ES.remove_position(pf_id, pos_id)
        return jsonify({"status": "closed", "realised_pnl": round(realised_pnl,2), "removed": True}), 200
    else:
        # Partial close — reduce qty
        ES.update_position(pf_id, pos_id, {"qty": remaining_qty, "closes": pos["closes"]})
        return jsonify({"status": "partial", "remaining_qty": remaining_qty, "realised_pnl": round(realised_pnl,2), "removed": False}), 200

@app.route("/api/enterprise/portfolios/<pf_id>/cash", methods=["POST"])
def ent_cash_adjustment(pf_id):
    body   = request.get_json(silent=True) or {}
    result = ES.adjust_cash(pf_id, float(body.get("amount", 0)), body.get("note",""))
    if result is None: return jsonify({"detail": "Portfolio not found"}), 404
    return jsonify(result), 200

@app.route("/api/enterprise/portfolios/<pf_id>/analytics")
def ent_portfolio_analytics(pf_id):
    """Enrich positions with live prices + per-position risk metrics."""
    pf = ES.get_portfolio(pf_id)
    if not pf: return jsonify({"detail": "Portfolio not found"}), 404

    positions = pf.get("positions", [])
    enriched  = []
    total_cost = 0.0; total_value = 0.0

    for pos in positions:
        ticker      = pos["ticker"]
        entry_price = float(pos.get("entry_price") or 0)
        qty         = float(pos.get("qty") or 0)
        cost        = entry_price * qty

        # Live price via data_utils
        live_px = D.fetch_live_price(ticker, atlas_get)

        mkt_value = (live_px or entry_price) * qty
        pnl       = mkt_value - cost
        pnl_pct   = (pnl/cost*100) if cost > 0 else 0

        # Risk metrics — try 30d first, fall back to 90d, then use entry vs live for delisted
        cs = D.fetch_candles(ticker, 30, atlas_get)
        if len(cs) < 2:
            cs = D.fetch_candles(ticker, 90, atlas_get)
        vol = None; shr = None; sor = None; mdd = None
        if len(cs) >= 2:
            cl  = [c["close"] for c in cs]
            vol = Q.ann_vol(cl); shr = Q.sharpe(cl)
            sor = Q.sortino(cl); mdd = Q.max_drawdown(cl)
        elif live_px and entry_price and entry_price > 0:
            # Delisted or no history — synthesise from entry vs live
            pnl_r = (live_px - entry_price) / entry_price
            mdd   = round(max(0, -pnl_r) * 100, 2)

        total_cost  += cost
        total_value += mkt_value
        enriched.append({
            **pos,
            "live_price":   round(live_px, 4) if live_px else None,
            "cost":         round(cost, 2),
            "market_value": round(mkt_value, 2),
            "pnl":          round(pnl, 2),
            "pnl_pct":      round(pnl_pct, 2),
            "weight_pct":   0,
            "ann_vol":      vol, "sharpe": shr,
            "sortino":      sor, "max_dd": mdd,
        })

    for e in enriched:
        e["weight_pct"] = round(e["market_value"]/total_value*100, 2) if total_value > 0 else 0

    cash            = float(pf.get("cash", 0))
    total_with_cash = total_value + cash
    total_pnl       = total_value - total_cost
    weights         = [e["weight_pct"]/100 for e in enriched]
    concentration_hhi = Q.hhi(weights) if weights else 0

    return jsonify({
        "id":       pf_id,
        "name":     pf.get("name",""),
        "client":   pf.get("client",""),
        "notes":    pf.get("notes",""),
        "strategy": pf.get("strategy",""),
        "positions": enriched,
        "cash":     cash,
        "cash_log": pf.get("cash_log", []),
        "summary": {
            "total_cost":       round(total_cost, 2),
            "total_value":      round(total_value, 2),
            "total_with_cash":  round(total_with_cash, 2),
            "total_pnl":        round(total_pnl, 2),
            "total_pnl_pct":    round(total_pnl/total_cost*100, 2) if total_cost > 0 else 0,
            "num_positions":    len(enriched),
            "hhi":              concentration_hhi,
            "concentration":    "HIGH" if concentration_hhi>3000 else "MEDIUM" if concentration_hhi>1500 else "LOW",
            "cash_pct":         round(cash/total_with_cash*100, 2) if total_with_cash > 0 else 100,
        },
        "realized_pnl":  round(sum(
            float(c.get("realised_pnl",0))
            for p in positions for c in p.get("closes",[])
        ), 2),
        "dividend_income": round(sum(
            float(d.get("amount",0))
            for p in positions for d in p.get("dividends",[])
        ), 2),
        "all_dividends": [
            {"ticker": p["ticker"], **d}
            for p in positions for d in p.get("dividends",[])
        ],
    }), 200


@app.route("/api/enterprise/portfolios/<pf_id>/full_analytics")
def ent_portfolio_full_analytics(pf_id):
    """Deep analytics: portfolio-level VaR, beta, correlation, attribution."""
    pf = ES.get_portfolio(pf_id)
    if not pf: return jsonify({"detail": "Portfolio not found"}), 404

    positions = pf.get("positions", [])
    days      = int(request.args.get("days", 60))

    # Fetch closes for all positions
    closes_map = {}
    for pos in positions:
        cs = D.fetch_candles(pos["ticker"], days, atlas_get)
        if len(cs) >= 5:
            closes_map[pos["ticker"]] = [c["close"] for c in cs]

    if not closes_map:
        return jsonify({"detail": "Insufficient price history", "var95":None, "attribution":[]}), 200

    # Align lengths
    min_len = min(len(v) for v in closes_map.values())
    aligned = {t: v[-min_len:] for t,v in closes_map.items()}

    # Weights from current positions
    total_cost = sum(float(p.get("qty",0))*float(p.get("entry_price",0)) for p in positions)
    weights = {}
    for p in positions:
        t = p["ticker"]
        if t in aligned:
            cost = float(p.get("qty",0))*float(p.get("entry_price",0))
            weights[t] = cost/total_cost if total_cost > 0 else 1/len(aligned)

    # Daily portfolio returns
    port_rets = []
    for i in range(1, min_len):
        day = sum(weights.get(t,0) * (aligned[t][i]-aligned[t][i-1])/aligned[t][i-1]
                  for t in aligned if aligned[t][i-1] > 0)
        port_rets.append(round(day, 6))

    # VaR
    var95, cvar95, var99 = Q.portfolio_var(port_rets) if hasattr(Q, 'portfolio_var') else (
        round(Q.var_hist(port_rets)*100, 3) if port_rets else None,
        round(Q.cvar_hist(port_rets)*100, 3) if port_rets else None,
        round(Q.var_hist(port_rets, 0.99)*100, 3) if len(port_rets)>=100 else None
    )

    # Portfolio-level metrics
    port_closes = [10000.0]
    for r in port_rets: port_closes.append(port_closes[-1]*(1+r))

    # Attribution
    attribution = []
    for t, cl in aligned.items():
        if len(cl) < 2 or not cl[0]: continue
        ret  = (cl[-1]-cl[0])/cl[0]*100
        cont = round(weights.get(t,0)*ret, 3)
        attribution.append({"ticker":t, "return_pct":round(ret,2), "weight_pct":round(weights.get(t,0)*100,2), "contribution_pct":cont})
    attribution.sort(key=lambda x: x["contribution_pct"], reverse=True)

    # Correlation matrix
    tickers = list(aligned.keys())
    rets_map = {}
    for t, cl in aligned.items():
        rets_map[t] = [(cl[i]-cl[i-1])/cl[i-1] for i in range(1, len(cl))]
    corr = {}
    for t1 in tickers:
        corr[t1] = {}
        for t2 in tickers:
            if t1==t2: corr[t1][t2]=1.0; continue
            a,b = rets_map[t1], rets_map[t2]
            n = min(len(a),len(b)); a,b = a[:n],b[:n]
            am,bm = sum(a)/n,sum(b)/n
            num = sum((a[i]-am)*(b[i]-bm) for i in range(n))
            da = (sum((x-am)**2 for x in a))**0.5; db=(sum((x-bm)**2 for x in b))**0.5
            corr[t1][t2] = round(num/da/db, 4) if da*db>0 else 0

    return jsonify({
        "var95_pct":    var95,
        "cvar95_pct":   cvar95,
        "var99_pct":    var99,
        "sharpe":       Q.sharpe(port_closes),
        "sortino":      Q.sortino(port_closes),
        "max_drawdown": Q.max_drawdown(port_closes),
        "ann_vol":      Q.ann_vol(port_closes),
        "attribution":  attribution,
        "correlation":  corr,
        "tickers":      tickers,
        "port_rets":    port_rets,
        "n_obs":        len(port_rets),
        "days":         days,
    }), 200


@app.route("/api/enterprise/portfolios/<pf_id>/deep_analytics")
def ent_deep_analytics(pf_id):
    """
    Full deep analytics: VaR, CVaR, correlation matrix, attribution,
    portfolio-level TWR, MWR/IRR, monthly returns, beta, rolling Sharpe,
    stress tests, and realized P&L from closed positions.
    """
    from datetime import datetime as _dt3, timezone as _tz3
    pf = ES.get_portfolio(pf_id)
    if not pf: return jsonify({"detail": "Portfolio not found"}), 404

    positions = pf.get("positions", [])
    days      = int(request.args.get("days", 90))

    # ── Price history for all positions ──────────────────────────────────────
    closes_map = {}
    dates_map  = {}
    for pos in positions:
        cs = D.fetch_candles(pos["ticker"], days, atlas_get)
        if len(cs) < 2:
            cs = D.fetch_candles(pos["ticker"], 180, atlas_get)
        if len(cs) >= 2:
            closes_map[pos["ticker"]] = [c["close"] for c in cs]
            dates_map[pos["ticker"]]  = [c["date"]  for c in cs]

    # ── Value-weighted weights ────────────────────────────────────────────────
    total_val = sum(
        float(p.get("qty",0)) * (D.fetch_live_price(p["ticker"], atlas_get) or float(p.get("entry_price",0)))
        for p in positions
    )
    weights = {}
    for p in positions:
        t = p["ticker"]
        live = D.fetch_live_price(t, atlas_get) or float(p.get("entry_price",0))
        val  = float(p.get("qty",0)) * live
        weights[t] = val / total_val if total_val > 0 else 0

    # ── Align closes ─────────────────────────────────────────────────────────
    tickers = list(closes_map.keys())
    if not tickers:
        return jsonify({"detail": "No price history available"}), 200

    min_len = min(len(v) for v in closes_map.values())
    aligned = {t: closes_map[t][-min_len:] for t in tickers}

    # ── Portfolio daily returns ───────────────────────────────────────────────
    port_rets = []
    for i in range(1, min_len):
        day = sum(
            weights.get(t, 0) * (aligned[t][i] - aligned[t][i-1]) / aligned[t][i-1]
            for t in tickers if aligned[t][i-1] > 0
        )
        port_rets.append(round(day, 6))

    # ── Portfolio equity curve ────────────────────────────────────────────────
    port_closes = [10000.0]
    for r in port_rets: port_closes.append(round(port_closes[-1] * (1 + r), 4))

    # ── VaR / CVaR ───────────────────────────────────────────────────────────
    var95  = round(Q.var_hist(port_rets, 0.95) * 100, 3) if port_rets else None
    cvar95 = round(Q.cvar_hist(port_rets, 0.95) * 100, 3) if port_rets else None
    var99  = round(Q.var_hist(port_rets, 0.99) * 100, 3) if len(port_rets) >= 20 else None

    # ── Rolling Sharpe (20-day window) ────────────────────────────────────────
    rolling_sharpe = []
    WINDOW = 20
    for i in range(WINDOW, len(port_rets) + 1):
        window_rets = port_rets[i - WINDOW:i]
        rolling_sharpe.append(round(Q.sharpe(
            [10000 * (1 + r) for r in window_rets]
        ), 3))

    # ── Monthly returns table ─────────────────────────────────────────────────
    # Build from portfolio returns using dates of the first ticker
    ref_dates = dates_map.get(tickers[0], [])[-min_len:] if tickers else []
    monthly = {}
    for i, r in enumerate(port_rets):
        if i < len(ref_dates):
            ym = ref_dates[i + 1][:7] if i + 1 < len(ref_dates) else ""
            if ym:
                monthly[ym] = monthly.get(ym, 0) + r
    monthly_list = [{"month": k, "return_pct": round(v * 100, 2)} for k, v in sorted(monthly.items())]

    # ── Correlation matrix ────────────────────────────────────────────────────
    rets_map = {t: [(aligned[t][i] - aligned[t][i-1]) / aligned[t][i-1]
                    for i in range(1, len(aligned[t])) if aligned[t][i-1] > 0]
                for t in tickers}
    corr = {}
    for t1 in tickers:
        corr[t1] = {}
        for t2 in tickers:
            if t1 == t2: corr[t1][t2] = 1.0; continue
            a, b = rets_map[t1], rets_map[t2]
            n = min(len(a), len(b)); a, b = a[:n], b[:n]
            if n < 2: corr[t1][t2] = 0; continue
            am, bm = sum(a)/n, sum(b)/n
            num = sum((a[i]-am)*(b[i]-bm) for i in range(n))
            da  = (sum((x-am)**2 for x in a))**0.5
            db  = (sum((x-bm)**2 for x in b))**0.5
            corr[t1][t2] = round(num / da / db, 4) if da * db > 0 else 0

    # ── Attribution (Brinson-style) ───────────────────────────────────────────
    attribution = []
    for t in tickers:
        cl = aligned[t]
        if len(cl) < 2 or not cl[0]: continue
        ret  = (cl[-1] - cl[0]) / cl[0] * 100
        cont = round(weights.get(t, 0) * ret, 3)
        attribution.append({
            "ticker": t,
            "return_pct":      round(ret, 2),
            "weight_pct":      round(weights.get(t, 0) * 100, 2),
            "contribution_pct": cont,
            "ann_vol":         round(Q.ann_vol(cl), 2),
            "sharpe":          round(Q.sharpe(cl), 3),
        })
    attribution.sort(key=lambda x: x["contribution_pct"], reverse=True)

    # ── Stress tests ─────────────────────────────────────────────────────────
    stress_scenarios = [
        {"name": "-10% market", "shocks": {t: -0.10 for t in tickers}},
        {"name": "-20% market", "shocks": {t: -0.20 for t in tickers}},
        {"name": "-30% crash",  "shocks": {t: -0.30 for t in tickers}},
        {"name": "+10% rally",  "shocks": {t: +0.10 for t in tickers}},
    ]
    # Per-ticker stress for largest positions
    pos_sorted = sorted(positions, key=lambda p: weights.get(p["ticker"], 0), reverse=True)
    for p in pos_sorted[:3]:
        t = p["ticker"]
        stress_scenarios.append({"name": t+" -20%", "shocks": {t: -0.20}})
        stress_scenarios.append({"name": t+" -40%", "shocks": {t: -0.40}})

    stress_results = []
    for sc in stress_scenarios:
        impact = sum(weights.get(t, 0) * sc["shocks"].get(t, 0) for t in tickers) * 100
        stress_results.append({"name": sc["name"], "portfolio_impact_pct": round(impact, 2)})

    # ── Realized P&L from closed positions ───────────────────────────────────
    realized_pnl = 0.0
    realized_trades = []
    for pos in positions:
        for close in pos.get("closes", []):
            realized_pnl += float(close.get("realised_pnl", 0))
            realized_trades.append({
                "ticker":       pos["ticker"],
                "close_price":  close.get("close_price"),
                "close_qty":    close.get("close_qty"),
                "close_date":   close.get("close_date", ""),
                "realised_pnl": close.get("realised_pnl"),
                "notes":        close.get("notes", ""),
            })
    realized_trades.sort(key=lambda x: x.get("close_date",""), reverse=True)

    # ── TWR (Time-Weighted Return) ────────────────────────────────────────────
    # Simplified: compound daily portfolio returns
    twr = round((port_closes[-1] / port_closes[0] - 1) * 100, 2) if len(port_closes) > 1 else 0

    # ── MWR / IRR (approximation using cash flows) ────────────────────────────
    # Cash flows: initial investment = total cost, current value = total_val
    total_cost = sum(float(p.get("qty",0)) * float(p.get("entry_price",0)) for p in positions)
    mwr = round((total_val - total_cost) / total_cost * 100, 2) if total_cost > 0 else 0

    return jsonify({
        "tickers":          tickers,
        "days":             days,
        "n_obs":            len(port_rets),
        "port_rets":        port_rets,
        "port_closes":      port_closes,
        "monthly_returns":  monthly_list,
        "rolling_sharpe":   rolling_sharpe,
        "var95_pct":        var95,
        "cvar95_pct":       cvar95,
        "var99_pct":        var99,
        "portfolio_metrics": {
            "sharpe":       Q.sharpe(port_closes),
            "sortino":      Q.sortino(port_closes),
            "calmar":       Q.calmar(port_closes),
            "ann_vol":      Q.ann_vol(port_closes),
            "max_drawdown": Q.max_drawdown(port_closes),
            "twr":          twr,
            "mwr":          mwr,
        },
        "attribution":      attribution,
        "correlation":      corr,
        "stress_tests":     stress_results,
        "realized_pnl":     round(realized_pnl, 2),
        "realized_trades":  realized_trades,
    }), 200


@app.route("/api/enterprise/portfolios/<pf_id>/benchmark")
def ent_benchmark(pf_id):
    """
    Compare portfolio vs a selected benchmark.
    benchmark param:
      SFP:SRI   — SRI stock (single ticker)
      B:NCOMP   — NER composite (all NER active securities, cap-weighted approx = equal-weight)
      B:TCOMP   — TSE composite (all TSE securities)
      B:NSTK    — NER stocks only (exclude VSP3/bonds: tickers containing 'SP' or known bonds)
      B:TSTK    — TSE stocks only (exclude commodity/bond TSE tickers)
      B:NCOM    — NER commodities only (NTR)
    """
    pf = ES.get_portfolio(pf_id)
    if not pf: return jsonify({"detail":"Not found"}), 404
    days      = int(request.args.get("days", 90))
    benchmark = request.args.get("benchmark", "B:NCOMP")

    # Get all securities
    _, secs = atlas_get("/securities", ttl=120)
    all_secs = secs if isinstance(secs, list) else []

    # NER active
    ner_active = [s["ticker"] for s in all_secs
                  if not s.get("ticker","").startswith("TSE:") and _is_active(s.get("ticker",""))]
    # TSE active
    tse_active = [s["ticker"] for s in all_secs
                  if s.get("ticker","").startswith("TSE:") and _is_active(s.get("ticker",""))]

    _BOND_LIKE = {"VSP3","VSP2","VSP1"}  # known bond/structured products
    _COMMODITY_LIKE = {"NTR"}            # NER commodities

    if benchmark == "SFP:SRI":
        bm_tickers = ["SRI"]
        bm_label   = "SRI (SFP)"
    elif benchmark == "B:NCOMP":
        bm_tickers = ner_active
        bm_label   = "NER Composite (B:NCOMP)"
    elif benchmark == "B:TCOMP":
        bm_tickers = tse_active
        bm_label   = "TSE Composite (B:TCOMP)"
    elif benchmark == "B:NSTK":
        bm_tickers = [t for t in ner_active if t not in _BOND_LIKE and t not in _COMMODITY_LIKE]
        bm_label   = "NER Stocks (B:NSTK)"
    elif benchmark == "B:TSTK":
        bm_tickers = [t for t in tse_active if "GOLD" not in t and "GCC" not in t]
        bm_label   = "TSE Stocks (B:TSTK)"
    elif benchmark == "B:NCOM":
        bm_tickers = list(_COMMODITY_LIKE & set(ner_active))
        bm_label   = "NER Commodities (B:NCOM)"
    else:
        bm_tickers = ner_active
        bm_label   = "NER Composite (B:NCOMP)"

    ner_tickers = bm_tickers[:12]  # keep existing variable name for compat

    # Portfolio closes
    positions = pf.get("positions", [])
    pf_closes_map = {}
    for pos in positions:
        cs = D.fetch_candles(pos["ticker"], days, atlas_get)
        if len(cs) >= 2:
            pf_closes_map[pos["ticker"]] = [c["close"] for c in cs]

    # Benchmark closes
    bm_closes_map = {}
    for t in ner_tickers[:8]:
        cs = D.fetch_candles(t, days, atlas_get)
        if len(cs) >= 2:
            bm_closes_map[t] = [c["close"] for c in cs]

    def _port_returns(closes_map, wts=None):
        if not closes_map: return []
        min_len = min(len(v) for v in closes_map.values())
        tickers = list(closes_map.keys())
        aligned = {t: closes_map[t][-min_len:] for t in tickers}
        n = len(tickers)
        rets = []
        for i in range(1, min_len):
            day = sum(
                (wts.get(t, 1/n) if wts else 1/n) *
                (aligned[t][i] - aligned[t][i-1]) / aligned[t][i-1]
                for t in tickers if aligned[t][i-1] > 0
            )
            rets.append(round(day, 6))
        return rets

    # Portfolio value-weighted
    total_val = sum(float(p.get("qty",0)) * (D.fetch_live_price(p["ticker"], atlas_get) or float(p.get("entry_price",0)))
                    for p in positions)
    pf_weights = {}
    for p in positions:
        t = p["ticker"]
        if t in pf_closes_map:
            live = D.fetch_live_price(t, atlas_get) or float(p.get("entry_price",0))
            pf_weights[t] = float(p.get("qty",0)) * live / total_val if total_val > 0 else 0

    pf_rets = _port_returns(pf_closes_map, pf_weights)
    bm_rets = _port_returns(bm_closes_map)  # equal-weight

    def _cum(rets):
        v = [10000.0]
        for r in rets: v.append(round(v[-1]*(1+r), 2))
        return v

    pf_curve = _cum(pf_rets)
    bm_curve = _cum(bm_rets)

    min_c = min(len(pf_curve), len(bm_curve))
    pf_ret_total = round((pf_curve[-1]/10000-1)*100, 2) if pf_curve else 0
    bm_ret_total = round((bm_curve[-1]/10000-1)*100, 2) if bm_curve else 0

    # Beta
    n = min(len(pf_rets), len(bm_rets))
    beta = None
    if n >= 5:
        pr, br = pf_rets[:n], bm_rets[:n]
        pm, bm_ = sum(pr)/n, sum(br)/n
        cov = sum((pr[i]-pm)*(br[i]-bm_) for i in range(n))/n
        var = sum((b-bm_)**2 for b in br)/n
        if var > 0: beta = round(cov/var, 3)

    alpha = round(pf_ret_total - (beta or 1) * bm_ret_total, 2) if beta else None

    return jsonify({
        "portfolio_curve":    pf_curve[:min_c],
        "benchmark_curve":    bm_curve[:min_c],
        "portfolio_return":   pf_ret_total,
        "benchmark_return":   bm_ret_total,
        "alpha":              alpha,
        "beta":               beta,
        "outperformance":     round(pf_ret_total - bm_ret_total, 2),
        "benchmark_tickers":  list(bm_closes_map.keys()),
        "benchmark_label":    bm_label,
        "days":               days,
    }), 200


@app.route("/api/enterprise/portfolios/<pf_id>/target_allocation", methods=["GET","POST"])
def ent_target_allocation(pf_id):
    """GET returns current target. POST saves a new target allocation."""
    pf = ES.get_portfolio(pf_id)
    if not pf: return jsonify({"detail": "Not found"}), 404
    if request.method == "POST":
        body = request.get_json(silent=True) or {}
        ES.update_portfolio_meta(pf_id, {"target_allocation": body.get("targets", {})})
        return jsonify({"ok": True, "targets": body.get("targets", {})}), 200
    return jsonify({"targets": pf.get("target_allocation", {})}), 200


@app.route("/api/enterprise/portfolios/<pf_id>/audit_log")
def ent_audit_log(pf_id):
    """Return the full audit trail for this portfolio."""
    pf = ES.get_portfolio(pf_id)
    if not pf: return jsonify({"detail": "Not found"}), 404
    return jsonify(pf.get("audit_log", [])), 200



# ── Per-position equity curve ──────────────────────────────────────────────────
@app.route("/api/enterprise/portfolios/<pf_id>/positions/<pos_id>/curve")
def ent_position_curve(pf_id, pos_id):
    pf = ES.get_portfolio(pf_id)
    if not pf: return jsonify({"detail":"Not found"}), 404
    pos = next((p for p in pf.get("positions",[]) if p["id"]==pos_id), None)
    if not pos: return jsonify({"detail":"Position not found"}), 404
    days = int(request.args.get("days", 180))
    cs   = D.fetch_candles(pos["ticker"], days, atlas_get)
    if len(cs) < 2: return jsonify({"dates":[],"closes":[],"entry":float(pos.get("entry_price",0))}), 200
    closes = [c["close"] for c in cs]
    dates  = [c["date"]  for c in cs]
    entry  = float(pos.get("entry_price", 0))
    entry_date = pos.get("entry_date","")
    # Trim to entry date if known
    if entry_date:
        try:
            idx = next((i for i,d in enumerate(dates) if d >= entry_date), 0)
            closes = closes[idx:]; dates = dates[idx:]
        except: pass
    # Compute pnl series from entry price normalised
    pnl_series = [round((c - entry)/entry*100, 3) if entry > 0 else 0 for c in closes]
    return jsonify({
        "ticker":     pos["ticker"],
        "dates":      dates,
        "closes":     closes,
        "pnl_series": pnl_series,
        "entry":      entry,
        "entry_date": entry_date,
        "current":    closes[-1] if closes else None,
        "stop_price": float(pos.get("stop_price") or 0) or None,
    }), 200


# ── Dividends ──────────────────────────────────────────────────────────────────
@app.route("/api/enterprise/portfolios/<pf_id>/positions/<pos_id>/dividends", methods=["GET","POST"])
def ent_position_dividends(pf_id, pos_id):
    from datetime import datetime as _dt4, timezone as _tz4
    pf = ES.get_portfolio(pf_id)
    if not pf: return jsonify({"detail":"Not found"}), 404
    data_store = next((None for _ in []), None)  # placeholder to load data
    # Load raw data for mutation
    import json as _json, os as _os
    _DATA_DIR = _os.environ.get("ENTERPRISE_DATA_DIR", _os.path.dirname(_os.path.abspath(__file__)))
    _ENT_FILE = _os.path.join(_DATA_DIR, "enterprise_state.json")
    raw = open(_ENT_FILE).read() if _os.path.exists(_ENT_FILE) else "{}"
    data = _json.loads(raw)
    pf_raw = next((p for p in data.get("portfolios",[]) if p["id"]==pf_id), None)
    if not pf_raw: return jsonify({"detail":"Not found"}), 404
    pos_raw = next((p for p in pf_raw.get("positions",[]) if p["id"]==pos_id), None)
    if not pos_raw: return jsonify({"detail":"Position not found"}), 404
    if request.method == "GET":
        return jsonify(pos_raw.get("dividends",[])), 200
    body = request.get_json(silent=True) or {}
    div = {
        "id":     str(__import__("uuid").uuid4())[:8],
        "date":   body.get("date",""),
        "amount": float(body.get("amount",0)),
        "note":   body.get("note",""),
        "ts":     _dt4.now(_tz4.utc).isoformat(),
    }
    pos_raw.setdefault("dividends",[]).append(div)
    # Add to cash
    pf_raw["cash"] = float(pf_raw.get("cash",0)) + div["amount"]
    pf_raw.setdefault("cash_log",[]).append({
        "amount": div["amount"], "note": f"Dividend {pos_raw['ticker']}: {div['note']}",
        "ts": div["ts"], "balance_after": pf_raw["cash"]
    })
    # Audit log
    pf_raw.setdefault("audit_log",[]).append({
        "ts": div["ts"], "action": "DIVIDEND",
        "detail": f"{pos_raw['ticker']} +${div['amount']:.2f} {div['note']}"
    })
    if len(pf_raw["audit_log"]) > 500:
        pf_raw["audit_log"] = pf_raw["audit_log"][-500:]
    tmp = _ENT_FILE+".tmp"
    with open(tmp,"w") as f: _json.dump(data,f,indent=2)
    _os.replace(tmp, _ENT_FILE)
    return jsonify(div), 200


# ── Report snapshots (scheduling) ─────────────────────────────────────────────
@app.route("/api/enterprise/portfolios/<pf_id>/report_snapshots", methods=["GET","POST","DELETE"])
def ent_report_snapshots(pf_id):
    from datetime import datetime as _dt5, timezone as _tz5
    import json as _json5, os as _os5
    _DATA_DIR5 = _os5.environ.get("ENTERPRISE_DATA_DIR", _os5.path.dirname(_os5.path.abspath(__file__)))
    _ENT_FILE5 = _os5.path.join(_DATA_DIR5, "enterprise_state.json")
    raw = open(_ENT_FILE5).read() if _os5.path.exists(_ENT_FILE5) else "{}"
    data = _json5.loads(raw)
    pf_raw = next((p for p in data.get("portfolios",[]) if p["id"]==pf_id), None)
    if not pf_raw: return jsonify({"detail":"Not found"}), 404
    if request.method == "GET":
        return jsonify(pf_raw.get("report_snapshots",[])), 200
    if request.method == "DELETE":
        snap_id = request.args.get("id","")
        pf_raw["report_snapshots"] = [s for s in pf_raw.get("report_snapshots",[]) if s.get("id")!=snap_id]
        tmp=_ENT_FILE5+".tmp"; open(tmp,"w").write(_json5.dumps(data,indent=2)); _os5.replace(tmp,_ENT_FILE5)
        return jsonify({"ok":True}), 200
    # POST — save a snapshot
    body = request.get_json(silent=True) or {}
    snap = {
        "id":       str(__import__("uuid").uuid4())[:8],
        "ts":       _dt5.now(_tz5.utc).isoformat(),
        "label":    body.get("label", _dt5.now(_tz5.utc).strftime("%Y-%m")),
        "html":     body.get("html",""),
        "commentary": body.get("commentary",""),
        "summary":  body.get("summary",{}),
    }
    pf_raw.setdefault("report_snapshots",[]).append(snap)
    # Keep last 24
    if len(pf_raw["report_snapshots"]) > 24:
        pf_raw["report_snapshots"] = pf_raw["report_snapshots"][-24:]
    tmp=_ENT_FILE5+".tmp"; open(tmp,"w").write(_json5.dumps(data,indent=2)); _os5.replace(tmp,_ENT_FILE5)
    return jsonify({"id":snap["id"],"label":snap["label"],"ts":snap["ts"]}), 200


# ── Scenario builder ──────────────────────────────────────────────────────────
@app.route("/api/enterprise/portfolios/<pf_id>/scenarios", methods=["GET","POST","DELETE"])
def ent_scenarios(pf_id):
    from datetime import datetime as _dt6, timezone as _tz6
    import json as _json6, os as _os6
    _DATA_DIR6 = _os6.environ.get("ENTERPRISE_DATA_DIR", _os6.path.dirname(_os6.path.abspath(__file__)))
    _ENT_FILE6 = _os6.path.join(_DATA_DIR6, "enterprise_state.json")
    raw = open(_ENT_FILE6).read() if _os6.path.exists(_ENT_FILE6) else "{}"
    data = _json6.loads(raw)
    pf_raw = next((p for p in data.get("portfolios",[]) if p["id"]==pf_id), None)
    if not pf_raw: return jsonify({"detail":"Not found"}), 404
    if request.method == "GET":
        return jsonify(pf_raw.get("scenarios",[])), 200
    if request.method == "DELETE":
        sid = request.args.get("id","")
        pf_raw["scenarios"] = [s for s in pf_raw.get("scenarios",[]) if s.get("id")!=sid]
        tmp=_ENT_FILE6+".tmp"; open(tmp,"w").write(_json6.dumps(data,indent=2)); _os6.replace(tmp,_ENT_FILE6)
        return jsonify({"ok":True}), 200
    body = request.get_json(silent=True) or {}
    # Run the scenario and return impact
    shocks = body.get("shocks",{})  # {ticker: pct_decimal}
    positions = pf_raw.get("positions",[])
    total_val = sum(float(p.get("qty",0)) * (D.fetch_live_price(p["ticker"],atlas_get) or float(p.get("entry_price",0))) for p in positions)
    impacts = []
    total_impact_pct = 0
    for pos in positions:
        t = pos["ticker"]
        live = D.fetch_live_price(t, atlas_get) or float(pos.get("entry_price",0))
        val  = float(pos.get("qty",0)) * live
        w    = val/total_val if total_val > 0 else 0
        shock = shocks.get(t, shocks.get("*", 0))
        contrib = w * shock * 100
        total_impact_pct += contrib
        if shock != 0:
            impacts.append({"ticker":t,"weight_pct":round(w*100,2),"shock_pct":round(shock*100,2),"contribution_pct":round(contrib,2)})
    scenario = {
        "id":          str(__import__("uuid").uuid4())[:8],
        "name":        body.get("name","Custom Scenario"),
        "shocks":      shocks,
        "impacts":     impacts,
        "total_impact_pct": round(total_impact_pct, 3),
        "total_val":   round(total_val, 2),
        "est_loss":    round(total_val * total_impact_pct/100, 2),
        "ts":          _dt6.now(_tz6.utc).isoformat(),
    }
    if body.get("save"):
        pf_raw.setdefault("scenarios",[]).append(scenario)
        tmp=_ENT_FILE6+".tmp"; open(tmp,"w").write(_json6.dumps(data,indent=2)); _os6.replace(tmp,_ENT_FILE6)
    return jsonify(scenario), 200


# ── Rolling beta ──────────────────────────────────────────────────────────────
@app.route("/api/enterprise/portfolios/<pf_id>/rolling_beta")
def ent_rolling_beta(pf_id):
    pf = ES.get_portfolio(pf_id)
    if not pf: return jsonify({"detail":"Not found"}), 404
    days   = int(request.args.get("days",90))
    window = int(request.args.get("window",20))
    benchmark = request.args.get("benchmark","B:NCOMP")
    positions = pf.get("positions",[])
    # Get portfolio candles
    closes_map = {}
    for pos in positions:
        cs = D.fetch_candles(pos["ticker"],days,atlas_get)
        if len(cs) >= 2: closes_map[pos["ticker"]] = [c["close"] for c in cs]
    if not closes_map: return jsonify({"beta30":[],"beta60":[],"dates":[]}), 200
    # Get benchmark
    _, secs = atlas_get("/securities", ttl=120)
    all_secs = secs if isinstance(secs,list) else []
    bm_tickers = [s["ticker"] for s in all_secs if not s.get("ticker","").startswith("TSE:") and _is_active(s.get("ticker",""))][:8]
    bm_map = {}
    for t in bm_tickers:
        cs = D.fetch_candles(t,days,atlas_get)
        if len(cs) >= 2: bm_map[t] = [c["close"] for c in cs]
    if not bm_map: return jsonify({"beta30":[],"beta60":[],"dates":[]}), 200
    min_len = min(min(len(v) for v in closes_map.values()), min(len(v) for v in bm_map.values()))
    # Portfolio returns (equal weight)
    tickers = list(closes_map.keys())
    pf_rets = []
    for i in range(1,min_len):
        r = sum((closes_map[t][i]-closes_map[t][i-1])/closes_map[t][i-1] for t in tickers if closes_map[t][i-1]>0)/max(len(tickers),1)
        pf_rets.append(r)
    bm_tks = list(bm_map.keys())
    bm_rets = []
    for i in range(1,min_len):
        r = sum((bm_map[t][i]-bm_map[t][i-1])/bm_map[t][i-1] for t in bm_tks if bm_map[t][i-1]>0)/max(len(bm_tks),1)
        bm_rets.append(r)
    def _rolling_beta(pf_r, bm_r, win):
        betas = []
        for i in range(win, len(pf_r)+1):
            pr = pf_r[i-win:i]; br = bm_r[i-win:i]
            pm = sum(pr)/win; bm_ = sum(br)/win
            cov = sum((pr[j]-pm)*(br[j]-bm_) for j in range(win))/win
            var = sum((b-bm_)**2 for b in br)/win
            betas.append(round(cov/var,3) if var>0 else 0)
        return betas
    n = min(len(pf_rets),len(bm_rets))
    pf_rets = pf_rets[:n]; bm_rets = bm_rets[:n]
    b20 = _rolling_beta(pf_rets, bm_rets, 20)
    b60 = _rolling_beta(pf_rets, bm_rets, min(60,n//2)) if n >= 30 else []
    return jsonify({"beta20":b20,"beta60":b60,"n_obs":n}), 200


# ── VaR backtest ──────────────────────────────────────────────────────────────
@app.route("/api/enterprise/portfolios/<pf_id>/var_backtest")
def ent_var_backtest(pf_id):
    pf = ES.get_portfolio(pf_id)
    if not pf: return jsonify({"detail":"Not found"}), 404
    days = int(request.args.get("days",180))
    positions = pf.get("positions",[])
    closes_map = {}
    for pos in positions:
        cs = D.fetch_candles(pos["ticker"],days,atlas_get)
        if len(cs) >= 5: closes_map[pos["ticker"]] = [c["close"] for c in cs]
    if not closes_map: return jsonify({"detail":"Insufficient data","coverage_ratio":None}), 200
    min_len = min(len(v) for v in closes_map.values())
    tickers = list(closes_map.keys())
    total_val = sum(float(p.get("qty",0)) * (D.fetch_live_price(p["ticker"],atlas_get) or float(p.get("entry_price",0))) for p in positions)
    weights = {}
    for p in positions:
        t = p["ticker"]
        if t in closes_map:
            live = D.fetch_live_price(t,atlas_get) or float(p.get("entry_price",0))
            weights[t] = float(p.get("qty",0))*live/total_val if total_val>0 else 0
    port_rets = []
    for i in range(1,min_len):
        r = sum(weights.get(t,0)*(closes_map[t][i]-closes_map[t][i-1])/closes_map[t][i-1] for t in tickers if closes_map[t][i-1]>0)
        port_rets.append(r)
    # Rolling VaR backtest: use first half to estimate, second half to test
    HALF = len(port_rets)//2
    if HALF < 10: return jsonify({"detail":"Need more history","coverage_ratio":None}), 200
    train = port_rets[:HALF]; test = port_rets[HALF:]
    var95 = Q.var_hist(train, 0.95)
    var99 = Q.var_hist(train, 0.99)
    breaches95 = sum(1 for r in test if -r > var95)
    breaches99 = sum(1 for r in test if -r > var99)
    expected95 = round(len(test)*0.05, 1)
    expected99 = round(len(test)*0.01, 1)
    return jsonify({
        "n_test":        len(test),
        "var95_pct":     round(var95*100,3),
        "var99_pct":     round(var99*100,3),
        "breaches95":    breaches95,
        "breaches99":    breaches99,
        "expected95":    expected95,
        "expected99":    expected99,
        "coverage95":    round(1-breaches95/len(test),3) if test else None,
        "coverage99":    round(1-breaches99/len(test),3) if test else None,
        "port_rets":     [round(r*100,4) for r in port_rets],
    }), 200


# ═══════════════════════════════════════════════════════════════════════════════
# ── CLIENT MANAGEMENT SPACE ────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

def _load_state():
    import json as _j, os as _o
    _DIR  = _o.environ.get("ENTERPRISE_DATA_DIR", _o.path.dirname(_o.path.abspath(__file__)))
    _FILE = _o.path.join(_DIR, "enterprise_state.json")
    if _o.path.exists(_FILE):
        try: return _j.loads(open(_FILE).read()), _FILE
        except: pass
    return {"portfolios": [], "clients": []}, _FILE

def _save_state(data, filepath):
    import json as _j, os as _o
    tmp = filepath + ".tmp"
    with open(tmp, "w") as f: _j.dump(data, f, indent=2)
    _o.replace(tmp, filepath)


@app.route("/api/enterprise/clients", methods=["GET", "POST"])
def ent_clients():
    from datetime import datetime as _dt, timezone as _tz
    data, fpath = _load_state()
    if request.method == "GET":
        return jsonify(data.get("clients", [])), 200
    body = request.get_json(silent=True) or {}
    clients = data.setdefault("clients", [])
    if body.get("id"):
        idx = next((i for i,c in enumerate(clients) if c["id"]==body["id"]), None)
        if idx is not None: clients[idx] = body
        else: clients.append(body)
    else:
        body["id"]         = str(__import__("uuid").uuid4())[:8]
        body["created_at"] = _dt.now(_tz.utc).isoformat()
        clients.append(body)
    _save_state(data, fpath)
    return jsonify(body), 200


@app.route("/api/enterprise/clients/<cid>", methods=["GET", "PATCH", "DELETE"])
def ent_client(cid):
    data, fpath = _load_state()
    clients = data.get("clients", [])
    client  = next((c for c in clients if c["id"] == cid), None)
    if not client: return jsonify({"detail": "Not found"}), 404
    if request.method == "GET":  return jsonify(client), 200
    if request.method == "DELETE":
        data["clients"] = [c for c in clients if c["id"] != cid]
        _save_state(data, fpath)
        return jsonify({"ok": True}), 200
    body = request.get_json(silent=True) or {}
    client.update({k: v for k, v in body.items() if k != "id"})
    _save_state(data, fpath)
    return jsonify(client), 200


@app.route("/api/enterprise/clients/<cid>/portal_token", methods=["POST"])
def ent_client_portal(cid):
    import secrets as _sec
    from datetime import datetime as _dt2, timezone as _tz2
    data, fpath = _load_state()
    client = next((c for c in data.get("clients", []) if c["id"] == cid), None)
    if not client: return jsonify({"detail": "Not found"}), 404
    token = _sec.token_urlsafe(20)
    client["portal_token"]   = token
    client["portal_created"] = _dt2.now(_tz2.utc).isoformat()
    # Find portfolios matching this client name
    pf_ids = [p["id"] for p in data.get("portfolios", [])
              if (p.get("client") or "").strip().lower() == (client.get("name") or "").strip().lower()]
    client["portal_pf_ids"] = pf_ids
    _save_state(data, fpath)
    host = request.host_url.rstrip("/")
    return jsonify({"token": token, "url": f"{host}/portal/{token}", "pf_ids": pf_ids}), 200


@app.route("/portal/<token>")
def client_portal(token):
    """Read-only client portal — simple static view."""
    data, _ = _load_state()
    client = next((c for c in data.get("clients", []) if c.get("portal_token") == token), None)
    if not client: return "<h2 style='font-family:monospace;color:#f44;padding:40px'>Invalid or expired portal link.</h2>", 404
    pf_ids = client.get("portal_pf_ids", [])
    portfolios = [p for p in data.get("portfolios", []) if p["id"] in pf_ids]
    return render_template("enterprise/portal.html", client=client, portfolios=portfolios, token=token)


@app.route("/api/enterprise/clients/<cid>/portfolios", methods=["GET", "POST"])
def ent_client_portfolios(cid):
    """GET: returns linked portfolio IDs. POST: sets linked_pf_ids list."""
    data, fpath = _load_state()
    client = next((c for c in data.get("clients", []) if c["id"] == cid), None)
    if not client: return jsonify({"detail": "Not found"}), 404
    if request.method == "GET":
        linked = client.get("linked_pf_ids", [])
        portfolios = [{"id": p["id"], "name": p.get("name",""), "client": p.get("client","")}
                      for p in data.get("portfolios", []) if p["id"] in linked]
        return jsonify({"linked_pf_ids": linked, "portfolios": portfolios}), 200
    body = request.get_json(silent=True) or {}
    client["linked_pf_ids"] = body.get("pf_ids", [])
    # Also update portal_pf_ids to stay in sync
    client["portal_pf_ids"] = client["linked_pf_ids"]
    _save_state(data, fpath)
    return jsonify({"linked_pf_ids": client["linked_pf_ids"]}), 200


@app.route("/api/enterprise/clients/<cid>/note", methods=["POST"])
def ent_client_note(cid):
    from datetime import datetime as _dt3, timezone as _tz3
    data, fpath = _load_state()
    client = next((c for c in data.get("clients", []) if c["id"] == cid), None)
    if not client: return jsonify({"detail": "Not found"}), 404
    body = request.get_json(silent=True) or {}
    note = {"id": str(__import__("uuid").uuid4())[:8],
            "ts": _dt3.now(_tz3.utc).isoformat(),
            "text": body.get("text", ""),
            "author": body.get("author", "PM")}
    client.setdefault("notes_log", []).append(note)
    _save_state(data, fpath)
    return jsonify(note), 200


# ═══════════════════════════════════════════════════════════════════════════════
# ── FIRM DASHBOARD ANALYTICS ──────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/api/enterprise/firm_analytics")
def ent_firm_analytics():
    """
    Firm-level: AUM over time, cross-portfolio exposure, model portfolio,
    revenue estimate, compliance flags.
    """
    data, _ = _load_state()
    portfolios = data.get("portfolios", [])

    # ── Per-portfolio enrichment ──────────────────────────────────────────────
    pf_rows = []
    grand_value = 0.0; grand_cost = 0.0; grand_cash = 0.0
    ticker_exposure = {}  # ticker → total $ across all portfolios

    for pf in portfolios:
        pf_value = 0.0; pf_cost = 0.0
        for pos in pf.get("positions", []):
            qty  = float(pos.get("qty", 0))
            ep   = float(pos.get("entry_price", 0))
            live = D.fetch_live_price(pos["ticker"], atlas_get) or ep
            val  = qty * live
            cost = qty * ep
            pf_value += val; pf_cost += cost
            ticker_exposure[pos["ticker"]] = ticker_exposure.get(pos["ticker"], 0) + val

        cash = float(pf.get("cash", 0))
        pnl  = pf_value - pf_cost
        pnl_pct = round(pnl / pf_cost * 100, 2) if pf_cost > 0 else 0
        aum  = pf_value + cash
        grand_value += pf_value; grand_cost += pf_cost; grand_cash += cash

        # Last activity date
        last_act = ""
        for pos in pf.get("positions", []):
            ts = pos.get("added_at", "")
            if ts > last_act: last_act = ts
        for e in pf.get("cash_log", []):
            ts = e.get("ts", "")
            if ts > last_act: last_act = ts

        pf_rows.append({
            "id":        pf["id"],
            "name":      pf.get("name", ""),
            "client":    pf.get("client", ""),
            "strategy":  pf.get("strategy", ""),
            "value":     round(pf_value, 2),
            "cost":      round(pf_cost, 2),
            "cash":      round(cash, 2),
            "aum":       round(aum, 2),
            "pnl":       round(pnl, 2),
            "pnl_pct":   pnl_pct,
            "positions": len(pf.get("positions", [])),
            "last_activity": last_act[:10] if last_act else "—",
        })

    grand_aum = grand_value + grand_cash
    grand_pnl = grand_value - grand_cost
    grand_pnl_pct = round(grand_pnl / grand_cost * 100, 2) if grand_cost > 0 else 0

    # ── Cross-portfolio exposure ──────────────────────────────────────────────
    top_exposures = sorted(
        [{"ticker": t, "value": round(v, 2), "pct_of_aum": round(v/grand_aum*100, 2) if grand_aum > 0 else 0}
         for t, v in ticker_exposure.items()],
        key=lambda x: x["value"], reverse=True
    )[:10]

    # Flag tickers >15% of firm AUM
    concentration_flags = [e for e in top_exposures if e["pct_of_aum"] > 15]

    # ── Revenue estimate (AUM * 0.01 / 12 monthly) ──────────────────────────
    mrr = round(grand_aum * 0.01 / 12, 2)
    arr = round(mrr * 12, 2)

    # ── Compliance flags ─────────────────────────────────────────────────────
    from datetime import datetime as _dtc, timezone as _tzc, timedelta as _tdc
    now_iso = _dtc.now(_tzc.utc).isoformat()
    compliance = []
    for pf in pf_rows:
        if pf["positions"] > 0 and pf["last_activity"] != "—":
            try:
                days_inactive = (_dtc.now(_tzc.utc) - _dtc.fromisoformat(pf["last_activity"].replace("Z","+00:00"))).days
                if days_inactive > 30:
                    compliance.append({"type": "INACTIVE", "pf": pf["name"], "detail": f"No activity in {days_inactive} days"})
            except: pass
        if pf["positions"] == 0 and pf["aum"] > 0:
            compliance.append({"type": "CASH_ONLY", "pf": pf["name"], "detail": "AUM held entirely in cash"})
    for flag in concentration_flags:
        compliance.append({"type": "CONCENTRATION", "pf": "FIRM", "detail": f"{flag['ticker']} is {flag['pct_of_aum']:.1f}% of firm AUM"})

    # ── AUM history (synthetic from oldest position dates) ───────────────────
    # Build a rough AUM timeline from cash_log deposit dates across all portfolios
    aum_events = []
    for pf in portfolios:
        for e in pf.get("cash_log", []):
            aum_events.append({"date": (e.get("ts","") or "")[:10], "amount": float(e.get("amount",0))})
    aum_events.sort(key=lambda x: x["date"])
    running = 0.0; aum_timeline = []
    for e in aum_events:
        running += e["amount"]
        if aum_timeline and aum_timeline[-1]["date"] == e["date"]:
            aum_timeline[-1]["aum"] = round(running, 2)
        else:
            aum_timeline.append({"date": e["date"], "aum": round(running, 2)})
    # Always include current AUM as latest point
    from datetime import datetime as _dtau
    aum_timeline.append({"date": _dtau.now().strftime("%Y-%m-%d"), "aum": round(grand_aum, 2)})

    return jsonify({
        "portfolios":           pf_rows,
        "summary": {
            "grand_aum":        round(grand_aum, 2),
            "grand_value":      round(grand_value, 2),
            "grand_cash":       round(grand_cash, 2),
            "grand_pnl":        round(grand_pnl, 2),
            "grand_pnl_pct":    grand_pnl_pct,
            "num_portfolios":   len(portfolios),
            "num_clients":      len(data.get("clients", [])),
            "mrr":              mrr,
            "arr":              arr,
        },
        "top_exposures":        top_exposures,
        "concentration_flags":  concentration_flags,
        "compliance":           compliance,
        "aum_timeline":         aum_timeline,
    }), 200



# ═══════════════════════════════════════════════════════════════════════════════
# ── FIRM SETTINGS ─────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/api/enterprise/firm_settings", methods=["GET", "POST"])
def ent_firm_settings():
    from datetime import datetime as _dts, timezone as _tzs
    data, fpath = _load_state()
    if request.method == "GET":
        return jsonify(data.get("firm_settings", {})), 200
    body = request.get_json(silent=True) or {}
    data["firm_settings"] = body
    _save_state(data, fpath)
    return jsonify(body), 200


# ═══════════════════════════════════════════════════════════════════════════════
# ── MORNING BRIEF ─────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/api/enterprise/morning_brief")
def ent_morning_brief():
    """
    Daily morning summary:
    - P&L delta vs last snapshot
    - Biggest movers in held positions
    - Overdue client reviews
    - Active compliance flags
    - Portfolios drifted from target
    - Total firm AUM
    """
    from datetime import datetime as _dtm, timezone as _tzm, timedelta as _tdm
    data, _ = _load_state()
    portfolios   = data.get("portfolios", [])
    clients      = data.get("clients", [])
    settings     = data.get("firm_settings", {})
    today        = _dtm.now(_tzm.utc).strftime("%Y-%m-%d")

    # ── Position movers ───────────────────────────────────────────────────────
    movers = []
    grand_value = 0.0; grand_cost = 0.0
    for pf in portfolios:
        for pos in pf.get("positions", []):
            ep   = float(pos.get("entry_price", 0))
            qty  = float(pos.get("qty", 0))
            live = D.fetch_live_price(pos["ticker"], atlas_get) or ep
            val  = qty * live; cost = qty * ep
            pnl_pct = (val - cost) / cost * 100 if cost > 0 else 0
            grand_value += val; grand_cost += cost
            movers.append({
                "ticker":   pos["ticker"],
                "pf_name":  pf.get("name",""),
                "pnl_pct":  round(pnl_pct, 2),
                "pnl":      round(val - cost, 2),
                "value":    round(val, 2),
            })
    movers.sort(key=lambda x: abs(x["pnl_pct"]), reverse=True)
    top_movers = movers[:8]

    # ── Overdue client reviews ────────────────────────────────────────────────
    overdue_reviews = []
    for c in clients:
        freq_days = {"monthly": 30, "quarterly": 90, "annual": 365}.get(
            c.get("review_frequency", "quarterly"), 90)
        last = c.get("last_review_date", c.get("created_at", ""))[:10]
        if last:
            try:
                days_since = (_dtm.now(_tzm.utc) - _dtm.fromisoformat(last + "T00:00:00+00:00")).days
                if days_since >= freq_days:
                    overdue_reviews.append({
                        "client_id":   c["id"],
                        "client_name": c.get("name",""),
                        "days_overdue": days_since - freq_days,
                        "last_review":  last,
                        "frequency":    c.get("review_frequency","quarterly"),
                    })
            except: pass

    # ── Compliance flags ──────────────────────────────────────────────────────
    max_pos_pct  = float(settings.get("max_position_pct", 40))
    max_hhi      = float(settings.get("max_hhi", 4000))
    flags = []
    for pf in portfolios:
        positions = pf.get("positions", [])
        total_val = sum(float(p.get("qty",0)) * (D.fetch_live_price(p["ticker"],atlas_get) or float(p.get("entry_price",0))) for p in positions)
        if total_val > 0:
            for pos in positions:
                live = D.fetch_live_price(pos["ticker"], atlas_get) or float(pos.get("entry_price",0))
                w = float(pos.get("qty",0)) * live / total_val * 100
                if w > max_pos_pct:
                    flags.append({"type":"OVERWEIGHT","pf":pf.get("name",""),"detail":f"{pos['ticker']} is {w:.1f}% of {pf.get('name','')}"})

    # ── Rebalancing needed ─────────────────────────────────────────────────────
    rebal_needed = []
    drift_threshold = float(settings.get("drift_alert_pct", 5))
    for pf in portfolios:
        targets = pf.get("target_allocation", {})
        if not targets: continue
        positions = pf.get("positions", [])
        total_val = sum(float(p.get("qty",0)) * (D.fetch_live_price(p["ticker"],atlas_get) or float(p.get("entry_price",0))) for p in positions)
        if total_val <= 0: continue
        for pos in positions:
            live = D.fetch_live_price(pos["ticker"],atlas_get) or float(pos.get("entry_price",0))
            actual = float(pos.get("qty",0)) * live / total_val * 100
            target = float(targets.get(pos["ticker"], 0))
            drift  = actual - target
            if abs(drift) >= drift_threshold:
                rebal_needed.append({"pf":pf.get("name",""),"ticker":pos["ticker"],"drift":round(drift,1),"actual":round(actual,1),"target":round(target,1)})

    return jsonify({
        "date":            today,
        "grand_aum":       round(grand_value + sum(float(p.get("cash",0)) for p in portfolios), 2),
        "grand_pnl":       round(grand_value - grand_cost, 2),
        "grand_pnl_pct":   round((grand_value - grand_cost) / grand_cost * 100, 2) if grand_cost > 0 else 0,
        "top_movers":      top_movers,
        "overdue_reviews": overdue_reviews,
        "flags":           flags,
        "rebal_needed":    rebal_needed,
        "num_portfolios":  len(portfolios),
        "num_clients":     len(clients),
    }), 200


# ═══════════════════════════════════════════════════════════════════════════════
# ── CLIENT REVIEW SCHEDULER ──────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/api/enterprise/clients/<cid>/review", methods=["POST"])
def ent_log_review(cid):
    """Log that a client review was completed."""
    from datetime import datetime as _dtr, timezone as _tzr
    data, fpath = _load_state()
    client = next((c for c in data.get("clients",[]) if c["id"]==cid), None)
    if not client: return jsonify({"detail":"Not found"}), 404
    body = request.get_json(silent=True) or {}
    today = _dtr.now(_tzr.utc).strftime("%Y-%m-%d")
    review = {
        "id":     str(__import__("uuid").uuid4())[:8],
        "date":   body.get("date", today),
        "notes":  body.get("notes",""),
        "ts":     _dtr.now(_tzr.utc).isoformat(),
    }
    client.setdefault("review_log", []).append(review)
    client["last_review_date"] = review["date"]
    if body.get("frequency"):
        client["review_frequency"] = body["frequency"]
    _save_state(data, fpath)
    return jsonify(review), 200


# ═══════════════════════════════════════════════════════════════════════════════
# ── FIRM-WIDE REALIZED P&L ────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/api/enterprise/firm_realized")
def ent_firm_realized():
    data, _ = _load_state()
    trades = []
    total_realized = 0.0; total_dividends = 0.0
    for pf in data.get("portfolios",[]):
        for pos in pf.get("positions",[]):
            for c in pos.get("closes",[]):
                pnl = float(c.get("realised_pnl",0))
                total_realized += pnl
                trades.append({
                    "pf_name":    pf.get("name",""),
                    "client":     pf.get("client",""),
                    "ticker":     pos["ticker"],
                    "close_date": c.get("close_date",""),
                    "close_qty":  c.get("close_qty"),
                    "close_price":c.get("close_price"),
                    "realised_pnl": pnl,
                    "notes":      c.get("notes",""),
                })
            for d in pos.get("dividends",[]):
                amt = float(d.get("amount",0))
                total_dividends += amt
                trades.append({
                    "pf_name":    pf.get("name",""),
                    "client":     pf.get("client",""),
                    "ticker":     pos["ticker"],
                    "close_date": d.get("date",""),
                    "close_qty":  None,
                    "close_price":None,
                    "realised_pnl": amt,
                    "notes":      f"Dividend: {d.get('note','')}",
                    "type":       "dividend",
                })
    trades.sort(key=lambda x: x.get("close_date",""), reverse=True)
    return jsonify({
        "trades":           trades,
        "total_realized":   round(total_realized, 2),
        "total_dividends":  round(total_dividends, 2),
        "total_income":     round(total_realized + total_dividends, 2),
    }), 200


# ═══════════════════════════════════════════════════════════════════════════════
# ── FIRM ATTRIBUTION OVER TIME ────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/api/enterprise/firm_attribution")
def ent_firm_attribution():
    """Monthly attribution: which portfolios drove firm-level returns each month."""
    data, _ = _load_state()
    portfolios = data.get("portfolios",[])
    days = int(request.args.get("days", 180))

    monthly_by_pf = {}  # {month: {pf_name: contribution_pct}}

    for pf in portfolios:
        positions = pf.get("positions",[])
        if not positions: continue
        for pos in positions:
            cs = D.fetch_candles(pos["ticker"], days, atlas_get)
            if len(cs) < 5: continue
            qty = float(pos.get("qty",0)); ep = float(pos.get("entry_price",0))
            cost = qty * ep
            if cost <= 0: continue
            # Group daily returns by month
            for i in range(1, len(cs)):
                if not cs[i-1]["close"]: continue
                day_ret = (cs[i]["close"] - cs[i-1]["close"]) / cs[i-1]["close"]
                month   = cs[i]["date"][:7]
                contrib = day_ret * (cost / 10000)  # normalised contribution
                monthly_by_pf.setdefault(month, {})
                monthly_by_pf[month][pf.get("name","")] =                     monthly_by_pf[month].get(pf.get("name",""),0) + round(contrib*100, 4)

    # Sort months
    months = sorted(monthly_by_pf.keys())
    pf_names = list({pf.get("name","") for pf in portfolios if pf.get("positions")})

    return jsonify({
        "months":   months,
        "pf_names": pf_names,
        "data":     {m: monthly_by_pf[m] for m in months},
    }), 200

@app.route("/api/enterprise/dashboard")
def ent_dashboard():
    """Aggregate analytics across all portfolios."""
    portfolios = ES.get_all_portfolios()
    results = []; grand_value = 0.0; grand_cost = 0.0; ticker_exposure = {}

    for pf in portfolios:
        pf_value = 0.0; pf_cost = 0.0
        for pos in pf.get("positions",[]):
            ticker      = pos["ticker"]
            entry_price = float(pos.get("entry_price") or 0)
            qty         = float(pos.get("qty") or 0)
            cost        = entry_price * qty
            live_px     = D.fetch_live_price(ticker, atlas_get)
            mkt_value   = (live_px or entry_price) * qty
            pf_value   += mkt_value; pf_cost += cost
            ticker_exposure[ticker] = ticker_exposure.get(ticker,0) + mkt_value

        cash = float(pf.get("cash",0)); pnl = pf_value - pf_cost
        results.append({
            "id": pf["id"], "name": pf.get("name",""), "client": pf.get("client",""),
            "value": round(pf_value,2), "cash": round(cash,2),
            "total": round(pf_value+cash,2), "cost": round(pf_cost,2),
            "pnl": round(pnl,2),
            "pnl_pct": round(pnl/pf_cost*100,2) if pf_cost>0 else 0,
            "positions": len(pf.get("positions",[])),
        })
        grand_value += pf_value; grand_cost += pf_cost

    top_exp = sorted(ticker_exposure.items(), key=lambda x: x[1], reverse=True)[:10]
    return jsonify({
        "portfolios": results,
        "summary": {
            "total_aum":      round(grand_value, 2),
            "total_pnl":      round(grand_value-grand_cost, 2),
            "total_pnl_pct":  round((grand_value-grand_cost)/grand_cost*100,2) if grand_cost>0 else 0,
            "num_portfolios": len(results),
            "top_exposures":  [{"ticker":t,"value":round(v,2)} for t,v in top_exp],
        }
    }), 200

# ── FUNDAMENTALS (uses /securities/{ticker}/stats) ────────────────────────────
@app.route("/api/fundamentals/<ticker>")
def fundamentals(ticker):
    s, d = atlas_get(f"/analytics/ticker_stats/{ticker}", ttl=600)
    if s != 200: return jsonify({"detail": d.get("detail","Not found")}), s
    s2, sec = atlas_get(f"/securities/{ticker}", ttl=60)
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
    limit  = request.args.get("limit", 50)
    ticker = request.args.get("ticker")
    offset = request.args.get("offset", 0)
    ttype  = request.args.get("type", "")
    atl_params = {"limit": limit}
    if ticker: atl_params["ticker"] = ticker
    s_atl, d_atl = atlas_get("/transactions", params=atl_params, ttl=10)
    if s_atl == 200: return jsonify(d_atl), 200
    params = {"limit": limit, "offset": offset}
    if ttype: params["type"] = ttype
    auth_hdrs = _get_user_auth(request)
    if not auth_hdrs:
        return jsonify({"detail": "Login required"}), 401
    try:
        r = _session.get(f"{NER_BASE}/transactions", headers=auth_hdrs, params=params, timeout=8)
        return jsonify(r.json()), r.status_code
    except Exception as e:
        return jsonify({"detail": str(e)}), 503

# ── OPEN ORDERS ────────────────────────────────────────────────────────────────
@app.route("/api/orders_open")
def orders_open():
    auth_hdrs = _get_user_auth(request)
    if not auth_hdrs:
        return jsonify({"detail": "Login required"}), 401
    try:
        r = _session.get(f"{NER_BASE}/orders", headers=auth_hdrs, timeout=8)
        return jsonify(r.json()), r.status_code
    except Exception as e:
        return jsonify({"detail": str(e)}), 503

@app.route("/api/orders_open/<order_id>", methods=["DELETE"])
def cancel_order(order_id):
    auth_hdrs = _get_user_auth(request)
    if not auth_hdrs:
        return jsonify({"detail": "Login required"}), 401
    try:
        r = _session.delete(f"{NER_BASE}/orders/{order_id}", headers=auth_hdrs, timeout=(4, 8))
        return jsonify(r.json()), r.status_code
    except: return jsonify({"detail":"Error"}), 503

# ── FUNDS ─────────────────────────────────────────────────────────────────────
@app.route("/api/funds")
def funds():
    auth_hdrs = _get_user_auth(request)
    if not auth_hdrs:
        return jsonify({"detail": "Login required"}), 401
    try:
        r = _session.get(f"{NER_BASE}/funds", headers=auth_hdrs, timeout=8)
        return jsonify(r.json()), r.status_code
    except Exception as e:
        return jsonify({"detail": str(e)}), 503

# ── ROLLING CORRELATION + PAIRS SPREAD ────────────────────────────────────────
@app.route("/api/rolling_correlation")
def rolling_correlation():
    t1 = request.args.get("t1",""); t2 = request.args.get("t2","")
    days   = int(request.args.get("days", 60))
    window = int(request.args.get("window", 14))
    if not t1 or not t2: return jsonify({"detail":"t1 and t2 required"}), 400
    _, r1 = atlas_get(f"/analytics/ohlcv/{t1}", params={"days":days}, ttl=60)
    _, r2 = atlas_get(f"/analytics/ohlcv/{t2}", params={"days":days}, ttl=60)
    c1 = [c["close"] for c in _norm_candles(r1.get("candles",[]))]
    c2 = [c["close"] for c in _norm_candles(r2.get("candles",[]))]
    dates = [c["date"] for c in _norm_candles(r1.get("candles",[]))]
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
    _, secs_data = atlas_get("/securities", ttl=60)
    tickers = [s["ticker"] for s in (secs_data if isinstance(secs_data,list) else [])][:20]  # cap at 20
    # Fetch closes for all
    closes_map = {}
    for t in tickers:
        _, raw = atlas_get(f"/analytics/ohlcv/{t}", params={"days":days}, ttl=600)
        cs = [c["close"] for c in _norm_candles(raw.get("candles",[]))]
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
    if ticker.upper().startswith("TSE:"):
        _, raw2 = atlas_get(f"/history/{ticker}", params={"days": 365, "limit": 5000}, ttl=60)
        pts2 = raw2.get("data", []) if isinstance(raw2, dict) else raw2
        cs = [c["close"] for c in _build_candles_from_history(pts2, days)]
    else:
        _, raw = atlas_get(f"/analytics/ohlcv/{ticker}", params={"days":days}, ttl=60)
        cs = [c["close"] for c in _norm_candles(raw.get("candles",[]))]
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
    body     = request.json or {}
    ticker   = body.get("ticker", "")
    days     = int(body.get("days", 90))
    strategy = body.get("strategy", "sma_cross")

    if ticker.upper().startswith("TSE:"):
        _, raw2 = atlas_get(f"/history/{ticker}", params={"days": 365, "limit": 5000}, ttl=600)
        pts2 = raw2.get("data", []) if isinstance(raw2, dict) else raw2
        cs = [c["close"] for c in _build_candles_from_history(pts2, days)]
    else:
        _, raw = atlas_get(f"/analytics/ohlcv/{ticker}", params={"days": days}, ttl=600)
        cs = [c["close"] for c in _norm_candles(raw.get("candles", []), days=days)]
    n  = len(cs)
    if n < 8:
        return jsonify({"detail": f"Not enough data ({n} candles, need at least 8)"}), 404

    def run_sma(fast, slow):
        if fast >= slow or slow >= n: return None
        sma_f = [None if i < fast-1 else sum(cs[i-fast+1:i+1])/fast for i in range(n)]
        sma_s = [None if i < slow-1 else sum(cs[i-slow+1:i+1])/slow for i in range(n)]
        pos = 0; equity = cs[0]
        for i in range(1, n):
            if sma_f[i] and sma_s[i] and sma_f[i-1] and sma_s[i-1]:
                if sma_f[i] > sma_s[i] and sma_f[i-1] <= sma_s[i-1] and pos == 0:
                    pos = equity / cs[i]
                elif sma_f[i] < sma_s[i] and sma_f[i-1] >= sma_s[i-1] and pos > 0:
                    equity = pos * cs[i]; pos = 0
        if pos > 0: equity = pos * cs[-1]
        return round((equity - cs[0]) / cs[0] * 100, 2)

    def run_rsi(period, ob, os_):
        if period >= n: return None
        r = [None] * period
        for i in range(period, n):
            g = [max(cs[j]-cs[j-1], 0) for j in range(i-period+1, i+1)]
            l = [max(cs[j-1]-cs[j], 0) for j in range(i-period+1, i+1)]
            ag, al = sum(g)/period, sum(l)/period
            r.append(100.0 if al == 0 else round(100 - 100/(1+ag/al), 2))
        pos = 0; equity = cs[0]
        for i in range(1, len(r)):
            if r[i] and r[i-1]:
                if r[i] > os_ and r[i-1] <= os_ and pos == 0:
                    pos = equity / cs[i]
                elif r[i] < ob and r[i-1] >= ob and pos > 0:
                    equity = pos * cs[i]; pos = 0
        if pos > 0: equity = pos * cs[-1]
        return round((equity - cs[0]) / cs[0] * 100, 2)

    # ── Dynamic grid: scale parameter ranges to available candles ────────────
    # SMA: max slow = floor(n * 0.6), capped at 50; generate ~5 meaningful steps
    def _steps(lo, hi, count):
        """Generate `count` evenly-spaced integer steps from lo to hi inclusive."""
        if hi <= lo: return [lo]
        step = max(1, (hi - lo) // (count - 1))
        vals = list(range(lo, hi + 1, step))
        if vals[-1] != hi: vals.append(hi)
        return vals[:count]

    results = {"strategy": strategy, "ticker": ticker, "candles": n}

    if strategy == "sma_cross":
        max_slow = max(4, min(50, int(n * 0.6)))
        max_fast = max(2, min(20, int(max_slow * 0.5)))
        slow_range = _steps(max(4, max_slow // 5), max_slow, 5)
        fast_range = _steps(2, max(3, slow_range[0] - 1), 5)
        results.update({
            "rows": fast_range, "cols": slow_range,
            "row_label": "Fast SMA", "col_label": "Slow SMA",
            "rows_data": [[run_sma(f, s) for s in slow_range] for f in fast_range],
        })

    elif strategy == "rsi":
        max_period = max(4, min(25, int(n * 0.4)))
        periods   = _steps(3, max_period, 5)
        ob_levels = [60, 65, 70, 75] if n >= 20 else [65, 70]
        results.update({
            "rows": periods, "cols": ob_levels,
            "row_label": "RSI Period", "col_label": "Overbought Level",
            "rows_data": [[run_rsi(p, ob, 100-ob) for ob in ob_levels] for p in periods],
        })

    results["bh_return"] = round((cs[-1] - cs[0]) / cs[0] * 100, 2) if cs[0] else 0
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
        _, raw = atlas_get(f"/analytics/ohlcv/{h['ticker']}", params={"days":days}, ttl=600)
        cs = [c["close"] for c in _norm_candles(raw.get("candles",[]))]
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
@app.route("/api/debug/ohlcv/<path:ticker>")
def debug_ohlcv(ticker):
    """Debug: shows raw Atlas ohlcv + price_history response shape — bypasses all caching."""
    import requests as _req
    try:
        r1 = _req.get(f"{ATLAS_BASE}/analytics/ohlcv/{ticker}",
                      headers=ATLAS_H, params={"days": 7}, timeout=10)
        raw1 = r1.json()
        candles = raw1.get("candles", []) if isinstance(raw1, dict) else []
        sample1 = candles[:2] if candles else []

        r2 = _req.get(f"{ATLAS_BASE}/history/{ticker}",
                      headers=ATLAS_H, params={"days": 7, "limit": 5}, timeout=10)
        raw2 = r2.json()
        pts = raw2.get("data", raw2) if isinstance(raw2, dict) else raw2
        sample2 = pts[:2] if isinstance(pts, list) else []

        return jsonify({
            "ohlcv": {
                "status": r1.status_code,
                "top_level_keys": list(raw1.keys()) if isinstance(raw1, dict) else str(type(raw1)),
                "candle_count": len(candles),
                "sample": sample1,
            },
            "price_history": {
                "status": r2.status_code,
                "type": str(type(raw2)),
                "top_level_keys": list(raw2.keys()) if isinstance(raw2, dict) else "list",
                "count": len(pts) if isinstance(pts, list) else "?",
                "sample": sample2,
            },
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/debug/shareholders/<ticker>")
def debug_shareholders(ticker):
    """Exposes raw NER /shareholders response for debugging."""
    try:
        r = _session.get(f"{NER_BASE}/shareholders", headers=AUTH_H,
                         params={"ticker": ticker}, timeout=6)
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

    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
