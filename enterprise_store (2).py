"""
enterprise_store.py — Portfolio persistence layer.
Thread-safe. Uses Railway Volume at /data if ENTERPRISE_DATA_DIR is set,
falls back to app directory for local dev.

To enable persistence on Railway:
  1. Add a Volume in Railway dashboard, mount path: /data
  2. Set env var: ENTERPRISE_DATA_DIR=/data
"""
import os, json, threading, uuid
from datetime import datetime, timezone

_LOCK = threading.Lock()

# Use ENTERPRISE_DATA_DIR env var (Railway Volume) or fall back to app dir
_DATA_DIR = os.environ.get("ENTERPRISE_DATA_DIR", os.path.dirname(os.path.abspath(__file__)))
_ENT_FILE = os.path.join(_DATA_DIR, "enterprise_state.json")

print(f"[store] data file: {_ENT_FILE}")


def _load() -> dict:
    with _LOCK:
        try:
            if os.path.exists(_ENT_FILE):
                with open(_ENT_FILE, "r") as f:
                    return json.load(f)
        except Exception as e:
            print(f"[store] load error: {e}")
        return {"portfolios": []}


def _save(data: dict):
    """Atomic write: write to .tmp then rename to prevent corruption."""
    with _LOCK:
        try:
            os.makedirs(_DATA_DIR, exist_ok=True)
            tmp = _ENT_FILE + ".tmp"
            with open(tmp, "w") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp, _ENT_FILE)  # atomic on POSIX
        except Exception as e:
            print(f"[store] save error: {e}")


# ── Portfolio CRUD ─────────────────────────────────────────────────────────────

def get_all_portfolios() -> list:
    return _load().get("portfolios", [])


def upsert_portfolio(body: dict) -> dict:
    data = _load()
    if not body.get("id"):
        body["id"] = str(uuid.uuid4())[:8]
    body.setdefault("positions", [])
    body.setdefault("cash", 0.0)
    body.setdefault("cash_log", [])
    body.setdefault("created_at", datetime.now(timezone.utc).isoformat())
    idx = next((i for i, p in enumerate(data["portfolios"]) if p["id"] == body["id"]), None)
    if idx is not None:
        existing = data["portfolios"][idx]
        # Preserve positions/cash/log if not explicitly in body
        for key in ("positions", "cash", "cash_log"):
            if key not in body:
                body[key] = existing.get(key, [] if key != "cash" else 0.0)
        data["portfolios"][idx] = body
    else:
        data["portfolios"].append(body)
    _save(data)
    return body


def delete_portfolio(pf_id: str) -> int:
    data = _load()
    before = len(data["portfolios"])
    data["portfolios"] = [p for p in data["portfolios"] if p["id"] != pf_id]
    _save(data)
    return before - len(data["portfolios"])


def get_portfolio(pf_id: str) -> dict | None:
    return next((p for p in _load()["portfolios"] if p["id"] == pf_id), None)


# ── Position CRUD ──────────────────────────────────────────────────────────────

def add_position(pf_id: str, pos: dict) -> dict | None:
    data = _load()
    pf = next((p for p in data["portfolios"] if p["id"] == pf_id), None)
    if not pf: return None
    pos["id"]          = str(uuid.uuid4())[:8]
    pos["added_at"]    = datetime.now(timezone.utc).isoformat()
    pos["ticker"]      = pos.get("ticker", "").upper()
    pos["qty"]         = float(pos.get("qty", 0))
    pos["entry_price"] = float(pos.get("entry_price", 0))
    pf.setdefault("positions", []).append(pos)
    _save(data)
    return pos


def bulk_add_positions(pf_id: str, positions: list) -> list:
    """Add multiple positions atomically — used by the import feature."""
    data = _load()
    pf = next((p for p in data["portfolios"] if p["id"] == pf_id), None)
    if not pf: return []
    added = []
    now = datetime.now(timezone.utc).isoformat()
    for pos in positions:
        pos["id"]          = str(uuid.uuid4())[:8]
        pos["added_at"]    = now
        pos["ticker"]      = pos.get("ticker", "").upper()
        pos["qty"]         = float(pos.get("qty", 0))
        pos["entry_price"] = float(pos.get("entry_price", 0))
        pf.setdefault("positions", []).append(pos)
        added.append(pos)
    _save(data)
    return added


def remove_position(pf_id: str, pos_id: str) -> int:
    data = _load()
    pf = next((p for p in data["portfolios"] if p["id"] == pf_id), None)
    if not pf: return 0
    before = len(pf.get("positions", []))
    pf["positions"] = [p for p in pf.get("positions", []) if p["id"] != pos_id]
    _save(data)
    return before - len(pf["positions"])


def update_position(pf_id: str, pos_id: str, updates: dict) -> dict | None:
    data = _load()
    pf = next((p for p in data["portfolios"] if p["id"] == pf_id), None)
    if not pf: return None
    pos = next((p for p in pf.get("positions", []) if p["id"] == pos_id), None)
    if not pos: return None
    pos.update(updates)
    _save(data)
    return pos


# ── Cash management ────────────────────────────────────────────────────────────

def adjust_cash(pf_id: str, amount: float, note: str = "") -> dict | None:
    data = _load()
    pf = next((p for p in data["portfolios"] if p["id"] == pf_id), None)
    if not pf: return None
    pf["cash"] = float(pf.get("cash", 0)) + amount
    pf.setdefault("cash_log", []).append({
        "amount":        amount,
        "note":          note,
        "ts":            datetime.now(timezone.utc).isoformat(),
        "balance_after": pf["cash"],
    })
    _save(data)
    return {"cash": pf["cash"], "log_entry": pf["cash_log"][-1]}


def update_portfolio_meta(pf_id: str, fields: dict) -> dict | None:
    data = _load()
    pf = next((p for p in data["portfolios"] if p["id"] == pf_id), None)
    if not pf: return None
    allowed = {"name", "client", "notes", "strategy", "benchmark", "currency"}
    for k, v in fields.items():
        if k in allowed:
            pf[k] = v
    _save(data)
    return pf
