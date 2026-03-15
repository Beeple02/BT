"""
enterprise_store.py — Portfolio persistence layer.
Currently uses a local JSON file. Swap out _load/_save to use Postgres/Redis later.
Thread-safe via a simple lock.
"""
import os, json, threading, uuid
from datetime import datetime, timezone

_LOCK = threading.Lock()
_ENT_FILE = os.path.join(os.path.dirname(__file__), "enterprise_state.json")


def _load():
    with _LOCK:
        try:
            if os.path.exists(_ENT_FILE):
                with open(_ENT_FILE) as f:
                    return json.load(f)
        except Exception as e:
            print(f"[store] load error: {e}")
        return {"portfolios": []}


def _save(data):
    with _LOCK:
        try:
            with open(_ENT_FILE, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"[store] save error: {e}")


# ── Portfolio CRUD ─────────────────────────────────────────────────────────────

def get_all_portfolios():
    return _load().get("portfolios", [])


def upsert_portfolio(body: dict) -> dict:
    data = _load()
    if not body.get("id"):
        body["id"] = str(uuid.uuid4())[:8]
    body.setdefault("positions", [])
    body.setdefault("cash", 0.0)
    body.setdefault("cash_log", [])
    body.setdefault("created_at", datetime.now(timezone.utc).isoformat())
    idx = next((i for i,p in enumerate(data["portfolios"]) if p["id"]==body["id"]), None)
    if idx is not None:
        # Preserve positions/cash unless explicitly passed
        existing = data["portfolios"][idx]
        body.setdefault("positions", existing.get("positions", []))
        body.setdefault("cash", existing.get("cash", 0.0))
        body.setdefault("cash_log", existing.get("cash_log", []))
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
    return next((p for p in _load()["portfolios"] if p["id"]==pf_id), None)


# ── Position CRUD ──────────────────────────────────────────────────────────────

def add_position(pf_id: str, pos: dict) -> dict | None:
    data = _load()
    pf = next((p for p in data["portfolios"] if p["id"]==pf_id), None)
    if not pf: return None
    pos["id"] = str(uuid.uuid4())[:8]
    pos["added_at"] = datetime.now(timezone.utc).isoformat()
    pos["ticker"] = pos.get("ticker","").upper()
    pos["qty"]    = float(pos.get("qty", 0))
    pos["entry_price"] = float(pos.get("entry_price", 0))
    pf.setdefault("positions", []).append(pos)
    _save(data)
    return pos


def remove_position(pf_id: str, pos_id: str) -> int:
    data = _load()
    pf = next((p for p in data["portfolios"] if p["id"]==pf_id), None)
    if not pf: return 0
    before = len(pf.get("positions",[]))
    pf["positions"] = [p for p in pf.get("positions",[]) if p["id"]!=pos_id]
    _save(data)
    return before - len(pf["positions"])


def update_position(pf_id: str, pos_id: str, updates: dict) -> dict | None:
    """Patch individual position fields (e.g. notes, qty)."""
    data = _load()
    pf = next((p for p in data["portfolios"] if p["id"]==pf_id), None)
    if not pf: return None
    pos = next((p for p in pf.get("positions",[]) if p["id"]==pos_id), None)
    if not pos: return None
    pos.update(updates)
    _save(data)
    return pos


# ── Cash management ────────────────────────────────────────────────────────────

def adjust_cash(pf_id: str, amount: float, note: str = "") -> dict | None:
    data = _load()
    pf = next((p for p in data["portfolios"] if p["id"]==pf_id), None)
    if not pf: return None
    pf["cash"] = float(pf.get("cash",0)) + amount
    pf.setdefault("cash_log",[]).append({
        "amount": amount,
        "note":   note,
        "ts":     datetime.now(timezone.utc).isoformat(),
        "balance_after": pf["cash"],
    })
    _save(data)
    return {"cash": pf["cash"], "log_entry": pf["cash_log"][-1]}


def update_portfolio_meta(pf_id: str, fields: dict) -> dict | None:
    """Update non-position fields: name, client, notes, strategy."""
    data = _load()
    pf = next((p for p in data["portfolios"] if p["id"]==pf_id), None)
    if not pf: return None
    allowed = {"name","client","notes","strategy","benchmark","currency"}
    for k,v in fields.items():
        if k in allowed: pf[k] = v
    _save(data)
    return pf
