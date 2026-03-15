"""
quant_utils.py — Pure financial math. No Flask, no HTTP, no side effects.
"""
import statistics as _stats


def sma(s, n):
    return [None if i < n-1 else round(sum(s[i-n+1:i+1])/n, 6) for i in range(len(s))]

def ema(s, n):
    k = 2/(n+1); r = [s[0]]
    for v in s[1:]: r.append(round(v*k + r[-1]*(1-k), 6))
    return r

def rsi(s, p=14):
    result = [None]*p
    for i in range(p, len(s)):
        g = [max(s[j]-s[j-1], 0) for j in range(i-p+1, i+1)]
        l = [max(s[j-1]-s[j], 0) for j in range(i-p+1, i+1)]
        ag, al = sum(g)/p, sum(l)/p
        result.append(round(100.0 if al==0 else 100-100/(1+ag/al), 2))
    return result

def atr(highs, lows, closes, p=14):
    trs = []
    for i in range(1, len(closes)):
        trs.append(max(highs[i]-lows[i], abs(highs[i]-closes[i-1]), abs(lows[i]-closes[i-1])))
    out = [None]
    for i in range(len(trs)):
        if i < p-1: out.append(None)
        else: out.append(round(sum(trs[max(0,i-p+1):i+1])/p, 6))
    return out

def bollinger(closes, p=20, k=2):
    upper, mid, lower = [], [], []
    for i in range(len(closes)):
        if i < p-1: upper.append(None); mid.append(None); lower.append(None)
        else:
            w = closes[i-p+1:i+1]; m = sum(w)/p
            sd = (sum((x-m)**2 for x in w)/p)**0.5
            mid.append(round(m,6)); upper.append(round(m+k*sd,6)); lower.append(round(m-k*sd,6))
    return upper, mid, lower

def macd(closes, fast=12, slow=26, signal=9):
    f = ema(closes, fast); s = ema(closes, slow)
    line = [round(a-b, 6) for a,b in zip(f, s)]
    sig  = ema(line, signal)
    hist = [round(m-s, 6) for m,s in zip(line, sig)]
    return line, sig, hist

def vwap_series(highs, lows, closes, volumes):
    typical = [(h+l+c)/3 for h,l,c in zip(highs,lows,closes)]
    cum_tv = 0; cum_v = 0; result = []
    for tp, v in zip(typical, volumes):
        cum_tv += tp*v; cum_v += v
        result.append(round(cum_tv/cum_v, 6) if cum_v else None)
    return result

def ann_vol(closes):
    if len(closes) < 2: return 0
    rets = [(closes[i]-closes[i-1])/closes[i-1] for i in range(1,len(closes))]
    if len(rets) < 2: return 0
    return round(_stats.stdev(rets) * (252**0.5) * 100, 2)

def max_drawdown(series):
    peak = series[0]; dd = 0
    for v in series:
        if v > peak: peak = v
        d = (peak-v)/peak if peak > 0 else 0
        if d > dd: dd = d
    return round(dd*100, 2)

def sharpe(closes, rf=0):
    if len(closes) < 2: return 0
    rets = [(closes[i]-closes[i-1])/closes[i-1] for i in range(1,len(closes))]
    if len(rets) < 2: return 0
    avg = sum(rets)/len(rets); std = _stats.stdev(rets)
    return round((avg - rf/252)/std*(252**0.5), 3) if std > 0 else 0

def sortino(closes, rf=0):
    if len(closes) < 2: return 0
    rets = [(closes[i]-closes[i-1])/closes[i-1] for i in range(1,len(closes))]
    neg  = [r for r in rets if r < 0]
    if not neg: return 0
    avg  = sum(rets)/len(rets)
    dd   = (sum(r**2 for r in neg)/len(neg))**0.5
    return round((avg - rf/252)/dd*(252**0.5), 3) if dd > 0 else 0

def calmar(closes):
    dd = max_drawdown(closes)
    if dd == 0 or not closes[0]: return 0
    return round(abs((closes[-1]-closes[0])/closes[0]*100/dd), 3)

def mean_reversion_score(closes):
    if len(closes) < 20: return None
    m = sum(closes[-20:])/20
    sd = (sum((c-m)**2 for c in closes[-20:])/20)**0.5
    return round((closes[-1]-m)/sd, 2) if sd else None

def downside_vol(closes):
    if len(closes) < 2: return None
    rets = [(closes[i]-closes[i-1])/closes[i-1] for i in range(1,len(closes))]
    neg  = [r for r in rets if r < 0]
    if not neg: return 0.0
    return round((sum(r**2 for r in neg)/len(neg))**0.5*(252**0.5)*100, 2)

def var_hist(returns, conf=0.95):
    if not returns: return 0
    return abs(sorted(returns)[int(len(returns)*(1-conf))])

def cvar_hist(returns, conf=0.95):
    if not returns: return 0
    s = sorted(returns); tail = s[:max(1, int(len(s)*(1-conf)))]
    return abs(sum(tail)/len(tail))

def hhi(weights):
    return round(sum(w**2 for w in weights)*10000, 1)

def gini(weights):
    n = len(weights)
    if n < 2: return 0
    return round(sum(abs(a-b) for a in weights for b in weights)/(2*n**2), 4)

def kelly(returns):
    if len(returns) < 2: return 0
    avg = sum(returns)/len(returns)
    var = sum((r-avg)**2 for r in returns)/len(returns)
    return round(avg/var, 4) if var > 0 else 0

# ── Backtest engine ────────────────────────────────────────────────────────────
COMMISSION = 0.005

def bt_run(closes, dates, signals, sl_pct=0.0, tp_pct=0.0):
    cash=10_000.0; shares=0; pos=False; entry=0.0; equity=[]; trades=[]
    for i, price in enumerate(closes):
        if price is None or price <= 0:
            equity.append(round(cash+shares*(closes[i-1] if i else 0), 4)); continue
        sig = signals[i]
        if pos and entry > 0:
            hit_sl = sl_pct>0 and price<=entry*(1-sl_pct)
            hit_tp = tp_pct>0 and price>=entry*(1+tp_pct)
            if hit_sl or hit_tp:
                proceeds=shares*price*(1-COMMISSION); cash+=proceeds
                trades.append({"date":dates[i],"action":"STOP" if hit_sl else "TP","price":round(price,4),"qty":shares,"value":round(proceeds,2)})
                shares=0; pos=False; entry=0.0
        if sig==1 and not pos:
            qty=int(cash/(price*(1+COMMISSION)))
            if qty>0:
                cost=qty*price*(1+COMMISSION); cash-=cost; shares=qty; pos=True; entry=price
                trades.append({"date":dates[i],"action":"BUY","price":round(price,4),"qty":qty,"value":round(cost,2)})
        elif sig==-1 and pos and shares>0:
            proceeds=shares*price*(1-COMMISSION); cash+=proceeds
            trades.append({"date":dates[i],"action":"SELL","price":round(price,4),"qty":shares,"value":round(proceeds,2)})
            shares=0; pos=False; entry=0.0
        equity.append(round(cash+shares*price, 4))
    return equity, trades

def bt_metrics(equity, closes, trades):
    end_eq    = equity[-1] if equity else 10_000
    total_ret = round((end_eq-10_000)/10_000*100, 2)
    bh_ret    = round((closes[-1]-closes[0])/closes[0]*100, 2) if len(closes)>=2 and closes[0] else 0
    dd = max_drawdown(equity); sh = sharpe(equity)
    buys  = [t for t in trades if t["action"]=="BUY"]
    exits = [t for t in trades if t["action"] in ("SELL","STOP","TP")]
    pairs = list(zip(buys, exits))
    wins  = sum(1 for b,e in pairs if e["price"]>b["price"])
    return {
        "total_return_pct":     total_ret,
        "benchmark_return_pct": bh_ret,
        "alpha_pct":            round(total_ret-bh_ret, 2),
        "max_drawdown_pct":     dd,
        "sharpe_ratio":         sh,
        "calmar_ratio":         round(abs(total_ret/dd), 3) if dd>0 else 0,
        "num_trades":           len(trades),
        "win_rate":             round(wins/len(pairs)*100,1) if pairs else 0.0,
        "final_equity":         round(end_eq, 2),
    }
