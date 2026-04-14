import math, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="ALHOWIFI SMART TRADING", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
.main-title{font-size:2.2rem;font-weight:900;letter-spacing:4px;color:#00f5ff;text-shadow:0 0 10px #00d4ff,0 0 20px #00d4ff;}
.sub{font-size:.85rem;color:#64748b;margin-bottom:1rem;letter-spacing:2px;}
.card{padding:14px;border-radius:12px;border:1px solid rgba(255,255,255,.08);background:rgba(255,255,255,.03);}
.call-card{padding:14px;border-radius:12px;border:1px solid rgba(0,255,136,.3);background:rgba(0,255,136,.07);}
.put-card{padding:14px;border-radius:12px;border:1px solid rgba(255,64,96,.3);background:rgba(255,64,96,.07);}
.wait-card{padding:14px;border-radius:12px;border:1px solid rgba(251,191,36,.3);background:rgba(251,191,36,.07);}
.safe-card{padding:12px;border-radius:10px;border:1px solid rgba(0,255,136,.3);background:rgba(0,255,136,.07);}
.risk-card{padding:12px;border-radius:10px;border:1px solid rgba(251,191,36,.3);background:rgba(251,191,36,.07);}
.trap-card{padding:12px;border-radius:10px;border:1px solid rgba(255,64,96,.3);background:rgba(255,64,96,.07);}
</style>
""", unsafe_allow_html=True)

SECTORS = {
    "تقنية": ["AAPL","MSFT","GOOGL","ORCL","XLK"],
    "AI/رقائق": ["NVDA","AMD","AVGO","MU","LRCX"],
    "سايبر/نمو": ["META","NFLX","CRWD","PANW"],
    "استهلاكي": ["TSLA","AMZN","COST"],
    "مالي": ["JPM","GS","XLF"],
    "صحة/طاقة": ["LLY","XOM","XLE","XLV"],
    "صناعي": ["GE","CAT","GLD"],
    "مؤشرات": ["SPY","QQQ"],
}
ALL_STOCKS = [s for lst in SECTORS.values() for s in lst]

TIMEFRAME_MAP = {
    "لحظي 1m":    {"interval":"1m",  "period":"1d"},
    "مضاربة 5m":  {"interval":"5m",  "period":"5d"},
    "مضاربة 15m": {"interval":"15m", "period":"60d"},
    "ساعة 1H":    {"interval":"60m", "period":"180d"},
    "يومي":       {"interval":"1d",  "period":"1y"},
    "أسبوعي":     {"interval":"1wk", "period":"5y"},
    "شهري":       {"interval":"1mo", "period":"10y"},
}
ATR_MULT = {"لحظي 1m":0.8,"مضاربة 5m":1.0,"مضاربة 15m":1.2,"ساعة 1H":1.5,"يومي":2.0,"أسبوعي":2.8,"شهري":3.8}
HIGHER_TF = {"1m":"5m","5m":"15m","15m":"60m","60m":"1d","1d":"1wk","1wk":"1mo","1mo":None}
HIGHER_PERIOD = {"5m":"5d","15m":"60d","60m":"180d","1d":"1y","1wk":"5y","1mo":"10y"}

# ── MATH ──────────────────────────────────────────────
def to1d(s):
    """تحويل أي Series أو Array لـ 1D numpy array"""
    arr = np.array(s).flatten().astype(float)
    return arr

def ema_calc(arr, n):
    k = 2.0/(n+1)
    result = np.zeros(len(arr))
    result[0] = arr[0]
    for i in range(1, len(arr)):
        result[i] = arr[i]*k + result[i-1]*(1-k)
    return result

def sma_calc(arr, n):
    result = np.full(len(arr), np.nan)
    for i in range(n-1, len(arr)):
        result[i] = np.mean(arr[i-n+1:i+1])
    return result

def wma_calc(arr, n):
    result = np.full(len(arr), np.nan)
    weights = np.arange(1, n+1, dtype=float)
    wsum = weights.sum()
    for i in range(n-1, len(arr)):
        result[i] = np.dot(arr[i-n+1:i+1], weights)/wsum
    return result

def hma_calc(arr, n):
    h = max(int(n/2), 1)
    sq = max(int(math.sqrt(n)), 1)
    wh = wma_calc(arr, h)
    wf = wma_calc(arr, n)
    diff = 2*wh - wf
    return wma_calc(diff, sq)

def rsi_calc(arr):
    result = np.full(len(arr), 50.0)
    if len(arr) < 15:
        return result
    gains = np.zeros(len(arr))
    losses = np.zeros(len(arr))
    for i in range(1, len(arr)):
        d = arr[i] - arr[i-1]
        if d > 0: gains[i] = d
        else: losses[i] = -d
    ag = np.mean(gains[1:15])
    al = np.mean(losses[1:15])
    if al == 0: result[14] = 100.0
    else: result[14] = 100.0 - 100.0/(1.0 + ag/al)
    for i in range(15, len(arr)):
        ag = (ag*13 + gains[i])/14
        al = (al*13 + losses[i])/14
        if al == 0: result[i] = 100.0
        else: result[i] = 100.0 - 100.0/(1.0 + ag/al)
    return result

def macd_calc(arr):
    el = ema_calc(arr, 5)
    es = ema_calc(arr, 13)
    line = el - es
    sig = ema_calc(line, 6)
    hist = line - sig
    return line, sig, hist

def atr_calc(high, low, close):
    tr = np.zeros(len(close))
    tr[0] = high[0] - low[0]
    for i in range(1, len(close)):
        tr[i] = max(high[i]-low[i], abs(high[i]-close[i-1]), abs(low[i]-close[i-1]))
    return ema_calc(tr, 14)

def vwap_calc(high, low, close, volume):
    tp = (high + low + close) / 3.0
    cv = np.cumsum(tp * volume)
    cv2 = np.cumsum(volume)
    result = np.where(cv2 > 0, cv/cv2, tp)
    return result

def bb_calc(arr, n=20, m=2):
    mid = sma_calc(arr, n)
    upper = np.full(len(arr), np.nan)
    lower = np.full(len(arr), np.nan)
    for i in range(n-1, len(arr)):
        std = np.std(arr[i-n+1:i+1])
        upper[i] = mid[i] + m*std
        lower[i] = mid[i] - m*std
    return mid, upper, lower

def nw_calc(arr, w=50, b=3.0):
    """Nadaraya-Watson - آمن تماماً"""
    n = len(arr)
    result = np.full(n, np.nan)
    w = min(w, n)
    if w < 5:
        return result
    # حساب الأوزان مرة واحدة
    positions = np.arange(w, dtype=float)
    center = float(w - 1)
    raw_weights = np.exp(-0.5 * ((positions - center) / b) ** 2)
    raw_weights = raw_weights / raw_weights.sum()
    for i in range(w-1, n):
        segment = arr[i-w+1:i+1]
        # تأكد أن الأبعاد متطابقة
        seg_flat = np.array(segment).flatten()
        if len(seg_flat) == w:
            result[i] = float(np.dot(seg_flat, raw_weights))
    return result

def rvol_calc(volume, n=20):
    avg = sma_calc(volume, n)
    result = np.where(avg > 0, volume/avg, 1.0)
    return result

# ── DATA ──────────────────────────────────────────────
@st.cache_data(ttl=60)
def get_raw(sym, interval, period):
    try:
        df = yf.download(sym, interval=interval, period=period,
                         auto_adjust=True, progress=False)
        if df is None or df.empty:
            return None
        # flatten MultiIndex
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        # keep only needed cols
        needed = ["Open","High","Low","Close","Volume"]
        for c in needed:
            if c not in df.columns:
                return None
        df = df[needed].copy()
        df = df.dropna(subset=["Close","High","Low"])
        if len(df) < 5:
            return None
        return df
    except:
        return None

def compute(df):
    if df is None or df.empty:
        return None
    try:
        C = to1d(df["Close"])
        H = to1d(df["High"])
        L = to1d(df["Low"])
        V = to1d(df["Volume"])
        n = len(C)
        if n < 5:
            return None

        r = df.copy()
        r["C"] = C
        r["H"] = H
        r["L"] = L
        r["V"] = V

        r["EMA200"] = ema_calc(C, min(200,n))
        r["RSI"]    = rsi_calc(C)
        r["MACD"], r["SIGNAL"], r["HIST"] = macd_calc(C)
        r["ATR"]   = atr_calc(H, L, C)
        r["RVOL"]  = rvol_calc(V)
        r["VWAP"]  = vwap_calc(H, L, C, V)
        r["HF"]    = hma_calc(C, min(18, max(3, n//3)))
        r["HS"]    = hma_calc(C, min(55, max(3, n//3)))
        r["HM"]    = hma_calc(C, min(14, max(3, n//3)))

        nwc        = nw_calc(C, min(50, max(5, n//2)), 3.0)
        atr        = r["ATR"].values
        r["NWC"]   = nwc
        r["NWU"]   = nwc + atr * 2.5
        r["NWL"]   = nwc - atr * 2.5

        mid, upper, lower = bb_calc(C)
        r["BBM"] = mid
        r["BBU"] = upper
        r["BBL"] = lower
        return r
    except Exception as e:
        return None

def get_df(sym, interval, period):
    raw = get_raw(sym, interval, period)
    return compute(raw)

# ── HMA TREND ─────────────────────────────────────────
def hma_trend(df):
    if df is None or df.empty:
        return "غير واضح"
    try:
        hf = to1d(df["HF"])
        hs = to1d(df["HS"])
        atr = to1d(df["ATR"])
        f, s, a = hf[-1], hs[-1], atr[-1] if atr[-1] > 0 else 0.01
        if abs(f-s) < a*0.05:
            return "عرضي"
        n = len(hf)
        fs = hf[-1]-hf[-4] if n > 4 else 0
        ss = hs[-1]-hs[-4] if n > 4 else 0
        if f > s and fs > 0 and ss >= 0: return "صاعد"
        if f < s and fs < 0 and ss <= 0: return "هابط"
        return "عرضي"
    except:
        return "غير واضح"

# ── SIGNAL ────────────────────────────────────────────
def build_signal(df, trend, trendH, tf_label):
    try:
        C   = to1d(df["C"])
        H   = to1d(df["H"])
        L   = to1d(df["L"])
        cv  = float(C[-1])
        a   = float(to1d(df["ATR"])[-1])
        ri  = float(to1d(df["RSI"])[-1])
        rv  = float(to1d(df["RVOL"])[-1])
        ml  = float(to1d(df["MACD"])[-1])
        ms  = float(to1d(df["SIGNAL"])[-1])
        vw  = float(to1d(df["VWAP"])[-1])
        e2  = float(to1d(df["EMA200"])[-1])
        nU  = float(to1d(df["NWU"])[-1])
        nL  = float(to1d(df["NWL"])[-1])
        bU  = float(to1d(df["BBU"])[-1])
        bL  = float(to1d(df["BBL"])[-1])

        if math.isnan(a) or a <= 0: a = cv * 0.01
        if math.isnan(nU): nU = cv * 1.05
        if math.isnan(nL): nL = cv * 0.95
        if math.isnan(bU): bU = cv * 1.02
        if math.isnan(bL): bL = cv * 0.98

        sup = float(np.nanmin(L[-50:]))
        res = float(np.nanmax(H[-50:]))
        atr_avg = float(np.nanmean(to1d(df["ATR"])[-20:]))
        if math.isnan(atr_avg) or atr_avg <= 0: atr_avg = a

        n = len(C)
        pH = float(H[-2]) if n > 1 else cv
        pL = float(L[-2]) if n > 1 else cv
        HL = n >= 3 and float(L[-1]) > float(L[-2])
        LH = n >= 3 and float(H[-1]) < float(H[-2])

        aV = cv > vw; aE = cv > e2; mU = ml > ms; mD = ml < ms
        rOK = not math.isnan(rv) and rv > 1.2
        aOK = a >= atr_avg * 0.9
        tHu = trendH in ["صاعد","عرضي"]
        tHd = trendH in ["هابط","عرضي"]

        CC = {
            "فوق VWAP": aV, "فوق EMA200": aE, "HMA صاعد": trend=="صاعد",
            "الفريم الأعلى": tHu, "RSI > 55": ri > 55, "MACD+": mU,
            "Higher Low": HL or cv > pH, "RVOL > 1.2": rOK, "ATR نشط": aOK,
            "منطقة دخول": cv > pH or cv <= (nL + a*0.35), "BB سليم": cv <= bU,
        }
        CP = {
            "تحت VWAP": not aV, "تحت EMA200": not aE, "HMA هابط": trend=="هابط",
            "الفريم الأعلى": tHd, "RSI < 45": ri < 45, "MACD-": mD,
            "Lower High": LH or cv < pL, "RVOL > 1.2": rOK, "ATR نشط": aOK,
            "منطقة دخول": cv < pL or cv >= (nU - a*0.35), "BB سليم": cv >= bL,
        }
        cs = int(sum(CC.values())/11*100)
        ps = int(sum(CP.values())/11*100)

        sig = "انتظار"; st_ = "WAIT"
        if cs >= 73 and cs > ps:   sig = "كول"; st_ = "CONFIRMED" if cs>=85 else "ARMED"
        elif ps >= 73 and ps > cs: sig = "بوت"; st_ = "CONFIRMED" if ps>=85 else "ARMED"
        elif cs >= 55 and cs > ps: sig = "كول"; st_ = "ARMED"
        elif ps >= 55 and ps > cs: sig = "بوت"; st_ = "ARMED"

        mW = abs(ml-ms) < max(0.02, a*0.02)
        trap = "SAFE"; tR = []
        if sig == "كول":
            if float(H[-1]) > res and cv < res: trap="TRAP"; tR.append("اختراق وهمي")
            if ri >= 75: trap="RISK" if trap!="TRAP" else trap; tR.append("RSI مشبع شراءً")
            if not aV: trap="RISK" if trap!="TRAP" else trap; tR.append("تحت VWAP")
            if mW: trap="RISK" if trap!="TRAP" else trap; tR.append("MACD ضعيف")
        if sig == "بوت":
            if float(L[-1]) < sup and cv > sup: trap="TRAP"; tR.append("كسر وهمي")
            if ri <= 25: trap="RISK" if trap!="TRAP" else trap; tR.append("RSI مشبع بيعاً")
            if aV: trap="RISK" if trap!="TRAP" else trap; tR.append("فوق VWAP")
            if mW: trap="RISK" if trap!="TRAP" else trap; tR.append("MACD ضعيف")

        if trap == "TRAP": sig = "انتظار"; st_ = "TRAP"
        mm = ATR_MULT.get(tf_label, 1.0)
        tgt = {"entry":cv,"stop":float("nan"),"t1":float("nan"),"t2":float("nan"),"t3":float("nan")}
        if sig == "كول":
            tgt = {"entry":cv,"stop":min(cv-a*mm,vw-a*0.3),"t1":cv+a*mm,"t2":cv+a*mm*2,"t3":res if res>cv else cv+a*mm*3}
        elif sig == "بوت":
            tgt = {"entry":cv,"stop":max(cv+a*mm,vw+a*0.3),"t1":cv-a*mm,"t2":cv-a*mm*2,"t3":sup if sup<cv else cv-a*mm*3}

        zone = "منطقة طلب" if cv<=nL else "منطقة عرض" if cv>=nU else "وسط النطاق"
        cont = "استمرار صاعد" if(aV and mU and ri>=55) else "ضعف" if(mD and ri<50) else "مراقبة"
        if sig == "بوت":
            cont = "استمرار هابط" if(not aV and mD and ri<=45) else "ضعف" if(mU and ri>50) else "مراقبة"

        prev_c = float(C[-2]) if n > 1 else cv
        chg = (cv - prev_c) / prev_c * 100 if prev_c != 0 else 0

        return dict(sig=sig,status=st_,cs=cs,ps=ps,trap=trap,trapR=tR,
                    CC=CC,CP=CP,trend=trend,trendH=trendH,zone=zone,cont=cont,
                    rsi=ri,rvol=rv,atr=a,aboveVWAP=aV,aboveEMA=aE,
                    macdHist=float(to1d(df["HIST"])[-1]),
                    bbPos="فوق BB" if cv>bU else "تحت BB" if cv<bL else "داخل",
                    price=cv,change=chg,sup=sup,res=res,tgt=tgt)
    except Exception as e:
        return None

def scan_sym(sym, tf_label):
    cfg = TIMEFRAME_MAP[tf_label]
    df = get_df(sym, cfg["interval"], cfg["period"])
    if df is None: return None
    trend = hma_trend(df)
    trendH = "غير واضح"
    hInt = HIGHER_TF.get(cfg["interval"])
    if hInt:
        hPer = HIGHER_PERIOD.get(hInt, "1y")
        dfH = get_df(sym, hInt, hPer)
        if dfH is not None: trendH = hma_trend(dfH)
    return build_signal(df, trend, trendH, tf_label)

# ── CHART ─────────────────────────────────────────────
def make_chart(df, sym, tgt):
    if df is None: return None
    chart = df.tail(120)
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        vertical_spacing=0.03, row_heights=[0.62,0.18,0.20])
    idx = chart.index
    fig.add_trace(go.Candlestick(x=idx, open=chart["Open"], high=chart["High"],
                                  low=chart["Low"], close=chart["Close"], name="السعر"), row=1, col=1)
    for col_, name_, color_, dash_ in [
        ("VWAP","VWAP","#00d4ff","solid"),("EMA200","EMA200","#475569","dot"),
        ("HF","HMA Fast","#00ff88","solid"),("HS","HMA Slow","#ff8c42","solid"),
        ("NWC","Nadaraya","#a78bfa","dash"),("NWU","NW Upper","#a78bfa","dot"),
        ("NWL","NW Lower","#a78bfa","dot"),("BBU","BB Upper","#00d4ff","dot"),
        ("BBL","BB Lower","#00d4ff","dot"),
    ]:
        if col_ in chart.columns:
            fig.add_trace(go.Scatter(x=idx, y=chart[col_], name=name_,
                line=dict(color=color_, dash=dash_, width=1.2), opacity=0.8), row=1, col=1)
    for lbl_, val_, color_ in [("دخول",tgt["entry"],"white"),("وقف",tgt["stop"],"#ff4060"),
                                 ("ه1",tgt["t1"],"#00ff88"),("ه2",tgt["t2"],"#00d4ff"),("ه3",tgt["t3"],"#a78bfa")]:
        if val_ and not (isinstance(val_,float) and math.isnan(val_)):
            fig.add_hline(y=val_, row=1, col=1, line_color=color_, line_dash="dot",
                          annotation_text=f"{lbl_} {val_:.2f}", annotation_position="right")
    if "RSI" in chart.columns:
        fig.add_trace(go.Scatter(x=idx, y=chart["RSI"], name="RSI",
            line=dict(color="#a78bfa", width=1.3)), row=2, col=1)
        fig.add_hline(y=70, row=2, col=1, line_color="#ff4060", line_dash="dot")
        fig.add_hline(y=30, row=2, col=1, line_color="#00ff88", line_dash="dot")
    if "MACD" in chart.columns:
        fig.add_trace(go.Scatter(x=idx, y=chart["MACD"], name="MACD",
            line=dict(color="#00d4ff", width=1.2)), row=3, col=1)
        fig.add_trace(go.Scatter(x=idx, y=chart["SIGNAL"], name="Signal",
            line=dict(color="#fbbf24", width=1.2)), row=3, col=1)
        fig.add_trace(go.Bar(x=idx, y=chart["HIST"], name="Hist",
            marker_color=["rgba(0,255,136,.5)" if v>=0 else "rgba(255,64,96,.5)"
                         for v in chart["HIST"].fillna(0)]), row=3, col=1)
    fig.update_layout(title=f"{sym} — ALHOWIFI", template="plotly_dark",
                      height=900, xaxis_rangeslider_visible=False,
                      legend=dict(orientation="h", yanchor="bottom", y=1.02))
    return fig

# ── SIDEBAR ───────────────────────────────────────────
st.sidebar.title("الإعدادات")
mode = st.sidebar.radio("نوع العرض", ["سهم واحد", "مسح القائمة"])
sym = st.sidebar.selectbox("الرمز", ALL_STOCKS, index=0)
tf_label = st.sidebar.selectbox("الفريم", list(TIMEFRAME_MAP.keys()), index=4)
st.sidebar.markdown("---")
for sec, stocks in SECTORS.items():
    st.sidebar.markdown(f"**{sec}:** {', '.join(stocks)}")

# ── HEADER ────────────────────────────────────────────
st.markdown('<div class="main-title">ALHOWIFI SMART TRADING</div>', unsafe_allow_html=True)
st.markdown('<div class="sub">HMA · NADARAYA · ATR · VWAP · EMA200 · RSI · MACD · BOLLINGER · TRAP FILTER · 12 CONDITIONS</div>', unsafe_allow_html=True)

# ── SCAN MODE ─────────────────────────────────────────
if mode == "مسح القائمة":
    st.subheader(f"مسح {len(ALL_STOCKS)} سهم — {tf_label}")
    if st.button("بدء المسح الكامل"):
        rows = []
        pb = st.progress(0)
        status_txt = st.empty()
        for i, s_ in enumerate(ALL_STOCKS):
            status_txt.text(f"تحليل {s_}...")
            try:
                res = scan_sym(s_, tf_label)
                if res:
                    sec = [k for k,v in SECTORS.items() if s_ in v]
                    rows.append({
                        "الرمز":s_,"القطاع":sec[0] if sec else "—",
                        "السعر":f"${res['price']:.2f}","التغيير":f"{res['change']:+.2f}%",
                        "الإشارة":res["sig"],"الاتجاه":res["trend"],
                        "CALL%":res["cs"],"PUT%":res["ps"],"Trap":res["trap"],
                        "الدخول":f"${res['tgt']['entry']:.2f}",
                        "الوقف":f"${res['tgt']['stop']:.2f}" if not math.isnan(res['tgt']['stop']) else "—",
                    })
            except: pass
            pb.progress((i+1)/len(ALL_STOCKS))
        status_txt.empty(); pb.empty()
        if rows:
            df_t = pd.DataFrame(rows).sort_values("CALL%", ascending=False)
            st.dataframe(df_t, use_container_width=True, hide_index=True)
    st.stop()

# ── SINGLE MODE ───────────────────────────────────────
cfg = TIMEFRAME_MAP[tf_label]
with st.spinner(f"تحليل {sym}..."):
    df = get_df(sym, cfg["interval"], cfg["period"])
    if df is None:
        st.error(f"لا توجد بيانات لـ {sym} — السوق مغلق أو الرمز غير صحيح")
        st.stop()
    trend = hma_trend(df)
    trendH = "غير واضح"
    hInt = HIGHER_TF.get(cfg["interval"])
    if hInt:
        hPer = HIGHER_PERIOD.get(hInt, "1y")
        dfH = get_df(sym, hInt, hPer)
        if dfH is not None: trendH = hma_trend(dfH)
    s = build_signal(df, trend, trendH, tf_label)
    if s is None:
        st.error("خطأ في حساب الإشارة")
        st.stop()

c1,c2,c3,c4,c5 = st.columns(5)
with c1:
    tc = "#00ff88" if trend=="صاعد" else "#ff4060" if trend=="هابط" else "#fbbf24"
    st.markdown(f'<div class="card"><div style="font-size:.8rem;color:#64748b;">الاتجاه</div><div style="font-size:1.6rem;font-weight:900;color:{tc}">{trend}</div><div style="font-size:.8rem;color:#64748b;">الأعلى: {trendH}</div></div>', unsafe_allow_html=True)
with c2:
    cls = "call-card" if s["sig"]=="كول" else "put-card" if s["sig"]=="بوت" else "wait-card"
    emoji = "🟢" if s["sig"]=="كول" else "🔴" if s["sig"]=="بوت" else "🟡"
    st.markdown(f'<div class="{cls}"><div style="font-size:.8rem;color:#64748b;">الإشارة</div><div style="font-size:1.6rem;font-weight:900;">{emoji} {s["sig"]}</div><div style="font-size:.8rem;color:#64748b;">{s["status"]} · {s["cont"]}</div></div>', unsafe_allow_html=True)
with c3:
    cls2 = "safe-card" if s["trap"]=="SAFE" else "risk-card" if s["trap"]=="RISK" else "trap-card"
    tc2 = "#00ff88" if s["trap"]=="SAFE" else "#fbbf24" if s["trap"]=="RISK" else "#ff4060"
    st.markdown(f'<div class="{cls2}"><div style="font-size:.8rem;color:#64748b;">Trap Filter</div><div style="font-size:1.6rem;font-weight:900;color:{tc2}">{s["trap"]}</div><div style="font-size:.8rem;color:#64748b;">{" | ".join(s["trapR"]) if s["trapR"] else "دخول أنظف"}</div></div>', unsafe_allow_html=True)
with c4:
    st.markdown(f'<div class="card"><div style="font-size:.8rem;color:#64748b;">القوة</div><div style="font-size:1.1rem;font-weight:700;color:#00ff88;">CALL {s["cs"]}%</div><div style="font-size:1.1rem;font-weight:700;color:#ff4060;">PUT {s["ps"]}%</div></div>', unsafe_allow_html=True)
with c5:
    st.markdown(f'<div class="card"><div style="font-size:.8rem;color:#64748b;">Zone / RSI / RVOL</div><div style="font-size:1.1rem;font-weight:700;">{s["zone"]}</div><div style="font-size:.8rem;color:#64748b;">RSI: {s["rsi"]:.1f} | RVOL: {s["rvol"]:.2f}</div></div>', unsafe_allow_html=True)

st.markdown("---")
col_chart, col_plan = st.columns([1.3, 0.7])
with col_chart:
    fig = make_chart(df, sym, s["tgt"])
    if fig: st.plotly_chart(fig, use_container_width=True)
with col_plan:
    st.markdown(f"### {sym} — ${s['price']:.2f} ({s['change']:+.2f}%)")
    st.markdown(f"**الفريم:** {tf_label} | **الاتجاه:** {trend} | **الأعلى:** {trendH}")
    st.markdown("---")
    t = s["tgt"]
    if s["sig"] != "انتظار":
        st.markdown("**خطة الصفقة:**")
        cc1, cc2 = st.columns(2)
        with cc1:
            st.metric("الدخول", f"${t['entry']:.2f}")
            st.metric("الهدف 1", f"${t['t1']:.2f}" if not math.isnan(t['t1']) else "—")
            st.metric("الهدف 3", f"${t['t3']:.2f}" if not math.isnan(t['t3']) else "—")
        with cc2:
            st.metric("الوقف", f"${t['stop']:.2f}" if not math.isnan(t['stop']) else "—")
            st.metric("الهدف 2", f"${t['t2']:.2f}" if not math.isnan(t['t2']) else "—")
            if not math.isnan(t['stop']) and t['stop'] != t['entry']:
                rr = abs(t['t1']-t['entry'])/abs(t['entry']-t['stop'])
                st.metric("R:R", f"1:{rr:.1f}")
    st.markdown("---")
    items = [
        ("الاتجاه",trend),("الأعلى",trendH),("الحالة",s["cont"]),
        ("VWAP","فوق" if s["aboveVWAP"] else "تحت"),
        ("EMA200","فوق" if s["aboveEMA"] else "تحت"),
        ("Nadaraya",s["zone"]),("Bollinger",s["bbPos"]),
    ]
    for k, v in items:
        st.write(f"- **{k}:** {v}")
    st.markdown("---")
    chks = s["CC"] if s["sig"]=="كول" else s["CP"]
    passed = [k for k,v in chks.items() if v]
    failed = [k for k,v in chks.items() if not v]
    st.markdown(f"**الشروط المتحققة ({len(passed)}/11):**")
    for c_ in passed: st.write(f"✅ {c_}")
    if failed:
        st.markdown("**غير متحققة:**")
        for c_ in failed: st.write(f"⚪ {c_}")

st.markdown("---")
st.caption("ALHOWIFI SMART TRADING © 2025 · للأغراض التحليلية فقط")
