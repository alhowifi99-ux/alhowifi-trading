import math, warnings, time, threading
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
from streamlit_autorefresh import st_autorefresh

st.set_page_config(page_title="ALHOWIFI FLOW PRO", layout="wide",
                   initial_sidebar_state="collapsed")

# ── AUTO REFRESH 30s ──────────────────────────────────
count = st_autorefresh(interval=30000, key="refresh")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@400;700;900&family=IBM+Plex+Mono:wght@400;700&display=swap');
* { font-family: 'Tajawal', sans-serif; }
.stApp { background: #010509; color: #f1f5f9; }
.block-container { padding: 0.5rem 1rem !important; max-width: 100% !important; }
[data-testid="stSidebar"] { display: none; }

.logo {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 1.8rem; font-weight: 900;
  color: #00f5ff;
  text-shadow: 0 0 10px #00d4ff, 0 0 20px #00d4ff;
  letter-spacing: 4px;
}
.sub-logo {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.6rem; color: #00ff88;
  letter-spacing: 6px;
  text-shadow: 0 0 6px #00ff88;
}

.alert-call {
  background: linear-gradient(135deg, rgba(0,255,136,0.1), rgba(0,212,255,0.05));
  border: 2px solid rgba(0,255,136,0.5);
  border-radius: 14px; padding: 16px 20px; margin: 8px 0;
  box-shadow: 0 0 40px rgba(0,255,136,0.2);
  direction: rtl;
}
.alert-put {
  background: linear-gradient(135deg, rgba(255,64,96,0.1), rgba(255,140,66,0.05));
  border: 2px solid rgba(255,64,96,0.5);
  border-radius: 14px; padding: 16px 20px; margin: 8px 0;
  box-shadow: 0 0 40px rgba(255,64,96,0.2);
  direction: rtl;
}
.alert-sym { font-size: 1.3rem; font-weight: 900; font-family: 'IBM Plex Mono', monospace; }
.alert-type-call { color: #00ff88; font-size: 1.1rem; font-weight: 900; }
.alert-type-put  { color: #ff4060; font-size: 1.1rem; font-weight: 900; }
.alert-detail { color: rgba(255,255,255,0.6); font-size: 0.85rem; margin-top: 4px; }
.alert-badge { background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1);
  border-radius: 20px; padding: 2px 10px; font-size: 0.75rem; color: #64748b; }

.sector-header {
  padding: 5px 12px; border-radius: 8px; margin: 8px 0 4px 0;
  font-weight: 800; font-size: 0.9rem;
  direction: rtl;
}

.stock-call {
  background: rgba(0,255,136,0.07);
  border: 1px solid rgba(0,255,136,0.35);
  border-top: 2px solid #00ff88;
  border-radius: 10px; padding: 10px 12px;
  box-shadow: 0 0 16px rgba(0,255,136,0.15);
  direction: rtl; margin: 3px;
}
.stock-put {
  background: rgba(255,64,96,0.07);
  border: 1px solid rgba(255,64,96,0.35);
  border-top: 2px solid #ff4060;
  border-radius: 10px; padding: 10px 12px;
  box-shadow: 0 0 16px rgba(255,64,96,0.15);
  direction: rtl; margin: 3px;
}
.stock-wait {
  background: rgba(3,7,18,0.9);
  border: 1px solid rgba(14,24,44,0.85);
  border-top: 2px solid #0f172a;
  border-radius: 10px; padding: 10px 12px;
  direction: rtl; margin: 3px;
}
.stock-sym { font-family: 'IBM Plex Mono', monospace; font-weight: 900; font-size: 1rem; color: #f1f5f9; }
.stock-price { font-family: 'IBM Plex Mono', monospace; font-weight: 800; font-size: 0.9rem; color: #f1f5f9; }
.stock-change-up   { color: #00ff88; font-size: 0.75rem; font-family: 'IBM Plex Mono', monospace; }
.stock-change-down { color: #ff4060; font-size: 0.75rem; font-family: 'IBM Plex Mono', monospace; }
.badge-conf { background: rgba(0,255,136,0.1); color: #00ff88; border: 1px solid rgba(0,255,136,0.3);
  border-radius: 20px; padding: 1px 8px; font-size: 0.7rem; font-weight: 700; }
.badge-armed { background: rgba(251,191,36,0.1); color: #fbbf24; border: 1px solid rgba(251,191,36,0.3);
  border-radius: 20px; padding: 1px 8px; font-size: 0.7rem; font-weight: 700; }
.badge-wait { background: rgba(71,85,105,0.1); color: #475569; border: 1px solid rgba(71,85,105,0.2);
  border-radius: 20px; padding: 1px 8px; font-size: 0.7rem; }

.dot-call { display: inline-block; width:12px; height:12px; border-radius:50%;
  background:#00ff88; box-shadow: 0 0 8px #00ff88; }
.dot-put  { display: inline-block; width:12px; height:12px; border-radius:50%;
  background:#ff4060; box-shadow: 0 0 8px #ff4060; }
.dot-wait { display: inline-block; width:10px; height:10px; border-radius:50%;
  background:#334155; }

.score-ring {
  display: inline-flex; align-items: center; justify-content: center;
  width: 44px; height: 44px; border-radius: 50%;
  font-family: 'IBM Plex Mono', monospace;
  font-weight: 700; font-size: 0.85rem;
}

.topbar {
  background: rgba(1,5,9,0.98);
  border-bottom: 1px solid rgba(10,20,40,0.9);
  padding: 8px 16px;
  display: flex; align-items: center; justify-content: space-between;
  direction: rtl;
}
.stat-box {
  background: rgba(255,255,255,0.02);
  border: 1px solid rgba(255,255,255,0.06);
  border-radius: 8px; padding: 4px 10px; text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
# DATA & SECTORS
# ══════════════════════════════════════════════════════
SECTORS = {
    "🖥️ تقنية":     {"color":"#00d4ff","stocks":["AAPL","MSFT","GOOGL","ORCL","XLK"]},
    "⚡ AI/رقائق":  {"color":"#a78bfa","stocks":["NVDA","AMD","AVGO","MU","LRCX"]},
    "🔒 سايبر/نمو": {"color":"#f97316","stocks":["META","NFLX","CRWD","PANW"]},
    "🛒 استهلاكي":  {"color":"#fbbf24","stocks":["TSLA","AMZN","COST"]},
    "💰 مالي":      {"color":"#34d399","stocks":["JPM","GS","XLF"]},
    "⚕️ صحة/طاقة": {"color":"#f472b6","stocks":["LLY","XOM","XLE","XLV"]},
    "🏭 صناعي":     {"color":"#94a3b8","stocks":["GE","CAT","GLD"]},
    "📊 مؤشرات":   {"color":"#ffffff","stocks":["SPY","QQQ"]},
}
ALL = [s for v in SECTORS.values() for s in v["stocks"]]

TFS = {
    "لحظي 1m":    {"interval":"1m",  "period":"1d"},
    "مضاربة 5m":  {"interval":"5m",  "period":"5d"},
    "مضاربة 15m": {"interval":"15m", "period":"60d"},
    "ساعة 1H":    {"interval":"60m", "period":"180d"},
    "يومي":       {"interval":"1d",  "period":"1y"},
    "أسبوعي":     {"interval":"1wk", "period":"5y"},
    "شهري":       {"interval":"1mo", "period":"10y"},
}
AMULT = {"لحظي 1m":0.8,"مضاربة 5m":1.0,"مضاربة 15m":1.2,"ساعة 1H":1.5,"يومي":2.0,"أسبوعي":2.8,"شهري":3.8}

# ══════════════════════════════════════════════════════
# MATH
# ══════════════════════════════════════════════════════
def flat(x): return np.array(x).flatten().astype(float)

def ema(a,n):
    a=flat(a); k=2/(n+1); r=np.zeros(len(a)); r[0]=a[0]
    for i in range(1,len(a)): r[i]=a[i]*k+r[i-1]*(1-k)
    return r

def wma(a,n):
    a=flat(a); r=np.full(len(a),np.nan); w=np.arange(1,n+1,dtype=float); ws=w.sum()
    for i in range(n-1,len(a)): r[i]=np.dot(a[i-n+1:i+1],w)/ws
    return r

def hma(a,n):
    a=flat(a); h=max(int(n/2),1); sq=max(int(math.sqrt(n)),1)
    return wma(2*wma(a,h)-wma(a,n),sq)

def rsi(a):
    a=flat(a); r=np.full(len(a),50.0)
    if len(a)<15: return r
    g=np.zeros(len(a)); l=np.zeros(len(a))
    for i in range(1,len(a)):
        d=a[i]-a[i-1]
        if d>0: g[i]=d
        else: l[i]=-d
    ag=np.mean(g[1:15]); al=np.mean(l[1:15])
    r[14]=100-100/(1+ag/(al or 1e-9))
    for i in range(15,len(a)):
        ag=(ag*13+g[i])/14; al=(al*13+l[i])/14
        r[i]=100-100/(1+ag/(al or 1e-9))
    return r

def macd(a):
    a=flat(a); el=ema(a,5); es=ema(a,13); ln=el-es; sg=ema(ln,6)
    return ln,sg,ln-sg

def atr(h,l,c):
    h=flat(h); l=flat(l); c=flat(c)
    tr=np.zeros(len(c)); tr[0]=h[0]-l[0]
    for i in range(1,len(c)): tr[i]=max(h[i]-l[i],abs(h[i]-c[i-1]),abs(l[i]-c[i-1]))
    return ema(tr,14)

def vwap(h,l,c,v):
    h=flat(h); l=flat(l); c=flat(c); v=flat(v)
    tp=(h+l+c)/3; cv=np.cumsum(tp*v); cv2=np.cumsum(v)
    return np.where(cv2>0,cv/cv2,tp)

def bb(c,n=20,m=2):
    c=flat(c); md=np.array([np.mean(c[max(0,i-n+1):i+1]) if i>=n-1 else np.nan for i in range(len(c))])
    up=np.array([md[i]+m*np.std(c[max(0,i-n+1):i+1]) if i>=n-1 else np.nan for i in range(len(c))])
    dn=np.array([md[i]-m*np.std(c[max(0,i-n+1):i+1]) if i>=n-1 else np.nan for i in range(len(c))])
    return md,up,dn

def nw(c,w=50,b=3):
    c=flat(c); n=len(c); r=np.full(n,np.nan); w=min(w,n)
    if w<5: return r
    pos=np.arange(w,dtype=float); wt=np.exp(-0.5*((pos-(w-1))/b)**2); wt/=wt.sum()
    for i in range(w-1,n):
        seg=c[i-w+1:i+1].flatten()
        if len(seg)==w: r[i]=float(np.dot(seg,wt))
    return r

def rvol(v,n=20):
    v=flat(v); avg=np.array([np.mean(v[max(0,i-n+1):i+1]) for i in range(len(v))])
    return np.where(avg>0,v/avg,1.0)

# ══════════════════════════════════════════════════════
# DATA FETCH
# ══════════════════════════════════════════════════════
@st.cache_data(ttl=30)
def fetch(sym, interval, period):
    try:
        df=yf.download(sym,interval=interval,period=period,auto_adjust=True,progress=False)
        if df is None or df.empty: return None
        if isinstance(df.columns,pd.MultiIndex): df.columns=df.columns.get_level_values(0)
        df=df[["Open","High","Low","Close","Volume"]].dropna(subset=["Close","High","Low"])
        if len(df)<5: return None
        C=flat(df.Close); H=flat(df.High); L=flat(df.Low); V=flat(df.Volume)
        n=len(C)
        df["C"]=C; df["H"]=H; df["L"]=L; df["V"]=V
        df["EMA200"]=ema(C,min(200,n))
        df["RSI"]=rsi(C)
        df["MACD"],df["SIG"],df["HIST"]=macd(C)
        df["ATR"]=atr(H,L,C)
        df["RVOL"]=rvol(V)
        df["VWAP"]=vwap(H,L,C,V)
        df["HF"]=hma(C,min(18,max(3,n//3)))
        df["HS"]=hma(C,min(55,max(3,n//3)))
        nc=nw(C,min(50,max(5,n//2)),3)
        at=flat(df.ATR)
        df["NWC"]=nc; df["NWU"]=nc+at*2.5; df["NWL"]=nc-at*2.5
        _,df["BBU"],df["BBL"]=bb(C)
        return df
    except: return None

def htrend(df):
    if df is None: return "غير واضح"
    try:
        hf=flat(df.HF); hs=flat(df.HS); a=flat(df.ATR)[-1]
        f=hf[-1]; s=hs[-1]; n=len(hf)
        if abs(f-s)<(a or 0.01)*0.05: return "عرضي"
        fs=hf[-1]-hf[-4] if n>4 else 0; ss=hs[-1]-hs[-4] if n>4 else 0
        if f>s and fs>0 and ss>=0: return "صاعد"
        if f<s and fs<0 and ss<=0: return "هابط"
        return "عرضي"
    except: return "غير واضح"

# ══════════════════════════════════════════════════════
# SIGNAL ENGINE (12 شرط)
# ══════════════════════════════════════════════════════
def signal(df, tH, tf_label):
    if df is None: return None
    try:
        C=flat(df.C); H=flat(df.H); L=flat(df.L)
        cv=C[-1]; a=float(flat(df.ATR)[-1]) or cv*0.01
        ri=float(flat(df.RSI)[-1]); rv=float(flat(df.RVOL)[-1])
        ml=float(flat(df.MACD)[-1]); ms=float(flat(df.SIG)[-1])
        vw=float(flat(df.VWAP)[-1]); e2=float(flat(df.EMA200)[-1])
        nU=float(flat(df.NWU)[-1]); nL=float(flat(df.NWL)[-1])
        bU=float(flat(df.BBU)[-1] if not math.isnan(flat(df.BBU)[-1]) else cv*1.02)
        bL=float(flat(df.BBL)[-1] if not math.isnan(flat(df.BBL)[-1]) else cv*0.98)
        for v in [nU,nL,bU,bL]:
            if math.isnan(v): nU=cv*1.05;nL=cv*0.95;bU=cv*1.02;bL=cv*0.98;break
        sup=float(np.nanmin(L[-50:])); res=float(np.nanmax(H[-50:]))
        aa=float(np.nanmean(flat(df.ATR)[-20:])) or a
        n=len(C); pH=H[-2] if n>1 else cv; pL=L[-2] if n>1 else cv
        HL=n>=3 and L[-1]>L[-2]; LH=n>=3 and H[-1]<H[-2]
        tr=htrend(df)
        aV=cv>vw; aE=cv>e2; mU=ml>ms; mD=ml<ms
        rOK=rv>1.2; aOK=a>=aa*0.9
        tHu=tH in["صاعد","عرضي"]; tHd=tH in["هابط","عرضي"]
        CC={"فوق VWAP":aV,"فوق EMA200":aE,"HMA صاعد":tr=="صاعد","الفريم الأعلى":tHu,
            "RSI > 55":ri>55,"MACD+":mU,"Higher Low":HL or cv>pH,
            "RVOL > 1.2":rOK,"ATR نشط":aOK,"منطقة دخول":cv>pH or cv<=nL+a*0.35,"BB سليم":cv<=bU}
        CP={"تحت VWAP":not aV,"تحت EMA200":not aE,"HMA هابط":tr=="هابط","الفريم الأعلى":tHd,
            "RSI < 45":ri<45,"MACD-":mD,"Lower High":LH or cv<pL,
            "RVOL > 1.2":rOK,"ATR نشط":aOK,"منطقة دخول":cv<pL or cv>=nU-a*0.35,"BB سليم":cv>=bL}
        cs=int(sum(CC.values())/11*100); ps=int(sum(CP.values())/11*100)
        sig="انتظار"; st="WAIT"
        if cs>=73 and cs>ps:   sig="CALL"; st="CONFIRMED" if cs>=85 else "ARMED"
        elif ps>=73 and ps>cs: sig="PUT";  st="CONFIRMED" if ps>=85 else "ARMED"
        elif cs>=55 and cs>ps: sig="CALL"; st="ARMED"
        elif ps>=55 and ps>cs: sig="PUT";  st="ARMED"
        mW=abs(ml-ms)<max(0.02,a*0.02)
        trap="SAFE"; tR=[]
        if sig=="CALL":
            if ri>=75: trap="RISK"; tR.append("RSI مشبع")
            if not aV: trap="RISK" if trap!="TRAP" else trap; tR.append("تحت VWAP")
            if mW: trap="RISK" if trap!="TRAP" else trap; tR.append("MACD ضعيف")
        if sig=="PUT":
            if ri<=25: trap="RISK"; tR.append("RSI مشبع")
            if aV: trap="RISK" if trap!="TRAP" else trap; tR.append("فوق VWAP")
        if trap=="TRAP": sig="انتظار"; st="TRAP"
        mm=AMULT.get(tf_label,1)
        tgt={"entry":cv,"stop":float("nan"),"t1":float("nan"),"t2":float("nan"),"t3":float("nan")}
        if sig=="CALL":
            tgt={"entry":cv,"stop":min(cv-a*mm,vw-a*0.3),"t1":cv+a*mm,"t2":cv+a*mm*2,"t3":res if res>cv else cv+a*mm*3}
        elif sig=="PUT":
            tgt={"entry":cv,"stop":max(cv+a*mm,vw+a*0.3),"t1":cv-a*mm,"t2":cv-a*mm*2,"t3":sup if sup<cv else cv-a*mm*3}
        prev=C[-2] if n>1 else cv
        chg=(cv-prev)/prev*100 if prev else 0
        return dict(sig=sig,st=st,cs=cs,ps=ps,trap=trap,tR=tR,
                    trend=tr,tH=tH,price=cv,change=chg,atr=a,rsi=ri,
                    sup=sup,res=res,tgt=tgt)
    except: return None

# ══════════════════════════════════════════════════════
# COLOR HELPERS
# ══════════════════════════════════════════════════════
def score_color(s):
    if s>=80: return "#00ff88"
    if s>=60: return "#fbbf24"
    if s>=40: return "#a78bfa"
    return "#475569"

def score_ring(score, sig):
    c = "#00ff88" if sig=="CALL" else "#ff4060" if sig=="PUT" else score_color(score)
    return f'<span style="font-family:IBM Plex Mono,monospace;font-weight:700;font-size:1rem;color:{c}">{score}</span>'

# ══════════════════════════════════════════════════════
# STATE
# ══════════════════════════════════════════════════════
if "tf" not in st.session_state: st.session_state.tf = "مضاربة 5m"
if "alerts" not in st.session_state: st.session_state.alerts = []
if "prev_sigs" not in st.session_state: st.session_state.prev_sigs = {}
if "results" not in st.session_state: st.session_state.results = {}
if "last_scan" not in st.session_state: st.session_state.last_scan = 0

# ══════════════════════════════════════════════════════
# SCAN
# ══════════════════════════════════════════════════════
def run_scan():
    cfg = TFS[st.session_state.tf]
    res = {}
    for sym in ALL:
        try:
            df = fetch(sym, cfg["interval"], cfg["period"])
            s = signal(df, "غير واضح", st.session_state.tf)
            if s:
                s["sym"] = sym
                res[sym] = s
                # detect new confirmed signal
                prev = st.session_state.prev_sigs.get(sym, {})
                if (s["st"] in ["CONFIRMED","ARMED"] and s["sig"]!="انتظار"
                        and prev.get("st") != s["st"]):
                    st.session_state.alerts.insert(0, {**s, "tf":st.session_state.tf, "time":time.strftime("%H:%M:%S")})
                    st.session_state.alerts = st.session_state.alerts[:8]
        except: pass
    st.session_state.prev_sigs = {k:{"st":v.get("st"),"sig":v.get("sig")} for k,v in res.items()}
    st.session_state.results = res
    st.session_state.last_scan = time.time()

# Auto scan on load / refresh
if time.time() - st.session_state.last_scan > 25:
    run_scan()

results = st.session_state.results
alerts  = st.session_state.alerts

# ══════════════════════════════════════════════════════
# TOP BAR
# ══════════════════════════════════════════════════════
conf_n  = sum(1 for v in results.values() if v.get("st")=="CONFIRMED")
armed_n = sum(1 for v in results.values() if v.get("st")=="ARMED")
call_n  = sum(1 for v in results.values() if v.get("sig")=="CALL")
put_n   = sum(1 for v in results.values() if v.get("sig")=="PUT")
live_n  = sum(1 for v in results.values() if v.get("price",0)>0)

col_logo, col_tf, col_stats, col_btn = st.columns([2,3,3,1])
with col_logo:
    st.markdown('<div class="logo">ALHOWIFI</div><div class="sub-logo">SMART TRADING</div>', unsafe_allow_html=True)
with col_tf:
    tf_opts = list(TFS.keys())
    chosen = st.selectbox("", tf_opts, index=tf_opts.index(st.session_state.tf), label_visibility="collapsed", key="tf_sel")
    if chosen != st.session_state.tf:
        st.session_state.tf = chosen
        run_scan()
        st.rerun()
with col_stats:
    st.markdown(f"""
    <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;direction:rtl;padding-top:6px">
      <div class="stat-box"><span style="color:#00ff88;font-weight:700">{conf_n}</span><br><span style="font-size:.65rem;color:#475569">مؤكد</span></div>
      <div class="stat-box"><span style="color:#fbbf24;font-weight:700">{armed_n}</span><br><span style="font-size:.65rem;color:#475569">قريب</span></div>
      <div class="stat-box"><span style="color:#00ff88;font-weight:700">{call_n}</span><br><span style="font-size:.65rem;color:#475569">كول</span></div>
      <div class="stat-box"><span style="color:#ff4060;font-weight:700">{put_n}</span><br><span style="font-size:.65rem;color:#475569">بوت</span></div>
      <div class="stat-box"><span style="color:#00d4ff;font-weight:700">{live_n}</span><br><span style="font-size:.65rem;color:#475569">سهم</span></div>
      <span style="font-size:.7rem;color:#334155">{time.strftime("%H:%M:%S")}</span>
    </div>""", unsafe_allow_html=True)
with col_btn:
    if st.button("🔍 مسح", use_container_width=True):
        run_scan(); st.rerun()

st.markdown("<hr style='border-color:rgba(14,24,44,.8);margin:8px 0'>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
# ALERTS
# ══════════════════════════════════════════════════════
if alerts:
    st.markdown(f"### 🚨 إشارات مؤكدة ({len(alerts)})")
    for i, a in enumerate(alerts[:6]):
        is_call = a["sig"]=="CALL"
        col_main, col_close = st.columns([20,1])
        with col_main:
            dot = '<span class="dot-call"></span>' if is_call else '<span class="dot-put"></span>'
            typ = f'<span class="alert-type-call">▲ CALL</span>' if is_call else f'<span class="alert-type-put">▼ PUT</span>'
            badge = f'<span class="badge-conf">مؤكد</span>' if a["st"]=="CONFIRMED" else f'<span class="badge-armed">قريب</span>'
            cls = "alert-call" if is_call else "alert-put"
            stop_str = f"${a['tgt']['stop']:.2f}" if not math.isnan(a['tgt']['stop']) else "—"
            st.markdown(f"""
            <div class="{cls}">
              <div style="display:flex;align-items:center;justify-content:flex-end;gap:10px">
                {dot}
                <span style="color:#64748b;font-size:.85rem">{a.get('tf','5m')} · {a.get('time','')}</span>
                {badge} {typ}
                <span class="alert-sym" style="color:{'#00ff88' if is_call else '#ff4060'}">{a['sym']}</span>
              </div>
              <div class="alert-detail" style="text-align:right">
                دخول <strong>${a['tgt']['entry']:.2f}</strong> &nbsp;·&nbsp;
                وقف <strong>{stop_str}</strong> &nbsp;·&nbsp;
                قوة <strong style="color:{'#00ff88' if is_call else '#ff4060'}">{max(a['cs'],a['ps'])}%</strong>
              </div>
            </div>""", unsafe_allow_html=True)
        with col_close:
            if st.button("✕", key=f"cl_{i}"):
                st.session_state.alerts.pop(i); st.rerun()
    st.markdown("<hr style='border-color:rgba(14,24,44,.8);margin:8px 0'>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
# SECTORS & STOCK CARDS
# ══════════════════════════════════════════════════════
for sec_name, sec_info in SECTORS.items():
    stocks = sec_info["stocks"]
    sec_results = [results.get(s) for s in stocks]
    sec_conf  = sum(1 for r in sec_results if r and r.get("st")=="CONFIRMED")
    sec_call  = sum(1 for r in sec_results if r and r.get("sig")=="CALL")
    sec_put   = sum(1 for r in sec_results if r and r.get("sig")=="PUT")
    dir_str = ("▲ صاعد" if sec_call>sec_put else "▼ هابط" if sec_put>sec_call else "—")
    dir_col = "#00ff88" if sec_call>sec_put else "#ff4060" if sec_put>sec_call else "#475569"

    conf_badge = f'<span class="badge-conf">✓{sec_conf}</span>' if sec_conf>0 else ""
    st.markdown(f"""
    <div class="sector-header" style="background:rgba(255,255,255,0.02);border:1px solid {sec_info['color']}22">
      <span style="color:{sec_info['color']}">{sec_name}</span>
      &nbsp;&nbsp;
      {conf_badge}
      &nbsp;
      <span style="color:{dir_col};font-size:.8rem">{dir_str}</span>
    </div>""", unsafe_allow_html=True)

    cols = st.columns(len(stocks))
    for i, (sym, r) in enumerate(zip(stocks, sec_results)):
        with cols[i]:
            if r is None or r.get("price",0)<=0:
                st.markdown(f'<div class="stock-wait"><div class="stock-sym">{sym}</div><div style="color:#334155;font-size:.7rem;margin-top:4px">—</div></div>', unsafe_allow_html=True)
                continue
            sig  = r.get("sig","انتظار")
            st_  = r.get("st","WAIT")
            price= r.get("price",0)
            chg  = r.get("change",0)
            cs   = r.get("cs",0)
            ps   = r.get("ps",0)
            score= cs if sig=="CALL" else ps if sig=="PUT" else max(cs,ps)
            sc   = score_color(score)
            up   = chg>=0
            chg_str = f"+{chg:.2f}%" if up else f"{chg:.2f}%"
            chg_cls = "stock-change-up" if up else "stock-change-down"

            if sig=="CALL": card_cls="stock-call"
            elif sig=="PUT": card_cls="stock-put"
            else: card_cls="stock-wait"

            dot_html = '<span class="dot-call"></span>' if sig=="CALL" else '<span class="dot-put"></span>' if sig=="PUT" else '<span class="dot-wait"></span>'
            typ_html = '<span style="color:#00ff88;font-size:.75rem;font-weight:700">&#9650; CALL</span>' if sig=="CALL" else '<span style="color:#ff4060;font-size:.75rem;font-weight:700">&#9660; PUT</span>' if sig=="PUT" else ""
            badge_html = '<span class="badge-conf">&#10003; مؤكد</span>' if st_=="CONFIRMED" else '<span class="badge-armed">قريب</span>' if st_=="ARMED" else '<span class="badge-wait">انتظار</span>'
            trend_str = r.get("trend","—")

            card_html = f"""<div class="{card_cls}">
  <div style="display:flex;justify-content:space-between;align-items:flex-start">
    <div>
      <div class="stock-sym">{sym}</div>
      {typ_html}
      <div style="color:{sc};font-size:.7rem">{trend_str}</div>
    </div>
    <div style="text-align:center">{dot_html}<br><span style="font-family:'IBM Plex Mono',monospace;font-size:1rem;font-weight:700;color:{sc}">{score}</span></div>
  </div>
  <div style="display:flex;justify-content:space-between;align-items:center;margin-top:6px">
    <div>
      <div class="stock-price">${price:.2f}</div>
      <div class="{chg_cls}">{chg_str}</div>
    </div>
    {badge_html}
  </div>
</div>"""
            st.markdown(card_html, unsafe_allow_html=True)

st.markdown("<hr style='border-color:rgba(14,24,44,.8);margin:12px 0'>", unsafe_allow_html=True)
st.markdown('<div style="text-align:center;color:#0f172a;font-size:.7rem">ALHOWIFI SMART TRADING © 2025 · للأغراض التحليلية فقط</div>', unsafe_allow_html=True)
