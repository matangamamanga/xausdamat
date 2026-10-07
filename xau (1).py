"""Bot sinyal XAUUSD: SMC (liquidity sweep + MSS + FVG/OB) M15 + bias H1 + filter berita + backtest.

Mode:
  python xau.py signal    -> dipanggil cron (hanya kirim kalau ada sinyal)
  python xau.py analisa   -> kirim analisa sekarang walau tidak ada sinyal
  python xau.py backtest  -> backtest ~60 hari terakhir, hasil dikirim ke Telegram
"""
import os
import sys
import datetime as dt

import numpy as np
import pandas as pd
import requests

# ---------------- konfigurasi ----------------
TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
SYMBOL = os.environ.get("SYMBOL", "GC=F")  # futures emas; beda beberapa $ dari spot XAUUSD
SPREAD = float(os.environ.get("SPREAD", "0.5"))  # biaya spread+slippage dalam $, dipakai di backtest

# strategi SMC: liquidity sweep -> MSS/CHoCH (konfirmasi) -> retest FVG/OB
PIV = 5            # swing high/low = ekstrem dalam +-5 candle (level likuiditas)
LOOKBACK = 120     # level likuiditas berlaku maks 120 candle (30 jam)
RECLAIM = 3        # setelah sweep, close harus kembali melewati level dalam 3 candle
MAX_MSS = 20       # MSS harus terjadi maks 20 candle setelah sweep
MAX_RETEST = 24    # retest zona + konfirmasi maks 24 candle setelah MSS
BUF = 0.1          # close harus tembus level MSS + 0.1*ATR
DISP = 0.8         # candle displacement: body >= 0.8*ATR
FVG_MIN = 0.15     # ukuran FVG minimal 0.15*ATR
SL_BUF = 0.3       # SL di balik ekstrem sweep + 0.3*ATR
MIN_RISK_ATR = 0.5
MAX_RISK_ATR = 4.0
TP1_R = 1.5
TP2_R = 2.5
BIAS_FILTER = os.environ.get("BIAS_FILTER", "1") == "1"  # hanya trade searah tren H1
SESSION = (7, 20)  # jam UTC: London + New York
MAX_BARS = 96      # backtest: posisi ditutup paksa setelah 96 candle (24 jam)

# filter berita (kalender ForexFactory, gratis, tanpa key)
NEWS_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
NEWS_BEFORE = 30   # menit sebelum rilis
NEWS_AFTER = 60    # menit sesudah rilis

STATE_FILE = "state/last.txt"
WIB = "Asia/Jakarta"


# ---------------- util ----------------
def send(text):
    print(text)
    if not TOKEN or not CHAT_ID:
        return
    r = requests.post(
        f"https://api.telegram.org/bot{TOKEN}/sendMessage",
        data={"chat_id": CHAT_ID, "text": text[:4000]},
        timeout=30,
    )
    r.raise_for_status()


def send_file(path, caption=""):
    if not TOKEN or not CHAT_ID:
        return
    with open(path, "rb") as f:
        requests.post(
            f"https://api.telegram.org/bot{TOKEN}/sendDocument",
            data={"chat_id": CHAT_ID, "caption": caption},
            files={"document": f},
            timeout=60,
        )


def ai_comment(ctx):
    if not GEMINI_KEY:
        return ""
    prompt = (
        "Kamu analis teknikal XAUUSD. Berdasarkan data berikut, tulis komentar singkat "
        "(maks 4 kalimat, Bahasa Indonesia): kondisi pasar dan risiko utama. "
        "Jangan menjanjikan profit.\n\n" + ctx
    )
    try:
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{GEMINI_MODEL}:generateContent?key={GEMINI_KEY}"
        )
        r = requests.post(url, json={"contents": [{"parts": [{"text": prompt}]}]}, timeout=40)
        return r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception:
        return ""


def get_data(interval, period):
    import yfinance as yf  # import di sini supaya bagian lain bisa dites tanpa jaringan

    df = yf.Ticker(SYMBOL).history(interval=interval, period=period).dropna()
    return df.iloc[:-1]  # buang candle yang belum close


# ---------------- indikator ----------------
def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def atr(df, n=14):
    pc = df["Close"].shift()
    tr = pd.concat(
        [df["High"] - df["Low"], (df["High"] - pc).abs(), (df["Low"] - pc).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def _utc(df):
    df = df.copy()
    idx = df.index
    df.index = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    return df


def prepare(m15, h1):
    """Tambahkan ATR, tren H1 (tanpa lookahead), dan pivot ke data M15."""
    m15, h1 = _utc(m15), _utc(h1)
    m15["atr"] = atr(m15)

    e50, e200 = ema(h1["Close"], 50), ema(h1["Close"], 200)
    trend = pd.Series(0, index=h1.index)
    trend[(e50 > e200) & (h1["Close"] > e50)] = 1
    trend[(e50 < e200) & (h1["Close"] < e50)] = -1
    trend.iloc[:200] = 0  # warmup EMA200
    # candle H1 baru boleh dipakai setelah closed (start + 1 jam)
    t = pd.DataFrame({"trend": trend.values}, index=h1.index + pd.Timedelta(hours=1))
    m15 = pd.merge_asof(m15, t, left_index=True, right_index=True, direction="backward")
    m15["trend"] = m15["trend"].fillna(0).astype(int)

    w = 2 * PIV + 1
    m15["piv_hi"] = m15["High"].rolling(w, center=True).max() == m15["High"]
    m15["piv_lo"] = m15["Low"].rolling(w, center=True).min() == m15["Low"]
    return m15


# ---------------- strategi: Liquidity sweep + MSS + FVG/OB ----------------
PENDING = []  # setup yang masih berjalan (ditampilkan di mode analisa)


def _arrays(df, d):
    """d=+1: setup BUY. d=-1: harga dicerminkan (dikali -1) supaya logika BUY dipakai ulang untuk SELL."""
    O, H, L, C = (df[c].values.astype(float) for c in ("Open", "High", "Low", "Close"))
    if d == 1:
        return O, H, L, C, df["piv_lo"].values, df["piv_hi"].values
    return -O, -L, -H, -C, df["piv_hi"].values, df["piv_lo"].values


def scan_side(df, d, session=True):
    """Urutan wajib (versi BUY; SELL = cermin):
    1) sweep   : low menembus swing low (sell-side liquidity)
    2) reclaim : close kembali di atas level yang disapu
    3) MSS     : close menembus swing high struktur terakhir, ada displacement + FVG
    4) retest  : harga kembali ke zona OB/FVG, lalu candle konfirmasi (bullish, close di atas
                 tengah zona) -> baru sinyal dikirim
    Tiap kondisi hanya memakai data sampai candle i (tanpa lookahead)."""
    O, H, L, C, liq_f, st_f = _arrays(df, d)
    A = df["atr"].values
    T = df["trend"].values * d
    hours = df.index.hour
    n = len(df)
    side = "BUY" if d == 1 else "SELL"

    liq, st, setups, sigs = [], [], [], []
    for i in range(30, n):
        j = i - PIV  # pivot baru terkonfirmasi setelah PIV candle
        if liq_f[j]:
            liq.append((L[j], j))
        if st_f[j]:
            st.append((H[j], j))
        liq = [x for x in liq if i - x[1] <= LOOKBACK]
        st = [x for x in st if i - x[1] <= LOOKBACK]
        a = A[i]
        if np.isnan(a):
            continue

        # 1) liquidity sweep
        swept = [x for x in liq if L[i] < x[0]]
        if swept:
            liq = [x for x in liq if L[i] >= x[0]]
            cands = [x for x in st if x[0] > C[i] + 0.3 * a]  # swing high yang harus ditembus (MSS)
            if cands and (T[i] == 1 or not BIAS_FILTER):
                lv = max(x[0] for x in swept)
                setups.append(
                    {
                        "stage": 0,
                        "lv": lv,
                        "i0": i,
                        "ext": L[i],
                        "ext_i": i,
                        "mss": cands[-1][0],
                        "reclaimed": bool(C[i] > lv),
                    }
                )
                setups = setups[-3:]

        for s in setups[:]:
            if s["i0"] == i:
                continue

            if s["stage"] == 0:
                # 2) reclaim + 3) MSS dengan displacement dan FVG
                if L[i] < s["ext"]:
                    s["ext"], s["ext_i"] = L[i], i
                if C[i] > s["lv"]:
                    s["reclaimed"] = True
                if i - s["i0"] > MAX_MSS or (not s["reclaimed"] and i - s["i0"] >= RECLAIM):
                    setups.remove(s)
                    continue
                if not (s["reclaimed"] and C[i] > s["mss"] + BUF * a):
                    continue
                e = s["ext_i"]
                if e >= i:
                    continue
                fvgs = [
                    (H[k - 2], L[k])
                    for k in range(e + 1, i + 1)
                    if L[k] - H[k - 2] >= FVG_MIN * A[k]
                ]
                body, dk = max((C[k] - O[k], k) for k in range(e + 1, i + 1))
                if not fvgs or body < DISP * a:
                    setups.remove(s)  # MSS tanpa displacement/FVG = lemah, abaikan
                    continue
                ob = None  # order block = candle bearish terakhir sebelum candle displacement
                for k in range(dk - 1, max(e - 2, 0), -1):
                    if C[k] < O[k]:
                        ob = (L[k], H[k])
                        break
                zones = [("FVG", lo, hi) for lo, hi in fvgs[-2:]]
                if ob:
                    zones.insert(0, ("OB", ob[0], ob[1]))
                s.update(stage=1, mss_i=i, zones=zones)

            else:
                # 4) retest zona + candle konfirmasi
                if i - s["mss_i"] > MAX_RETEST or C[i] < s["ext"]:
                    setups.remove(s)
                    continue
                if C[i] < min(z[1] for z in s["zones"]) - 0.3 * a:
                    setups.remove(s)  # zona ditembus ke bawah = setup gagal
                    continue
                hit = next(
                    (
                        (nm, zl, zh)
                        for nm, zl, zh in s["zones"]
                        if L[i] <= zh and C[i] > zl and C[i] > O[i] and C[i] >= (zl + zh) / 2
                    ),
                    None,
                )
                if hit is None:
                    continue
                setups.remove(s)
                entry, sl = C[i], s["ext"] - SL_BUF * a
                risk = entry - sl
                if not (MIN_RISK_ATR * a <= risk <= MAX_RISK_ATR * a):
                    continue
                if session and not (SESSION[0] <= hours[i] < SESSION[1]):
                    continue
                # target likuiditas: swing high terdekat di atas entry yang belum disapu
                tg = [
                    x[0]
                    for x in st
                    if x[0] > entry and H[x[1] + 1 : i + 1].max(initial=-np.inf) < x[0]
                ]
                overlap = any(
                    not (fz[2] < oz[1] or fz[1] > oz[2])
                    for oz in s["zones"]
                    if oz[0] == "OB"
                    for fz in s["zones"]
                    if fz[0] == "FVG"
                )
                zlo, zhi = sorted((d * hit[1], d * hit[2]))
                sigs.append(
                    {
                        "i": i,
                        "time": df.index[i],
                        "side": side,
                        "level": d * s["lv"],
                        "mss": d * s["mss"],
                        "zone": "OB+FVG" if overlap else hit[0],
                        "zlo": zlo,
                        "zhi": zhi,
                        "entry": d * entry,
                        "sl": d * sl,
                        "tp1": d * (entry + TP1_R * risk),
                        "tp2": d * (entry + TP2_R * risk),
                        "target": d * min(tg) if tg else None,
                        "risk": risk,
                        "atr": a,
                    }
                )

    for s in setups:
        if s["stage"] == 0:
            arah = "di atas" if d == 1 else "di bawah"
            PENDING.append(
                f"{side}: sweep {d * s['lv']:.2f}, menunggu reclaim + MSS (close {arah} {d * s['mss']:.2f})"
            )
        else:
            PENDING.append(f"{side}: MSS terkonfirmasi, menunggu retest zona + candle konfirmasi")
    return sigs


def scan(df, session=True):
    PENDING.clear()
    sigs = scan_side(df, 1, session) + scan_side(df, -1, session)
    sigs.sort(key=lambda s: s["i"])
    out = []
    for s in sigs:
        if not out or out[-1]["i"] != s["i"]:
            out.append(s)
    return out


# ---------------- backtest ----------------
def simulate(df, sigs):
    """Satu posisi per waktu. 50% ditutup di TP1, SL digeser ke entry, sisanya ke TP2.
    Kalau SL dan TP kena di candle yang sama, dianggap SL (konservatif)."""
    H, L, C = df["High"].values, df["Low"].values, df["Close"].values
    n = len(df)
    part = 0.5 * TP1_R
    trades, busy = [], -1
    for s in sigs:
        i = s["i"]
        if i <= busy:
            continue
        buy = s["side"] == "BUY"
        e, sl, risk = s["entry"], s["sl"], s["risk"]
        stage, r, why = 0, None, None
        end = min(i + MAX_BARS, n - 1)
        k = i
        for k in range(i + 1, end + 1):
            stop = sl if stage == 0 else e
            hit_stop = (L[k] <= stop) if buy else (H[k] >= stop)
            if stage == 0:
                if hit_stop:
                    r, why = -1.0, "SL"
                    break
                if (H[k] >= s["tp1"]) if buy else (L[k] <= s["tp1"]):
                    stage = 1
            else:
                if hit_stop:
                    r, why = part, "BE"
                    break
                if (H[k] >= s["tp2"]) if buy else (L[k] <= s["tp2"]):
                    r, why = part + 0.5 * TP2_R, "TP2"
                    break
        if r is None:
            k = end
            m = ((C[end] - e) if buy else (e - C[end])) / risk
            r = m if stage == 0 else part + 0.5 * m
            why = "TIMEOUT"
        r -= SPREAD / risk
        busy = k
        trades.append(
            {
                "time_wib": s["time"].tz_convert(WIB).strftime("%Y-%m-%d %H:%M"),
                "side": s["side"],
                "entry": round(e, 2),
                "sl": round(sl, 2),
                "risk_usd": round(risk, 2),
                "exit": why,
                "bars": k - i,
                "R": round(r, 2),
            }
        )
    return trades


def backtest_report(trades, days):
    if not trades:
        return "BACKTEST: tidak ada trade pada periode ini."
    r = np.array([t["R"] for t in trades])
    wins, losses = r[r > 0], r[r <= 0]
    pf = wins.sum() / abs(losses.sum()) if losses.sum() < 0 else float("inf")
    eq = np.r_[0.0, np.cumsum(r)]
    dd = (np.maximum.accumulate(eq) - eq).max()
    streak = cur = 0
    for x in r:
        cur = cur + 1 if x <= 0 else 0
        streak = max(streak, cur)

    def side_line(side):
        rr = np.array([t["R"] for t in trades if t["side"] == side])
        if len(rr) == 0:
            return f"{side}: 0 trade"
        return f"{side}: {len(rr)} trade, winrate {100 * (rr > 0).mean():.0f}%, total {rr.sum():+.1f}R"

    txt = (
        f"BACKTEST XAUUSD SMC sweep+MSS+FVG/OB M15 (~{days} hari, data {SYMBOL})\n"
        f"Trade: {len(r)} | Winrate: {100 * (r > 0).mean():.0f}% | Rata-rata: {r.mean():+.2f}R\n"
        f"Total: {r.sum():+.1f}R | Profit factor: {pf:.2f}\n"
        f"Max drawdown: {dd:.1f}R | Rugi beruntun terpanjang: {streak}\n"
        f"{side_line('BUY')}\n{side_line('SELL')}\n"
        f"Biaya spread/slippage dimasukkan: ${SPREAD}\n\n"
    )
    if len(r) < 30:
        txt += "PERINGATAN: sampel < 30 trade, hasil belum bisa dipercaya.\n"
    if r.mean() <= 0:
        txt += "Expectancy <= 0: strategi ini TIDAK menguntungkan di data ini. Jangan dipakai uang asli.\n"
    txt += (
        "Catatan: filter berita TIDAK ikut dibacktest (tidak ada data kalender historis gratis). "
        "Yahoo hanya menyediakan 15m ~60 hari, jadi sampel kecil. Hasil lalu bukan jaminan."
    )
    return txt


def run_backtest():
    m15, h1 = get_data("15m", "60d"), get_data("1h", "60d")
    df = prepare(m15, h1)
    trades = simulate(df, scan(df))
    days = max(1, (df.index[-1] - df.index[0]).days)
    send(backtest_report(trades, days))
    if trades:
        pd.DataFrame(trades).to_csv("trades.csv", index=False)
        send_file("trades.csv", "Daftar trade backtest")


# ---------------- filter berita ----------------
def fetch_events():
    try:
        r = requests.get(NEWS_URL, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        out = []
        for e in r.json():
            if e.get("country") == "USD" and e.get("impact") == "High":
                out.append({"title": e.get("title", ""), "time": pd.to_datetime(e["date"], utc=True)})
        return out
    except Exception as ex:
        print("kalender berita gagal:", ex)
        return None


def news_block(ts, events):
    for e in events or []:
        if e["time"] - pd.Timedelta(minutes=NEWS_BEFORE) <= ts <= e["time"] + pd.Timedelta(minutes=NEWS_AFTER):
            return e
    return None


def upcoming(ts, events, hours=24):
    lim = ts + pd.Timedelta(hours=hours)
    return [e for e in (events or []) if ts <= e["time"] <= lim]


def fmt_ev(e):
    return f"{e['time'].tz_convert(WIB).strftime('%d %b %H:%M')} WIB - {e['title']}"


# ---------------- sinyal live ----------------
def already_sent(sid):
    try:
        return open(STATE_FILE).read().strip() == sid
    except OSError:
        return False


def mark_sent(sid):
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    with open(STATE_FILE, "w") as f:
        f.write(sid)


def run_signal(force):
    now = pd.Timestamp.now(tz="UTC")
    if not force and not (SESSION[0] <= now.hour < SESSION[1] and now.weekday() < 5):
        print("Di luar sesi, skip.")
        return

    df = prepare(get_data("15m", "10d"), get_data("1h", "60d"))
    sigs = scan(df)
    last_i = len(df) - 1
    bar_close = df.index[-1] + pd.Timedelta(minutes=15)
    fresh = (now - bar_close) <= pd.Timedelta(minutes=45)
    live = [s for s in sigs if s["i"] == last_i and fresh]

    events = fetch_events()
    last = df.iloc[-1]
    trend = {1: "NAIK", -1: "TURUN", 0: "SIDEWAYS"}[int(last["trend"])]
    nxt = upcoming(now, events)
    news_txt = "\n".join("- " + fmt_ev(e) for e in nxt) if nxt else "- tidak ada berita USD high-impact 24 jam ke depan"
    if events is None:
        news_txt = "- kalender berita GAGAL dimuat, cek manual sebelum entry"
    ctx = (
        f"Harga {last['Close']:.2f}, tren H1 {trend}, ATR M15 {last['atr']:.2f}. "
        f"Berita 24 jam: {news_txt}"
    )

    if live:
        s = live[-1]
        sid = f"{s['time'].isoformat()}-{s['side']}"
        if already_sent(sid):
            print("Sinyal sudah dikirim.")
            return
        mark_sent(sid)
        ev = news_block(bar_close, events)
        if ev:
            send(
                f"SINYAL {s['side']} DITAHAN (berita)\n"
                f"Dekat rilis: {fmt_ev(ev)}\nEntry kalau jadi: {s['entry']:.2f}. Lewati saja."
            )
            return
        tg = f"Target likuiditas: {s['target']:.2f}\n" if s.get("target") else ""
        text = (
            f"SINYAL {s['side']} XAUUSD (SMC M15, sudah terkonfirmasi)\n"
            f"1) Liquidity sweep di {s['level']:.2f}\n"
            f"2) MSS/CHoCH: close tembus {s['mss']:.2f}\n"
            f"3) Retest {s['zone']} {s['zlo']:.2f}-{s['zhi']:.2f} + candle konfirmasi\n\n"
            f"Entry: {s['entry']:.2f}\nSL: {s['sl']:.2f}\n"
            f"TP1: {s['tp1']:.2f} (tutup 50%, SL ke entry)\nTP2: {s['tp2']:.2f}\n"
            f"{tg}Jarak SL: ${s['risk']:.2f} | Tren H1: {trend}\n"
            f"Harga data futures; cek selisih dengan spot di MT5 sebelum entry.\n\n"
            f"Berita:\n{news_txt}\n"
        )
        c = ai_comment(ctx + f" Sinyal {s['side']} SMC: sweep {s['level']:.2f}, MSS {s['mss']:.2f}, retest {s['zone']}.")
        if c:
            text += "\nAI: " + c + "\n"
        text += "\nRisiko maks 1% per trade. Bukan nasihat keuangan. Tes di demo dulu."
        send(text)
    elif force:
        pend = "\n".join("- " + p for p in PENDING) or "- belum ada sweep likuiditas aktif"
        text = f"ANALISA XAUUSD\n{ctx}\nSinyal: TIDAK ADA.\nSetup berjalan:\n{pend}"
        c = ai_comment(ctx + " Tidak ada sinyal.")
        if c:
            text += "\n\nAI: " + c
        send(text)
    else:
        print("Tidak ada sinyal.", ctx)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "signal"
    if mode == "backtest":
        run_backtest()
    else:
        run_signal(force=(mode == "analisa"))
