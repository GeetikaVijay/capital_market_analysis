# Market Explorer (yfinance) — Start-End Dates only
# - Choose market category (US, India, Crypto, Forex, Indices, ETFs)
# - Select one or many tickers (or add custom tickers)
# - Pick exact Start–End date + interval (1d/1wk/1mo)
# - Visualize prices, indicators, returns, volume, correlation
# - Forecasting: LSTM (simple) + (S)ARIMA vs LSTM + optional GARCH volatility
# - Export data

import io
import math
import warnings
import datetime as dt
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st

# Optional (used for correlation heatmap & plots)
import matplotlib.pyplot as plt
import seaborn as sns

# Forecasting deps
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX

# TensorFlow
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# arch is optional for GARCH volatility
try:
    from arch import arch_model
    HAVE_ARCH = True
except Exception:
    HAVE_ARCH = False

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Market Explorer (yfinance)",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------- Helpers -------------

@st.cache_data(show_spinner=False)
def fetch_prices(
    tickers: List[str],
    start: str | dt.date | None = None,
    end: str | dt.date | None = None,
    period: str | None = None,           # kept for signature compatibility; not used in UI
    interval: str = "1d",
    auto_adjust: bool = True
) -> pd.DataFrame:
    """
    Fetch OHLCV for 1+ tickers using yfinance.
    - Ensures `end` is treated INCLUSIVE (yfinance end is exclusive).
    - Removes timezone from index to avoid plot/date mismatches.
    - Defensively re-slices to [start, end] inclusive after download.
    Returns a MultiIndex-column DataFrame: (Field, Ticker).
    """
    if not tickers:
        return pd.DataFrame()

    # Normalize incoming dates to pandas Timestamps
    start_ts = pd.Timestamp(start) if start else None
    end_ts   = pd.Timestamp(end) if end else None

    # yfinance's `end` is EXCLUSIVE; make it inclusive by +1 day for download
    yf_end = None
    if end_ts is not None:
        yf_end = (end_ts + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    data = yf.download(
        tickers=tickers,
        start=start_ts.strftime("%Y-%m-%d") if start_ts is not None else None,
        end=yf_end,                       # inclusive behavior
        period=None,                      # Period not used when start/end provided
        interval=interval,
        auto_adjust=auto_adjust,
        group_by="column",
        progress=False
    )

    # Normalize columns to MultiIndex (Field, Ticker)
    if not isinstance(data.columns, pd.MultiIndex) and not data.empty:
        fields = data.columns
        tkr = tickers[0]
        data.columns = pd.MultiIndex.from_product([fields, [tkr]])

    if data.empty:
        return data

    # ---- Fix timezone mismatches on index ----
    idx = data.index
    try:
        # If tz-aware (common for intraday), convert to UTC then drop tz
        if getattr(idx, "tz", None) is not None:
            data.index = idx.tz_convert("UTC").tz_localize(None)
    except Exception:
        # Fall back: just drop tz if conversion not possible
        try:
            data.index = idx.tz_localize(None)
        except Exception:
            pass

    # ---- Defensive inclusive slice after download ----
    if start_ts is not None or end_ts is not None:
        if interval in ("1d", "1wk", "1mo"):
            s = start_ts if start_ts is not None else data.index.min()
            e = end_ts   if end_ts   is not None else data.index.max()
            data = data.loc[s:e]
        else:
            # Intraday (not used in this UI): include entire end day
            s = start_ts if start_ts is not None else data.index.min()
            e_excl = (end_ts + pd.Timedelta(days=1)) if end_ts is not None else (data.index.max() + pd.Timedelta(seconds=1))
            data = data[(data.index >= s) & (data.index < e_excl)]

    data = data.dropna(how="all")
    return data


def list_to_clean_tickers(raw_list: List[str]) -> List[str]:
    toks = []
    for t in raw_list:
        t = (t or "").strip().upper()
        if t:
            toks.append(t)
    seen, out = set(), []
    for t in toks:
        if t not in seen:
            seen.add(t); out.append(t)
    return out


def parse_custom_tickers(text: str) -> List[str]:
    if not text:
        return []
    parts = [p.strip().upper() for p in text.replace("\n", ",").split(",")]
    return [p for p in parts if p]


def compute_indicators(close: pd.DataFrame, window_short=20, window_long=50) -> Dict[str, pd.DataFrame]:
    out = {}
    out["MA_short"] = close.rolling(window_short, min_periods=1).mean()
    out["MA_long"]  = close.rolling(window_long,  min_periods=1).mean()
    rolling = close.rolling(window_short, min_periods=1)
    mid = out["MA_short"]
    std = rolling.std(ddof=0)
    out["BB_mid"] = mid
    out["BB_up"]  = mid + 2 * std
    out["BB_low"] = mid - 2 * std
    return out


def to_csv_download(df: pd.DataFrame, filename: str = "prices.csv") -> Tuple[bytes, str]:
    csv = df.to_csv(index=True).encode("utf-8")
    return csv, filename


# ---------- Shared Forecast utilities (clean + robust) ----------

def rmse(y_true, y_pred):
    return math.sqrt(mean_squared_error(y_true, y_pred))

def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true, float), np.array(y_pred, float)
    eps = 1e-8
    return np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), eps))) * 100

def split_series(df: pd.DataFrame, train_frac: float = 0.80):
    cut = int(len(df)*train_frac)
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()

def cum_from_returns(last_price: float, returns) -> np.ndarray:
    """
    Robust cumulative price path from log-returns.
    Forces scalar math at each step; returns 1-D numpy array (h,).
    """
    rets = np.asarray(returns, float).reshape(-1)
    out = np.empty(len(rets), dtype=float)
    prev = float(last_price)
    exp = math.exp
    for i, r in enumerate(rets):
        prev = prev * exp(float(r))
        out[i] = prev
    return out

# ---------- LSTM (levels) — for comparison tab ----------

def make_supervised(series_scaled: np.ndarray, lookback: int):
    """
    X[t] = series[t:t+lookback], y[t] = series[t+lookback].
    Drop the final window so X and y lengths match.
    """
    n = len(series_scaled)
    if n <= lookback:
        raise ValueError(f"Series too short for lookback={lookback} (n={n}). Lower LOOKBACK.")
    X_win = np.lib.stride_tricks.sliding_window_view(series_scaled, lookback)
    X = X_win[:-1]
    y = series_scaled[lookback:]
    n_min = min(len(X), len(y))
    X = X[:n_min].astype(np.float32)[..., np.newaxis]
    y = y[:n_min].astype(np.float32)
    return X, y

def build_lstm(input_shape, units, dropout, lr=1e-3):
    model = Sequential([Input(shape=input_shape),
                        LSTM(units),
                        Dropout(dropout),
                        Dense(1)])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss="mse")
    return model

def train_lstm_and_predict(train_close: pd.Series, test_close: pd.Series,
                           lookback: int, epochs: int, units: int, dropout: float):
    # Scale on TRAIN only (levels)
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_close.values.reshape(-1,1)).ravel()
    X_tr, y_tr = make_supervised(train_scaled, lookback)

    model = build_lstm((lookback,1), units, dropout, lr=1e-3)
    es = EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)
    model.fit(X_tr, y_tr, epochs=epochs, batch_size=64, validation_split=0.15,
              callbacks=[es], verbose=0)

    # Recursive prediction across TEST
    test_scaled = scaler.transform(test_close.values.reshape(-1,1)).ravel()
    window = train_scaled[-lookback:].copy()
    preds_scaled = np.empty(len(test_scaled), dtype=np.float32)
    for i in range(len(test_scaled)):
        x = window.reshape(1, lookback, 1)
        yhat = model.predict(x, verbose=0)[0,0]
        preds_scaled[i] = yhat
        window = np.concatenate([window[1:], [yhat]])
    preds = scaler.inverse_transform(preds_scaled.reshape(-1,1)).ravel()
    return pd.Series(preds, index=test_close.index)

# ---------- (S)ARIMA & GARCH helpers ----------

def quick_sarima(y_train: pd.Series, candidates, maxiter=150):
    best = None; best_aic = np.inf; best_order=None; best_seasonal=None
    for (order, seasonal_order) in candidates:
        try:
            m = SARIMAX(y_train, order=order,
                        seasonal_order=seasonal_order,
                        trend='c',
                        enforce_stationarity=False, enforce_invertibility=False)
            r = m.fit(disp=False, maxiter=maxiter)
            if r.aic < best_aic:
                best, best_aic = r, r.aic
                best_order, best_seasonal = order, seasonal_order
        except Exception:
            continue
    return best, best_aic, best_order, best_seasonal

def fast_garch(y_train: pd.Series, last_price: float, horizon: int, maxiter=300):
    am = arch_model(y_train*100.0, p=1, q=1, mean='constant', vol='GARCH', dist='t')
    res = am.fit(disp='off', options={"maxiter": maxiter})
    fc = res.forecast(horizon=horizon)
    mu  = (fc.mean.iloc[-1].values / 100.0)            # mean returns (h steps)
    var = (fc.variance.iloc[-1].values / (100.0**2))   # variance of returns
    vol = np.sqrt(var)
    price_path = cum_from_returns(last_price, mu)
    return mu, vol, pd.Series(price_path)

# ---------- Simple LSTM tab helpers (log1p + StandardScaler + walk-forward) ----------

def _to_supervised(arr_1d: np.ndarray, lookback: int) -> Tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for i in range(lookback, len(arr_1d)):
        X.append(arr_1d[i - lookback:i])
        y.append(arr_1d[i])
    X = np.array(X)
    y = np.array(y)
    return X[..., None], y[..., None]

def _walk_forward_predict(model: tf.keras.Model, series: np.ndarray, lookback: int) -> np.ndarray:
    preds = []
    for i in range(lookback, len(series)):
        window = series[i - lookback:i].reshape(1, lookback, 1)
        p = model.predict(window, verbose=0)
        preds.append(float(p.squeeze()))
    return np.array(preds)

def _recover_prices_from_log(std_values: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    inv = scaler.inverse_transform(std_values.reshape(-1, 1)).ravel()
    inv = np.clip(inv, a_min=-1e6, a_max=1e6)
    levels = np.expm1(inv)  # inverse of log1p
    return levels

def _build_lstm_simple(lookback: int, units: int = 64, dropout: float = 0.2, lr: float = 1e-3) -> tf.keras.Model:
    model = Sequential([
        Input(shape=(lookback, 1)),
        LSTM(units, return_sequences=False),
        Dropout(dropout),
        Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss="mse")
    return model


# ------------- Sidebar Controls -------------

st.sidebar.title("⚙️ Controls")

MARKET_PRESETS: Dict[str, List[str]] = {
    "US Stocks": [
        "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "NFLX", "BRK-B", "JPM",
        "DIS", "BAC", "V", "MA", "HD", "KO", "PEP", "INTC", "AMD", "MRK"
    ],
    "Indian Stocks (NSE)": [
        "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS",
        "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS",
        "BAJFINANCE.NS", "LT.NS", "TATAMOTORS.NS", "MARUTI.NS", "ASIANPAINT.NS"
    ],
    "Cryptos": [
        "BTC-USD", "ETH-USD", "BNB-USD", "XRP-USD", "ADA-USD", "DOGE-USD",
        "SOL-USD", "TRX-USD", "MATIC-USD"
    ],
    "Forex": [
        "EURUSD=X", "USDJPY=X", "GBPUSD=X", "AUDUSD=X", "USDCAD=X",
        "USDCHF=X", "NZDUSD=X", "EURINR=X", "USDINR=X", "GBPINR=X"
    ],
    "Indices": [
        "^GSPC", "^DJI", "^IXIC", "^RUT", "^VIX",
        "^NSEI", "^NSEBANK", "^BSESN", "^FTSE", "^N225", "^HSI", "^STOXX50E"
    ],
    "ETFs": [
        "SPY", "QQQ", "DIA", "IWM", "VTI", "IVV",
        "EFA", "EEM", "HDFC.NS", "HINDPETRO.NS"  # mixed example
    ]
}

market = st.sidebar.selectbox("Market category", list(MARKET_PRESETS.keys()), index=0)

preset = MARKET_PRESETS[market]
sel_tickers = st.sidebar.multiselect("Select tickers", preset, default=preset[:3])

custom = st.sidebar.text_area(
    "Add custom tickers (comma or newline separated)",
    placeholder="e.g., NIFTYBEES.NS, BTC-USD, USDINR=X"
)
custom_list = parse_custom_tickers(custom)

all_tickers = list_to_clean_tickers(sel_tickers + custom_list)

st.sidebar.caption(
    "Tip: Indian stocks on Yahoo Finance typically use the **.NS** suffix (e.g., RELIANCE.NS). "
    "Forex pairs end with **=X** (e.g., USDINR=X). Cryptos use **-USD** (e.g., BTC-USD)."
)

st.sidebar.markdown("---")

st.sidebar.subheader("Time range (Start–End only)")

today = dt.date.today()
default_start = today - dt.timedelta(days=365)
start_date = st.sidebar.date_input("Start date", value=default_start)
end_date = st.sidebar.date_input("End date", value=today)

interval = st.sidebar.selectbox(
    "Interval",
    ["1d", "1wk", "1mo"],
    index=0,
    help="Daily/Weekly/Monthly only for stable forecasting tabs."
)

st.sidebar.markdown("---")
auto_adjust = st.sidebar.checkbox(
    "Auto-adjust for splits/dividends (recommended for stocks/ETFs)",
    value=True
)

st.sidebar.markdown("---")
st.sidebar.caption("Time Series Analysis Project created by Rajendra Panda (M24DE3091) & Geetika Vijay (M24DE3035)")

# ------------- Main UI -------------

st.title("📊 Market Analysis 📈")

if not all_tickers:
    st.info("Select at least one ticker from the sidebar or add custom tickers.")
    st.stop()

colA, colB = st.columns([2, 1])
with colA:
    st.write(f"**Selected tickers:** {', '.join(all_tickers)}")
with colB:
    st.write(f"**Interval:** `{interval}`")

with st.spinner("Fetching data from Yahoo Finance..."):
    data = fetch_prices(
        tickers=all_tickers,
        start=start_date,   # inclusive handling inside fetch_prices
        end=end_date,       # inclusive handling inside fetch_prices
        interval=interval,
        auto_adjust=auto_adjust
    )

if data.empty:
    st.warning("No data returned. Try a different ticker set, date range, or interval.")
    st.stop()

# Normalize fields
if "Close" not in data.columns.get_level_values(0):
    st.error("Response doesn't contain 'Close' prices — try a different interval/date range.")
    st.stop()

# Build a "Close only" wide frame: index=Date, columns=Tickers
close = pd.DataFrame({t: data["Close"][t] for t in data["Close"].columns})
volume = (pd.DataFrame({t: data["Volume"][t]
                        for t in data["Volume"].columns
                        if ("Volume", t) in data.columns})
          if "Volume" in data.columns.get_level_values(0)
          else pd.DataFrame(index=close.index))

shown_start = data.index.min()
shown_end   = data.index.max()
st.caption(f"**Data Window:** {shown_start.date()} → {shown_end.date()}  |  **Interval:** `{interval}`")

# -------------------- TABS --------------------
# Order: tab6 (LSTM simple) immediately followed by tab7 (comparison)
tab1, tab2, tab3, tab4, tab5, tab6_simple, tab7_cmp = st.tabs(
    ["📊 Prices", "📐 Indicators", "📈 Returns", "🔗 Correlation", "📥 Export",
     "🤖 LSTM Forecast (simple)", "🤖 Forecast (LSTM vs (S)ARIMA + GARCH)"]
)

# --- TAB 1: Prices ---
with tab1:
    st.subheader("Adjusted Close Prices")
    st.line_chart(close)
    if not volume.empty:
        st.subheader("Volume")
        st.bar_chart(volume)

# --- TAB 2: Indicators ---
with tab2:
    st.subheader("Moving Averages & Bollinger Bands")
    w1 = st.slider("Short window (MA + Bollinger)", min_value=5, max_value=60, value=20, step=1)
    w2 = st.slider("Long window (MA)", min_value=10, max_value=200, value=50, step=5)

    inds = compute_indicators(close, window_short=w1, window_long=w2)

    sel = st.selectbox("Choose a single ticker to visualize indicators", all_tickers, index=0)
    if sel in close.columns:
        df_plot = pd.DataFrame({
            "Close": close[sel],
            f"MA{w1}": inds["MA_short"][sel],
            f"MA{w2}": inds["MA_long"][sel],
            f"BB{w1}_UP": inds["BB_up"][sel],
            f"BB{w1}_MID": inds["BB_mid"][sel],
            f"BB{w1}_LOW": inds["BB_low"][sel]
        })
        st.line_chart(df_plot)
    else:
        st.warning("Selected ticker not found in data.")

# --- TAB 3: Returns ---
with tab3:
    st.subheader("Daily/Weekly/Monthly Returns (per interval)")
    returns = close.pct_change().dropna(how="all")
    st.line_chart(returns)

    st.markdown("**Summary Stats (Returns)**")
    stats = returns.agg(["mean", "std", "min", "max"]).T
    stats.rename(columns={"mean": "Mean", "std": "Std Dev", "min": "Min", "max": "Max"}, inplace=True)
    st.dataframe(stats.style.format("{:.4f}"))

    st.markdown("**Histogram (choose one ticker)**")
    sel_r = st.selectbox("Ticker", all_tickers, index=0, key="ret_hist")
    if sel_r in returns.columns:
        fig, ax = plt.subplots(figsize=(6, 4))
        returns[sel_r].dropna().hist(bins=50, ax=ax)
        ax.set_title(f"Histogram of Returns — {sel_r}")
        ax.set_xlabel("Return"); ax.set_ylabel("Frequency")
        st.pyplot(fig)
    else:
        st.info("Selected ticker not available in returns data.")

# --- TAB 4: Correlation ---
with tab4:
    st.subheader("Correlation of Returns")
    returns = close.pct_change().dropna(how="all")
    if returns.shape[1] >= 2:
        corr = returns.corr()
        fig, ax = plt.subplots(figsize=(min(10, 2+len(all_tickers)), min(8, 1+len(all_tickers)*0.5)))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="viridis", ax=ax)
        ax.set_title("Correlation Heatmap (Returns)")
        st.pyplot(fig)
        st.caption("Higher values (close to 1) mean the two tickers move together more often.")
    else:
        st.info("Select at least two tickers to see correlation.")

# --- TAB 5: Export ---
with tab5:
    st.subheader("Download Data")
    # Flatten MultiIndex columns for CSV
    flat = data.copy()
    flat.columns = ["_".join(col) for col in flat.columns.to_flat_index()]
    csv_bytes, fname = to_csv_download(flat, filename="market_data.csv")
    st.download_button(
        label="Download CSV",
        data=csv_bytes,
        file_name=fname,
        mime="text/csv"
    )
    st.caption("CSV includes all OHLCV fields (if available) for each ticker.")

# --- TAB 6 (Simple first): LSTM Forecast (simple) ---
with tab6_simple:
    st.subheader("Next-Period Close Forecast — LSTM (log1p + StandardScaler + walk-forward)")

    # Choose one ticker for forecasting
    tkr = st.selectbox("Select a ticker", all_tickers, index=0, key="simple_tkr")
    series = close[tkr].dropna()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        lookback = st.number_input("Lookback (window size)", min_value=10, max_value=400, value=60, step=5, key="simple_lb")
    with c2:
        test_frac = st.slider("Test fraction", 0.05, 0.40, 0.20, 0.01, key="simple_tf")
    with c3:
        val_frac = st.slider("Validation fraction (of training)", 0.05, 0.40, 0.20, 0.01, key="simple_vf")
    with c4:
        epochs = st.number_input("Epochs", min_value=10, max_value=500, value=60, step=10, key="simple_ep")

    c5, c6, c7 = st.columns(3)
    with c5:
        units = st.number_input("LSTM units", min_value=16, max_value=256, value=64, step=16, key="simple_units")
    with c6:
        dropout = st.slider("Dropout", 0.0, 0.8, 0.2, 0.05, key="simple_do")
    with c7:
        batch_size = st.number_input("Batch size", min_value=8, max_value=512, value=64, step=8, key="simple_bs")

    run_btn = st.button("Train & Forecast (LSTM Simple)", type="primary", key="simple_run")

    if run_btn:
        if len(series) < max(100, lookback + 30):
            st.warning(
                f"Not enough data points for a stable train/validation/test split with lookback={lookback}. "
                f"Have {len(series)} points; try reducing lookback or extending the period."
            )
        else:
            with st.spinner("Training LSTM (simple)…"):
                # 1) Log transform to stabilize scale, ensure positive
                ser = series.astype(float)
                if (ser <= 0).any():
                    shift = abs(float(ser.min())) + 1e-6
                    ser = ser + shift
                logy = np.log1p(ser.values)

                # 2) Standardize
                scaler = StandardScaler()
                logy_std = scaler.fit_transform(logy.reshape(-1, 1)).ravel()

                # 3) Train/val/test split (by time)
                n = len(logy_std)
                n_test = int(n * test_frac)
                n_trainval = n - n_test
                if n_trainval <= lookback + 5:
                    st.error("Training portion too small for given lookback. Reduce test fraction or lookback.")
                    st.stop()

                n_val = int(n_trainval * val_frac)
                n_train = n_trainval - n_val

                train_std = logy_std[:n_train]
                val_std   = logy_std[n_train:n_trainval]
                test_std  = logy_std[n_trainval:]

                # 4) Supervised datasets
                Xtr, ytr = _to_supervised(train_std, lookback)
                if len(val_std) <= lookback + 1:
                    st.warning("Validation segment very small; merging into training.")
                    train_std = logy_std[:n_trainval]
                    val_std   = np.array([], dtype=float)
                    Xtr, ytr = _to_supervised(train_std, lookback)
                    Xval = yval = None
                else:
                    Xval, yval = _to_supervised(val_std, lookback)

                # 5) Model
                model = _build_lstm_simple(lookback=lookback, units=int(units), dropout=float(dropout), lr=1e-3)
                callbacks = [EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)]

                # 6) Fit
                if Xval is None:
                    history = model.fit(
                        Xtr, ytr,
                        epochs=int(epochs),
                        batch_size=int(batch_size),
                        verbose=0
                    )
                else:
                    history = model.fit(
                        Xtr, ytr,
                        validation_data=(Xval, yval),
                        epochs=int(epochs),
                        batch_size=int(batch_size),
                        callbacks=callbacks,
                        verbose=0
                    )

                # 7) Walk-forward predictions on the full range; then slice test part
                full_std = logy_std
                preds_std = _walk_forward_predict(model, full_std, lookback=lookback)

                # Align indices
                pred_index = series.index[lookback:]
                preds_std_aligned = preds_std[-(n - lookback):]

                test_index = series.index[n_trainval:]
                start_idx_for_test_preds = (n_trainval) - lookback
                if start_idx_for_test_preds < 0:
                    start_idx_for_test_preds = 0
                preds_test_std = preds_std_aligned[start_idx_for_test_preds:]

                # 8) Inverse transform to price levels
                y_test_levels = _recover_prices_from_log(test_std, scaler)
                y_pred_levels = _recover_prices_from_log(preds_test_std, scaler)

                # Align lengths
                m = min(len(test_index), len(y_test_levels), len(y_pred_levels))
                test_index = test_index[:m]
                y_test_levels = y_test_levels[:m]
                y_pred_levels = y_pred_levels[:m]

                # 9) Metrics
                _rmse = math.sqrt(mean_squared_error(y_test_levels, y_pred_levels))
                _mape = mean_absolute_percentage_error(
                    np.clip(y_test_levels, 1e-9, None),
                    np.clip(y_pred_levels, 1e-9, None)
                ) * 100.0

                # 10) Next-period forecast
                last_window_std = logy_std[-lookback:].reshape(1, lookback, 1)
                next_std = float(model.predict(last_window_std, verbose=0).squeeze())
                next_level = float(np.expm1(scaler.inverse_transform([[next_std]])[0, 0]))

                st.success(f"Model trained for **{tkr}**. Test RMSE: **{_rmse:,.4f}**, MAPE: **{_mape:.2f}%**")

                # Plot
                plot_df = pd.DataFrame({
                    "Actual": pd.Series(y_test_levels, index=test_index),
                    "Predicted": pd.Series(y_pred_levels, index=test_index)
                })
                st.line_chart(plot_df)

                # Next timestamp label (best-effort)
                if len(series.index) >= 2 and hasattr(series.index, "inferred_freq") and series.index.inferred_freq:
                    next_ts = series.index[-1] + pd.tseries.frequencies.to_offset(series.index.inferred_freq)
                else:
                    next_ts = series.index[-1] + pd.Timedelta(days=1)

                st.markdown(f"**Next {interval} forecast for {tkr}:** `{next_ts}` → **{next_level:,.4f}**")

                # Export predictions
                out_pred = plot_df.copy()
                out_pred.loc[next_ts, "Predicted_Next"] = next_level
                csv_bytes, fname = to_csv_download(out_pred, filename=f"{tkr}_lstm_simple_forecast.csv")
                st.download_button(
                    "Download forecast CSV",
                    data=csv_bytes,
                    file_name=fname,
                    mime="text/csv"
                )

                with st.expander("Training details"):
                    st.write({
                        "lookback": lookback,
                        "epochs": int(epochs),
                        "batch_size": int(batch_size),
                        "units": int(units),
                        "dropout": float(dropout),
                        "train_points": len(train_std),
                        "val_points": len(val_std),
                        "test_points": len(test_std),
                    })

# --- TAB 7 (Second): Forecast (LSTM vs (S)ARIMA + GARCH) ---
with tab7_cmp:
    st.subheader("LSTM (levels) vs quick (S)ARIMA (returns → price) + GARCH(1,1) Volatility")

    # Interval guard (models expect daily/weekly/monthly cadence)
    if interval not in ["1d", "1wk", "1mo"]:
        st.warning("Forecast tab supports only daily/weekly/monthly intervals. Please switch interval.")
        st.stop()

    sel_f = st.selectbox("Choose a ticker to forecast", all_tickers, index=0, key="cmp_sel")
    train_frac = st.slider("Train fraction", min_value=0.6, max_value=0.95, value=0.80, step=0.01, key="cmp_tr")

    speed = st.radio("Speed preset", ["mini", "fast", "full"], index=0, horizontal=True,
                     help="mini=quick test | fast=balanced | full=better search & epochs", key="cmp_speed")

    # Presets
    if speed == "mini":
        LOOKBACK, LSTM_EPOCHS, LSTM_UNITS, LSTM_DROPOUT = 30, 8, 32, 0.10
        SARIMA_CANDS = [((1,0,1),(0,0,0,0)), ((1,1,1),(0,0,0,0)), ((1,0,1),(1,0,1,5))]
        GARCH_LAST_N, SARIMAX_MAXITER, GARCH_MAXITER = 800, 100, 200
    elif speed == "fast":
        LOOKBACK, LSTM_EPOCHS, LSTM_UNITS, LSTM_DROPOUT = 45, 18, 48, 0.15
        SARIMA_CANDS = [
            ((1,0,1),(0,0,0,0)), ((1,1,1),(0,0,0,0)),
            ((2,0,1),(0,0,0,0)), ((1,0,2),(0,0,0,0)),
            ((1,0,1),(1,0,1,5)), ((1,0,1),(1,0,1,21)),
        ]
        GARCH_LAST_N, SARIMAX_MAXITER, GARCH_MAXITER = 1200, 200, 300
    else:
        LOOKBACK, LSTM_EPOCHS, LSTM_UNITS, LSTM_DROPOUT = 60, 30, 64, 0.20
        SARIMA_CANDS = [((p,d,q),(0,0,0,0)) for p in (0,1,2) for d in (0,1) for q in (0,1,2)]
        SARIMA_CANDS += [((1,0,1),(1,0,1,5)), ((1,0,1),(1,0,1,21))]
        GARCH_LAST_N, SARIMAX_MAXITER, GARCH_MAXITER = 2000, 300, 400

    # Pull series
    if sel_f not in close.columns:
        st.error("Selected ticker not found in data.")
        st.stop()

    series = close[sel_f].dropna()
    if len(series) < max(250, LOOKBACK + 30):
        st.warning("Not enough data for a stable comparison. Try a longer period or different ticker.")
        st.stop()

    st.caption("Models: LSTM trains on **levels**; (S)ARIMA fits **log-returns** and rebuilds to price. "
               "GARCH(1,1) forecasts **volatility** on returns (optional).")

    run = st.button("Run comparison", type="primary", key="cmp_run")

    if run:
        with st.spinner("Training & forecasting…"):
            # Prepare dataframe with Close & LogRet
            df_f = pd.DataFrame({"Close": series})
            df_f["LogRet"] = np.log(df_f["Close"]).diff()

            train_df, test_df = split_series(df_f, train_frac=train_frac)
            h = len(test_df)

            # LSTM on levels
            lstm_pred = train_lstm_and_predict(
                train_close=train_df["Close"], test_close=test_df["Close"],
                lookback=LOOKBACK, epochs=LSTM_EPOCHS, units=LSTM_UNITS, dropout=LSTM_DROPOUT
            )

            # (S)ARIMA on returns → price
            y_tr = train_df["LogRet"].dropna()
            sarima_res, aic, best_order, best_seasonal = quick_sarima(
                y_tr, SARIMA_CANDS, maxiter=SARIMAX_MAXITER
            )
            if sarima_res is None:
                st.error("All SARIMA candidates failed; try speed='fast' or increase the period.")
                st.stop()

            fc_obj = sarima_res.get_forecast(steps=h)
            rets_1d = np.asarray(fc_obj.predicted_mean, dtype=float).reshape(-1)
            last_close = float(train_df["Close"].iloc[-1])
            sarima_prices = cum_from_returns(last_close, rets_1d)
            sarima_pred = pd.Series(np.ravel(sarima_prices), index=test_df.index)

            # GARCH(1,1) volatility (optional)
            ann_vol_series, garch_price_series = None, None
            if HAVE_ARCH:
                y_g = y_tr.tail(GARCH_LAST_N).dropna()
                try:
                    mu, vol, garch_price = fast_garch(
                        y_g, last_close, horizon=h, maxiter=GARCH_MAXITER
                    )
                    ann_vol_series = pd.Series(vol*np.sqrt(252)*100.0, index=test_df.index)  # annualized %
                    garch_price_series = pd.Series(np.ravel(garch_price.values), index=test_df.index)
                except Exception as ex:
                    st.info(f"GARCH skipped due to error: {ex}")

            # Metrics
            def add_metrics(name, actual, pred):
                return {"Model": name,
                        "MAE": mean_absolute_error(actual, pred),
                        "RMSE": rmse(actual, pred),
                        "MAPE%": mape(actual, pred)}

            metrics = [
                add_metrics("LSTM (levels)",     test_df["Close"], lstm_pred),
                add_metrics("(S)ARIMA (levels)", test_df["Close"], sarima_pred),
            ]
            if garch_price_series is not None:
                metrics.append(add_metrics("GARCH mean-only (levels)", test_df["Close"], garch_price_series))

            metrics_df = pd.DataFrame(metrics).sort_values("RMSE")
            st.markdown("#### Test Metrics")
            st.dataframe(metrics_df.style.format({"MAE":"{:.4f}", "RMSE":"{:.4f}", "MAPE%":"{:.2f}"}), use_container_width=True)

            # Model choice hint
            best_row = metrics_df.iloc[0]
            st.success(f"**Current best (by RMSE):** {best_row['Model']} | RMSE: {best_row['RMSE']:.4f} | "
                       f"MAPE: {best_row['MAPE%']:.2f}%")

            # Plots
            fig1, ax1 = plt.subplots(figsize=(10,4.5))
            ax1.plot(test_df.index, test_df["Close"].values, label="Actual", linewidth=2)
            ax1.plot(lstm_pred.index, lstm_pred.values, label="LSTM", alpha=0.9)
            ax1.plot(sarima_pred.index, sarima_pred.values, label="(S)ARIMA", alpha=0.9)
            if garch_price_series is not None:
                ax1.plot(garch_price_series.index, garch_price_series.values, label="GARCH mean-only", alpha=0.9)
            ax1.set_title(f"{sel_f} — Levels: LSTM vs (S)ARIMA")
            ax1.grid(True); ax1.legend()
            st.pyplot(fig1)

            # SARIMA details
            st.caption(f"SARIMA chosen (AIC {aic:.2f}): order={best_order}, seasonal_order={best_seasonal}")

            # Volatility
            if ann_vol_series is not None:
                fig2, ax2 = plt.subplots(figsize=(10,3.8))
                ax2.plot(ann_vol_series.index, ann_vol_series.values)
                ax2.set_title(f"{sel_f} — GARCH(1,1) Annualized Volatility (fast)")
                ax2.set_ylabel("Volatility (%)"); ax2.grid(True)
                st.pyplot(fig2)

            st.info("Notes: LSTM uses levels; SARIMA uses returns and is rebuilt to price. "
                    "Volatility comes from GARCH(1,1) on returns. Results vary by ticker, date range, and preset.")

st.markdown("---")
st.caption(
    "Data provided by Yahoo Finance via **yfinance**. "
    "Some tickers/intervals may not be available for all markets. "
    "Forecasts are for research/education only."
)
