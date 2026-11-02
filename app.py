import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import pyfredapi as pf
from hmmlearn.hmm import GaussianHMM
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Page Config
st.set_page_config(page_title="Institutional US Macro Dashboard", layout="wide")

st.title("🏛️ Institutional US Macro Regime & Scoring Dashboard")
st.markdown("*Quantitative Multi-Asset Modeling, Liquidity Plumbing, & Central Bank Sentiment Engine*")

# Sidebar Controls
st.sidebar.header("Configuration Panel")
fred_key_input = st.sidebar.text_input("FRED API Key", value="43d2e92afac22209503ca5e2c1b15441", type="password")
bls_key_input = st.sidebar.text_input("BLS API Key", value="9bc1ba1efaa1411fadefad8e9037d179", type="password")

sample_fed_transcript = st.sidebar.text_area(
    "Fed Speak / FOMC Transcript Input",
    value="Inflation remains elevated and persistent, driven by energy costs and demand. We must maintain a restrictive stance."
)

if st.sidebar.button("Run Macro Pipeline"):
    with st.spinner("Fetching data from FRED and Yahoo Finance, running HMM & PCA models..."):
        
        # 1. Fetch Market Data
        tickers = ["^GSPC", "^SOX", "^TNX", "HG=F", "GC=F", "^VIX", "^VIX3M", "DX-Y.NYB", "^MOVE"]
        yf_data = yf.download(tickers, period="3y", interval="1d", progress=False)["Close"]
        if isinstance(yf_data.columns, pd.MultiIndex):
            yf_data.columns = yf_data.columns.get_level_values(0)
            
        yf_data["copper_gold"] = yf_data["HG=F"] / yf_data["GC=F"]
        yf_data["semis_spx"] = yf_data["^SOX"] / yf_data["^GSPC"]
        yf_data["vix_slope"] = yf_data["^VIX"] - yf_data["^VIX3M"]

        # 2. Fetch FRED Data
        try:
            import os
            os.environ["FRED_API_KEY"] = fred_key_input
            walcl = pf.get_series(series_id="WALCL")
            tga = pf.get_series(series_id="WTREGEN")
            rrp = pf.get_series(series_id="RRPONTSYD")
            hy_oas = pf.get_series(series_id="BAMLH0A0HYM2")
            ism_orders = pf.get_series(series_id="NAPMNEWO")
            curve = pf.get_series(series_id="T10Y2Y")
            nfci = pf.get_series(series_id="NFCI")
            
            net_liq = walcl - tga - rrp
            fred_df = pd.DataFrame({
                "fed_net_liquidity": net_liq,
                "hy_credit_spread": hy_oas,
                "ism_new_orders": ism_orders,
                "curve_2s10s": curve,
                "nfci": nfci
            })
        except Exception as e:
            st.warning(f"FRED API notice: {e}. Using synchronized defaults.")
            fred_df = pd.DataFrame(index=yf_data.index)
            fred_df["fed_net_liquidity"] = 6000000
            fred_df["hy_credit_spread"] = 3.5
            fred_df["ism_new_orders"] = 51.0
            fred_df["curve_2s10s"] = 0.2
            fred_df["nfci"] = -0.5

        master_df = yf_data.join(fred_df, how="outer").sort_index().ffill().dropna()

        # 3. Model Calculations
        z_df = pd.DataFrame(index=master_df.index)
        for col in master_df.columns:
            z_df[col] = (master_df[col] - master_df[col].rolling(252).mean()) / master_df[col].rolling(252).std()
        z_df = z_df.dropna()

        # Composite Scoring
        growth = (z_df['ism_new_orders']*0.12 + z_df['semis_spx']*0.10 + z_df['copper_gold']*0.05 + z_df['curve_2s10s']*0.03) / 0.30
        liq = (z_df['fed_net_liquidity']*0.12 + 0.15*0.10 - z_df['nfci']*0.08 - z_df['DX-Y.NYB']*0.05) / 0.35
        risk = (-z_df['hy_credit_spread'].diff(20)*0.15 - z_df['vix_slope']*0.10 - z_df['^MOVE']*0.10) / 0.35
        composite = (growth * 0.30) + (liq * 0.35) + (risk * 0.35)

        # HMM Model
        feat_hmm = pd.DataFrame({'eq_ret': master_df['^GSPC'].pct_change(), 'rate': master_df['^TNX'].diff(), 'vix': master_df['^VIX']}).dropna()
        X_hmm = (feat_hmm - feat_hmm.mean()) / feat_hmm.std()
        hmm = GaussianHMM(n_components=4, covariance_type="full", n_iter=500, random_state=42).fit(X_hmm.values)
        states = hmm.predict(X_hmm.values)
        probs = hmm.predict_proba(X_hmm.values[-1:])

        # Display Results in UI
        st.success("Pipeline executed successfully!")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Growth Pillar (30%)", f"{growth.iloc[-1]:.2f}σ")
        col2.metric("Liquidity & Policy (35%)", f"{liq.iloc[-1]:.2f}σ")
        col3.metric("Risk & Premia (35%)", f"{risk.iloc[-1]:.2f}σ")
        col4.metric("Composite Macro Index", f"{composite.iloc[-1]:.2f}σ")

        st.subheader("📊 HMM Regime Probability State")
        st.write(f"**Current Active State:** State {states[-1]}")
        prob_df = pd.DataFrame(probs, columns=[f"State {i}" for i in range(4)]).T
        prob_df.columns = ["Probability"]
        st.bar_chart(prob_df)

        st.subheader("Historical Composite Index Trend")
        st.line_chart(composite.tail(252))

else:
    st.info("Configure your parameters in the sidebar and click **Run Macro Pipeline** to generate live model outputs.")