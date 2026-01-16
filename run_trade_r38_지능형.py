import streamlit as st
import pandas as pd
import FinanceDataReader as fdr
import yfinance as yf
import datetime, time, requests, os, json, gc
import joblib
import traceback
from datetime import timezone, timedelta
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from concurrent.futures import ThreadPoolExecutor, as_completed
from streamlit_gsheets import GSheetsConnection
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from scipy.signal import find_peaks
from scipy.stats import norm
from bs4 import BeautifulSoup

# [V81.58 Update] 딥러닝/규제 라이브러리
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import MinMaxScaler

# ==========================================
# ⚙️ 1. 시스템 설정 및 초기화
# ==========================================
MODEL_FILE = "ai_ensemble_model.pkl"
LSTM_MODEL_FILE = "ai_lstm_model.h5"
SCALER_FILE = "ai_lstm_scaler.pkl"

def get_now_kst():
    return datetime.datetime.now(timezone(timedelta(hours=9)))

def check_market_open():
    now = get_now_kst()
    if now.weekday() >= 5: return False
    start_time = datetime.time(9, 0)
    end_time = datetime.time(15, 30)
    return start_time <= now.time() <= end_time

st.set_page_config(page_title="AI Master V81.58 Stable", page_icon="🧬", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #f0f2f6; }
    .metric-card { background: white; padding: 20px; border-radius: 12px; border-left: 5px solid #00897b; box-shadow: 0 2px 8px rgba(0,0,0,0.05); margin-bottom: 15px; }
    .recovery-card { background: #fff3e0; padding: 15px; border-radius: 10px; border: 1px solid #ffe0b2; margin-top: 10px; font-size: 0.9em; }
    .rebal-card { background: #e8f5e9; padding: 15px; border-radius: 10px; border: 1px solid #c8e6c9; margin-top: 10px; font-size: 0.9em; }
    .scanner-card { padding: 20px; border-radius: 15px; border: 1px solid #e0e0e0; margin-bottom: 15px; background-color: white; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }
    .strategy-grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; margin-top: 15px; width: 100%; }
    .buy-box { background-color: #e0f7fa; padding: 15px; border-radius: 10px; border: 1px solid #b2ebf2; color: #006064; font-size: 0.9em; line-height: 1.5; }
    .sell-box { background-color: #ffebee; padding: 15px; border-radius: 10px; border: 1px solid #ffcdd2; color: #b71c1c; font-size: 0.9em; line-height: 1.5; }
    .stop-box { background-color: #f3e5f5; padding: 15px; border-radius: 10px; border: 1px solid #e1bee7; color: #4a148c; font-size: 0.9em; line-height: 1.5; }
    .current-price { font-size: 1.6em; font-weight: 800; color: #212121; }
    .ai-badge { background-color: #00897b; color: white; padding: 3px 8px; border-radius: 4px; font-weight: bold; font-size: 0.8em; }
    .mode-badge { background-color: #37474f; color: #00e676; padding: 3px 8px; border-radius: 4px; font-weight: bold; font-size: 0.8em; }
    .style-badge { background-color: #512da8; color: #fff; padding: 3px 8px; border-radius: 4px; font-weight: bold; font-size: 0.8em; margin-right: 5px; }
    .mtf-badge { background-color: #e3f2fd; color: #1565c0; padding: 2px 6px; border-radius: 4px; font-size: 0.8em; font-weight: bold; border: 1px solid #bbdefb; }
    .pattern-badge { background-color: #fff8e1; color: #f57f17; padding: 2px 6px; border-radius: 4px; font-size: 0.8em; font-weight: bold; border: 1px solid #ffecb3; margin-left: 3px; }
    .hit-tag { background-color: #e8f5e9; color: #2e7d32; padding: 2px 6px; border-radius: 4px; border: 1px solid #c8e6c9; margin-right: 4px; font-size: 0.85em; }
    .alpha-tag { background-color: #f3e5f5; color: #7b1fa2; padding: 2px 6px; border-radius: 4px; border: 1px solid #e1bee7; margin-right: 4px; font-size: 0.85em; font-weight:bold; }
    .break-tag { background-color: #ffcdd2; color: #b71c1c; padding: 2px 6px; border-radius: 4px; border: 1px solid #ef5350; margin-right: 4px; font-size: 0.85em; font-weight:bold; }
    .whipsaw-box { background-color: #fff3e0; padding: 10px; border-radius: 6px; border: 1px solid #ffe0b2; color: #e65100; font-weight: bold; margin: 10px 0; font-size: 0.9em; }
    .pro-tag { background-color: #e3f2fd; color: #0d47a1; font-size: 0.75em; padding: 2px 5px; border-radius: 4px; border: 1px solid #90caf9; font-weight:bold; margin-left: 5px; }
    .clock-box { font-size: 1.2em; font-weight: bold; color: #333; text-align: center; margin-bottom: 5px; padding: 10px; background: #e0f7fa; border-radius: 8px; border: 1px solid #b2ebf2; }
    .source-box { background-color: #37474f; color: #fff; padding: 8px; border-radius: 6px; text-align: center; font-size: 0.9em; margin-bottom: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
    .list-box { background-color: #546e7a; color: #fff; padding: 8px; border-radius: 6px; text-align: center; font-size: 0.9em; margin-bottom: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
    .status-open { color: #2e7d32; font-weight: bold; text-align: center; margin-bottom: 15px; }
    .status-closed { color: #c62828; font-weight: bold; text-align: center; margin-bottom: 15px; }
    @media (max-width: 640px) { .strategy-grid { grid-template-columns: 1fr; } }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 🧠 2. LSTM 엔진 (V81.58 수정 완료: L2 규제 및 Input Layer 명시)
# ==========================================
class LSTMEngine:
    def __init__(self, lookback=20):
        self.lookback = lookback
        self.model = None
        self.scaler = None

    # 🔹 헬퍼 함수: 지연 로딩 (필요할 때만 TensorFlow 로드)
    def _import_tf(self):
        try:
            import tensorflow as tf
            from tensorflow.keras.models import Sequential, load_model
            from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
            
            # [V81.58 Fix] l2 규제 모듈 명시적 임포트
            from tensorflow.keras.regularizers import l2
            from sklearn.preprocessing import MinMaxScaler
            
            # 반환 값 순서: tf, Seq, load, LSTM, Dense, Drop, Input, l2, MMS
            return tf, Sequential, load_model, LSTM, Dense, Dropout, Input, l2, MinMaxScaler
        except ImportError:
            return None, None, None, None, None, None, None, None, None

    def create_model(self, input_shape):
        # [V81.58 Fix] 변수 받아오기
        tf, Sequential, _, LSTM, Dense, Dropout, Input, l2, _ = self._import_tf()
        
        if not tf: return None
        
        # 모델 구조 정의
        model = Sequential()
        # [V81.58 Fix] Input Layer 명시
        model.add(Input(shape=input_shape))
        
        # [V81.58 Fix] LSTM 층에 kernel_regularizer=l2(0.01) 적용 및 유닛 64로 상향
        model.add(LSTM(64, return_sequences=False, kernel_regularizer=l2(0.01)))
        model.add(Dropout(0.3)) # 드롭아웃 비율 상향
        model.add(Dense(1, activation='sigmoid'))
        
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def prepare_data(self, df, training=False):
        # MinMaxScaler 받아오기
        _, _, _, _, _, _, _, _, MinMaxScaler = self._import_tf()
        
        features = ['Close', 'Volume', 'RSI', 'MACD', 'Stoch_20']
        if len(df) < self.lookback + 5: return None, None
        
        if 'RSI' not in df.columns: df = get_all_indicators(df)
        if df is None: return None, None
            
        temp_df = df[features].fillna(0)
        data = temp_df.values
        
        if training:
            self.scaler = MinMaxScaler()
            scaled_data = self.scaler.fit_transform(data)
        else:
            if self.scaler is None: 
                try: self.scaler = joblib.load(SCALER_FILE)
                except: return None, None
            scaled_data = self.scaler.transform(data)

        X, y = [], []
        if training:
            for i in range(self.lookback, len(scaled_data) - 1):
                X.append(scaled_data[i-self.lookback:i])
                target = 1 if data[i+1][0] > data[i][0] * 1.02 else 0 
                y.append(target)
            return np.array(X), np.array(y)
        else:
            last_sequence = scaled_data[-self.lookback:]
            return np.array([last_sequence]), None

    def train_and_save(self, df_list):
        if len(df_list) > 10: df_list = df_list[:10] 
        
        all_X, all_y = [], []
        for df in df_list:
            X, y = self.prepare_data(df, training=True)
            if X is not None:
                all_X.append(X); all_y.append(y)
        
        if not all_X: return False, "데이터 부족"
        
        X_final = np.concatenate(all_X)
        y_final = np.concatenate(all_y)
        
        self.model = self.create_model((self.lookback, X_final.shape[2]))
        if self.model:
            self.model.fit(X_final, y_final, epochs=3, batch_size=16, verbose=0)
            self.model.save(LSTM_MODEL_FILE)
            joblib.dump(self.scaler, SCALER_FILE)
            return True, f"LSTM 경량 학습 완료 ({len(X_final)}샘플)"
        return False, "TensorFlow 로딩 실패"

    def predict_score(self, df):
        try:
            _, _, load_model, _, _, _, _, _, _ = self._import_tf()
            
            if self.model is None:
                if os.path.exists(LSTM_MODEL_FILE): self.model = load_model(LSTM_MODEL_FILE)
                else: return 50
            
            X_pred, _ = self.prepare_data(df, training=False)
            if X_pred is None: return 50
            
            prob = self.model.predict(X_pred, verbose=0)[0][0]
            return int(prob * 100)
        except: return 50

# 엔진 초기화
lstm_engine = LSTMEngine()

# --- [KIS API Client] (재시도 로직 강화) ---
class KIS_Data_Client:
    def __init__(self, app_key, app_secret, mock=False):
        self.app_key = app_key
        self.app_secret = app_secret
        self.base_url = "https://openapivts.koreainvestment.com:29443" if mock else "https://openapi.koreainvestment.com:9443"
        self.token = None
        self.token_issued = None
        
    def get_access_token(self):
        headers = {"content-type": "application/json"}
        body = {"grant_type": "client_credentials", "appkey": self.app_key, "appsecret": self.app_secret}
        url = f"{self.base_url}/oauth2/tokenP"
        try:
            res = requests.post(url, headers=headers, data=json.dumps(body), timeout=5)
            if res.status_code == 200:
                self.token = res.json()['access_token']
                self.token_issued = datetime.datetime.now()
                return True
        except: pass
        return False

    def check_token(self):
        if self.token is None or self.token_issued is None: return self.get_access_token()
        if (datetime.datetime.now() - self.token_issued).total_seconds() > 21000: return self.get_access_token()
        return True

    def get_current_price(self, code):
        if not self.check_token(): return None
        headers = {
            "content-type": "application/json", "authorization": f"Bearer {self.token}",
            "appkey": self.app_key, "appsecret": self.app_secret, "tr_id": "FHKST01010100"
        }
        params = {"FID_COND_MRKT_DIV_CODE": "J", "FID_INPUT_ISCD": code}
        url = f"{self.base_url}/uapi/domestic-stock/v1/quotations/inquire-price"
        
        for i in range(3): 
            try:
                res = requests.get(url, headers=headers, params=params, timeout=3)
                if res.status_code == 200:
                    data = res.json()
                    if 'output' in data: return int(data['output']['stck_prpr'])
                elif res.status_code in [401, 403]:
                    self.get_access_token()
                    headers["authorization"] = f"Bearer {self.token}"
                    continue 
            except: pass
            time.sleep(0.5 * (2 ** i)) 
        return None

    def get_daily_chart(self, code):
        if not self.check_token(): return None
        now = datetime.datetime.now()
        start_dt = (now - datetime.timedelta(days=150)).strftime("%Y%m%d") 
        end_dt = now.strftime("%Y%m%d")
        headers = {
            "content-type": "application/json", "authorization": f"Bearer {self.token}",
            "appkey": self.app_key, "appsecret": self.app_secret, "tr_id": "FHKST01010400"
        }
        params = {
            "FID_COND_MRKT_DIV_CODE": "J", "FID_INPUT_ISCD": code,
            "FID_INPUT_DATE_1": start_dt, "FID_INPUT_DATE_2": end_dt,
            "FID_PERIOD_DIV_CODE": "D", "FID_ORG_ADJ_PRC": "1"
        }
        url = f"{self.base_url}/uapi/domestic-stock/v1/quotations/inquire-daily-itemchartprice"
        
        for i in range(3):
            try:
                res = requests.get(url, headers=headers, params=params, timeout=5)
                if res.status_code == 200:
                    data = res.json()
                    if 'output2' in data and data['output2']:
                        df = pd.DataFrame(data['output2'])
                        df = df.rename(columns={
                            'stck_bsop_date': 'Date', 'stck_oprc': 'Open', 'stck_hgpr': 'High',
                            'stck_lwpr': 'Low', 'stck_clpr': 'Close', 'acml_vol': 'Volume'
                        })
                        df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')
                        cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                        for c in cols: df[c] = pd.to_numeric(df[c])
                        return df.sort_values('Date').set_index('Date')
            except: pass
            time.sleep(0.5 * (2 ** i))
        return None

kis_client = None

# --- [Data Loader] ---
def get_naver_realtime_price(code):
    try:
        url = f"https://m.stock.naver.com/api/stock/{code}/basic"
        headers = {'User-Agent': 'Mozilla/5.0', 'Referer': 'https://m.stock.naver.com/'}
        res = requests.get(url, headers=headers, timeout=1.5)
        if res.status_code == 200:
            data = res.json()
            if 'closePrice' in data: return int(data['closePrice'].replace(',', ''))
    except: pass 
    return None

@st.cache_data(ttl=3600*12) 
def get_consensus_data(code):
    try:
        url = f"https://finance.naver.com/item/main.naver?code={code}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        res = requests.get(url, headers=headers, timeout=2)
        soup = BeautifulSoup(res.text, 'html.parser')
        wrapper = soup.find('div', {'id': 'content'})
        target_price = 0; opinion = 0.0
        if wrapper:
            r_width = wrapper.find_all('em')
            for em in r_width:
                if em.parent and "목표주가" in str(em.parent):
                    try: target_price = int(em.text.strip().replace(',', ''))
                    except: pass
                if em.parent and "투자의견" in str(em.parent):
                      try: opinion = float(em.text.strip())
                      except: pass
        return target_price, opinion
    except: return 0, 0.0

@st.cache_data(ttl=300) 
def get_data_safe(code, days=2000, interval="1d"):
    error_logs = []
    global kis_client
    if kis_client and kis_client.token and interval == "1d":
        try:
            df_kis = kis_client.get_daily_chart(code)
            if df_kis is not None and len(df_kis) >= 60:
                df_kis.attrs['source'] = "⚡ KIS (Premium)"
                return df_kis, None
            else: error_logs.append("KIS Data too short")
        except Exception as e: error_logs.append(f"KIS Error: {e}")
            
    if interval == "15m": days = 59 
    elif interval == "60m": days = 700 
    elif interval == "1w": days = 3650 
    start_date = (get_now_kst() - timedelta(days=days)).strftime('%Y-%m-%d')
    df = None; source = ""

    if interval == "1d":
        try:
            if code in ['KS11', 'KQ11']: df = fdr.DataReader(code, start_date)
            else: df = fdr.DataReader(code, start_date)
            if df is not None and not df.empty:
                source = "⚡ KRX (FDR)"; df = df.loc[:, ~df.columns.duplicated()]
        except Exception as e: error_logs.append(f"FDR: {e}")
    elif interval == "1w":
        try:
            df_d = fdr.DataReader(code, start_date)
            if df_d is not None and not df_d.empty:
                logic = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}
                df = df_d.resample('W-FRI').agg(logic).dropna(); source = "⚡ KRX (W)"
        except: pass

    if df is None or df.empty:
        yf_interval = interval if interval in ["1d","15m","60m"] else "1wk"
        for i in range(2): 
            try:
                time.sleep(0.3)
                yf_code = "^KS11" if code == 'KS11' else "^KQ11" if code == 'KQ11' else f"{code}.KS"
                df = yf.download(yf_code, start=start_date if interval=='1d' else None, 
                                 period=f"{days}d" if interval not in ['1d', '1w'] else None,
                                 interval=yf_interval, progress=False, threads=False)
                if not df.empty:
                    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
                    df = df.loc[:, ~df.columns.duplicated()]; source = f"⚡ Yahoo {interval}"; break
                if code not in ['KS11', 'KQ11']:
                    df = yf.download(f"{code}.KQ", start=start_date if interval=='1d' else None,
                                     period=f"{days}d" if interval not in ['1d', '1w'] else None,
                                     interval=yf_interval, progress=False, threads=False)
                    if not df.empty:
                        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
                        df = df.loc[:, ~df.columns.duplicated()]; source = f"⚡ Yahoo {interval}"; break
            except Exception as e: 
                if i == 1: error_logs.append(f"YF({interval}): {e}"); time.sleep(1)
    
    if df is not None and not df.empty and interval == "1d" and code not in ['KS11', 'KQ11']:
        real_price = None
        if kis_client:
            real_price = kis_client.get_current_price(code)
            if real_price: source += " + KIS-Live"
        if real_price is None:
            real_price = get_naver_realtime_price(code)
            if real_price: source += " + N-Patch"
        if real_price is not None:
            try:
                df.iloc[-1, df.columns.get_loc('Close')] = float(real_price)
                if real_price > df.iloc[-1]['High']: df.iloc[-1, df.columns.get_loc('High')] = float(real_price)
                if real_price < df.iloc[-1]['Low']: df.iloc[-1, df.columns.get_loc('Low')] = float(real_price)
            except: pass
        
    if df is not None:
        df.attrs['source'] = source
        return df, None
    return None, " / ".join(error_logs)

@st.cache_data(ttl=86400)
def get_safe_stock_listing():
    file_path = "krx_code_list.csv"
    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path, converters={'Code': str})
            if len(df) > 50 and 'Sector' in df.columns: 
                return df, "💾 Saved List"
            else: 
                os.remove(file_path)
        except: pass

    try:
        df = fdr.StockListing('KRX')
        if not df.empty and len(df) > 100:
            if 'Sector' not in df.columns: df['Sector'] = "Unknown"
            df.to_csv(file_path, index=False)
            return df, "⚡ KRX Live"
    except Exception as e: print(f"KRX Load Fail: {e}")

    try:
        time.sleep(1)
        df_k = fdr.StockListing('KOSPI')
        df_q = fdr.StockListing('KOSDAQ')
        df = pd.concat([df_k, df_q])
        if not df.empty and len(df) > 100:
            if 'Sector' not in df.columns: df['Sector'] = "Unknown"
            df = df.drop_duplicates(subset=['Code'])
            df.to_csv(file_path, index=False)
            return df, "⚡ Split Load"
    except: pass

    fb_data = [
        ['005930', '삼성전자', '전기전자', 400000000000000], ['000660', 'SK하이닉스', '전기전자', 100000000000000],
        ['373220', 'LG에너지솔루션', '전기전자', 90000000000000], ['207940', '삼성바이오로직스', '의약품', 50000000000000],
        ['005380', '현대차', '운수장비', 40000000000000], ['000270', '기아', '운수장비', 35000000000000],
        ['005490', 'POSCO홀딩스', '철강금속', 30000000000000], ['035420', 'NAVER', '서비스업', 25000000000000],
        ['006400', '삼성SDI', '전기전자', 20000000000000], ['051910', 'LG화학', '화학', 20000000000000],
        ['068270', '셀트리온', '의약품', 30000000000000], ['035720', '카카오', '서비스업', 20000000000000],
        ['105560', 'KB금융', '금융업', 20000000000000], ['028260', '삼성물산', '유통업', 20000000000000],
        ['012330', '현대모비스', '운수장비', 20000000000000], ['055550', '신한지주', '금융업', 18000000000000],
        ['066570', 'LG전자', '전기전자', 15000000000000], ['003670', '포스코퓨처엠', '전기전자', 15000000000000],
        ['096770', 'SK이노베이션', '석유화학', 13000000000000], ['032830', '삼성생명', '보험', 13000000000000]
    ]
    df_fb = pd.DataFrame(fb_data, columns=['Code', 'Name', 'Sector', 'Marcap'])
    return df_fb, "⚠️ Emergency List (20)"

@st.cache_data(ttl=3600)
def get_sector_performance_map(df_krx):
    sector_map = {}
    try:
        if 'Sector' not in df_krx.columns: return {}
        df_valid = df_krx[df_krx['Sector'].notna()]
        top_sectors = df_valid['Sector'].value_counts().head(30).index.tolist()
        for sector in top_sectors:
            top_stocks = df_valid[df_valid['Sector'] == sector].sort_values('Marcap', ascending=False).head(3)['Code'].tolist()
            changes = []
            for code in top_stocks:
                d, _ = get_data_safe(code, 5)
                if d is not None and len(d) >= 2:
                    curr = d['Close'].iloc[-1]; prev = d['Close'].iloc[-2]
                    changes.append((curr - prev) / prev * 100)
            if changes: sector_map[sector] = sum(changes) / len(changes)
    except Exception as e: print(f"Sector Analysis Error: {e}")
    return sector_map

# [V81.58 Fix] Decorator: Return empty DataFrame instead of None on failure
def retry_gsheets(func):
    def wrapper(*args, **kwargs):
        for i in range(3):
            try: return func(*args, **kwargs)
            except: time.sleep(1)
        return pd.DataFrame() # Return empty DF to prevent NoneType error
    return wrapper

@retry_gsheets
def get_portfolio_gsheets():
    conn = st.connection("gsheets", type=GSheetsConnection)
    df = conn.read(worksheet="portfolio", ttl="0")
    if df is not None and not df.empty:
        df.columns = [str(c).strip().replace(" ", "_") for c in df.columns]
        rename_map = {'코드':'Code','종목코드':'Code','Code':'Code','종목명':'Name','Name':'Name','평단가':'Buy_Price','Buy_Price':'Buy_Price','수량':'Qty','Qty':'Qty'}
        df = df.rename(columns=rename_map)
        if 'Code' in df.columns:
            df = df.dropna(subset=['Code'])
            df['Code'] = df['Code'].astype(str).str.split('.').str[0].str.zfill(6)
            df['Buy_Price'] = pd.to_numeric(df['Buy_Price'], errors='coerce').fillna(0)
            df['Qty'] = pd.to_numeric(df['Qty'], errors='coerce').fillna(0)
            return df[['Code', 'Name', 'Buy_Price', 'Qty']]
    return pd.DataFrame(columns=['Code', 'Name', 'Buy_Price', 'Qty'])

def save_bulk_results(items):
    if not items: return
    new_df = pd.DataFrame([{
        "Date": get_now_kst().strftime('%Y-%m-%d'),
        "Code": str(item['code']).zfill(6), 
        "Name": item['name'],
        "Entry_Price": item['strat']['buy'][0][0], 
        "Target_Price": item['strat']['sell'][0][0],
        "Stop_Price": item['strat']['final_stop'][0], 
        "Strategy": item['strat']['logic'],
        "Buys_Info": json.dumps([b[0] for b in item['strat']['buy']]),
        "Sells_Info": json.dumps([s[0] for s in item['strat']['sell']])
    } for item in items])
    for i in range(3):
        try:
            conn = st.connection("gsheets", type=GSheetsConnection)
            try:
                existing_df = conn.read(worksheet="history", ttl="0")
                if existing_df is not None and not existing_df.empty:
                    existing_df['Code'] = existing_df['Code'].astype(str).str.zfill(6)
                    existing_df['Date'] = existing_df['Date'].astype(str)
                    if 'Buys_Info' not in existing_df.columns: existing_df['Buys_Info'] = "[]"; existing_df['Sells_Info'] = "[]"
                    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                    final_df = combined_df.drop_duplicates(subset=['Date', 'Code'], keep='last')
                else: final_df = new_df
                conn.update(worksheet="history", data=final_df)
            except: conn.update(worksheet="history", data=new_df)
            break
        except: time.sleep(1)

@retry_gsheets
def get_scan_history():
    conn = st.connection("gsheets", type=GSheetsConnection)
    df = conn.read(worksheet="history", ttl="0")
    if df is not None and not df.empty and 'Date' in df.columns:
        df['Code'] = df['Code'].astype(str).str.split('.').str[0].str.zfill(6) 
        return df
    return pd.DataFrame(columns=['Date', 'Code', 'Name', 'Entry_Price', 'Target_Price', 'Stop_Price', 'Strategy'])

def analyze_market_condition(idx_code):
    # 1. 기술적 분석 (기존 유지)
    df, _ = get_data_safe(idx_code, days=300)
    tech_score = 0
    adx = 0
    if df is not None and len(df) >= 60:
        close = df['Close']; ma20 = close.rolling(20).mean(); ma60 = close.rolling(60).mean()
        tr1 = df['High'] - df['Low']; tr2 = (df['High'] - df['Close'].shift(1)).abs(); tr3 = (df['Low'] - df['Close'].shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1); atr = tr.rolling(14).mean()
        up_move = df['High'] - df['High'].shift(1); down_move = df['Low'].shift(1) - df['Low']
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        plus_di = 100 * (pd.Series(plus_dm).rolling(14).mean() / atr)
        minus_di = 100 * (pd.Series(minus_dm).rolling(14).mean() / atr)
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)
        adx = dx.rolling(14).mean().iloc[-1]
        
        curr = close.iloc[-1]
        if curr > ma20.iloc[-1] and ma20.iloc[-1] > ma60.iloc[-1]: tech_score = -10 
        elif curr < ma20.iloc[-1] and ma20.iloc[-1] < ma60.iloc[-1]: tech_score = 15 
        
    # 2. 매크로 분석 (에러 방지 강화)
    macro_score = 0
    macro_msg = []
    
    # [수정됨] 야후 파이낸스 호출 안정화
    try:
        # threads=False로 설정하여 차단 확률 낮춤
        usd_data = yf.download("KRW=X", period="5d", progress=False, threads=False)
        if not usd_data.empty:
            if isinstance(usd_data.columns, pd.MultiIndex): usd_data.columns = usd_data.columns.get_level_values(0)
            usd_krw = float(usd_data['Close'].iloc[-1])
            if usd_krw > 1400: macro_score += 10; macro_msg.append(f"환율주의({int(usd_krw)})")
            
        bond_data = yf.download("^TNX", period="5d", progress=False, threads=False)
        if not bond_data.empty:
            if isinstance(bond_data.columns, pd.MultiIndex): bond_data.columns = bond_data.columns.get_level_values(0)
            us_bond = float(bond_data['Close'].iloc[-1])
            if us_bond > 4.5: macro_score += 5; macro_msg.append(f"금리부담({us_bond:.1f}%)")
            
    except Exception as e:
        # 에러 발생 시 로그만 남기고 0점 처리 (앱 멈춤 방지)
        print(f"Macro Data Error: {e}") 

    final_score = tech_score + macro_score
    status_txt = f"Tech:{tech_score} + Macro:{macro_score}"
    if macro_msg: status_txt += f" ({', '.join(macro_msg)})"
    status_color = "#4caf50" if final_score <= 0 else "#f44336" if final_score >= 10 else "#ff9800"
    return final_score, status_txt, status_color

def get_ai_condition():
    k_score, k_stat, k_col = analyze_market_condition("KS11")
    q_score, q_stat, q_col = analyze_market_condition("KQ11")
    final_penalty = max(k_score, q_score)
    market_msg = f"KOSPI:{k_stat} / KOSDAQ:{q_stat}"
    if final_penalty <= -5: status = f"🚀 공격 모드 (기준 {final_penalty} 완화) - {market_msg}"
    elif final_penalty >= 10: status = f"🛡️ 방어 모드 (기준 +{final_penalty} 상향) - {market_msg}"
    else: status = f"⚖️ 균형 모드 (기준 +{final_penalty} 조정) - {market_msg}"
    return final_penalty, status, k_stat

def send_telegram_msg(token, chat_id, message):
    if token and chat_id and message:
        try: requests.post(f"https://api.telegram.org/bot{token}/sendMessage", json={"chat_id": chat_id, "text": message, "parse_mode": "HTML"}, timeout=5)
        except: pass

def calc_stoch(df, n, m, t):
    l = df['Low'].rolling(n).min(); h = df['High'].rolling(n).max()
    return ((df['Close'] - l) / (h - l + 1e-9) * 100).rolling(m).mean().rolling(t).mean()

def get_all_indicators(df):
    if df is None or len(df) < 5: return None 
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    df = df.loc[:, ~df.columns.duplicated()]
    close = df['Close']; high = df['High']; low = df['Low']; vol = df['Volume']
    
    df['MA5'] = close.rolling(5).mean(); df['MA10'] = close.rolling(10).mean(); df['MA20'] = close.rolling(20).mean()
    df['MA60'] = close.rolling(60).mean(); df['MA120'] = close.rolling(120).mean(); df['MA200'] = close.rolling(200).mean()
    k = close.ewm(span=12, adjust=False).mean(); d = close.ewm(span=26, adjust=False).mean()
    df['MACD'] = k - d; df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    
    p9_high = high.rolling(9).max(); p9_low = low.rolling(9).min()
    df['Ichi_Tenkan'] = (p9_high + p9_low) / 2
    p26_high = high.rolling(26).max(); p26_low = low.rolling(26).min()
    df['Ichi_Kijun'] = (p26_high + p26_low) / 2
    df['Ichi_SpanA'] = ((df['Ichi_Tenkan'] + df['Ichi_Kijun']) / 2).shift(26)
    p52_high = high.rolling(52).max(); p52_low = low.rolling(52).min()
    df['Ichi_SpanB'] = ((p52_high + p52_low) / 2).shift(26)
    df['Kumo_Top'] = df[['Ichi_SpanA', 'Ichi_SpanB']].max(axis=1)
    df['Kumo_Bot'] = df[['Ichi_SpanA', 'Ichi_SpanB']].min(axis=1)
    
    recent_high = high.rolling(60).max(); recent_low = low.rolling(60).min(); diff = recent_high - recent_low
    df['Fibo_0.382'] = recent_high - (diff * 0.382); df['Fibo_0.5'] = recent_high - (diff * 0.5); df['Fibo_0.618'] = recent_high - (diff * 0.618)
    
    is_red = close.shift(1) < df['Open'].shift(1); is_green = close > df['Open']
    engulfing = (close > df['Open'].shift(1)) & (df['Open'] < close.shift(1))
    vol_up = vol > vol.rolling(20).mean()
    df['OB_Bull'] = 0; mask_ob = is_red & is_green & engulfing & vol_up
    df.loc[mask_ob, 'OB_Bull'] = df['Open'].shift(1)
    df['OB_Support'] = df['OB_Bull'].replace(0, np.nan).ffill(limit=10).fillna(0)
    
    # Stochastic Slow 로직 적용
    df['Stoch_5'] = calc_stoch(df, 5, 3, 3); df['Stoch_10'] = calc_stoch(df, 10, 6, 6); df['Stoch_20'] = calc_stoch(df, 20, 12, 12)
    
    tr1 = high - low; tr2 = (high - close.shift(1)).abs(); tr3 = (low - close.shift(1)).abs()
    df['ATR'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).rolling(14).mean()
    tp = (high + low + close) / 3
    df['MVWAP'] = (tp * vol).rolling(20).sum() / (vol.rolling(20).sum() + 1e-9)
    ma_bb = close.rolling(20).mean(); std_bb = close.rolling(20).std()
    df['BB_Up'] = ma_bb + (std_bb * 2); df['BB_Lo'] = ma_bb - (std_bb * 2)
    df['BB_Pos'] = (close - df['BB_Lo']) / (df['BB_Up'] - df['BB_Lo'] + 1e-9)
    delta = close.diff(); g = delta.where(delta>0,0).rolling(14).mean(); l_idx = -delta.where(delta<0,0).rolling(14).mean()
    df['RSI'] = 100 - (100/(1+(g/(l_idx+1e-9))))
    mad = tp.rolling(14).apply(lambda x: (x - x.mean()).abs().mean())
    df['CCI'] = (tp - tp.rolling(14).mean()) / (0.015 * mad + 1e-9)
    raw_mf = tp * vol; pos_mf = raw_mf.where(tp > tp.shift(1), 0).rolling(14).sum(); neg_mf = raw_mf.where(tp < tp.shift(1), 0).rolling(14).sum()
    df['MFI'] = 100 - (100 / (1 + (pos_mf / (neg_mf + 1e-9))))
    dm_pos = high.diff().clip(lower=0); dm_neg = -low.diff().clip(upper=0)
    di_pos = 100 * (dm_pos.ewm(alpha=1/14).mean() / df['ATR'])
    di_neg = 100 * (dm_neg.ewm(alpha=1/14).mean() / df['ATR'])
    df['ADX'] = (100 * abs(di_pos - di_neg) / (di_pos + di_neg + 1e-9)).rolling(14).mean()
    df['Vol_Z'] = (vol - vol.rolling(20).mean()) / (vol.rolling(20).std() + 1e-9)
    df['ER'] = close.diff(10).abs() / (close.diff().abs().rolling(10).sum() + 1e-9)
    df['Rel_Close'] = df['Close'].pct_change() * 100
    return df

def get_market_trend(code, name):
    d, _ = get_data_safe(code, 200)
    if d is not None:
        d = get_all_indicators(d)
        if d is not None:
            curr = d.iloc[-1]; cp = curr['Close']; ma20 = curr['MA20']; ma60 = curr['MA60']
            if abs(cp - ma20) / ma20 < 0.005: return f"🦀 {name}: 횡보/혼조", "#ff9800"
            if cp > ma20:
                if ma20 > ma60: return f"🔥 {name}: 대세상승", "#d32f2f"
                else: return f"🔺 {name}: 상승세", "#f44336"
            else:
                if ma20 < ma60: return f"❄️ {name}: 대세하락", "#1976d2"
                else: return f"💧 {name}: 하락세", "#2196f3"
    return f"❓ {name}: 데이터없음", "gray"

@st.cache_data(ttl=1800)
def get_benchmark_data(days=2600):
    ks11, _ = get_data_safe('KS11', days)
    kq11, _ = get_data_safe('KQ11', days)
    return ks11, kq11

# [V81.58 Patch] 학습 로직 (Safe Guard: NoneType + 컷오프)
def train_global_model(stock_list, limit=50, mode="update"):
    all_X = []; all_y = []
    collected_dfs = [] 
    
    features = ['RSI', 'Stoch_20', 'CCI', 'MFI', 'ADX', 'Vol_Z', 'BB_Pos', 'ER', 'Rel_Close', 'KOSPI_Trend']
    status_text = st.empty(); progress_bar = st.progress(0)
    
    ks11_df, kq11_df = get_benchmark_data()
    if ks11_df is None or kq11_df is None: return False, "지수 데이터 로딩 실패"
    
    for b_df in [ks11_df, kq11_df]:
        b_df.index = pd.to_datetime(b_df.index)
        b_df['Idx_Chg'] = b_df['Close'].pct_change() * 100
    
    trend_map_ks = ks11_df['Close'].rolling(20).mean().to_dict() 
    
    if mode == "initial": targets = stock_list.head(limit)['Code'].tolist(); days_to_fetch = 730
    elif mode == "full_initial": targets = stock_list.head(300)['Code'].tolist(); days_to_fetch = 730
    else: targets = stock_list.head(limit)['Code'].tolist(); days_to_fetch = 15

    success_count = 0; total_targets = len(targets)
    print(f"=== 학습 시작: 대상 {total_targets}개 ===") 

    with ThreadPoolExecutor(max_workers=2) as ex: 
        fut_map = {ex.submit(get_data_safe, code, days_to_fetch): code for code in targets}
        for i, fut in enumerate(as_completed(fut_map)):
            code = fut_map[fut]
            try:
                result = fut.result()
                if not result: continue 
                
                df, _ = result 
                if df is None: continue 
                if df.empty: continue 
                if len(df) <= 60: continue 

                # 성능 최적화: 거래대금 컷오프
                if 'Close' not in df.columns or 'Volume' not in df.columns: continue
                avg_amt = (df['Close'] * df['Volume']).rolling(5).mean().iloc[-1]
                if avg_amt < 1000000000: continue 

                df.index = pd.to_datetime(df.index)
                df = get_all_indicators(df)
                
                if df is not None:
                    if len(collected_dfs) < 200: collected_dfs.append(df.copy())
                    
                    market_type = 'KQ' if code not in ['005930'] and int(code) > 100000 else 'KS' 
                    benchmark = kq11_df if market_type == 'KQ' else ks11_df
                    
                    aligned_idx = benchmark['Idx_Chg'].reindex(df.index).fillna(0)
                    
                    df['Idx_Chg'] = aligned_idx
                    df['Stock_Chg'] = df['Close'].pct_change() * 100
                    df['Rel_Close'] = df['Stock_Chg'] - df['Idx_Chg']
                    
                    ma20_series = pd.Series(trend_map_ks).reindex(df.index).ffill()
                    ks_close = ks11_df['Close'].reindex(df.index).ffill()
                    df['KOSPI_Trend'] = (ks_close > ma20_series).astype(int)

                    data_ml = df[features].copy().dropna()
                    future_close = df['Close'].shift(-5)
                    target = (future_close > df['Close'] * 1.02).astype(int)
                    common_idx = data_ml.index.intersection(target.index[:-5])
                    
                    if len(common_idx) > 10:
                        all_X.append(data_ml.loc[common_idx])
                        all_y.append(target.loc[common_idx])
                        success_count += 1
            except Exception as e:
                print(f"Error processing {code}: {e}") 
                
            if i % 10 == 0:
                progress_bar.progress((i + 1) / total_targets)
                status_text.text(f"📥 수집 중... ({success_count}/{total_targets} 성공)")
                gc.collect() 

    if not all_X: 
        print("!!! 데이터 수집 실패: all_X가 비어있음 !!!")
        return False, "데이터 수집 실패 (0건)"
    
    status_text.text("💾 데이터 병합 및 학습 시작...")
    X_new = pd.concat(all_X).sort_index(); y_new = pd.concat(all_y).sort_index()
    del all_X, all_y; gc.collect()
    
    print(f"학습 데이터 크기: {len(X_new)} rows") 

    xgb_model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=5, n_jobs=1, random_state=42)
    xgb_model.fit(X_new, y_new)
    
    rf = RandomForestClassifier(n_estimators=50, max_depth=6, random_state=42, n_jobs=1, oob_score=True).fit(X_new, y_new)
    
    status_text.text(f"🧠 LSTM 학습 중... ({len(collected_dfs)}개)")
    if collected_dfs:
        lstm_res, lstm_msg = lstm_engine.train_and_save(collected_dfs)
    else:
        lstm_res = False
    del collected_dfs; gc.collect()
    
    model_data = {
        "xgb": xgb_model, "rf": rf, "date": get_now_kst().strftime('%Y-%m-%d'), 
        "sample_size": len(targets), "feature_names": features,
        "feature_importance": xgb_model.feature_importances_, 
        "oob_score": rf.oob_score_ if hasattr(rf, 'oob_score_') else 0,
        "lstm_status": lstm_res
    }
    joblib.dump(model_data, MODEL_FILE)
    print(f"파일 저장 완료: {MODEL_FILE}") 
    return True, f"학습 완료! (총 {success_count}개 종목 성공)"

@st.cache_resource
def load_ai_model():
    if os.path.exists(MODEL_FILE):
        try: return joblib.load(MODEL_FILE)
        except: pass
    return None

def get_ai_score_fast(df, market_code='KS11'):
    features = ['RSI', 'Stoch_20', 'CCI', 'MFI', 'ADX', 'Vol_Z', 'BB_Pos', 'ER', 'Rel_Close', 'KOSPI_Trend']
    model_data = load_ai_model()
    if model_data is None: return 50 
    
    try:
        if 'Rel_Close' not in df.columns: df['Rel_Close'] = df['Close'].pct_change() * 100 
        if 'KOSPI_Trend' not in df.columns: df['KOSPI_Trend'] = 1 
        data_ml = df[features].iloc[-1:].fillna(0)
        p_xgb = model_data['xgb'].predict_proba(data_ml)[0][1]
        p_rf = model_data['rf'].predict_proba(data_ml)[0][1]
        lstm_score = lstm_engine.predict_score(df) / 100.0
        final_prob = (p_xgb * 0.5) + (p_rf * 0.2) + (lstm_score * 0.3)
        return int(final_prob * 100)
    except Exception as e:
        return 50

# [V80.74] Updated Rebalancing Logic
def analyze_rebalancing_suggestion(pf_list):
    if not pf_list: return []
    suggestions = []
    for p in pf_list:
        score = p['score']; profit = p['profit_pct']; vol = p['vol']
        action = "유지 (Hold)"; color = "black"; reason = "특이사항 없음 (관망)"
        if score >= 80:
            if profit > 0:
                action = "🚀 불타기/보유 (Let Profit Run)"; color = "#2e7d32" 
                reason = f"수익 중({profit:.1f}%)이며 상승 여력(AI:{score})도 높음. 수익 극대화."
            else:
                action = "🟢 물타기/비중확대 (Add)"; color = "green"
                reason = f"현재 손실이나 AI 확신(AI:{score})이 강함. 저점 매수 기회."
        elif score < 50:
            if profit > 3.0:
                action = "💰 익절 (Take Profit)"; color = "#fbc02d" 
                reason = f"수익({profit:.1f}%) 확보 권장. 상승 탄력(AI:{score}) 둔화됨."
            elif profit < -3.0:
                if vol > 0.4:
                    action = "🔴 교체 매매 (Swap)"; color = "red"
                    reason = "손실 중이며 변동성 위험 높음. 기회비용 고려 교체."
                else:
                    action = "🟡 비중 축소 (Reduce)"; color = "#f57f17" 
                    reason = "상승 모멘텀 부족. 현금화 후 대기."
            else:
                action = "🟡 매도 후 관망"; color = "#f57f17"
                reason = "탄력 둔화. 재미없는 흐름 예상."
        else:
            if profit > 5.0:
                action = "🛡️ 수익 실현/홀딩"; color = "blue"
                reason = "안정적 흐름. 일부 실현 후 나머지는 추세 추종."
        suggestions.append({"name": p['name'], "action": action, "color": color, "reason": reason, "score": score, "profit": profit})
    return sorted(suggestions, key=lambda x: x['score'], reverse=True)

def analyze_supply(df):
    supply_bonus = 0; supply_msg = []
    curr = df.iloc[-1]; vol_avg = df['Volume'].rolling(20).mean().iloc[-1]
    if curr['Close'] > curr['Open'] and curr['Volume'] > vol_avg * 1.5:
        supply_bonus += 5; supply_msg.append("거래량폭발")
    if curr['Low'] > curr['MVWAP']:
        supply_bonus += 5; supply_msg.append("세력평단위")
    return supply_bonus, supply_msg

def analyze_patterns(df):
    pattern_score = 0; pattern_msg = []
    if len(df) < 60: return 0, []
    close = df['Close'].values
    peaks, _ = find_peaks(-close[-60:], distance=10)
    if len(peaks) >= 3:
        p1, p2, p3 = peaks[-3:]
        v1, v2, v3 = close[-60:][p1], close[-60:][p2], close[-60:][p3]
        if v2 < v1 and v2 < v3 and v3 > v2: pattern_score += 15; pattern_msg.append("역헤드앤숄더")
    peaks_high, _ = find_peaks(close[-60:], distance=8); peaks_low, _ = find_peaks(-close[-60:], distance=8)
    if len(peaks_low) >= 2 and len(peaks_high) >= 1:
        last_low = close[-60:][peaks_low[-1]]; prev_low = close[-60:][peaks_low[-2]]; last_high = close[-60:][peaks_high[-1]]
        if last_low > prev_low and close[-1] > last_low:
            if close[-1] < last_high: pattern_score += 5; pattern_msg.append("상승N자패턴")
            elif close[-1] > last_high: pattern_score += 20; pattern_msg.append("엘리어트3파")
    return pattern_score, pattern_msg

def analyze_advanced_features(df):
    bonus = 0; msgs = []
    if len(df) < 20: return 0, []
    curr = df.iloc[-1]
    if curr['Close'] > curr['MA60']:
        recent = df.tail(3)
        if (recent['Close'].iloc[-1] < recent['Close'].iloc[-2]): 
            vol_mean = df['Volume'].rolling(20).mean().iloc[-1]
            if curr['Volume'] < vol_mean * 0.7: bonus += 10; msgs.append("📉건전한눌림")
    prev_low_idx = df['Low'].iloc[-10:-1].idxmin() 
    if pd.notna(prev_low_idx):
        prev_low_val = df.loc[prev_low_idx, 'Low']; prev_rsi_val = df.loc[prev_low_idx, 'RSI']
        if curr['Low'] < prev_low_val: 
            if curr['RSI'] > prev_rsi_val: bonus += 15; msgs.append("✨상승다이버전스")
    return bonus, msgs

def analyze_mtf_comprehensive(code, daily_score):
    mtf_bonus = 0; mtf_msg = []
    dw, _ = get_data_safe(code, interval="1w")
    if dw is not None:
        dfw = get_all_indicators(dw)
        if dfw is not None and len(dfw) > 20:
            curw = dfw.iloc[-1]
            if curw['Close'] > curw['MA20']: mtf_bonus += 10; mtf_msg.append("주봉상승")
            if curw['MACD_Hist'] > 0 and curw['MACD_Hist'] > dfw.iloc[-2]['MACD_Hist']: mtf_bonus += 5; mtf_msg.append("주봉MACD개선")
    d60, _ = get_data_safe(code, interval="60m")
    if d60 is not None:
        df60 = get_all_indicators(d60)
        if df60 is not None and len(df60) > 20:
            cur60 = df60.iloc[-1]
            if cur60['Close'] > cur60['MA20']: mtf_bonus += 5; mtf_msg.append("60m상승")
            if cur60['Stoch_10'] < 30 and cur60['Stoch_10'] > df60.iloc[-2]['Stoch_10']: mtf_bonus += 3; mtf_msg.append("60m반등")
    d15, _ = get_data_safe(code, interval="15m")
    if d15 is not None:
        df15 = get_all_indicators(d15)
        if df15 is not None and len(df15) > 20:
            cur15 = df15.iloc[-1]; prev15 = df15.iloc[-2]
            if cur15['MA5'] > cur15['MA20'] and prev15['MA5'] <= prev15['MA20']: mtf_bonus += 7; mtf_msg.append("15m골크")
            if cur15['Stoch_5'] < 20 and cur15['Stoch_5'] > prev15['Stoch_5']: mtf_bonus += 5; mtf_msg.append("15m타점")
    return mtf_bonus, mtf_msg

def calc_recovery_math(buy_price, curr_price, volatility):
    if buy_price <= 0 or curr_price <= 0: return None
    if curr_price >= buy_price: return None
    loss_amt = buy_price - curr_price
    loss_pct = (loss_amt / buy_price) * 100
    req_return = (loss_amt / curr_price) * 100
    t = 60 / 252; dist = np.log(buy_price / curr_price); drift = 0.05 
    z = (dist - (drift - 0.5 * volatility**2) * t) / (volatility * np.sqrt(t))
    prob_3m = (1 - norm.cdf(z)) * 100
    return {"loss_pct": loss_pct, "req_return": req_return, "prob_3m": prob_3m, "volatility": volatility * 100}

def calc_reach_prob(curr_price, target_price, atr, ai_score, type='target', base_days=5):
    if curr_price <= 0: return 0
    dist = abs(target_price - curr_price)
    ai_factor = ai_score / 50.0 
    if type == 'stop': days_expected = base_days / (ai_factor if ai_factor > 0 else 1)
    else: days_expected = base_days * (ai_factor if ai_factor >= 1 else 1)
    expected_range = atr * (days_expected ** 0.5) 
    if dist == 0: return 100
    ratio = dist / (expected_range + 1e-9)
    raw_prob = np.exp(-1.0 * ratio) * 100 
    final_prob = min(99, int(raw_prob))
    if ai_score >= 80 and type == 'sell': final_prob = max(final_prob, int(30 + (ai_score-80)))
    return max(1, final_prob)

def determine_best_horizon(df):
    curr = df.iloc[-1]
    if curr['Vol_Z'] > 2.0 or (curr['Stoch_5'] < 20 and curr['Stoch_5'] > df.iloc[-2]['Stoch_5']): return 'short', "⚡단타"
    if curr['Close'] > curr['MA200'] and curr['MA60'] > curr['MA120']: return 'long', "🌳장기"
    return 'swing', "🌊스윙"

def calculate_sizing(score, curr_price, min_invest, max_invest, ai_prob):
    if score >= 80 and ai_prob >= 60: allocation = max_invest
    elif score >= 80 and ai_prob < 40: allocation = min_invest
    elif score >= 60: allocation = (min_invest + max_invest) / 2
    else: allocation = min_invest
    if curr_price <= 0: return 0, 0, 0, 0
    q1 = int((allocation * 0.3) // curr_price); q2 = int((allocation * 0.3) // curr_price); q3 = int((allocation * 0.4) // curr_price)
    return q1, q2, q3, int(allocation)

def analyze_portfolio_action(score, ai_prob, loss_pct, rsi):
    action_txt = "관망"; action_col = "black"; tag = "Hold"
    if score >= 80:
        if loss_pct < 0: action_txt = "💧물타기 추천"; action_col = "green"; tag = "Add"
        else: action_txt = "🔥불타기 가능"; action_col = "#2e7d32"; tag = "BuyMore"
    elif score < 40:
        if loss_pct < -5: action_txt = "✂️손절/교체 검토"; action_col = "red"; tag = "Cut"
        elif loss_pct > 3: action_txt = "💰익절 권장"; action_col = "#fbc02d"; tag = "Profit"
    return action_txt, action_col, tag

# [V81.10 Update] 이안 트레이더 패치
def get_darwin_strategy(df, buy_price=0, code=None, use_mtf=False, min_inv=3000000, max_inv=5000000, market_status="Neutral", sec_score=0):
    if df is None or len(df) < 100: return None
    
    curr = df.iloc[-1]
    prev = df.iloc[-2]
    cp = curr['Close']
    atr = curr['ATR']
    ai_prob = get_ai_score_fast(df)
    
    score = 0
    hit_reasons = []
    logic_mode = "⚖️ Balanced"
    alpha_stock = False
    breakout = False
    
    horizon, horizon_tag = determine_best_horizon(df)
    is_bear = "하락" in market_status or "눌림" in market_status
    if is_bear:
        if curr['Close'] > curr['MA20'] and curr.get('Rel_Close', 0) > 2.0:
            score += 20; hit_reasons.append("🛡️하락장방어"); alpha_stock = True; logic_mode = "🐻 Crisis Hunter"
        else: score -= 15 

    vol_mean = df['Volume'].rolling(20).mean().iloc[-1]
    is_vol_explosive = curr['Volume'] > vol_mean * 1.5

    if sec_score > 1.0: 
        if is_vol_explosive:
            score += 25
            hit_reasons.append(f"🔥주도섹터+수급폭발({sec_score:.1f}%)")
        else:
            score += 10
            hit_reasons.append(f"🏭주도섹터({sec_score:.1f}%)")
    elif sec_score < -1.0: score -= 10

    if cp > curr['Kumo_Top']:
        if prev['Close'] <= prev['Kumo_Top']:
            score += 25; hit_reasons.append("☁️구름대강력돌파"); breakout = True
        else:
            score += 10; hit_reasons.append("☁️구름대위(정배열)")
    elif cp < curr['Kumo_Bot']:
        score -= 10
    
    rng = prev['High'] - prev['Low']; breakout_price = curr['Open'] + (rng * 0.5)
    if cp > breakout_price: score += 20; hit_reasons.append("💥변동성돌파"); breakout = True

    consensus_info = {"price": 0, "upside": 0, "prob": 0, "opinion": 0.0}
    if code:
        target_price_consensus, opinion_score = get_consensus_data(code)
        if target_price_consensus > 0:
            upside_potential = (target_price_consensus - cp) / cp * 100
            consensus_prob = calc_reach_prob(cp, target_price_consensus, atr, ai_prob, type='target', base_days=120)
            consensus_info = {"price": target_price_consensus, "upside": upside_potential, "prob": consensus_prob, "opinion": opinion_score}
            if upside_potential > 10: score += 10; hit_reasons.append(f"🎯목표가괴리({upside_potential:.1f}%)")
            if opinion_score >= 3.8: score += 5; hit_reasons.append(f"👍기관강력매수({opinion_score})")

    if cp > curr['MA200']: score += 10; hit_reasons.append("📈장기정배열")
    if cp >= curr['MVWAP']: score += 10; hit_reasons.append("기관수급")
    if ai_prob >= 70: score += 20; hit_reasons.append(f"🤖AI확신({ai_prob}%)")
    
    if curr['MACD_Hist'] > 0 and prev['MACD_Hist'] <= 0: score += 15; hit_reasons.append("🌊MACD반전")
    elif curr['MACD'] > curr['MACD_Signal'] and curr['MACD'] > 0: score += 5
    
    if curr['Fibo_0.618'] <= cp <= curr['Fibo_0.5'] * 1.02: score += 20; hit_reasons.append("✨황금비율지지")
    if curr['OB_Support'] > 0 and abs(cp - curr['OB_Support']) / cp < 0.03: score += 20; hit_reasons.append("🧱오더블럭지점")

    s5, s10, s20 = curr['Stoch_5'], curr['Stoch_10'], curr['Stoch_20']
    
    is_strong_trend = (curr.get('ADX', 0) >= 25) and (curr['Close'] > curr['MA20']) and (curr['MA20'] > curr['MA60'])
    bull_ride_triggered = False

    if is_strong_trend:
        if 35 <= s5 <= 65 and s5 > prev['Stoch_5']:
            score += 30
            hit_reasons.append(f"🚀강세눌림목(ADX:{curr.get('ADX',0):.1f})")
            logic_mode = "🐂 Bull Ride" 
            bull_ride_triggered = True

    if not bull_ride_triggered:
        if s5 < 25 and s10 < 25 and s20 < 30:
            if s5 > prev['Stoch_5']: score += 40; hit_reasons.append("💎대바닥반등"); logic_mode = "🛡️ Sniper"
        elif s20 > 50 and s5 < 20: 
            score += 35; hit_reasons.append("⚡상승중눌림목"); logic_mode = "🐆 Hunter"

    sup_score, sup_msg = analyze_supply(df); score += sup_score; hit_reasons.extend(sup_msg)
    pat_score, pat_msg = analyze_patterns(df); score += pat_score; pattern_reasons = pat_msg 
    adv_score, adv_msg = analyze_advanced_features(df); score += adv_score; pattern_reasons.extend(adv_msg)

    mtf_reasons = []
    if use_mtf and code and score >= 40:
        mtf_score, mtf_msgs = analyze_mtf_comprehensive(code, score)
        score += mtf_score; mtf_reasons = mtf_msgs
        if mtf_score > 0: hit_reasons.append(f"⏱️MTF가산({mtf_score})")

    whipsaw_warnings = []
    if curr['RSI'] > 80: whipsaw_warnings.append("RSI과열")
    if curr['Stoch_20'] > 90: whipsaw_warnings.append("스토캐과열")
    if cp > curr['MA20'] and curr['Volume'] < vol_mean * 0.4: whipsaw_warnings.append("거래량부족")

    def adj(p):
        if np.isnan(p) or p <= 0: return 0
        t = 1 if p<2000 else 5 if p<5000 else 10 if p<20000 else 50 if p<50000 else 100 if p<200000 else 500
        return int(round(p/t)*t)

    pp = (prev['High'] + prev['Low'] + prev['Close']) / 3
    s1 = (2 * pp) - prev['High']
    low_60 = df['Low'].tail(60).min(); low_120 = df['Low'].tail(120).min()
    
    support_candidates = []
    calc_days = 5; gap_mul = 0.8
    if horizon == 'short':
        support_candidates = [(curr['MA5'], "5일선"), (curr['MA10'], "10일선"), (curr['MA20'], "20일선"), (curr['BB_Lo'], "BB하단"), (s1, "피봇S1")]
        calc_days = 3; gap_mul = 0.5
    elif horizon == 'long':
        support_candidates = [(curr['MA60'], "60일선"), (curr['MA120'], "120일선"), (curr['MA200'], "200일선"), (curr['Fibo_0.618'], "Fibo 0.618")]
        calc_days = 40; gap_mul = 2.0
    else: 
        support_candidates = [(curr['MA20'], "20일선"), (curr['MA60'], "60일선"), (curr['Fibo_0.5'], "Fibo 0.5")]
        calc_days = 10; gap_mul = 1.0

    valid_buys = []
    for p, label in support_candidates:
        if 0 < p <= cp: valid_buys.append((p, label))
    valid_buys.sort(key=lambda x: x[0], reverse=True) 

    final_buys = []
    min_gap = atr * gap_mul
    if valid_buys:
        b1 = valid_buys[0]
        if (cp - b1[0]) < atr * 0.1: 
             if len(valid_buys) > 1: b1 = valid_buys[1]
        final_buys.append((adj(b1[0]), b1[1]))
    else: final_buys.append((cp, "현재가"))

    last_buy = final_buys[0][0]
    for p, label in valid_buys:
        if len(final_buys) >= 3: break
        if last_buy - p >= min_gap: final_buys.append((adj(p), label)); last_buy = p
    while len(final_buys) < 3:
        ref = final_buys[-1][0]; final_buys.append((adj(ref * 0.97), "지지선없음"))

    if bull_ride_triggered:
        final_buys.insert(0, (adj(cp), "🚀시장가진입"))
        if len(final_buys) > 3: final_buys.pop()

    resist_candidates = [
        (curr['BB_Up'], "BB상단"), (curr['MA120'], "120선"), (curr['MA200'], "200선"),
        (cp + atr*2.5 * (2 if horizon=='long' else 1), "ATR목표"),
        (curr['Fibo_0.382'] + (curr['Fibo_0.382']-curr['Fibo_0.618']), "Fibo확장")
    ]
    valid_sells = []
    for p, label in resist_candidates:
        if p >= cp * 1.02: valid_sells.append((p, label)) 
    valid_sells.sort(key=lambda x: x[0]) 
    
    final_sells = []
    if valid_sells: final_sells.append((adj(valid_sells[0][0]), valid_sells[0][1]))
    else: final_sells.append((adj(cp * 1.05), "목표가(+5%)"))
    
    last_sell = final_sells[0][0]
    for p, label in valid_sells:
        if len(final_sells) >= 3: break
        if p - last_sell >= min_gap: final_sells.append((adj(p), label)); last_sell = p
    while len(final_sells) < 3:
        ref = final_sells[-1][0]; final_sells.append((adj(ref * 1.05), "추가상승"))

    if score >= 80 and curr['Close'] > curr['MA5']:
        new_target = final_sells[0][0] * 1.05 
        final_sells[0] = (adj(new_target), "📈추세지속홀딩")
        hit_reasons.append("Profit Run(목표상향)")

    volatility_mult = 1.0 + (score / 200.0)
    tb_stop = cp - (atr * 1.5)
    
    entry_p = final_buys[0][0]; last_entry_p = final_buys[-1][0]
    final_stop = (adj(entry_p * 0.97), "-3%손절") 
    if "20일선" in final_buys[0][1]: final_stop = (adj(curr['BB_Lo']), "BB하단이탈")
    elif "60일선" in final_buys[0][1]: final_stop = (adj(low_120), "전저점이탈")
    
    if tb_stop > final_stop[0]: final_stop = (adj(tb_stop), "🛡️트리플배리어")
    min_stop_limit = adj(last_entry_p * 0.97)
    if final_stop[0] >= last_entry_p: final_stop = (min_stop_limit, "최종지지이탈")

    q1, q2, q3, total_alloc = calculate_sizing(score, cp, min_inv, max_inv, ai_prob)

    buys_w_prob = []
    shares_map = [q1, q2, q3]
    for i, (p, desc) in enumerate(final_buys):
        qty = shares_map[i] if i < 3 else 0
        buys_w_prob.append((p, desc, calc_reach_prob(cp, p, atr, ai_prob, 'buy', base_days=calc_days), qty))
        
    sells_w_prob = []
    for p, desc in final_sells: sells_w_prob.append((p, desc, calc_reach_prob(cp, p, atr, ai_prob, 'sell', base_days=calc_days)))
    stop_prob = calc_reach_prob(cp, final_stop[0], atr, ai_prob, 'stop', base_days=calc_days)

    status = {"type": "💤 관망", "color": "#78909c", "msg": "대기"}
    if buy_price > 0:
        pct = (cp - buy_price) / buy_price * 100
        status = {"type": "💰 수익" if pct > 0 else "❄️ 손실", "color": "#2e7d32" if pct > 0 else "#1976d2", "msg": f"{pct:+.2f}%"}

    return {
        "buy": buys_w_prob, "sell": sells_w_prob, "score": int(score), "ai": ai_prob, 
        "stops": {"ma5": adj(curr['MA5']), "ma20": adj(curr['MA20']), "bb_lo": adj(curr['BB_Lo'])},
        "final_stop": (final_stop[0], final_stop[1], stop_prob), 
        "status": status, "logic": logic_mode, "reasons": hit_reasons, 
        "pattern_reasons": pattern_reasons, "horizon_tag": horizon_tag,
        "mtf_reasons": mtf_reasons, "mvwap": curr['MVWAP'], "rsi": curr['RSI'], "whipsaw": whipsaw_warnings,
        "allocation": total_alloc, "alpha": alpha_stock, "breakout": breakout,
        "consensus": consensus_info
    }

def format_3split_msg(name, s, prefix=""):
    alpha_mark = "🔮" if s.get('alpha') else ""
    break_mark = "💥" if s.get('breakout') else ""
    msg = f"{prefix} <b>{name}</b> {s['horizon_tag']} {alpha_mark}{break_mark} ({s['score']}점/{s['ai']}%)\n"
    if s['consensus']['price'] > 0: msg += f"🎯 목표가: {s['consensus']['price']:,}원 (괴리율 {s['consensus']['upside']:.1f}%)\n"
    if s['pattern_reasons']: msg += f"🧩 패턴: {', '.join(s['pattern_reasons'])}\n"
    msg += f"전략: {s['logic']} (배정: {int(s['allocation']/10000)}만원)\n"
    msg += "🔵 <b>[분할 매수]</b>\n"
    for i, (p, d, prob, qty) in enumerate(s['buy']): msg += f" {i+1}차: {p:,}원 ({qty}주, {prob}%)\n"
    msg += "🔴 <b>[분할 매도]</b>\n"
    for i, (p, d, prob) in enumerate(s['sell']): msg += f" {i+1}차: {p:,}원 ({prob}%)\n"
    msg += f"🛑 손절: {s['final_stop'][0]:,}원 ({s['final_stop'][2]}%)\n"
    return msg + "\n"

# ==========================================
# 🖥️ 4. 메인 UI (Sidebar)
# ==========================================
with st.sidebar:
    now = get_now_kst()
    is_market_open = check_market_open()
    st.markdown(f'<div class="clock-box">⏰ {now.strftime("%H:%M:%S")}</div>', unsafe_allow_html=True)
    if is_market_open: st.markdown('<div class="status-open">🟢 장중 (Active)</div>', unsafe_allow_html=True)
    else: st.markdown('<div class="status-closed">🔴 장 마감 (Closed)</div>', unsafe_allow_html=True)
    
    source_container = st.empty()
    source_container.markdown('<div class="source-box">📡 준비 완료</div>', unsafe_allow_html=True)
    krx_list, list_src = get_safe_stock_listing()
    st.markdown(f'<div class="list-box">📋 {list_src}</div>', unsafe_allow_html=True)

    st.title("✨ AI Master V81.58")
    st.caption("Hybrid Engine (XGB+RF+LSTM-L2)")
    
    with st.expander("💰 자금 관리 설정 (Money Mgmt)", expanded=True):
        invest_min = st.number_input("종목당 최소 투자금", value=3000000, step=500000)
        invest_max = st.number_input("종목당 최대 투자금", value=5000000, step=500000)
        st.caption(f"AI 점수에 따라 {int(invest_min/10000)}~{int(invest_max/10000)}만원 사이에서 자동 배정됩니다.")
    
    with st.expander("🔐 한국투자증권(KIS) 데이터 설정", expanded=False):
        st.caption("계좌번호 없이 시세 조회용으로만 사용합니다.")
        k_conf = "kis_config.json"
        
        def_k, def_s, def_m = "", "", False
        if "kis" in st.secrets:
            def_k = st.secrets["kis"]["app_key"]
            def_s = st.secrets["kis"]["app_secret"]
        elif os.path.exists(k_conf):
            try:
                with open(k_conf, "r") as f:
                    d = json.load(f)
                    def_k = d.get("key", "")
                    def_s = d.get("secret", "")
                    def_m = d.get("mock", False)
            except: pass

        kis_app_key = st.text_input("App Key", value=def_k, type="password", key="kis_key")
        kis_app_secret = st.text_input("App Secret", value=def_s, type="password", key="kis_sec")
        kis_mock = st.checkbox("모의투자 서버", value=def_m)
        
        if st.button("설정 저장 (KIS)"):
            with open(k_conf, "w") as f:
                json.dump({"key": kis_app_key, "secret": kis_app_secret, "mock": kis_mock}, f)
            st.toast("설정이 저장되었습니다!", icon="💾")

        if st.button("KIS 데이터 연동 (토큰발급)"):
            if kis_app_key and kis_app_secret:
                kis_client = KIS_Data_Client(kis_app_key, kis_app_secret, kis_mock)
                if kis_client.get_access_token():
                    st.success("✅ 인증 성공! (시세 조회에 KIS 사용)")
                else: st.error("❌ 인증 실패 (Key/Secret 확인)")
            else: st.warning("Key와 Secret을 입력하세요.")
    
    with st.expander("⚙️ 설정 및 알림", expanded=False):
        config_file = "telegram_config.json"
        default_token = ""; default_id = ""
        
        if "telegram" in st.secrets:
            default_token = st.secrets["telegram"]["token"]
            default_id = st.secrets["telegram"]["chat_id"]
        elif os.path.exists(config_file):
            try:
                with open(config_file, "r") as f:
                    config = json.load(f)
                    default_token = config.get("token", ""); default_id = config.get("chat_id", "")
            except: pass

        tg_token = st.text_input("Bot Token", value=default_token, type="password")
        tg_id = st.text_input("Chat ID", value=default_id)
        if st.button("설정 저장"):
            with open(config_file, "w") as f: json.dump({"token": tg_token, "chat_id": tg_id}, f)
        
        st.markdown("---")
        st.markdown("🧠 **AI 모델 관리 (Hybrid)**")
        
        model_exists = False
        model_info = "⚠️ 모델 없음 (학습필요)"; model_size = "-"
        if os.path.exists(MODEL_FILE):
             try:
                 m = joblib.load(MODEL_FILE)
                 model_info = m.get('date', 'Unknown')
                 model_size = m.get('sample_size', 0)
                 model_exists = True
             except: pass
        
        if not model_exists:
            st.error("🚨 AI 모델이 없습니다! 아래 [Top 50개 학습]을 눌러주세요.")
        else:
            st.caption(f"최근 학습일: {model_info} (샘플: {model_size}개)")
            
        train_limit = st.slider("학습 종목 수 (상위 N개)", 20, 100, 50, step=10)
        
        st.markdown("##### 1️⃣ 약식 테스트")
        if st.button(f"⚡ Top {train_limit}개만 학습"):
            with st.spinner(f"Top {train_limit}개 종목으로 빠르게 학습합니다... (LSTM 포함)"):
                success, msg = train_global_model(krx_list, limit=train_limit, mode="initial") 
                if success:
                    st.success(msg); st.cache_resource.clear(); st.cache_data.clear(); time.sleep(2); st.rerun()
                else: st.error(msg)

        st.markdown("##### 2️⃣ 실전 정밀 학습")
        if st.button("💰 시총 3000억 이상 전수 학습 (10년)"):
            with st.spinner("시가총액 3000억 이상 전 종목 10년치 데이터를 학습합니다. (Safe Mode)"):
                try:
                    min_marcap = 300000000000 
                    target_pool = krx_list[krx_list['Marcap'] >= min_marcap].copy()
                    target_pool['Code'] = target_pool['Code'].astype(str).str.zfill(6)
                    success, msg = train_global_model(target_pool, limit=len(target_pool), mode="full_initial")
                    if success:
                        st.success(f"✅ 학습 완료! ({len(target_pool)}개)"); st.cache_resource.clear(); st.cache_data.clear(); time.sleep(2); st.rerun()
                    else: st.error(msg)
                except Exception as e: st.error(f"데이터 준비 실패: {e}")

        st.markdown("##### 3️⃣ 데일리 업데이트")
        if st.button("📅 일일 데이터 갱신 (누적)"):
            with st.spinner("오늘치 데이터를 추가하여 모델을 업데이트합니다..."):
                success, msg = train_global_model(krx_list, limit=100, mode="update")
                if success:
                    st.success(msg); st.cache_resource.clear(); st.cache_data.clear(); time.sleep(2); st.rerun()
                else: st.error(msg)

        st.markdown("---")
        c_auto, c_min = st.columns([1.5, 1])
        with c_auto: auto_scan_on = st.toggle("🤖 자동 스캔", value=False)
        with c_min: scan_interval_min = st.number_input("분", min_value=10, max_value=120, value=30, step=10, label_visibility="collapsed")
        
        auto_report = st.checkbox("✅ 자동 리포트 (장마감)", value=True)
        report_time = st.time_input("발송 시간", datetime.time(16, 0))
        if st.button("🗑️ 캐시 초기화"): st.cache_data.clear(); st.rerun()

    min_m = st.number_input("최소 시총(억)", value=3000) * 100000000

    if 'last_scan_time' not in st.session_state:
        st.session_state['last_scan_time'] = datetime.datetime.min

    should_run_auto = False
    if auto_scan_on:
        if is_market_open: 
            elapsed = get_now_kst().replace(tzinfo=None) - st.session_state['last_scan_time'].replace(tzinfo=None)
            if elapsed.total_seconds() > (scan_interval_min * 60): 
                should_run_auto = True
                st.session_state['last_scan_time'] = get_now_kst().replace(tzinfo=None)
            else:
                time.sleep(1)
                st.rerun()
        else: st.sidebar.warning("🌙 장 마감: 자동 스캔 대기 중")

    def generate_closing_report():
        report = []
        now = get_now_kst()
        report.append(f"<b>🌅 [AI Master] {now.strftime('%Y-%m-%d')} 마감 리포트</b>\n")
        try:
            us_indices = {'나스닥': '^IXIC', 'S&P500': '^GSPC'}
            report.append("<b>[🌎 글로벌 마감]</b>")
            for name, ticker in us_indices.items():
                idx_data = yf.download(ticker, period='2d', progress=False, threads=False) 
                if not idx_data.empty and len(idx_data) >= 2:
                    if isinstance(idx_data.columns, pd.MultiIndex): idx_data.columns = idx_data.columns.get_level_values(0)
                    cp_idx = idx_data['Close'].iloc[-1]; pp_idx = idx_data['Close'].iloc[-2]
                    chg = (cp_idx - pp_idx) / pp_idx * 100
                    symbol = "🔺" if chg > 0 else "🔻"
                    report.append(f"{symbol} {name}: {cp_idx:,.2f} ({chg:+.2f}%)")
            report.append("")
        except: pass

        pf_df = get_portfolio_gsheets()
        if not pf_df.empty:
            report.append("<b>[💼 내 포트폴리오(시트): 물타기 추천]</b>")
            watering_needed = False
            for _, r in pf_df.iterrows():
                d, _ = get_data_safe(r['Code'], days=300)
                if d is not None:
                    df_ind = get_all_indicators(d)
                    if df_ind is not None:
                        cp = df_ind['Close'].iloc[-1]; buy_price = float(r['Buy_Price'])
                        _, _, mkt_stat = get_ai_condition()
                        s = get_darwin_strategy(df_ind, buy_price, code=r['Code'], use_mtf=True, min_inv=invest_min, max_inv=invest_max, market_status=mkt_stat)
                        if s and cp < buy_price and s['score'] >= 70 and s['ai'] >= 60:
                            watering_needed = True
                            loss_pct = (cp - buy_price) / buy_price * 100
                            prefix = f"💧 <b>[물타기적합]</b> (손실 {loss_pct:.2f}%) "
                            report.append(format_3split_msg(r['Name'], s, prefix=prefix))
            if not watering_needed: report.append("📌 현재 물타기 권장 종목 없음 (보수적 기준 미달)\n")
        
        report.append("<b>[⭐ 명일 주력 추천 (교차검증)]</b>")
        report.append("<i>대상: KOSPI200/KOSDAQ150 중 AI+MTF+수급+Fibo 우량주</i>\n")
        try:
            k200 = fdr.StockListing('KOSPI 200')['Code'].tolist()
            kd150 = fdr.StockListing('KOSDAQ 150')['Code'].tolist()
            target_codes = list(set(k200 + kd150))
        except: target_codes = krx_list.head(200)['Code'].tolist()

        found_count = 0
        with ThreadPoolExecutor(max_workers=2) as executor:
            fut_map = {executor.submit(get_data_safe, c, 300): c for c in target_codes}
            for fut in as_completed(fut_map):
                try:
                    d_raw, _ = fut.result()
                    if d_raw is not None:
                        if len(d_raw) < 60: continue
                        cur_amt = (d_raw['Close'].iloc[-1] * d_raw['Volume'].iloc[-1])
                        if cur_amt < 1000000000: continue

                        df_ind = get_all_indicators(d_raw)
                        if df_ind is not None:
                            _, _, mkt_stat = get_ai_condition()
                            s_res = get_darwin_strategy(df_ind, code=fut_map[fut], use_mtf=True, min_inv=invest_min, max_inv=invest_max, market_status=mkt_stat) 
                            if s_res and s_res['score'] >= 70 and s_res['ai'] >= 65:
                                name = krx_list[krx_list['Code'] == fut_map[fut]]['Name'].values[0]
                                report.append(format_3split_msg(name, s_res, prefix="🔥"))
                                found_count += 1
                                if found_count >= 5: break
                except: continue
        if found_count == 0: report.append("🚩 명일 강력 추천 종목 없음 (관망 권장)")
        return "\n".join(report)

    if 'sent_report_date' not in st.session_state:
        st.session_state['sent_report_date'] = None

    cur_date = now.strftime("%Y-%m-%d")
    target_dt = now.replace(hour=report_time.hour, minute=report_time.minute, second=0, microsecond=0)
    valid_window = timedelta(minutes=30)

    if auto_report and (target_dt <= now <= target_dt + valid_window):
        if st.session_state['sent_report_date'] != cur_date:
            if tg_token and tg_id:
                with st.spinner("📧 마감 리포트 자동 발송 중..."):
                    rpt = generate_closing_report()
                    send_telegram_msg(tg_token, tg_id, rpt)
                    st.session_state['sent_report_date'] = cur_date 
                    st.toast(f"{report_time.strftime('%H:%M')} 리포트 발송 완료!", icon="✅")

    st.markdown("---")
    if st.button("📧 마감 리포트(명일전략) 생성"):
        with st.spinner("데이터 분석 및 리포트 작성 중..."):
            rpt_text = generate_closing_report()
            st.session_state['generated_report'] = rpt_text 
            if tg_token and tg_id: 
                send_telegram_msg(tg_token, tg_id, rpt_text)
                st.toast("텔레그램 발송 완료!", icon="✈️")
            else: st.toast("리포트 생성 완료", icon="⚠️")

    # 🧮 목표가 도달 확률 계산기
    st.markdown("---")
    with st.expander("🧮 목표가 도달 확률 계산기", expanded=True):
        st.caption("AI와 변동성(ATR) 기반 예측")
        calc_code = st.text_input("종목코드", value="035720") 
        calc_target = st.number_input("희망 목표가", value=80000, step=1000)
        calc_days = st.selectbox("기간 설정", [60, 120, 240], index=1, format_func=lambda x: f"{x}거래일 (약 {x//20}개월)")
        
        if st.button("🎲 확률 계산 실행"):
            with st.spinner("AI가 시뮬레이션 중..."):
                d_cal, _ = get_data_safe(calc_code, 300)
                if d_cal is not None:
                    df_cal = get_all_indicators(d_cal)
                    if df_cal is not None:
                        curr_p = df_cal['Close'].iloc[-1]
                        atr = df_cal['ATR'].iloc[-1]
                        ai_s = get_ai_score_fast(df_cal) 
                        prob = calc_reach_prob(curr_p, calc_target, atr, ai_s, base_days=calc_days)
                        st.write(f"**현재가:** {int(curr_p):,}원")
                        st.write(f"**AI 점수:** {ai_s}점")
                        dist_pct = (calc_target - curr_p) / curr_p * 100
                        if prob > 50: st.success(f"🎉 도달 확률: **{prob}%** (매우 높음)")
                        elif prob > 20: st.warning(f"⚠️ 도달 확률: **{prob}%** (도전적)")
                        else: st.error(f"📉 도달 확률: **{prob}%** (희박함)")
                        st.caption(f"💡 {dist_pct:.1f}% 상승은 현재 변동성으로 쉽지 않습니다.")
                else: st.error("데이터 로딩 실패")

# --- Tabs Implementation ---
tabs = st.tabs(["📊 대시보드", "🔍 MTF 스캐너", "🧬 백테스트", "💼 분석", "➕ 관리(GSheets)", "🔄 회복 시뮬레이션", "📈 AI 성장 일기", "💾 동기화"])

with tabs[0]: # 대시보드
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        txt, color = get_market_trend("KS11", "KOSPI")
        st.markdown(f"<div class='market-card'><h4 style='color:{color}; margin:0;'>{txt}</h4></div>", unsafe_allow_html=True)
    with col_m2:
        txt, color = get_market_trend("KQ11", "KOSDAQ")
        st.markdown(f"<div class='market-card'><h4 style='color:{color}; margin:0;'>{txt}</h4></div>", unsafe_allow_html=True)
    
    st.write("")
    if os.path.exists(MODEL_FILE):
        try:
            m_data = joblib.load(MODEL_FILE)
            if 'feature_importance' in m_data:
                with st.expander("🧠 AI 모델 브리핑 (XGB + RF + LSTM-L2)", expanded=True):
                    ic1, ic2, ic3 = st.columns(3)
                    acc_score = m_data.get('oob_score', 0) * 100
                    ic1.metric("RF OOB 정확도", f"{acc_score:.1f}%")
                    ic2.metric("학습 샘플 수", f"{m_data.get('sample_size', 0)}개")
                    
                    lstm_stat = "✅ 적용됨" if m_data.get('lstm_status') else "❌ 미적용"
                    ic3.metric("LSTM 엔진 상태", lstm_stat)
                    
                    fi_df = pd.DataFrame({
                        'Feature': m_data['feature_names'],
                        'Importance': m_data['feature_importance']
                    }).sort_values('Importance', ascending=True)
                    
                    fig_fi = px.bar(fi_df, x='Importance', y='Feature', orientation='h', title="AI가 중요하게 보는 핵심 지표 Top", template="plotly_white")
                    fig_fi.update_traces(marker_color='#00897b')
                    st.plotly_chart(fig_fi, use_container_width=True)
        except: pass
    else: st.error("🚨 현재 AI 모델이 없습니다! 왼쪽 사이드바에서 [⚡ Top 50개만 학습]을 꼭 눌러주세요.")

    pf = get_portfolio_gsheets()
    if not pf.empty:
        t_buy, t_eval, dash_list = 0, 0, []
        for idx, row in pf.iterrows():
            d, err = get_data_safe(row['Code'], days=300)
            if d is None:
                st.markdown(f"<div class='error-box'>⚠️ {row['Name']} 로딩 실패: {err}</div>", unsafe_allow_html=True); continue
            source_container.markdown(f'<div class="source-box">{d.attrs.get("source", "Unknown")}</div>', unsafe_allow_html=True)
            df = get_all_indicators(d)
            if df is not None:
                _, _, mkt_stat = get_ai_condition()
                res = get_darwin_strategy(df, row['Buy_Price'], min_inv=invest_min, max_inv=invest_max, market_status=mkt_stat)
                if res:
                    cp = df['Close'].iloc[-1]; t_buy += (row['Buy_Price']*row['Qty']); t_eval += (cp*row['Qty'])
                    dash_list.append({"종목": row['Name'], "수익": (cp-row['Buy_Price'])*row['Qty'], "상태": res['status']['type']})
        c1, c2, c3 = st.columns(3)
        c1.metric("총 매수", f"{int(t_buy):,}원")
        c2.metric("총 평가", f"{int(t_eval):,}원", f"{(t_eval-t_buy)/t_buy*100:+.2f}%" if t_buy>0 else "0%")
        c3.metric("평가 손익", f"{int(t_eval-t_buy):,}원")
        if dash_list: st.plotly_chart(px.bar(pd.DataFrame(dash_list), x='종목', y='수익', color='상태', template="plotly_white"), use_container_width=True)
    else: st.info(f"📌 등록된 포트폴리오가 없습니다. [관리] 탭에서 추가해주세요.")

with tabs[1]: # 스캐너
    st.markdown("### ⚡ 실시간 AI/MTF/수급 유망 종목 발굴")
    manual_start = st.button("💎 주봉+일봉+60분+15분 MTF 정밀 스캔 시작", type="primary", use_container_width=True)
    if manual_start or should_run_auto:
        score_penalty, ai_status, mkt_stat = get_ai_condition()
        target_score = 65 + score_penalty
        st.info(f"🤖 **AI 자가 진단:** {ai_status} (추천 기준점: {target_score}점)")
        pf_df = get_portfolio_gsheets()
        my_stock_map = {} 
        if not pf_df.empty: my_stock_map = pf_df.set_index('Code')['Buy_Price'].to_dict()
        targets = krx_list[krx_list['Marcap'] >= min_m].sort_values('Marcap', ascending=False)
        target_count = len(targets); found = []; prog_bar = st.progress(0); status_txt = st.empty()
        
        with st.spinner("1단계: 섹터 동향 분석 중..."):
            sec_map = get_sector_performance_map(krx_list)

        with ThreadPoolExecutor(max_workers=2) as ex:
            fut_map = {ex.submit(get_data_safe, r['Code'], 300, "1d"): (r['Name'], r['Code'], r['Marcap'], r['Sector']) for _, r in targets.iterrows()}
            for i, f in enumerate(as_completed(fut_map)):
                name, code, marcap, sector = fut_map[f]
                try:
                    d_raw, err = f.result()
                    status_txt.markdown(f"📡 **{name}** 분석 중... ({i+1}/{target_count})")
                    
                    if d_raw is not None and not d_raw.empty:
                        # [Improvement 1] 스캐너 컷오프 적용: 거래대금 20억 미만 즉시 Skip
                        cur_amt = (d_raw['Close'].iloc[-1] * d_raw['Volume'].iloc[-1])
                        if cur_amt < 2000000000: continue

                        res = get_all_indicators(d_raw)
                        if res is not None:
                            sec_s = sec_map.get(sector, 0)
                            s = get_darwin_strategy(res, code=code, use_mtf=True, min_inv=invest_min, max_inv=invest_max, market_status=mkt_stat, sec_score=sec_s) 
                            if s and s['score'] >= target_score: 
                                cp = res['Close'].iloc[-1]
                                is_watering = False; my_avg = 0; loss_pct = 0; action_msg = ""; action_color = "#333"
                                if code in my_stock_map:
                                    my_avg = my_stock_map[code]; loss_pct = (cp - my_avg) / my_avg * 100
                                    action_txt, action_col, action_tag = analyze_portfolio_action(s['score'], s['ai'], loss_pct, s['rsi'])
                                    action_msg = f"<br><span style='color:{action_col}; font-weight:bold; font-size:0.9em; background-color:#fff3e0; padding:2px 6px; border-radius:4px;'>{action_txt}</span>"
                                    if "물타기" in action_txt: is_watering = True
                                data_src = d_raw.attrs.get('source', 'Unknown')
                                item = {"name": name, "code": code, "marcap": marcap, "score": s['score'], "strat": s, "cp": cp, "source": data_src, "is_watering": is_watering, "my_avg": my_avg, "loss_pct": loss_pct, "action_msg": action_msg}
                                found.append(item) 
                except Exception as e: print(f"Error: {e}")
                prog_bar.progress((i+1)/target_count)
        
        history_df = get_scan_history()
        already_sent_today = []
        if history_df is not None and not history_df.empty:
            today_str = get_now_kst().strftime('%Y-%m-%d')
            if 'Date' in history_df.columns:
                already_sent_today = history_df[history_df['Date'] == today_str]['Code'].tolist()
        
        large_cap = []; mid_cap = []; small_cap = []
        for item in found:
            m = item['marcap']
            if str(item['code']).zfill(6) in already_sent_today: continue 
            if m >= 1000000000000: large_cap.append(item) 
            elif m >= 300000000000: mid_cap.append(item) 
            else: small_cap.append(item) 
        if tg_token and tg_id and (large_cap or mid_cap or small_cap):
            with st.spinner("🚀 신규 포착 종목 텔레그램 전송 중..."):
                def send_batch(title, items):
                    if not items: return
                    send_telegram_msg(tg_token, tg_id, f"=== 🏆 {title} New Pick ===")
                    for d in items[:5]: 
                        s = d['strat']; prefix_msg = f"💧 <b>[물타기]</b> " if d['is_watering'] else "🐲 <b>[AI포착]</b> "
                        msg = format_3split_msg(d['name'], s, prefix=prefix_msg)
                        send_telegram_msg(tg_token, tg_id, msg); time.sleep(0.5)
                send_batch("대형주(1조↑)", large_cap); send_batch("중형주(3천억~1조)", mid_cap); send_batch("소형주(3천억↓)", small_cap)
            st.toast("신규 알림 발송 완료!", icon="✅")
        elif found and not (large_cap or mid_cap or small_cap): st.toast("새로운 종목이 없습니다 (중복 제외)", icon="ℹ️")
        if found: save_bulk_results(found)
        status_txt.success(f"✅ 스캔 및 저장 완료! 총 {len(found)}개 종목이 포착되었습니다.")
        display_list = sorted(found, key=lambda x: x['score'], reverse=True)
        if not display_list: st.warning("조건에 부합하는 종목이 없습니다.")
        for d in display_list:
            s = d['strat']
            reasons_html = "".join([f"<span class='hit-tag'>{r}</span>" for r in s['reasons']])
            mtf_html = "".join([f"<span class='mtf-badge'>{r}</span>" for r in s['mtf_reasons']]) if s['mtf_reasons'] else ""
            pattern_html = "".join([f"<span class='pattern-badge'>{p}</span>" for p in s['pattern_reasons']]) if s.get('pattern_reasons') else ""
            whipsaw_html = f"<div class='whipsaw-box'>⚠️ 주의: {', '.join(s['whipsaw'])} (가짜 신호 가능성)</div>" if s['whipsaw'] else ""
            
            con_html = ""
            if s['consensus']['price'] > 0:
                con_html = f"<div style='background:#f3e5f5; padding:6px; border-radius:4px; margin-top:5px; font-size:0.85em;'>🎯 <b>목표가:</b> {s['consensus']['price']:,}원 <span style='color:#d32f2f;'>({s['consensus']['upside']:.1f}%)</span> / 의견: {s['consensus']['opinion']}</div>"

            avg_info = ""; card_border = ""
            if s.get('alpha'): card_border = "border-left: 5px solid #7b1fa2;" 
            mode_badge = f"<span class='mode-badge'>{s['logic']}</span>"
            style_badge = f"<span class='style-badge'>{s['horizon_tag']}</span>"
            alpha_badge = "<span class='alpha-tag'>🔮 Alpha Hunter</span>" if s.get('alpha') else ""
            break_tag = "<span class='break-tag'>💥 돌파</span>" if s.get('breakout') else ""
            
            if d['my_avg'] > 0:
                card_border = "border-left: 5px solid #d32f2f;" if d['is_watering'] else "border-left: 5px solid #2e7d32;"
                avg_info = f"<br><span style='color:#555; font-size:0.85em;'>📉 내 평단: {int(d['my_avg']):,}원 ({d['loss_pct']:.2f}%)</span> {d['action_msg']}"
            cap_tag = "🦖대형" if d['marcap']>=1000000000000 else "🐅중형" if d['marcap']>=300000000000 else "🐇소형"
            card_html = f"""
<div class="scanner-card" style="{card_border}">
<div style="display:flex; justify-content:space-between; align-items:center;">
<div>
<h3 style="margin:0;">{d['name']} {style_badge} {alpha_badge} {break_tag} <span style="font-size:0.7em; color:#666;">({d['code']})</span> <span style="font-size:0.6em; background:#eee; padding:2px 4px; border-radius:3px;">{cap_tag}</span></h3>
<span class="current-price">{d['cp']:,}원</span> {avg_info}
<br><span class="pro-tag">Source: {d['source']}</span>
</div>
<div style="text-align:right;">
<span class="ai-badge">AI확률: {s['ai']}%</span><br>
<span style="color:#00897b; font-weight:bold; font-size:1.2em;">Score: {s['score']}</span><br>
<span style="font-size:0.9em; background:#e3f2fd; padding:3px 6px; border-radius:4px; font-weight:bold; color:#1565c0;">💰 {int(s['allocation']/10000)}만원</span>
</div>
</div>
<div style="margin:10px 0; line-height:1.6;">
{reasons_html} {mtf_html} {pattern_html} {con_html}
</div>
{whipsaw_html}
<div class="strategy-grid">
<div class="buy-box">
<b>🔵 분할 매수 ({int(s['allocation']/10000)}만)</b><br>
1차: <b>{s['buy'][0][0]:,}</b> <span style="font-size:0.8em">({s['buy'][0][3]}주, {s['buy'][0][2]}%)</span><br>
2차: {s['buy'][1][0]:,} <span style="font-size:0.8em">({s['buy'][1][3]}주, {s['buy'][1][2]}%)</span><br>
3차: {s['buy'][2][0]:,} <span style="font-size:0.8em">({s['buy'][2][3]}주, {s['buy'][2][2]}%)</span>
</div>
<div class="sell-box">
<b>🔴 분할 매도</b><br>
1차: <b>{s['sell'][0][0]:,}</b> <span style="font-size:0.8em">({s['sell'][0][2]}%)</span><br>
2차: {s['sell'][1][0]:,} <span style="font-size:0.8em">({s['sell'][1][2]}%)</span><br>
3차: {s['sell'][2][0]:,} <span style="font-size:0.8em">({s['sell'][2][2]}%)</span>
</div>
<div class="stop-box">
<b>🛑 리스크 관리</b><br>
{s['final_stop'][1]}: <b>{s['final_stop'][0]:,}</b><br>
<span style="font-size:0.8em">(터치확률: {s['final_stop'][2]}%)</span>
</div>
</div>
</div>"""
            st.markdown(card_html, unsafe_allow_html=True)

with tabs[2]: # 백테스트
    if st.button("🚀 샘플 검증 (Top 10)"):
        targets = krx_list.head(10)['Code'].tolist()
        results = []; prog = st.progress(0)
        for idx, code in enumerate(targets):
            raw_data, _ = get_data_safe(code, days=1000)
            df = get_all_indicators(raw_data)
            if df is not None:
                for i in range(50, 0, -2):
                    past = df.iloc[:-i*5]; future = df.iloc[-i*5:]
                    if len(future) >= 5:
                        s = get_darwin_strategy(past)
                        if s and s['score'] >= 65:
                            entry = past['Close'].iloc[-1]; exit_p = future['Close'].iloc[4]
                            results.append({"Date": past.index[-1], "Win": 1 if exit_p > entry else 0})
            prog.progress((idx+1)/len(targets))
        if results:
            df_res = pd.DataFrame(results)
            win_rate = df_res['Win'].mean() * 100
            st.metric("예측 승률 (Score>=65)", f"{win_rate:.1f}%", f"총 {len(df_res)}회")

with tabs[3]: # 분석
    st.subheader("🔍 기업 정밀 분석 & 리포트")
    c_mode, c_input = st.columns([1, 3])
    with c_mode: mode = st.radio("분석 대상", ["내 포트폴리오", "종목 직접 검색"], horizontal=False)
    target_code = None; target_name = None; target_price = 0
    with c_input:
        if mode == "내 포트폴리오":
            pf_gs = get_portfolio_gsheets()
            if not pf_gs.empty:
                pf_gs['Display'] = pf_gs.apply(lambda x: f"{x['Name']} (평단 {int(x['Buy_Price']):,}원)", axis=1)
                sel_display = st.selectbox("보유 종목 선택", pf_gs['Display'].unique())
                row = pf_gs[pf_gs['Display'] == sel_display].iloc[0]
                target_code = row['Code']; target_name = row['Name']; target_price = row['Buy_Price']
            else: st.warning("⚠️ 포트폴리오 비어있음 (관리 탭에서 추가)")
        else:
            col_search, col_btn = st.columns([4, 1])
            with col_search: search_txt = st.text_input("종목명/코드 입력", placeholder="삼성전자 or 005930")
            with col_btn: 
                st.write(""); st.write("")
                search_trigger = st.button("검색")
            if search_txt:
                clean_txt = search_txt.strip().upper()
                res = krx_list[ (krx_list['Name'].str.upper() == clean_txt) | (krx_list['Code'] == clean_txt) ]
                if not res.empty:
                    target_code = res.iloc[0]['Code']; target_name = res.iloc[0]['Name']; target_price = 0
                else: st.error("❌ 종목을 찾을 수 없습니다.")

    if target_code:
        st.markdown("---")
        with st.spinner(f"📡 '{target_name}' 정밀 분석 중..."):
            raw_data, err = get_data_safe(target_code, days=400)
            if raw_data is None: st.error(f"데이터 로드 실패: {err}")
            else:
                df = get_all_indicators(raw_data)
                if df is not None:
                    _, _, mkt_stat = get_ai_condition()
                    sec_score = 0
                    if 'Sector' in krx_list.columns:
                        try:
                            sec_row = krx_list[krx_list['Code'] == target_code]
                            if not sec_row.empty: pass
                        except: pass
                    
                    s = get_darwin_strategy(df, target_price, code=target_code, use_mtf=True, min_inv=invest_min, max_inv=invest_max, market_status=mkt_stat, sec_score=sec_score)
                    if s:
                        reasons_html = "".join([f"<span class='hit-tag'>{r}</span>" for r in s['reasons']])
                        mtf_html = "".join([f"<span class='mtf-badge'>{r}</span>" for r in s['mtf_reasons']]) if s['mtf_reasons'] else ""
                        pattern_html = "".join([f"<span class='pattern-badge'>{p}</span>" for p in s['pattern_reasons']]) if s.get('pattern_reasons') else ""
                        whipsaw_html = f"<div class='whipsaw-box'>⚠️ {', '.join(s['whipsaw'])}</div>" if s['whipsaw'] else ""
                        
                        style_badge = f"<span class='style-badge'>{s['horizon_tag']}</span>"
                        mode_badge = f"<span class='mode-badge'>{s['logic']}</span>"
                        alpha_badge = "<span class='alpha-tag'>🔮 Alpha Hunter</span>" if s.get('alpha') else ""
                        break_tag = "<span class='break-tag'>💥 돌파</span>" if s.get('breakout') else ""
                        
                        con_info = ""
                        if s['consensus']['price'] > 0:
                            con_info = f"""
                            <div style="background:#f3e5f5; padding:10px; border-radius:8px; margin-top:8px; font-size:0.95em; border:1px solid #e1bee7;">
                                🎯 <b>증권사 목표가:</b> {s['consensus']['price']:,}원 
                                <span style="color:#d32f2f; font-weight:bold;">(괴리율 +{s['consensus']['upside']:.1f}%)</span><br>
                                📊 <b>도달 확률(6개월):</b> {s['consensus']['prob']}% 
                                (투자의견: {s['consensus']['opinion']}/5.0)
                            </div>
                            """

                        analysis_html = f"""
<div class="metric-card" style="border-left:10px solid {s['status']['color']};">
<div style="display:flex; justify-content:space-between; align-items:center;">
<div>
<h2 style="margin:0;">{target_name} {style_badge} {mode_badge} {alpha_badge} {break_tag}</h2>
<p style="font-size:1.1em; color:{s['status']['color']}; font-weight:bold; margin-top:5px;">
{s['status']['msg']} <span style="font-size:0.8em; color:#666;">(AI확률: {s['ai']}%)</span>
</p>
</div>
<div style="text-align:right;">
<h2 style="color:#333; margin:0;">{df['Close'].iloc[-1]:,}원</h2>
<span class="pro-tag">MVWAP: {int(s['mvwap']):,}</span><br>
<span style="font-size:0.9em; background:#e3f2fd; padding:3px 6px; border-radius:4px; font-weight:bold; color:#1565c0;">💰 {int(s['allocation']/10000)}만원</span>
</div>
</div>
<div style="margin:10px 0;">
<b>포착 근거:</b> {reasons_html} {mtf_html} {pattern_html}
</div>
{con_info}
{whipsaw_html}
<div class="strategy-grid">
<div class="buy-box">
<b>🔵 분할 매수 ({int(s['allocation']/10000)}만)</b><br>
1차: <b>{s['buy'][0][0]:,}</b> <span style="font-size:0.8em">({s['buy'][0][3]}주, {s['buy'][0][2]}%)</span><br>
2차: {s['buy'][1][0]:,} <span style="font-size:0.8em">({s['buy'][1][3]}주, {s['buy'][1][2]}%)</span><br>
3차: {s['buy'][2][0]:,} <span style="font-size:0.8em">({s['buy'][2][3]}주, {s['buy'][2][2]}%)</span>
</div>
<div class="sell-box">
<b>🔴 분할 매도</b><br>
1차: <b>{s['sell'][0][0]:,}</b> <span style="font-size:0.8em">({s['sell'][0][2]}%)</span><br>
2차: {s['sell'][1][0]:,} <span style="font-size:0.8em">({s['sell'][1][2]}%)</span><br>
3차: {s['sell'][2][0]:,} <span style="font-size:0.8em">({s['sell'][2][2]}%)</span>
</div>
<div class="stop-box">
<b>🛑 리스크 관리 (Stop)</b><br>
{s['final_stop'][1]}: <b>{s['final_stop'][0]:,}</b><br>
<span style="font-size:0.8em">(터치확률: {s['final_stop'][2]}%)</span>
</div>
</div>
</div>
"""
                        st.markdown(analysis_html, unsafe_allow_html=True)
                        col_dummy, col_send = st.columns([4, 1.5])
                        with col_send:
                            if st.button("✈️ 분석 결과 텔레그램 전송", key="btn_send_anl", use_container_width=True):
                                if tg_token and tg_id:
                                    msg = format_3split_msg(target_name, s, prefix="📊 <b>[정밀분석]</b> ")
                                    send_telegram_msg(tg_token, tg_id, msg)
                                    st.toast("전송 완료!", icon="✅")
                                else: st.error("토큰 설정 필요")

                        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3], subplot_titles=("가격 및 이동평균선", "스토캐스틱"))
                        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Candle', increasing_line_color='#ef5350', decreasing_line_color='#2962ff'), row=1, col=1)
                        # Ichimoku Cloud (Span A, B fill)
                        fig.add_trace(go.Scatter(x=df.index, y=df['Ichi_SpanA'], line=dict(color='rgba(0,0,0,0)'), showlegend=False), row=1, col=1)
                        fig.add_trace(go.Scatter(x=df.index, y=df['Ichi_SpanB'], fill='tonexty', fillcolor='rgba(135, 206, 235, 0.2)', line=dict(color='rgba(0,0,0,0)'), name='구름대'), row=1, col=1)

                        fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], line=dict(color='blue', width=2), name='20일선'), row=1, col=1)
                        fig.add_trace(go.Scatter(x=df.index, y=df['MA60'], line=dict(color='green', width=1.5), name='60일선'), row=1, col=1)
                        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Up'], line=dict(color='gray', width=1, dash='dot'), name='BB상단', showlegend=False), row=1, col=1)
                        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lo'], line=dict(color='gray', width=1, dash='dot'), name='BB하단', showlegend=False), row=1, col=1)
                        fig.add_trace(go.Scatter(x=df.index, y=df['Stoch_5'], line=dict(color='#2962ff', width=1.5), name='Fast(5-3-3)'), row=2, col=1)
                        fig.add_trace(go.Scatter(x=df.index, y=df['Stoch_10'], line=dict(color='#00c853', width=1.5), name='Mid(10-6-6)'), row=2, col=1)
                        fig.add_trace(go.Scatter(x=df.index, y=df['Stoch_20'], line=dict(color='#ff6d00', width=2), name='Slow(20-12-12)'), row=2, col=1)
                        fig.add_hline(y=80, line_dash="dot", line_color="red", row=2, col=1)
                        fig.add_hline(y=20, line_dash="dot", line_color="blue", row=2, col=1)
                        fig.update_layout(height=700, template="plotly_white", xaxis_rangeslider_visible=False, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                        st.plotly_chart(fig, use_container_width=True)

with tabs[4]: # 관리 (GSheets Manual Input)
    st.subheader("📝 내 포트폴리오 관리 (구글 시트 연동)")
    st.info("타 증권사 계좌의 보유 종목을 수기로 입력하면 대시보드에서 분석됩니다.")
    df_p = get_portfolio_gsheets()
    with st.form("add_pf"):
        c1, c2, c3 = st.columns(3)
        n = c1.text_input("종목명 (정확히 입력)")
        p = c2.number_input("평단가", min_value=0)
        q = c3.number_input("수량", min_value=0)
        if st.form_submit_button("추가 / 업데이트"):
            try:
                conn = st.connection("gsheets", type=GSheetsConnection)
                m = krx_list[krx_list['Name']==n]
                if not m.empty:
                    code = m.iloc[0]['Code']
                    if 'Code' in df_p.columns: df_p = df_p[df_p['Code'] != code]
                    new_row = pd.DataFrame([[code, n, p, q]], columns=['Code','Name','Buy_Price','Qty'])
                    updated_p = pd.concat([df_p, new_row], ignore_index=True)
                    conn.update(worksheet="portfolio", data=updated_p)
                    st.success(f"{n} 추가 완료!"); time.sleep(1); st.rerun()
                else: st.error("정확한 종목명을 입력해주세요.")
            except Exception as e: st.error(f"시트 연결 오류: {e}")
    
    if not df_p.empty:
        st.write("▼ 현재 등록된 종목")
        st.dataframe(df_p, use_container_width=True)
        if st.button("선택 종목 삭제"): 
             st.info("구글 스프레드시트에서 직접 행을 삭제해주세요.")

with tabs[5]: # Recovery & Rebalance Tab
    st.subheader("🔄 원금 회복 & 포트폴리오 시뮬레이터")
    pf = get_portfolio_gsheets()
    if pf.empty: st.warning("⚠️ 포트폴리오가 비어있습니다. [관리] 탭에서 종목을 먼저 추가해주세요.")
    else:
        st.markdown("#### 💰 전체 계좌 원금 회복 시나리오")
        with st.spinner("전체 포트폴리오 정밀 분석 중..."):
            total_buy = 0; total_eval = 0; weighted_vol_sum = 0; rebal_data = []
            with ThreadPoolExecutor(max_workers=5) as ex:
                fut_map = {ex.submit(get_data_safe, row['Code'], 200): row for _, row in pf.iterrows()}
                for fut in as_completed(fut_map):
                    row = fut_map[fut]
                    try:
                        d, _ = fut.result()
                        if d is not None and not d.empty:
                            cp = d['Close'].iloc[-1]; val = cp * row['Qty']; buy_val = row['Buy_Price'] * row['Qty']
                            total_buy += buy_val; total_eval += val
                            profit_pct = (cp - row['Buy_Price']) / row['Buy_Price'] * 100
                            daily_ret = d['Close'].pct_change().dropna()
                            vol = daily_ret.std() * np.sqrt(252); weighted_vol_sum += (vol * val) 
                            df_ind = get_all_indicators(d)
                            if df_ind is not None:
                                ai_s = get_ai_score_fast(df_ind)
                                rebal_data.append({'name': row['Name'], 'code': row['Code'], 'score': ai_s, 'vol': vol, 'value': val, 'profit_pct': profit_pct})
                    except: pass
            
            if total_eval > 0:
                port_volatility = weighted_vol_sum / total_eval
                total_loss_pct = (total_eval - total_buy) / total_buy * 100
                t_stats = calc_recovery_math(total_buy, total_eval, port_volatility)
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("총 매수금", f"{int(total_buy):,}원"); c2.metric("총 평가금", f"{int(total_eval):,}원")
                c3.metric("총 손익률", f"{total_loss_pct:.2f}%", delta_color="inverse")
                if t_stats:
                    c4.metric("원금회복 필요수익", f"+{t_stats['req_return']:.2f}%")
                    st.markdown(f"""<div class="recovery-card">📊 <b>진단 결과:</b> 3개월(60영업일) 내 회복 확률: <span style="font-size:1.5em; color:#d32f2f; font-weight:bold;">{t_stats['prob_3m']:.1f}%</span></div>""", unsafe_allow_html=True)
                
                st.markdown("---")
                st.markdown("#### ⚖️ AI & 수익률 기반 리밸런싱 제안")
                rebal_res = analyze_rebalancing_suggestion(rebal_data)
                if rebal_res:
                    col_r1, col_r2 = st.columns(2)
                    with col_r1:
                        for rb in rebal_res[:len(rebal_res)//2 + 1]:
                            st.markdown(f"""<div class="rebal-card" style="border-left: 5px solid {rb['color']}"><b>{rb['name']}</b> (AI:{rb['score']}점 / {rb['profit']:.1f}%) → <span style="color:{rb['color']}; font-weight:bold;">{rb['action']}</span><br><span style="color:#555; font-size:0.85em;">{rb['reason']}</span></div>""", unsafe_allow_html=True)
                    with col_r2:
                        for rb in rebal_res[len(rebal_res)//2 + 1:]:
                            st.markdown(f"""<div class="rebal-card" style="border-left: 5px solid {rb['color']}"><b>{rb['name']}</b> (AI:{rb['score']}점 / {rb['profit']:.1f}%) → <span style="color:{rb['color']}; font-weight:bold;">{rb['action']}</span><br><span style="color:#555; font-size:0.85em;">{rb['reason']}</span></div>""", unsafe_allow_html=True)

with tabs[6]:
    st.subheader("📈 AI 성장 일기 (Portfolio Performance)")
    
    col_btn, _ = st.columns([1, 4])
    with col_btn:
        if st.button("🗑️ 기록 초기화", type="primary", use_container_width=True, key="del_final_v8158"):
            try:
                conn = st.connection("gsheets", type=GSheetsConnection)
                empty_df = pd.DataFrame(columns=['Date', 'Code', 'Name', 'Entry_Price', 'Target_Price', 'Stop_Price', 'Strategy', 'Buys_Info', 'Sells_Info'])
                conn.update(worksheet="history", data=empty_df)
                st.cache_data.clear()
                st.toast("초기화 완료!", icon="✨"); time.sleep(1); st.rerun()
            except: pass

    df_history = get_scan_history()

    # [V81.58 Fix] Check for None explicitly before accessing .empty
    if df_history is not None and not df_history.empty:
        if 'Date' in df_history.columns:
            df_history = df_history.sort_values('Date', ascending=False)

        st.markdown("### 📊 Overall Statistics")
        total_cnt = len(df_history)
        try:
            avg_plan_profit = ((pd.to_numeric(df_history['Target_Price'], errors='coerce') - pd.to_numeric(df_history['Entry_Price'], errors='coerce')) / pd.to_numeric(df_history['Entry_Price'], errors='coerce') * 100).mean()
        except: avg_plan_profit = 0.0

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("📋 전체 포착", f"{total_cnt}개")
        m2.metric("🎯 평균 목표수익", f"{avg_plan_profit:.1f}%")
        
        p_filled = m3.empty()
        p_win = m4.empty()
        p_filled.metric("🛒 실제 진입", "계산중...")
        p_win.metric("💯 누적 승률", "계산중...")
        
        st.divider()

        filled_cnt = 0 
        win_cnt = 0       

        st.caption(f"👇 총 {total_cnt}개 종목의 최신 시세와 컨센서스를 분석합니다...")
        progress_bar = st.progress(0)

        for i, (idx, row) in enumerate(df_history.iterrows()):
            try:
                progress_bar.progress((i + 1) / total_cnt)
                code = str(row['Code']).zfill(6)
                entry_p = float(row['Entry_Price'])
                target_p = float(row['Target_Price'])

                try:
                    buys = json.loads(row.get('Buys_Info', '[]'))
                    sells = json.loads(row.get('Sells_Info', '[]'))
                except: buys, sells = [], []
                
                if not buys: buys = [entry_p]
                if not sells: sells = [target_p]

                curr_p, day_low, day_high = entry_p, 0, 0
                try:
                    df_now = fdr.DataReader(code, datetime.datetime.now() - timedelta(days=5))
                    if not df_now.empty:
                        curr_p = float(df_now['Close'].iloc[-1])
                        day_low = float(df_now['Low'].min())
                        day_high = float(df_now['High'].max())
                except: pass

                con_price, con_opinion = 0, 0.0
                try:
                    con_price, con_opinion = get_consensus_data(code)
                except: pass

                buy_step = 0
                for bp in buys:
                    if day_low <= float(bp) * 1.01: buy_step += 1
                
                sell_step = 0
                if buy_step > 0:
                    filled_cnt += 1 
                    for sp in sells:
                        if day_high >= float(sp): sell_step += 1
                    if sell_step > 0: win_cnt += 1 

                status_emoji = "⏳"
                status_msg = "대기"
                profit_str = ""
                
                if buy_step > 0:
                    profit = (curr_p - float(buys[0])) / float(buys[0]) * 100
                    profit_str = f"({profit:+.2f}%)"
                    if sell_step > 0:
                        status_emoji = "🎉"; status_msg = f"{sell_step}차 익절"
                    else:
                        status_emoji = "🔴" if profit > 0 else "🔵"
                        status_msg = f"{buy_step}차 보유"
                else:
                    gap = (float(buys[0]) - curr_p) / curr_p * 100
                    profit_str = f"(괴리 {gap:.1f}%)"
                    status_msg = "미체결"

                label = f"{status_emoji} **{row['Name']}** │ {status_msg} │ 현재: {curr_p:,.0f}원 {profit_str}"
                
                with st.expander(label):
                    if con_price > 0:
                        up_pot = (con_price - curr_p) / curr_p * 100
                        con_msg = f"🎯 **증권사 컨센서스**: 목표가 **{con_price:,}원** (괴리율 {up_pot:+.1f}%) │ 투자의견: {con_opinion}/5.0"
                        if up_pot > 0: st.info(con_msg)
                        else: st.warning(con_msg)
                    else:
                        st.caption("📉 증권사 컨센서스(목표가) 데이터가 없습니다.")

                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown("##### 🔵 매수 단계 (Buying)")
                        for idx_b, p in enumerate(buys):
                            chk = "✅ **체결**" if day_low <= float(p)*1.01 else "⏳"
                            st.write(f"- {idx_b+1}차: {float(p):,.0f}원 {chk}")
                    with c2:
                        st.markdown("##### 🔴 매도 단계 (Selling)")
                        for idx_s, p in enumerate(sells):
                            chk = "🎉 **달성**" if (buy_step > 0 and day_high >= float(p)) else "🎯"
                            st.write(f"- {idx_s+1}차: {float(p):,.0f}원 {chk}")
                    
                    st.caption(f"Captured Strategy: {row['Strategy']}")

            except Exception as e: continue
        
        progress_bar.empty()
        win_rate = (win_cnt / filled_cnt * 100) if filled_cnt > 0 else 0.0
        p_filled.metric("🛒 실제 진입", f"{filled_cnt}개")
        p_win.metric("💯 누적 승률", f"{win_rate:.1f}%")

    else: st.info("📭 기록이 없습니다.")

with tabs[7]:
    st.subheader("💾 AI 모델 동기화")
    col_export, col_import = st.columns(2)
    with col_export:
        st.markdown("#### 📤 내보내기")
        if os.path.exists(MODEL_FILE):
            with open(MODEL_FILE, "rb") as f:
                st.download_button(label="🧠 AI 모델 파일 다운로드 (.pkl)", data=f, file_name="ai_ensemble_model.pkl", mime="application/octet-stream")
        else: st.warning("⚠️ 학습된 모델이 없습니다.")
    with col_import:
        st.markdown("#### 📥 가져오기")
        uploaded_file = st.file_uploader("외부 모델 파일 업로드", type=["pkl"])
        if uploaded_file is not None:
            if st.button("🧠 모델 덮어쓰기"):
                with open(MODEL_FILE, "wb") as f: f.write(uploaded_file.getbuffer())
                st.success("✅ 적용 완료 (새로고침 필요)"); time.sleep(2); st.rerun()

if 'generated_report' in st.session_state and st.session_state['generated_report']:
    st.markdown("---"); st.subheader("📝 생성된 마감 리포트")
    with st.expander("▼ 리포트 내용 확인하기 (클릭)", expanded=True):
        st.markdown(st.session_state['generated_report'], unsafe_allow_html=True)
        if st.button("닫기 (화면 지우기)"): del st.session_state['generated_report']; st.rerun()
