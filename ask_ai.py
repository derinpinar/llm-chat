import sys, os
from datetime import datetime, timedelta
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_openai import ChatOpenAI
from langchain_core.runnables.history import RunnableWithMessageHistory
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
import uuid
import json
import requests
import warnings
from os.path import join, dirname
#import logging
import pickle
import pandas as pd


# logger = logging.getLogger('ASK-AI')
# logger.setLevel(logging.INFO)
# today_ymd = datetime.today().strftime('%Y-%m-%d')
# file_handler = logging.FileHandler(f"chat_{today_ymd}.log")
# file_handler.setLevel(logging.INFO)
#
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# file_handler.setFormatter(formatter)
# logger.addHandler(file_handler)
# console_handler = logging.StreamHandler()
# console_handler.setFormatter(formatter)
# logger.addHandler(console_handler)
#
# logging.getLogger().setLevel(logging.WARNING)

warnings.filterwarnings("ignore")
dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)

store = {}  # memory chain dışında tutulmalı saatlik temizleniyor ilerde redis e geçilebilir.

class ResponseModel:
    def __init__(self, message, session_id, is_new_session):
        self.message = message
        self.session_id = session_id
        self.is_new_session = is_new_session

    def to_json(self):
        return {
            'message': self.message,
            'session_id': self.session_id,
            'is_new_session': self.is_new_session}


def remove_old_objects(objects, time_threshold):
    current_time = datetime.now()
    keys_to_delete = [key for key, value in objects.items() if current_time - value['last_processed'] > time_threshold]
    for key in keys_to_delete:
        del objects[key]


def init_history(_context):
    history = ChatMessageHistory()
    history.add_message(_context)
    return history


def start_session(session_id, is_new_session,  prompt, context):
    if is_new_session is False:
        remove_old_objects(store, timedelta(hours=1))
        if session_id in store:
            history = store[session_id]['session_obj']
        else:
            history = init_history(context)
    else:
        is_new_session = True
        history = init_history(context)

    def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
        if session_id not in store:
            data = {'last_processed': datetime.now(),
                    'session_obj': history}
            store[session_id] = data
        return store[session_id]['session_obj']

    def generate_response(langchain_llm):
        chain = RunnableWithMessageHistory(langchain_llm, get_session_history)
        _response = chain.invoke(
            [HumanMessage(content=prompt)
             ],
            config={"configurable": {"session_id": session_id}},
        )
        return _response

    # DeepSeek v3.2 configuration
    llm = ChatOpenAI(
        model='deepseek-chat',
        temperature=0.2,
        base_url='https://api.deepseek.com',
        api_key=os.getenv('DEEPSEEK_API_KEY')
    )
    model_name = 'deepseek-chat'
    response = generate_response(llm)

    session_data = {'is_new_session': is_new_session,
                    'session_id': session_id,
                    'model_name': model_name,
                    'total_token': response.usage_metadata['total_tokens']}

    return str(response.content), session_data


def select_context_prompt():
    today = datetime.today()
    today_string = f""" If you need to calculate a date or age, today is {today.strftime("%Y-%m-%d")}."""

    system_message = """# Trading Analysis AI - System Prompt
    Sen bir kripto para trading analiz asistanısın. Görevin, kullanıcılara aktif işlemler hakkında kısa ve öz bilgi vermek.
    
    ## Sana Verilen Veriler
    
    ### 1. İşlem Bilgisi
    - **coin_name**: İşlem yapılan pair (örn: BTCUSDT)
    - **Kar-Zarar Modeli Tahmini**: ML model tahmini
        - > 1.0: Karlı işlem beklentisi
        - = 1.0: Nötr
        - < 1.0: Zararlı işlem beklentisi
    - **kill_zone**: İşlem zamanlaması (Asian/London/NY)
    - **session_main**: Ana seans bilgisi
    
    ### 2. Coin Glass Verileri (coin_glass_info)
    Gerçek zamanlı piyasa verileri:
    - current_price: Güncel fiyat
    - price_change_percent_1h/4h/12h/24h: Fiyat değişim yüzdeleri
    - volume_usd_24h: 24 saatlik işlem hacmi
    - buy_volume vs sell_volume: Alım/satım baskısı
    - net_flows: Net para akışı (negatif = satış baskısı, pozitif = alım baskısı)
    
    
    ## Yanıt Formatı
    1. **Durum Özeti**: İşlemin genel durumu (2-3 cümle)
    2. **Piyasa Analizi**: Fiyat ve hacim bazlı kısa yorum
    3. **Zamanlama**: Kill zone ve session değerlendirmesi
    4. **Sonuç**: Net tavsiye veya uyarı
    
    ## Kurallar
    - Kısa ve net ol, maksimum 150 kelime
    - Teknik jargon kullan ama anlaşılır ol
    - Net flows ve volume değişimlerini mutlaka yorumla
    - Modelin değerini her zaman vurgula
    - Analist ismini açıklmanda vurgula vurgula
    - Emoji kullanabilirsin (📈📉⚠️✅)
    - Türkçe yanıt ver
    - Kesin al/sat tavsiyesi verme, sadece analiz yap
    """ + today_string

    return system_message


def check_coin_pair_exists(coin_pair: str) -> bool:
    """
    1. Coin pair'in desteklenip desteklenmediğini kontrol eder
    2. Destekleniyorsa OKX exchange verisini döndürür

    Args:
        coin_pair: Coin sembolü (örn: "BTC", "ETH")

    Returns:
        dict: OKX verisi (bulunursa)
        False: Pair desteklenmiyorsa veya OKX verisi yoksa
    """
    api_key = os.getenv('CG-API-KEY')

    headers = {
        "accept": "application/json",
        "CG-API-KEY": api_key
    }

    # 1- COIN supported mı diye bak.
    try:
        check_url = "https://open-api-v4.coinglass.com/api/spot/supported-exchange-pairs"
        response = requests.get(check_url, headers=headers, timeout=10)
        data = response.json()

        if data.get("code") != "0":
            return False

        exchanges_data = data.get("data", {})

        pair_exists = False
        for exchange_name, pairs in exchanges_data.items():
            for pair in pairs:
                if pair.get("instrument_id") == coin_pair:
                    pair_exists = True
                    break
            if pair_exists:
                break

        if not pair_exists:
            print(f"Pair {coin_pair} bulunamadı")
            return False

    except Exception as e:
        print(f"Pair kontrol hatası: {e}")
        return False

    # 2- OKX verisini al
    try:
        # coin_pair'den base asset'i çıkar (BTCUSDT -> BTC)
        symbol = coin_pair.replace("USDT", "").replace("USDC", "").replace("BUSD", "")

        market_url = f"https://open-api-v4.coinglass.com/api/spot/pairs-markets?symbol={symbol}"
        response = requests.get(market_url, headers=headers, timeout=10)
        data = response.json()

        if data.get("code") != "0":
            return False

        markets = data.get("data", [])

        # OKX'i bul ve döndür
        for market in markets:
            if market.get("exchange_name") == "OKX":
                return market

        print(f"OKX bulunamadı, ilk sıradaki döndürülüyor: {markets[0].get('exchangeName')}")
        return markets[0]

    except Exception as e:
        print(f"Market veri hatası: {e}")
        return False


def analyst_model_prediction(analyst, coin_pair, position, activation_time):
    """
    Predict Net R using trained XGBoost model
    
    Args:
        analyst: Analyst name (e.g., 'FLASHH')
        coin_pair: Coin pair (e.g., 'ETHUSDT')
        position: Position type (default: 'long')
        activation_time: Activation time string in UTC+3 (e.g., '2026-01-19 01:22:00')
    
    Returns:
        dict: Prediction result with Net R and features
    """
    
    try:
        # Load model and feature columns
        model_path = join(dirname(__file__), 'xgboost_model.pkl')
        features_path = join(dirname(__file__), 'feature_columns.pkl')
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        with open(features_path, 'rb') as f:
            feature_columns = pickle.load(f)
        
        # Convert activation_time from UTC+3 to UTC
        if activation_time:
            activation_dt_utc3 = pd.to_datetime(activation_time)
            activation_dt_utc = activation_dt_utc3 - pd.Timedelta(hours=3)
        else:
            # Use current time if not provided
            activation_dt_utc3 = datetime.now()
            activation_dt_utc = activation_dt_utc3 - pd.Timedelta(hours=3)
        
        # Calculate kill_zone and session_main
        def get_kill_zone(dt):
            hour = dt.hour
            if 0 <= hour < 3:
                return 'AsiaKZ'
            elif 7 <= hour < 10:
                return 'LondonKZ'
            elif 13 <= hour < 16:
                return 'NewYorkKZ'
            else:
                return None
        
        def get_main_session(dt):
            hour = dt.hour
            if 0 <= hour < 7:
                return 'Asia'
            elif 7 <= hour < 13:
                return 'London'
            elif 13 <= hour < 22:
                return 'NewYork'
            else:
                return 'OffHours'
        
        kill_zone = get_kill_zone(activation_dt_utc)
        session_main = get_main_session(activation_dt_utc)
        
        # Create input data
        input_data = pd.DataFrame({
            'analysts': [analyst],
            'coin_name': [coin_pair],
            'position': [position],
            'kill_zone': [kill_zone],
            'session_main': [session_main]
        })
        
        # One-hot encode
        categorical_features = ['analysts', 'coin_name', 'position', 'kill_zone', 'session_main']
        input_encoded = pd.get_dummies(input_data, columns=categorical_features, drop_first=True)
        
        # Align with training features
        for col in feature_columns:
            if col not in input_encoded.columns:
                input_encoded[col] = 0
        
        # Reorder columns to match training data
        input_encoded = input_encoded[feature_columns]
        
        # Predict
        prediction = model.predict(input_encoded)
        
        return {
            'success': True,
            'predicted_net_r': float(prediction[0]),
            'kill_zone': kill_zone,
            'session_main': session_main,
            'activation_time_utc': str(activation_dt_utc),
            'activation_time_utc3': str(activation_dt_utc3)
        }
        
    except Exception as e:
        #logger.error(f"Model prediction error: {str(e)}")
        return {
            'success': False,
            'error': str(e),
            'predicted_net_r': None
        }


def generate_user_prompt(analyst, coin_name, position, analyst_result, coin_glass_info, user_prompt=None):
    user_message = f""" {analyst} Analistinin İşlem Açma Bilgisi
    Modellenen kazanç/kayıp oranı {analyst} traderın işlem açma zamanlarının kill zone 
    {analyst_result['kill_zone']}, aktif sessionları {analyst_result['session_main']}, coin geçmiş verileri
    modellenerek {coin_name} {position} işlemi için yapılan kazanç/kayıp tahmini: {analyst_result['predicted_net_r']} dir. 
    
    Coin Güncel Piyasa Verileri:
    {json.dumps(coin_glass_info, indent=2)}
    """

    if user_prompt:
        user_message += f"""Kullanıcı Sorusu:
    "{user_prompt}"
    Yukarıdaki verileri kullanarak kullanıcının sorusunu yanıtla."""
    else:
        user_message += f"""Bu işlem hakkında analiz yap. """

    return user_message


def main(headers, item):
    session_id = getattr(item, 'SessionId', None) or str(uuid.uuid4())
    user_prompt = item.Prompt
    coin_pair = item.CoinPair
    analyst = item.Analyst
    activation_time = item.ActivationDate
    position = item.Position
    is_new_session = item.IsNewSession

    if headers.get('Bottom-API-Key'):
        print('Validate Token')
    else:
        print('Not Validated')

    coin_glass_info = check_coin_pair_exists(coin_pair)

    if coin_glass_info:
        analyst_result = analyst_model_prediction(analyst, coin_pair, position, activation_time)
        context = select_context_prompt()
        prompt = generate_user_prompt(analyst, coin_pair, position, analyst_result, coin_glass_info, user_prompt)
        response, session_data = start_session(session_id, is_new_session, prompt, context)
        return ResponseModel(response, session_data['session_id'], session_data['is_new_session']).to_json()
    else:
        response= 'There is no sufficient information for analysis and review.'
        return ResponseModel(response, None, False).to_json()


