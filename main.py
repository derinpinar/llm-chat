import os
from fastapi import FastAPI, HTTPException, Depends, Header, Request
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv
from os.path import join, dirname

# Load environment variables
dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)

from ask_ai import main as ask_ai_main

app = FastAPI(
    title="Kripto Trading AI API",
    description="Trading analiz ve tahmin API'si",
    version="1.0.0"
)

# API Key Security
API_KEY_NAME = "Bottom-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

def get_api_key():
    return os.getenv("BOTTOM-API_KEY")


async def verify_api_key(api_key: str = Depends(api_key_header)):
    expected_key = get_api_key()
    if not expected_key:
        raise HTTPException(status_code=500, detail="API key not configured on server")
    if api_key != expected_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return api_key


# Request Model
class AskAIRequest(BaseModel):
    SessionId: Optional[str] = None
    Prompt: Optional[str] = None
    CoinPair: str
    Analyst: str
    ActivationDate: str
    Position: str = "long"
    IsNewSession: bool = False


# Response Model
class AskAIResponse(BaseModel):
    message: str
    session_id: str
    is_new_session: bool


@app.post("/ask-ai", response_model=AskAIResponse, dependencies=[Depends(verify_api_key)])
async def ask_ai_endpoint(request_body: AskAIRequest, request: Request):
    """
    Trading analizi için AI'ya soru sor.
    
    - **SessionId**: Oturum ID'si (opsiyonel, otomatik oluşturulur)
    - **Prompt**: Kullanıcı sorusu (opsiyonel)
    - **CoinPair**: İşlem çifti (örn: BTCUSDT, ETHUSDT)
    - **Analyst**: Analist adı (örn: FLASHH)
    - **ActivationDate**: İşlem açılış tarihi (örn: 2026-01-19 01:22:00)
    - **Position**: İşlem pozisyonu (long/short)
    - **IsNewSession**: Yeni oturum mu?
    """
    try:
        # Convert request to namespace-like object
        class RequestItem:
            pass
        
        item = RequestItem()
        item.SessionId = request_body.SessionId
        item.Prompt = request_body.Prompt
        item.CoinPair = request_body.CoinPair
        item.Analyst = request_body.Analyst
        item.ActivationDate = request_body.ActivationDate
        item.Position = request_body.Position
        item.IsNewSession = request_body.IsNewSession
        
        # Get headers as dict
        headers = dict(request.headers)
        
        result = ask_ai_main(headers, item)
        
        return AskAIResponse(**result)
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "version": "1.0.0"}
