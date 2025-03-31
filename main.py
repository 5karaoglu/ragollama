from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import json
from langchain_community.llms import Ollama
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import pickle
from pathlib import Path
import subprocess
import time

app = FastAPI(title="RAG SQL Uygulaması")

# Sabit yapılandırmalar
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
MODEL_NAME = "unsloth/DeepSeek-R1-Distill-Qwen-14B-unsloth-bnb-4bit"
CACHE_DIR = Path("./cache")
VECTOR_STORE_DIR = Path("./chroma_db")
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_CACHE_FILE = CACHE_DIR / "model_status.json"

# Cache dizinini oluştur
CACHE_DIR.mkdir(exist_ok=True)
VECTOR_STORE_DIR.mkdir(exist_ok=True)

# Veri yapıları
class Query(BaseModel):
    question: str

# Global değişkenler
vector_store = None
llm = None
embeddings = None
raw_data = None

def load_raw_data():
    """JSON verisini yükler"""
    global raw_data
    try:
        with open("Book1.json", "r") as f:
            raw_data = json.load(f)
        return raw_data
    except Exception as e:
        print(f"Veri yükleme hatası: {e}")
        return None

def check_model_exists():
    """Ollama modelinin yüklü olup olmadığını kontrol eder"""
    try:
        result = subprocess.run(
            ["curl", f"{OLLAMA_HOST}/api/tags"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            models = json.loads(result.stdout).get("models", [])
            return any(model.get("name") == MODEL_NAME for model in models)
    except Exception as e:
        print(f"Model kontrolü sırasında hata: {e}")
    return False

def pull_model():
    """Ollama modelini indirir"""
    try:
        print(f"{MODEL_NAME} modeli indiriliyor...")
        result = subprocess.run(
            ["curl", "-X", "POST", f"{OLLAMA_HOST}/api/pull", "-d", json.dumps({"name": MODEL_NAME})],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print("Model başarıyla indirildi!")
            return True
        else:
            print(f"Model indirme hatası: {result.stderr}")
            return False
    except Exception as e:
        print(f"Model indirme sırasında hata: {e}")
        return False

def load_or_create_llm():
    """LLM modelini yükler veya indirir"""
    global llm
    
    # Model durumunu kontrol et
    if MODEL_CACHE_FILE.exists():
        with open(MODEL_CACHE_FILE, "r") as f:
            cache_data = json.load(f)
            if cache_data.get("model_name") == MODEL_NAME and cache_data.get("last_check", 0) > time.time() - 3600:  # 1 saat
                print("Model cache'den yükleniyor...")
                llm = Ollama(
                    base_url=OLLAMA_HOST,
                    model=MODEL_NAME,
                    temperature=0.1
                )
                return llm
    
    # Model yüklü değilse indir
    if not check_model_exists():
        if not pull_model():
            raise Exception("Model indirilemedi!")
    
    # Model durumunu cache'le
    with open(MODEL_CACHE_FILE, "w") as f:
        json.dump({
            "model_name": MODEL_NAME,
            "last_check": time.time()
        }, f)
    
    # LLM'i başlat
    llm = Ollama(
        base_url=OLLAMA_HOST,
        model=MODEL_NAME,
        temperature=0.1
    )
    return llm

def load_or_create_embeddings():
    """Embedding modelini cache'den yükler veya yeni oluşturur"""
    global embeddings
    cache_file = CACHE_DIR / "embeddings.pkl"
    
    if cache_file.exists():
        print("Embedding modeli cache'den yükleniyor...")
        with open(cache_file, "rb") as f:
            embeddings = pickle.load(f)
    else:
        print("Yeni embedding modeli oluşturuluyor...")
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        with open(cache_file, "wb") as f:
            pickle.dump(embeddings, f)
    
    return embeddings

def initialize_rag():
    """RAG sistemini başlatır ve gerekli modelleri yükler"""
    global vector_store, llm, raw_data
    
    # Ham veriyi yükle
    raw_data = load_raw_data()
    if raw_data is None:
        raise Exception("Veri yüklenemedi!")
    
    # LLM modelini yükle
    llm = load_or_create_llm()
    
    # Embedding modelini yükle
    embeddings = load_or_create_embeddings()
    
    # Vector store'u kontrol et
    if (VECTOR_STORE_DIR / "chroma.sqlite3").exists():
        print("Vector store cache'den yükleniyor...")
        vector_store = Chroma(
            persist_directory=str(VECTOR_STORE_DIR),
            embedding_function=embeddings
        )
    else:
        print("Yeni vector store oluşturuluyor...")
        # Veriyi metin formatına dönüştür
        texts = []
        for item in raw_data:
            texts.append(json.dumps(item, ensure_ascii=False))
        
        # Metinleri böl
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        splits = text_splitter.create_documents(texts)
        
        # Vector store'u oluştur
        vector_store = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory=str(VECTOR_STORE_DIR)
        )
        vector_store.persist()

def process_query(question: str, context: str) -> str:
    """Kullanıcı sorusunu işler ve yanıt üretir"""
    prompt = f"""Aşağıdaki veri seti ve soru için yanıt oluştur:

Veri Seti:
{context}

Soru: {question}

Lütfen soruya uygun, doğru ve anlaşılır bir yanıt ver. Eğer veri setinde yeterli bilgi yoksa, bunu belirt."""

    return llm.predict(prompt)

@app.on_event("startup")
async def startup_event():
    """Uygulama başlatıldığında RAG sistemini başlatır"""
    initialize_rag()

@app.post("/query")
async def query_endpoint(query: Query):
    """Kullanıcı sorgusunu işler ve yanıt üretir"""
    try:
        # Benzer dokümanları bul
        docs = vector_store.similarity_search(query.question, k=3)
        
        # Context'i hazırla
        context = "\n".join([doc.page_content for doc in docs])
        
        # Yanıtı oluştur
        response = process_query(query.question, context)
        
        return {
            "question": query.question,
            "answer": response.strip(),
            "context": context
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    """Ana sayfa"""
    return {"message": "RAG SQL Uygulamasına Hoş Geldiniz!"} 