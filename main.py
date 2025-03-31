from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import json
from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import pickle
from pathlib import Path
import subprocess
import time
import re

app = FastAPI(title="RAG SQL Uygulaması")

# Sabit yapılandırmalar
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
MODEL_NAME = "deepseek-r1:14b-qwen-distill-q4_K_M"  # DeepSeek-R1-Distill-Qwen-14B 4-bit quantization modeli
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

class QueryResponse(BaseModel):
    question: str
    answer: str
    context: str
    think: str  # Yeni eklenen alan

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

@app.post("/query", response_model=QueryResponse)
async def query(question: Query):
    try:
        # Veri setini oku
        with open("Book1.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Veriyi metin formatına dönüştür
        text_data = json.dumps(data, ensure_ascii=False, indent=2)
        
        # Metni parçalara ayır
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_text(text_data)
        
        # Embedding modelini yükle
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={'device': 'cuda'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Vektör veritabanını oluştur
        vectorstore = Chroma.from_texts(
            texts=chunks,
            embedding=embeddings,
            persist_directory=str(VECTOR_STORE_DIR)
        )
        
        # En alakalı parçaları bul
        relevant_docs = vectorstore.similarity_search(question.question, k=3)
        context = "\n".join([doc.page_content for doc in relevant_docs])
        
        # LLM modelini yükle
        llm = Ollama(
            base_url=OLLAMA_HOST,
            model=MODEL_NAME,
            temperature=0.1,
            num_ctx=4096,
            num_gpu=1,
            num_thread=8,
            repeat_penalty=1.1,
            top_k=10,
            top_p=0.7,
            tfs_z=1,
            num_predict=512,
            stop=["</think>", "</sql>", "</result>"]
        )
        
        # Prompt oluştur
        prompt = f"""Veri seti:
{context}

Soru: {question.question}

Lütfen bu soruyu yanıtlamak için bir SQL sorgusu oluştur. Sorgu JSON verisini analiz etmeli ve soruyu yanıtlamalı.

<think>
1. Veri setinin yapısını analiz et
2. Hangi alanları kullanmam gerekiyor?
3. Nasıl bir SQL sorgusu oluşturmalıyım?
</think>

<sql>
SQL sorgusunu buraya yaz
</sql>

<result>
Sorgu sonucunu buraya yaz
</result>"""

        # LLM'den yanıt al
        response = llm(prompt)
        
        # Yanıtı parçalara ayır
        think_match = re.search(r'<think>(.*?)</think>', response, re.DOTALL)
        sql_match = re.search(r'<sql>(.*?)</sql>', response, re.DOTALL)
        result_match = re.search(r'<result>(.*?)</result>', response, re.DOTALL)
        
        think_content = think_match.group(1).strip() if think_match else ""
        sql_content = sql_match.group(1).strip() if sql_match else ""
        result_content = result_match.group(1).strip() if result_match else ""
        
        return QueryResponse(
            question=question.question,
            answer=result_content,
            context=context,
            think=think_content  # Think tag'i içindeki değerleri döndür
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    """Ana sayfa"""
    return {"message": "RAG SQL Uygulamasına Hoş Geldiniz!"} 