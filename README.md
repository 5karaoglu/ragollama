# RAG SQL Uygulaması

Bu uygulama, doğal dil sorgularını SQL sorgularına dönüştüren bir RAG (Retrieval Augmented Generation) uygulamasıdır.

## Özellikler

- Ollama ile LLM entegrasyonu
- ChromaDB ile vektör veritabanı
- FastAPI ile REST API
- Docker Compose ile kolay kurulum

## Gereksinimler

- Docker
- Docker Compose
- NVIDIA GPU (Ollama için)

## Kurulum

1. Projeyi klonlayın:
```bash
git clone <repo-url>
cd <repo-name>
```

2. Docker Compose ile uygulamayı başlatın:
```bash
docker-compose up --build
```

3. Ollama modelini indirin:
```bash
docker exec -it ragollama-ollama-1 ollama pull unsloth/DeepSeek-R1-Distill-Qwen-14B-unsloth-bnb-4bit
```

## Kullanım

Uygulama http://localhost:8000 adresinde çalışır.

### API Endpointleri

- `GET /`: Ana sayfa
- `POST /query`: SQL sorgusu oluşturma

### Örnek Kullanım

```bash
curl -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -d '{"question": "Tüm kullanıcıları listele"}'
```

## Lisans

MIT 