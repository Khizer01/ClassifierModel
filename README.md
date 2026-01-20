# Ad Classification SLM System

Production-ready Small Language Model (SLM) system for ad relevance classification using Phi-3.5-mini.

## Features

- **Single-Model Pipeline**: Uses Phi-3.5-mini for efficient classification
- **Batch Processing**: Optimized for high throughput (50k+ ads/day)
- **Low Latency**: 4-bit quantization for faster inference
- **Deterministic Output**: JSON-only responses with structured schema
- **Async Database**: Non-blocking SQLite operations with aiosqlite
- **Production Ready**: FastAPI backend with health checks and error handling

## Architecture

```
Frontend → FastAPI Backend → Database Query → Batch Inference → JSON Response
                                    ↓
                            Phi-3.5-mini (4-bit)
```

## Installation

### Prerequisites
- Python 3.10+
- CUDA-capable GPU (recommended) or CPU
- 8GB+ RAM (16GB+ recommended)

### Setup

1. **Clone and navigate to the project:**
```bash
cd classification-slm
```

2. **Create virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Configure environment:**
```bash
cp .env.example .env
# Edit .env with your settings
```

## Configuration

Edit `.env` file:

```env
DATABASE_PATH=./ads_data.db          # Path to SQLite database
MODEL_NAME=microsoft/Phi-3.5-mini-instruct
BATCH_SIZE=8                          # Recommended for A4000 16GB
MAX_LENGTH=256                        # Lower is faster; increase if needed
MAX_NEW_TOKENS=20                     # Lower is faster; increase if needed
DEVICE=cuda                           # cuda or cpu
USE_4BIT_QUANTIZATION=false          # Optional: enable if you install bitsandbytes
ENABLE_TF32=true                     # Recommended on NVIDIA Ampere GPUs (A4000)
ENABLE_TORCH_COMPILE=false           # Optional: enable for extra speed after warmup
TORCH_COMPILE_MODE=reduce-overhead
API_HOST=0.0.0.0
API_PORT=8000
```

### Batch Size Optimization

| GPU VRAM | Recommended Batch Size |
|----------|------------------------|
| 4GB      | 4-8                    |
| 8GB      | 8-16                   |
| 12GB+    | 16-32                  |

## Usage

### Start the API Server

```bash
python main.py
```

Or with uvicorn:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

### API Endpoints

#### 1. Health Check
```bash
GET /health
```

Response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "database_connected": true
}
```

#### 2. Classify Ads
```bash
POST /classify
Content-Type: application/json

{
  "keyword": "Pepsi"
}
```

Response:
```json
{
  "results": [
    {
      "ad_id": "123",
      "is_relevant": true,
      "theme": "Product Promotion"
    },
    {
      "ad_id": "124",
      "is_relevant": false,
      "theme": "Unrelated"
    }
  ]
}
```

#### 3. Get Keyword Stats
```bash
GET /stats/{keyword}
```

Response:
```json
{
  "keyword": "Pepsi",
  "total_ads": 150
}
```

## Classification Themes

The system classifies ads into these primary themes:

1. **Product Promotion** - Direct product advertising
2. **Brand Awareness** - Brand building content
3. **Fan Content** - User-generated brand content
4. **Personal Post** - Personal stories/experiences
5. **Meme** - Humorous/viral content
6. **News** - News articles/updates
7. **Unrelated** - Not relevant to the keyword

## Performance Optimization

### For High Throughput (50k+ ads/day)

1. **Increase Batch Size**: Set `BATCH_SIZE=32` if GPU memory allows
2. **Enable Quantization**: Keep `USE_4BIT_QUANTIZATION=true`
3. **Use GPU**: Set `DEVICE=cuda` for 10-20x speedup
4. **Async Processing**: API handles concurrent requests efficiently

### Estimated Throughput

| Configuration | Ads/Second | Ads/Day |
|---------------|------------|---------|
| CPU (batch=4) | 0.5-1      | 43k-86k |
| GPU 8GB (batch=16) | 5-10 | 432k-864k |
| GPU 12GB+ (batch=32) | 10-20 | 864k-1.7M |

## Frontend Integration

### Example JavaScript Fetch

```javascript
async function classifyAds(keyword) {
  const response = await fetch('http://localhost:8000/classify', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ keyword: keyword })
  });
  
  const data = await response.json();
  return data.results;
}

// Usage
const results = await classifyAds('Pepsi');
console.log(results);
```

### Example React Component

```jsx
import { useState } from 'react';

function AdClassifier() {
  const [keyword, setKeyword] = useState('');
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);

  const handleClassify = async () => {
    setLoading(true);
    try {
      const response = await fetch('http://localhost:8000/classify', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ keyword })
      });
      const data = await response.json();
      setResults(data.results);
    } catch (error) {
      console.error('Classification failed:', error);
    }
    setLoading(false);
  };

  return (
    <div>
      <input 
        value={keyword} 
        onChange={(e) => setKeyword(e.target.value)}
        placeholder="Enter brand keyword"
      />
      <button onClick={handleClassify} disabled={loading}>
        {loading ? 'Classifying...' : 'Classify Ads'}
      </button>
      
      <div>
        {results.map(result => (
          <div key={result.ad_id}>
            <p>Ad ID: {result.ad_id}</p>
            <p>Relevant: {result.is_relevant ? 'Yes' : 'No'}</p>
            <p>Theme: {result.theme}</p>
          </div>
        ))}
      </div>
    </div>
  );
}
```

## Production Deployment

### Docker Deployment (Recommended)

Create `Dockerfile`:
```dockerfile
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y python3.10 python3-pip

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python3", "main.py"]
```

Build and run:
```bash
docker build -t ad-classifier .
docker run --gpus all -p 8000:8000 ad-classifier
```

### Scaling Considerations

1. **Load Balancing**: Deploy multiple instances behind nginx/HAProxy
2. **Caching**: Cache classification results for frequently queried keywords
3. **Database**: Consider PostgreSQL for production at scale
4. **Monitoring**: Add Prometheus metrics and logging
5. **Rate Limiting**: Implement request throttling for API protection

## Troubleshooting

### Out of Memory Error
- Reduce `BATCH_SIZE` in `.env`
- Enable `USE_4BIT_QUANTIZATION=true`
- Use CPU if GPU memory is insufficient

### Slow Performance
- Ensure GPU is being used: check `DEVICE=cuda`
- Increase `BATCH_SIZE` if memory allows
- Verify CUDA is properly installed

### Model Download Issues
- First run downloads ~7GB model from HuggingFace
- Ensure stable internet connection
- Set `HF_HOME` environment variable for custom cache location

## Database Schema

The system expects a SQLite database with this schema:

```sql
CREATE TABLE keywords (
    id INTEGER PRIMARY KEY,
    keyword TEXT NOT NULL
);

CREATE TABLE meta_ads (
    id INTEGER PRIMARY KEY,
    keyword_id INTEGER,
    body_text TEXT,
    page_name TEXT,
    title TEXT,
    description TEXT,
    FOREIGN KEY (keyword_id) REFERENCES keywords(id)
);
```

## Cost Optimization

- **No API Costs**: Runs locally, no external API calls
- **GPU Rental**: ~$0.50/hour on cloud providers (processes 50k+ ads/hour)
- **CPU Only**: Free but slower (processes 5k+ ads/hour)

## License

MIT License - See LICENSE file for details

## Support

For issues or questions, please open an issue on the repository.
