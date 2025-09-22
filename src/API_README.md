# Chess API

This directory contains a FastAPI-based REST API for the transformer chess model.

## Quick Start

1. **Install dependencies** (if not already done):
   ```bash
   pip install -r requirements.txt
   ```

2. **Test the model locally**:
   ```bash
   python test_api.py
   ```

3. **Start the API server**:
   ```bash
   python run_api.py
   ```
   
   The API will be available at `http://localhost:8000`
   Interactive documentation: `http://localhost:8000/docs`

4. **Test the API** (in a separate terminal):
   ```bash
   python test_client.py
   ```

## API Endpoints

### POST /best-move

Get the best move for a given chess position.

**Request Body:**
```json
{
  "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
}
```

**Response:**
```json
{
  "uci": "e2e4"
}
```

### GET /

Health check endpoint.

**Response:**
```json
{
  "message": "Transformer-Chess API is running"
}
```

## Model Details

- **Model Type**: Bidirectional Transformer (BidirectionalPredictor)
- **Architecture**: 8 layers, 8 attention heads, 256 embedding dimensions
- **Input**: Chess position (FEN) + candidate move
- **Output**: Win probability distribution over 128 buckets
- **Selection**: Best move based on expected win probability

## Files

- `app.py` - FastAPI application
- `utils.py` - Model inference utilities  
- `run_api.py` - Server startup script
- `test_api.py` - Local model testing
- `test_client.py` - API client testing

## Development

To run the server in development mode with auto-reload:
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

## Production Deployment

For production, consider using:
- Gunicorn with multiple workers
- Docker containerization
- HTTPS/TLS termination
- Rate limiting
- Authentication if needed
