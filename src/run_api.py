#!/usr/bin/env python3
"""
Startup script for the Chess API.
Run this to start the FastAPI server.
"""
import uvicorn

if __name__ == "__main__":
    print("Starting Chess API server...")
    print("The API will be available at: http://localhost:8000")
    print("Interactive docs available at: http://localhost:8000/docs")
    print("Press Ctrl+C to stop the server")
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
