from fastapi import FastAPI
from contextlib import asynccontextmanager
import logging
from fastapi.middleware.cors import CORSMiddleware
from routes.Dataroutes import router as DataRouter
import torch
import threading
import time
from file_queue import FileQueue
from controllers.ExtractData import process_extraction

logging.basicConfig(level=logging.INFO)
logger: logging.Logger = logging.getLogger(__name__)

logger.info(f"Running on GPU instance: {torch.cuda.is_available()}")

queue = FileQueue()

def worker_loop() -> None:
    while True:
        item = queue.dequeue()
        if not item:
            time.sleep(0.5)
            continue
        job_id, payload = item
        try:
            process_extraction(
                pdf_path=payload["pdf_path"],
                pdf_filename=payload["pdf_filename"],
                pages=payload["pages"],
            )
            queue.mark_completed(job_id, {"message": "done"})
        except Exception as e:
            queue.mark_failed(job_id, str(e))

@asynccontextmanager
async def lifespan(app: FastAPI):
    # start background worker on startup
    t = threading.Thread(target=worker_loop, daemon=True)
    t.start()
    try:
        yield
    finally:
        # No explicit shutdown; daemon thread exits with process
        pass

app = FastAPI(title="Data Extractor AI", description="API for Data Extraction", version="0.0.1", lifespan=lifespan)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://smhxghmg-5173.inc1.devtunnels.ms/", "https://smhxghmg-5173.inc1.devtunnels.ms"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# register router
app.include_router(DataRouter, prefix="/ai/api", tags=["Data Extraction"])

if __name__ == '__main__':
    import uvicorn
    uvicorn.run("main:app", port=1234, host="0.0.0.0", reload=True)