from fastapi import FastAPI
import logging
from fastapi.middleware.cors import CORSMiddleware
from routes.Dataroutes import router as DataRouter
import torch

logging.basicConfig(level=logging.INFO)
logger: logging.Logger = logging.getLogger(__name__)

logger.info(f"Running on GPU instance: {torch.cuda.is_available()}")

app = FastAPI(title="Data Extractor AI", description="API for Data Extraction", version="0.0.1")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# register router
app.include_router(DataRouter, prefix="/ai/api", tags=["Data Extraction"])

if __name__ == '__main__':
    import uvicorn
    uvicorn.run("main:app", port=1234, host="0.0.0.0", reload=True)