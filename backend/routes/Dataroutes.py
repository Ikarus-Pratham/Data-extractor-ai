from fastapi import APIRouter, File, Form, UploadFile
from controllers.ExtractData import extractData
from file_queue import FileQueue
import json
import os

router = APIRouter()
queue = FileQueue()

@router.get("/health")
async def healthCheckRoute():
    """
    Health check endpoint to verify the API is running.

    Returns:
        dict: A simple status message indicating the API is running.
    """
    return {"status": "ok"}

@router.post("/extract")
async def extractDataRoute(
    pdf_file: UploadFile = File(...),
    pages: str = Form(...)
):
    """
    Enqueue a data extraction job and return job_id.
    """
    # Stream save the uploaded PDF to disk to avoid loading the whole file into memory
    from config.config import Config
    config = Config()
    if not os.path.exists(config.Dirs.PDF_DIR):
        os.makedirs(config.Dirs.PDF_DIR, exist_ok=True)
    target_path = config.Dirs.PDF_DIR + "/" + pdf_file.filename
    CHUNK_SIZE = 1024 * 1024  # 1 MB chunks
    with open(target_path, "wb") as f:
        while True:
            chunk = await pdf_file.read(CHUNK_SIZE)
            if not chunk:
                break
            f.write(chunk)
    # Reset file pointer (not strictly necessary after streaming)
    await pdf_file.close()
    pages_dict = json.loads(pages)
    job_id = queue.enqueue({
        "pdf_path": target_path,
        "pdf_filename": pdf_file.filename,
        "pages": pages_dict
    })
    return {"job_id": job_id, "status": "queued"}

@router.get("/status/{job_id}")
async def get_status(job_id: str):
    status = queue.get_status(job_id)
    return {"job_id": job_id, "status": status}