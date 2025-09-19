from fastapi import APIRouter, File, Form, UploadFile
from controllers.ExtractData import extractData

router = APIRouter()

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
    Extract data from a PDF file and apply operations.

    Returns:
        dict: A simple status message indicating the data extraction is complete.
    """
    return await extractData(pdf_file, pages)