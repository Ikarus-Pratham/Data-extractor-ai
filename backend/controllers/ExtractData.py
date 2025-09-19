from fastapi import HTTPException
from modals.DataRequestModal import DataReqModal
from middlewares.ExtractAndSave import ExtractAndSaveData
from config.config import Config
import logging
import json
import time
import os

config = Config()
logger: logging.Logger = logging.getLogger(__name__)

def process_extraction(pdf_path: str, pdf_filename: str, pages: dict) -> None:
    """
    Pure processing function: performs extraction and saves outputs to disk.
    """
    extractDataAndSave = ExtractAndSaveData(
        pdf_path=pdf_path,
        pdf_dir=config.Dirs.PDF_DIR,
        json_dir=config.Dirs.JSON_DIR,
        result_dir=config.Dirs.RESULT_DIR,
        progress_dir=config.Dirs.PROGRESS_DIR,
        pages=pages,
        model_name=config.Model.NAME,
        temperature=config.Model.TEMPERATURE,
        top_p=config.Model.TOP_P,
        num_ctx=config.Model.NUM_CTX,
        num_predict=config.Model.NUM_PREDICT,
        pdf_filename=pdf_filename
    )

    for k, v in pages.items():
        start_idx = v[0]
        end_idx = v[1]
        pages_to_extract = []
        for i in range(start_idx, end_idx+1):
            pages_to_extract.append(i)

        if not os.path.exists(config.Dirs.PROGRESS_DIR + "/" + str(k) + "_" + pdf_filename + "/" + config.Dirs.PROGRESS_FILE):
            os.makedirs(config.Dirs.PROGRESS_DIR + "/" + str(k) + "_" + pdf_filename, exist_ok=True)

        processed_pages = extractDataAndSave.load_progress(
            progress_path=config.Dirs.PROGRESS_DIR + "/" + str(k) + "_" + pdf_filename + "/" + config.Dirs.PROGRESS_FILE
        )

        pages_remaining = list(set(pages_to_extract) - set(processed_pages))

        if pages_remaining:
            extracted_pages_map = extractDataAndSave.extract_pages(config.Dirs.PDF_DIR + "/" + pdf_filename, pages_remaining)
            for page_num, text in extracted_pages_map.items():
                logger.info(f"Starting for page {page_num} for {pdf_filename}")

                structured_text = extractDataAndSave.extract_structured_text_pass1(text)

                if not structured_text:
                    logger.error(f"Failed pass 1 extraction for page {page_num}")
                else:
                    if not os.path.exists(config.Dirs.JSON_DIR + "/" + str(k) + "_" + pdf_filename):
                        os.makedirs(config.Dirs.JSON_DIR + "/" + str(k) + "_" + pdf_filename, exist_ok=True)

                    extractDataAndSave.save_text_file(
                        structured_text,
                        page_num,
                        output_dir=config.Dirs.JSON_DIR + "/" + str(k) + "_" + pdf_filename
                    )

                    processed_pages.add(page_num)

                    extractDataAndSave.save_progress(
                        processed_pages, progress_path=config.Dirs.PROGRESS_DIR + "/" + str(k) + "_" + pdf_filename + "/" + config.Dirs.PROGRESS_FILE
                    )

                if page_num < max(extracted_pages_map.keys()):
                    logger.info("Waiting 3 seconds before next page...")
                    time.sleep(3)

            logger.info(f"\nPass 1 complete. Total processed pages: {len(processed_pages)} / {len(pages_to_extract)}")

        logger.info("\nStarting pass 2 (JSON conversion) for all target pages...")
        successful_json_pages = 0

        for page_num in pages_to_extract:
            txt_path = os.path.join(config.Dirs.JSON_DIR + "/" + str(k) + "_" + pdf_filename, f"page_{page_num}_extracted.txt")
            json_path = os.path.join(config.Dirs.JSON_DIR + "/" + str(k) + "_" + pdf_filename, f"page_{page_num}_data.json")

            if os.path.exists(json_path) and os.path.getsize(json_path) > 0:
                logger.info(f"Skipping JSON conversion for page {page_num}: {json_path} already exists")
                successful_json_pages += 1
                continue

            if not os.path.exists(txt_path):
                logger.info(f"Missing extracted text for page {page_num} at {txt_path}. Attempting pass 1 now...")
                extracted_map = extractDataAndSave.extract_pages(config.Dirs.PDF_DIR + "/" + pdf_filename, [page_num])
                text = extracted_map.get(page_num, "")
                if not text:
                    logger.error(f"Could not extract text for page {page_num}")
                    continue

                structured_text = extractDataAndSave.extract_structured_text_pass1(text)
                if not structured_text:
                    logger.error(f"Failed to extract structured text for page {page_num}")
                    continue

                extractDataAndSave.save_text_file(
                    structured_text,
                    page_num,
                    output_dir=config.Dirs.JSON_DIR + "/" + str(k) + "_" + pdf_filename
                )

                processed_pages.add(page_num)

                extractDataAndSave.save_progress(
                    processed_pages,
                    progress_path=config.Dirs.PROGRESS_DIR + "/" + str(k) + "_" + pdf_filename + "/" + config.Dirs.PROGRESS_FILE
                )

            try:
                with open(txt_path, 'r', encoding='utf-8') as f:
                    structured_text = f.read()
            except Exception as e:
                logger.error(f"Error reading structured text for page {page_num}: {e}")
                continue

            if not structured_text.strip():
                logger.error(f"Skipping page {page_num}: empty structured text")
                continue

            json_result = extractDataAndSave.convert_structured_text_to_json_pass2(structured_text)
            if json_result:
                saved = extractDataAndSave.save_json_file(json_result, page_num, output_dir=config.Dirs.JSON_DIR + "/" + str(k) + "_" + pdf_filename)
                if saved:
                    logger.info(f"JSON saved for page {page_num}")
                    successful_json_pages += 1
            else:
                logger.error(f"Failed JSON conversion for page {page_num}")

        logger.info(f"\nJSON generated for {successful_json_pages} / {len(pages_to_extract)} pages")

        if successful_json_pages > 0:
            logger.info("\nNow combining all data using Ollama (structured outputs where possible)...")

            combined_json = extractDataAndSave.combine_with_ollama_structured(config.Dirs.JSON_DIR + "/" + str(k) + "_" + pdf_filename, pages=pages_to_extract, batch_size=3)

            if not combined_json:
                logger.warning("Falling back to non-structured Ollama combine...")
                combined_json = extractDataAndSave.combine_with_ollama_from_json(config.Dirs.JSON_DIR + "/" + str(k) + "_" + pdf_filename, pages=pages_to_extract, batch_size=3)

            if combined_json:
                logger.info("Successfully combined data")
                if not os.path.exists(config.Dirs.RESULT_DIR + "/" + str(k) + "_" + pdf_filename):
                        os.makedirs(config.Dirs.RESULT_DIR + "/" + str(k) + "_" + pdf_filename, exist_ok=True)
                if extractDataAndSave.save_final_data(combined_json, config.Dirs.RESULT_DIR + "/" + str(k) + "_" + pdf_filename + "/" + "combined.json"):
                    logger.info(f"\nProcess completed successfully!")
                    logger.info(f"Individual JSON files saved in: {config.Dirs.JSON_DIR + '/' + str(k) + '_' + pdf_filename}/")
                    logger.info(f"Final combined JSON: final_combined_data.json")
                else:
                    logger.error("Error saving final combined data")
            else:
                logger.error("Failed to combine data with both Ollama and Gemini")


async def extractData(pdf_file, pages):
    try:
        logging.basicConfig(level=logging.INFO)
        pdf_bytes = await pdf_file.read()
        pages_dict: DataReqModal = DataReqModal(pages=json.loads(pages))
        
        if not os.path.exists(config.Dirs.PDF_DIR):        
            os.makedirs(config.Dirs.PDF_DIR, exist_ok=True)

        # save pdf file
        with open(config.Dirs.PDF_DIR + "/" + pdf_file.filename, "wb") as f:
            f.write(pdf_bytes)
        process_extraction(
            pdf_path=config.Dirs.PDF_DIR + "/" + pdf_file.filename,
            pdf_filename=pdf_file.filename,
            pages=pages_dict.pages
        )
        return

    except Exception as e:
        logger.error(f"Error Processing Data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error Processing Data: {str(e)}")