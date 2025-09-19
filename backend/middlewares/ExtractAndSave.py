import fitz  # PyMuPDF
import requests
import re
import json
import csv
import os
import time
from dotenv import load_dotenv
from typing import Dict, Any
import logging

# set up logging
logging.basicConfig(level=logging.INFO)
logger: logging.Logger = logging.getLogger(__name__)

try:
    from pydantic import BaseModel, Field, model_validator
except Exception:
    BaseModel = object  # type: ignore
    Field = lambda *args, **kwargs: None  # type: ignore
    def model_validator(*args, **kwargs):  # type: ignore
        def wrap(fn):
            return fn
        return wrap

try:
    from langchain_ollama import ChatOllama
except Exception:
    ChatOllama = None  # Optional

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except Exception:
    ChatGoogleGenerativeAI = None  # Optional


load_dotenv()


class CombinedSpec(BaseModel):
    name: str
    general_specification: Dict[str, Any]
    description: Dict[str, Any]
    colors: Dict[str, Any]
    pricing: Dict[str, Any]
    dimensions: Dict[str, Any]
    components: Dict[str, Any]
    adjustments: Dict[str, Any]
    extra_remaining_info: Dict[str, Any] = Field(default_factory=dict, alias="extra/remaining_info")

    class Config:
        populate_by_name = True

    @staticmethod
    def _ensure_obj(value: Any) -> Dict[str, Any]:
        if value is None:
            return {}
        if isinstance(value, dict):
            return value
        if isinstance(value, list):
            return {"items": value}
        if isinstance(value, str):
            return {"text": value}
        return {"value": value}

    @model_validator(mode="before")
    def coerce_sections_to_objects(cls, data: Any):  # type: ignore
        if not isinstance(data, dict):
            return data
        section_keys = [
            "general_specification",
            "description",
            "colors",
            "pricing",
            "dimensions",
            "components",
            "adjustments",
            "extra_remaining_info",
            "extra/remaining_info",
        ]
        for key in section_keys:
            if key in data:
                data[key] = CombinedSpec._ensure_obj(data[key])
        # Ensure alias maps correctly if provider emits alias key
        if "extra/remaining_info" in data and "extra_remaining_info" not in data:
            data["extra_remaining_info"] = data.pop("extra/remaining_info")
        return data

class ExtractAndSaveData:
    def __init__(self, pdf_path, pdf_dir, json_dir, result_dir, progress_dir, pages, model_name, temperature, top_p, num_ctx, num_predict, pdf_filename):
        self.pdf_path = pdf_path
        self.pdf_dir = pdf_dir
        self.json_dir = json_dir
        self.result_dir = result_dir
        self.progress_dir = progress_dir
        self.pages = pages
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.num_ctx = num_ctx
        self.num_predict = num_predict
        self.pdf_filename = pdf_filename

        os.makedirs(self.pdf_dir, exist_ok=True)
        os.makedirs(self.json_dir, exist_ok=True)
        os.makedirs(self.result_dir, exist_ok=True)
        os.makedirs(self.progress_dir, exist_ok=True)

    def clean_text(self, text):
        """
        Clean extracted text and fix common OCR/extraction issues with numbers and fractions.
        """
        # Replace common fraction character issues
        replacements = {
            '1⁄4': '¼', '1⁄2': '½', '3⁄4': '¾', '1⁄8': '⅛',
            '3⁄8': '⅜', '5⁄8': '⅝', '7⁄8': '⅞',
            '1/4': '¼', '1/2': '½', '3/4': '¾', '1/8': '⅛',
            '3/8': '⅜', '5/8': '⅝', '7/8': '⅞'
        }
        for wrong, right in replacements.items():
            text = text.replace(wrong, right)
        
        # Fix patterns like "91/4" or "23/4" to "9¼" or "2¾"
        text = re.sub(r'(\d+)/4', r'\1¼', text)
        text = re.sub(r'(\d+)/2', r'\1½', text)
        text = re.sub(r'(\d+)/8', r'\1⅛', text)
        text = re.sub(r'(\d+)(?:3/4|¾)', r'\1¾', text)
        text = re.sub(r'(\d+)(?:3/8|⅜)', r'\1⅜', text)
        text = re.sub(r'(\d+)(?:5/8|⅝)', r'\1⅝', text)
        text = re.sub(r'(\d+)(?:7/8|⅞)', r'\1⅞', text)
        
        return text

    def clean_json_fractions(self, json_str):
        """
        Clean fraction formats in a JSON string after Ollama processing.
        """
        try:
            # Load JSON to manipulate as dictionary
            data = json.loads(json_str)
            # Recursively clean fractions in all string values
            def clean_value(value):
                if isinstance(value, str):
                    for wrong, right in {
                        '1/4': '¼', '1/2': '½', '3/4': '¾', '1/8': '⅛',
                        '3/8': '⅜', '5/8': '⅝', '7/8': '⅞'
                    }.items():
                        value = value.replace(wrong, right)
                    value = re.sub(r'(\d+)/4', r'\1¼', value)
                    value = re.sub(r'(\d+)/2', r'\1½', value)
                    value = re.sub(r'(\d+)/8', r'\1⅛', value)
                    value = re.sub(r'(\d+)(?:3/4|¾)', r'\1¾', value)
                    value = re.sub(r'(\d+)(?:3/8|⅜)', r'\1⅜', value)
                    value = re.sub(r'(\d+)(?:5/8|⅝)', r'\1⅝', value)
                    value = re.sub(r'(\d+)(?:7/8|⅞)', r'\1⅞', value)
                    return value
                elif isinstance(value, dict):
                    return {k: clean_value(v) for k, v in value.items()}
                elif isinstance(value, list):
                    return [clean_value(item) for item in value]
                return value
            
            cleaned_data = clean_value(data)
            return json.dumps(cleaned_data)
        except json.JSONDecodeError as e:
            logger.error(f"Error cleaning JSON fractions: {e}")
            return json_str

    def extract_pages(self, pdf_path, page_numbers):
        """
        Extract text from specified pages using PyMuPDF with improved text cleaning.
        """
        extracted_data = {}
        pdf = fitz.open(pdf_path)
        total_pages = pdf.page_count
        
        for page_num in page_numbers:
            if 1 <= page_num <= total_pages:
                page = pdf.load_page(page_num - 1)
                text = page.get_text()
                logger.info(f"Raw text (page {page_num}): {text[:200]}...")  # Debug raw output
                text = self.clean_text(text)
                logger.info(f"Cleaned text (page {page_num}): {text[:200]}...")  # Debug cleaned output
                extracted_data[page_num] = text
            else:
                logger.error(f"Page {page_num} is out of range. PDF has {total_pages} pages.")
        
        pdf.close()
        return extracted_data

    def extract_json_from_response(self, response_text):
        """
        Extract clean JSON from Ollama response that may contain markdown formatting.
        """
        try:
            # Remove any markdown code block formatting
            json_match = re.search(r'```(?:json)?\s*($$   .*?   $$|\{.*?\})\s*```', response_text, re.DOTALL)
            if json_match:
                return json_match.group(1).strip()
            
            # Look for JSON without code blocks
            json_match = re.search(r'(\[.*?\]|\{.*?\})', response_text, re.DOTALL)
            if json_match:
                return json_match.group(1).strip()
            
            return response_text.strip()
        except Exception as e:
            logger.error(f"Error extracting JSON from response: {e}")
            return response_text.strip()
    
    def check_ollama_status():
        """
        Check if Ollama is running and responsive.
        """
        try:
            url = "http://localhost:11434/api/tags"
            response = requests.get(url, timeout=10)
            return response.status_code == 200
        except:
            return False

    def _get_chat_ollama(self, model):
        if ChatOllama is None:
            raise RuntimeError("langchain_ollama is not installed. Please install it to proceed.")
        return ChatOllama(
            model=model,
            temperature=self.temperature,
            top_p=self.top_p,
            num_ctx=self.num_ctx,
            num_predict=self.num_predict,
        )
    
    def parse_with_ollama(self, prompt, max_retries=3):
        """
        Send the prompt via LangChain ChatOllama and return cleaned JSON text.
        """
        try:
            llm = self._get_chat_ollama(model=self.model_name)
        except Exception as e:
            logger.error(f"Error initializing ChatOllama: {e}")
            return None

        for attempt in range(max_retries):
            try:
                if not self.check_ollama_status():
                    logger.info("Error: Ollama is not responding. Please check if Ollama is running.")
                    return None
                messages = [("human", prompt)]
                resp = llm.invoke(messages)
                content = getattr(resp, "content", None) or ""
                if content.strip():
                    clean_json = self.extract_json_from_response(content)
                    clean_json = self.clean_json_fractions(clean_json)
                    logger.info(f"Ollama request successful (attempt {attempt + 1})")
                    return clean_json
                if attempt < max_retries - 1:
                    logger.info("Warning: Empty response from Ollama, retrying in 5 seconds...")
                    time.sleep(5)
                    continue
                return None
            except Exception as e:
                logger.error(f"Error calling ChatOllama (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(5)
                    continue
                return None
        logger.error(f"Failed after {max_retries} attempts")
        return None

    def flatten_json(self, data, parent_key='', sep='_'):
        """
        Recursively flatten a nested JSON structure into a dictionary with single-level keys.
        """
        items = []
        if isinstance(data, dict):
            for k, v in data.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                if isinstance(v, (dict, list)):
                    items.extend(self.flatten_json(v, new_key, sep).items())
                else:
                    items.append((new_key, v))
        elif isinstance(data, list):
            for i, v in enumerate(data):
                new_key = f"{parent_key}{sep}{i}" if parent_key else str(i)
                if isinstance(v, (dict, list)):
                    items.extend(self.flatten_json(v, new_key, sep).items())
                else:
                    items.append((new_key, v))
        else:
            items.append((parent_key, data))
        return dict(items)
    
    def call_ollama_raw(self, prompt, max_retries=3):
        """
        Low-level call via ChatOllama that returns raw content without JSON cleanup.
        Used for pass-1 plain extraction and pass-2 JSON formatting prompts.
        """
        try:
            llm = self._get_chat_ollama(model=self.model_name)
        except Exception as e:
            logger.error(f"Error initializing ChatOllama: {e}")
            return None
        for attempt in range(max_retries):
            try:
                resp = llm.invoke([("human", prompt)])
                content = getattr(resp, "content", None) or ""
                if content and content.strip():
                    return content
                if attempt < max_retries - 1:
                    time.sleep(5)
                    continue
                return None
            except Exception as e:
                logger.error(f"Error calling ChatOllama (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(5)
                    continue
                return None

    def save_text_file(self, text_data, page_num, output_dir="json_output"):
        """
        Save intermediate extracted text (pass 1) for each page.
        """
        try:
            os.makedirs(output_dir, exist_ok=True)
            filepath = os.path.join(output_dir, f"page_{page_num}_extracted.txt")
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(text_data or "")
            logger.info(f"Extracted text saved: {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Error saving extracted text for page {page_num}: {e}")
            return None
    
    def load_progress(self, progress_path):
        """
        Load processed page numbers from progress file. Auto-create if missing.
        """
        try:
            if not os.path.exists(progress_path):
                with open(progress_path, 'w', encoding='utf-8') as f:
                    json.dump({"processed_pages": []}, f, indent=2)
                return set()
            with open(progress_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                pages = data.get("processed_pages", [])
                return set(int(p) for p in pages)
        except Exception as e:
            logger.error(f"Error loading progress: {e}")
            return set()

    def save_progress(self, processed_pages, progress_path):
        """
        Save processed page numbers to progress file.
        """
        try:
            with open(progress_path, 'w', encoding='utf-8') as f:
                json.dump({"processed_pages": sorted(list(processed_pages))}, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving progress: {e}")

    def save_json_file(self, json_data, page_num, output_dir="json_output"):
        """
        Save JSON data to individual files for each page.
        """
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            if not json_data or not json_data.strip():
                logger.warning(f"Warning: No data to save for page {page_num}")
                return None
            
            try:
                data = json.loads(json_data)
                logger.info(f"Successfully parsed JSON for page {page_num}")
            except json.JSONDecodeError as e:
                logger.info(f"JSON parsing failed for page {page_num}: {e}")
                logger.info(f"Raw content preview: {json_data[:200]}...")
                data = {
                    "page_number": page_num,
                    "raw_content": json_data,
                    "parse_error": True,
                    "error_message": str(e),
                    "timestamp": __import__('datetime').datetime.now().isoformat()
                }
                logger.info(f"Saved raw content for page {page_num}")
            
            json_filename = f"page_{page_num}_data.json"
            json_filepath = os.path.join(output_dir, json_filename)
            
            with open(json_filepath, 'w', encoding='utf-8') as jsonfile:
                json.dump(data, jsonfile, indent=2, ensure_ascii=False)
            
            logger.info(f"JSON file saved: {json_filepath}")
            return json_filepath
        except Exception as e:
            logger.error(f"Error saving JSON for page {page_num}: {e}")
            return None
    
    def extract_structured_text_pass1(self, page_text):
        """
        Pass 1: Ask the model to extract all information faithfully as structured, labeled text
        (not JSON). The output should be human-readable, sectioned, and comprehensive.
        """
        prompt = f"""Extract ALL furniture specs and tabular data from the following text.

    Rules:
    - Do NOT output JSON in this step.
    - Preserve all details, prices, tables, options, colors, descriptions, measurements, and notes.
    - Keep fractions as Unicode characters (¼, ½, ¾, ⅛, ⅜, ⅝, ⅞).
    - Normalize inches by writing the word inches instead of ".
    - Use clear headings and bullet points or simple table-like layouts.
    - If text content doesn't have any details, return nothing.

    Input Text:
    {page_text}

    Output: A clean, well-structured plain-text summary with headings and lists.
    """
        return self.call_ollama_raw(prompt)
    
    def convert_structured_text_to_json_pass2(self, structured_text):
        """
        Pass 2: Convert the structured text into strict, valid JSON.
        """
        prompt = f"""Convert the following structured text into STRICT, VALID JSON.

    Rules:
    - Output ONLY JSON. No code fences, no explanations.
    - When doing for price, put the currency sign and positive or negative sign if any. 
    - Keep all information. Use arrays/objects where appropriate.
    - Keep fractions as Unicode characters (¼, ½, ¾, ⅛, ⅜, ⅝, ⅞).
    - Replace any remaining \" with the word inches in measurement contexts.
    - Ensure valid JSON with correct commas and quotes.
    - Remove any stray newlines within string values.

    Structured text:
    {structured_text}

    JSON only:
    """
        content = self.call_ollama_raw(prompt)
        if not content:
            return None
        # Try to extract a JSON object/array if model added extra text
        cleaned = self.extract_json_from_response(content)
        cleaned = self.clean_json_fractions(cleaned)
        return cleaned

    def combine_with_ollama_structured(self, json_dir="json_outputs", pages=None, batch_size=3):
        """
        Combine per-page JSON files using Ollama with structured outputs constrained by the CombinedSpec schema.
        Returns a JSON string conforming to CombinedSpec or None.
        """
        try:
            if ChatOllama is None or not hasattr(CombinedSpec, "model_json_schema"):
                logger.info("langchain_ollama or Pydantic not available; falling back to unstructured combine.")
                return self.combine_with_ollama_from_json(json_dir=json_dir, pages=pages, batch_size=batch_size)

            # Build list of json files to include
            if pages:
                filenames = [f"page_{p}_data.json" for p in pages]
            else:
                filenames = [f for f in os.listdir(json_dir) if f.endswith('_data.json') or f.endswith('.json')]
            filenames = [f for f in sorted(filenames) if os.path.exists(os.path.join(json_dir, f))]
            if not filenames:
                logger.warning("No JSON files found to combine")
                return None

            logger.info(f"Found {filenames} JSON files to combine.")

            # Load all JSON entries at once instead of processing in batches
            all_json_entries = []
            for filename in filenames:
                path = os.path.join(json_dir, filename)
                try:
                    with open(path, 'r', encoding='utf-8') as jf:
                        data = json.load(jf)
                    m = re.search(r"page_(\d+)_data\\.json", filename)
                    page_num = int(m.group(1)) if m else None
                    all_json_entries.append({"filename": filename, "page": page_num, "data": data})
                except Exception as e:
                    logger.error(f"Skipping {filename}: {e}")

            if not all_json_entries:
                logger.warning("No valid JSON entries found to combine")
                return None

            logger.info(f"Combining {len(all_json_entries)} pages at once...")
            
            # Process all pages in a single API call
            llm = self._get_chat_ollama(model=self.model_name)
            structured_llm = llm.with_structured_output(CombinedSpec)
            prompt = (
                "NOTE: KEEP EVERYTHING DETAILED. DO NOT OMIT ANYTHING. "
                "When doing for price, put the currency sign and positive or negative sign if any. "
                "You are an expert data merger. Merge furniture spec JSONs from all pages into a single CombinedSpec. "
                "The name should be the full product name. "
                "If any parameter is not found with a similar key in the inputs, keep it empty. "
                "Keep all details, deduplicate intelligently, preserve fractions as Unicode (¼, ½, ¾, ⅛, ⅜, ⅝, ⅞), "
                "and use the word inches instead of \". Return only the CombinedSpec JSON.\n\n"
                f"Input JSONs from all pages: {json.dumps(all_json_entries, ensure_ascii=False)}"
            )
            
            combined_obj: CombinedSpec = structured_llm.invoke(prompt)

            if combined_obj is None:
                return None

            return json.dumps(combined_obj.model_dump(by_alias=True), ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error combining with Ollama structured outputs: {e}")
            return None
    
    def combine_with_ollama_from_json(self, json_dir="json_output", pages=None, batch_size=3):
        """
        Combine per-page JSON files using Ollama in small batches to reduce timeouts.
        Returns a single JSON string on success, or None on failure.
        """
        try:
            # Build list of json files to include
            if pages:
                filenames = [f"page_{p}_data.json" for p in pages]
            else:
                filenames = [f for f in os.listdir(json_dir) if f.endswith('_data.json') or f.endswith('.json')]
            filenames = [f for f in sorted(filenames) if os.path.exists(os.path.join(json_dir, f))]

            if not filenames:
                logger.warning("No JSON files found to combine")
                return None

            def load_json_entries(files):
                entries = []
                for filename in files:
                    path = os.path.join(json_dir, filename)
                    try:
                        with open(path, 'r', encoding='utf-8') as jf:
                            data = json.load(jf)
                        # Derive page number for traceability
                        m = re.search(r"page_(\d+)_data\\.json", filename)
                        page_num = int(m.group(1)) if m else None
                        entries.append({"filename": filename, "page": page_num, "data": data})
                    except Exception as e:
                        logger.error(f"Skipping {filename}: {e}")
                return entries

            # Split into batches
            batches = [filenames[i:i+batch_size] for i in range(0, len(filenames), batch_size)]
            current_combined = None

            for idx, batch in enumerate(batches, start=1):
                batch_entries = load_json_entries(batch)
                if not batch_entries:
                    continue

                if current_combined is None:
                    prompt = f"""You are merging structured furniture spec JSONs coming from multiple pages.
    Rules:
    - Produce ONE unified, STRICT, VALID JSON as output (no code fences, no commentary).
    - Merge identical products and variants where appropriate; remove duplicates.
    - Keep all details: descriptions, tables, options, colors, measurements, prices, notes.
    - Keep fractions as Unicode (¼, ½, ¾, ⅛, ⅜, ⅝, ⅞). Use the word inches instead of ".

    Input batch JSONs (array of objects with filename, page, data):
    {json.dumps(batch_entries, ensure_ascii=False)}

    Output the Combined JSON only:"""
                else:
                    prompt = f"""You are merging additional page JSONs into an existing combined JSON.
    Rules:
    - Output STRICT, VALID JSON only. Keep all information and deduplicate logically.
    - Keep fractions as Unicode (¼, ½, ¾, ⅛, ⅜, ⅝, ⅞). Use the word inches instead of ".

    Existing combined JSON:
    {current_combined}

    New batch JSONs (array of objects with filename, page, data):
    {json.dumps(batch_entries, ensure_ascii=False)}

    Output the updated Combined JSON only:"""

                logger.info(f"Sending JSON batch {idx}/{len(batches)} to Ollama for combining...")
                result = self.parse_with_ollama(prompt)
                if not result:
                    logger.warning(f"Batch {idx} failed to combine; aborting.")
                    return None
                current_combined = self.extract_json_from_response(result)

            return current_combined
        except Exception as e:
            logger.error(f"Error combining JSON data with Ollama: {e}")
            return None

    def save_final_data(self, combined_json, output_json_path):
        """
        Save the final combined data to both JSON and CSV formats.
        """
        try:
            if not combined_json or not combined_json.strip():
                logger.warning("No combined data to save")
                return False
            
            logger.info(f"Attempting to parse combined JSON...")
            logger.info(f"JSON preview: {combined_json[:300]}...")
            
            try:
                parsed_data = json.loads(combined_json)
                logger.info("Successfully parsed combined JSON")
            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing failed: {e}")
                clean_json = self.extract_json_from_response(combined_json)
                logger.info(f"Trying with cleaned JSON: {clean_json[:200]}...")
                try:
                    parsed_data = json.loads(clean_json)
                    logger.info("Successfully parsed cleaned JSON")
                except json.JSONDecodeError as e2:
                    logger.error(f"❌ Still failed to parse JSON: {e2}")
                    return False
            
            with open(output_json_path, 'w', encoding='utf-8') as jsonfile:
                json.dump(parsed_data, jsonfile, indent=2, ensure_ascii=False)
            logger.info(f"Final combined JSON saved: {output_json_path}")
            
            return True
        except Exception as e:
            logger.error(f"Error saving final data: {e}")
            return False



