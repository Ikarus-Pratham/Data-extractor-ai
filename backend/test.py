import fitz  # PyMuPDF
import requests
import re
import json
import csv
import os
import time
from dotenv import load_dotenv
from typing import Dict, Any

startTime = time.time()

# Optional dependencies; only used in the combine step
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

def clean_text(text):
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

def clean_json_fractions(json_str):
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
        print(f"Error cleaning JSON fractions: {e}")
        return json_str

def extract_pages(pdf_path, page_numbers):
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
            print(f"Raw text (page {page_num}): {text[:200]}...")  # Debug raw output
            text = clean_text(text)
            print(f"Cleaned text (page {page_num}): {text[:200]}...")  # Debug cleaned output
            extracted_data[page_num] = text
        else:
            print(f"Page {page_num} is out of range. PDF has {total_pages} pages.")
    
    pdf.close()
    return extracted_data

def extract_json_from_response(response_text):
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
        print(f"Error extracting JSON from response: {e}")
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

def _get_chat_ollama(model: str = "gemma3:latest"):
    if ChatOllama is None:
        raise RuntimeError("langchain_ollama is not installed. Please install it to proceed.")
    return ChatOllama(
        model=model,
        temperature=0.1,
        top_p=0.9,
        num_ctx=65536,
        num_predict=-1,
    )

def parse_with_ollama(prompt, model="gemma3:latest", max_retries=3):
    """
    Send the prompt via LangChain ChatOllama and return cleaned JSON text.
    """
    try:
        llm = _get_chat_ollama(model=model)
    except Exception as e:
        print(f"Error initializing ChatOllama: {e}")
        return None

    for attempt in range(max_retries):
        try:
            if not check_ollama_status():
                print("Error: Ollama is not responding. Please check if Ollama is running.")
                return None
            messages = [("human", prompt)]
            resp = llm.invoke(messages)
            content = getattr(resp, "content", None) or ""
            if content.strip():
                clean_json = extract_json_from_response(content)
                clean_json = clean_json_fractions(clean_json)
                print(f"✅ Ollama request successful (attempt {attempt + 1})")
                return clean_json
            if attempt < max_retries - 1:
                print("Warning: Empty response from Ollama, retrying in 5 seconds...")
                time.sleep(5)
                continue
            return None
        except Exception as e:
            print(f"Error calling ChatOllama (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(5)
                continue
            return None
    print(f"Failed after {max_retries} attempts")
    return None

def flatten_json(data, parent_key='', sep='_'):
    """
    Recursively flatten a nested JSON structure into a dictionary with single-level keys.
    """
    items = []
    if isinstance(data, dict):
        for k, v in data.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, (dict, list)):
                items.extend(flatten_json(v, new_key, sep).items())
            else:
                items.append((new_key, v))
    elif isinstance(data, list):
        for i, v in enumerate(data):
            new_key = f"{parent_key}{sep}{i}" if parent_key else str(i)
            if isinstance(v, (dict, list)):
                items.extend(flatten_json(v, new_key, sep).items())
            else:
                items.append((new_key, v))
    else:
        items.append((parent_key, data))
    return dict(items)

def call_ollama_raw(prompt, model="gemma3:latest", max_retries=3):
    """
    Low-level call via ChatOllama that returns raw content without JSON cleanup.
    Used for pass-1 plain extraction and pass-2 JSON formatting prompts.
    """
    try:
        llm = _get_chat_ollama(model=model)
    except Exception as e:
        print(f"Error initializing ChatOllama: {e}")
        return None
    for attempt in range(max_retries):
        try:
            if not check_ollama_status():
                print("Error: Ollama is not responding. Please check if Ollama is running.")
                return None
            resp = llm.invoke([("human", prompt)])
            content = getattr(resp, "content", None) or ""
            if content and content.strip():
                return content
            if attempt < max_retries - 1:
                time.sleep(5)
                continue
            return None
        except Exception as e:
            print(f"Error calling ChatOllama (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(5)
                continue
            return None

def save_text_file(text_data, page_num, output_dir="json_output"):
    """
    Save intermediate extracted text (pass 1) for each page.
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, f"page_{page_num}_extracted.txt")
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(text_data or "")
        print(f"Extracted text saved: {filepath}")
        return filepath
    except Exception as e:
        print(f"Error saving extracted text for page {page_num}: {e}")
        return None

def extract_structured_text_pass1(page_text):
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
    return call_ollama_raw(prompt)

def convert_structured_text_to_json_pass2(structured_text):
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
    content = call_ollama_raw(prompt)
    if not content:
        return None
    # Try to extract a JSON object/array if model added extra text
    cleaned = extract_json_from_response(content)
    cleaned = clean_json_fractions(cleaned)
    return cleaned

PROGRESS_FILE = "progress.json"

def load_progress(progress_path=PROGRESS_FILE):
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
        print(f"Error loading progress: {e}")
        return set()

def save_progress(processed_pages, progress_path=PROGRESS_FILE):
    """
    Save processed page numbers to progress file.
    """
    try:
        with open(progress_path, 'w', encoding='utf-8') as f:
            json.dump({"processed_pages": sorted(list(processed_pages))}, f, indent=2)
    except Exception as e:
        print(f"Error saving progress: {e}")

def save_json_file(json_data, page_num, output_dir="json_output"):
    """
    Save JSON data to individual files for each page.
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        if not json_data or not json_data.strip():
            print(f"Warning: No data to save for page {page_num}")
            return None
        
        try:
            data = json.loads(json_data)
            print(f"Successfully parsed JSON for page {page_num}")
        except json.JSONDecodeError as e:
            print(f"JSON parsing failed for page {page_num}: {e}")
            print(f"Raw content preview: {json_data[:200]}...")
            data = {
                "page_number": page_num,
                "raw_content": json_data,
                "parse_error": True,
                "error_message": str(e),
                "timestamp": __import__('datetime').datetime.now().isoformat()
            }
            print(f"Saved raw content for page {page_num}")
        
        json_filename = f"page_{page_num}_data.json"
        json_filepath = os.path.join(output_dir, json_filename)
        
        with open(json_filepath, 'w', encoding='utf-8') as jsonfile:
            json.dump(data, jsonfile, indent=2, ensure_ascii=False)
        
        print(f"JSON file saved: {json_filepath}")
        return json_filepath
    except Exception as e:
        print(f"Error saving JSON for page {page_num}: {e}")
        return None

def combine_with_ollama_from_text(text_dir="json_output", pages=None, batch_size=3):
    """
    Combine page-level extracted TXT files using Ollama in small batches to avoid timeouts.
    Returns a single JSON string on success, or None on failure.
    """
    try:
        # Build list of text files to include
        if pages:
            filenames = [f"page_{p}_extracted.txt" for p in pages]
        else:
            filenames = [f for f in os.listdir(text_dir) if f.endswith('_extracted.txt')]
        filenames = [f for f in sorted(filenames) if os.path.exists(os.path.join(text_dir, f))]

        if not filenames:
            print("No extracted TXT files found to combine")
            return None

        def load_text_entries(files):
            entries = []
            for filename in files:
                path = os.path.join(text_dir, filename)
                try:
                    with open(path, 'r', encoding='utf-8') as tf:
                        content = tf.read()
                    # Derive page number for traceability
                    try:
                        page_num = int(re.search(r"page_(\d+)_extracted\\.txt", filename).group(1))
                    except Exception:
                        page_num = None
                    entries.append({"filename": filename, "page": page_num, "text": content})
                except Exception as e:
                    print(f"Skipping {filename}: {e}")
            return entries

        # Split into batches
        batches = [filenames[i:i+batch_size] for i in range(0, len(filenames), batch_size)]
        current_combined = None

        for idx, batch in enumerate(batches, start=1):
            batch_entries = load_text_entries(batch)
            if not batch_entries:
                continue

            if current_combined is None:
                prompt = f"""You are merging structured furniture spec information that is provided as plain text extracted from PDF pages.
Rules:
- Produce ONE unified, STRICT, VALID JSON as output (no code fences, no commentary).
- Merge identical products and variants where appropriate; remove duplicates.
- Keep all details: descriptions, tables, options, colors, measurements, prices, notes.
- Keep fractions as Unicode (¼, ½, ¾, ⅛, ⅜, ⅝, ⅞). Use the word inches instead of ".

Plain-text inputs per page (do not output these, only use to build JSON):
{json.dumps(batch_entries, ensure_ascii=False)}

Output the Combined JSON only:"""
            else:
                prompt = f"""You are merging additional plain-text furniture spec inputs into an existing combined JSON.
Rules:
- Output STRICT, VALID JSON only. Keep all information and deduplicate logically.
- Keep fractions as Unicode (¼, ½, ¾, ⅛, ⅜, ⅝, ⅞). Use the word inches instead of ".

Existing combined JSON:
{current_combined}

New plain-text inputs (do not output these, only use to update JSON):
{json.dumps(batch_entries, ensure_ascii=False)}

Output the updated Combined JSON only:"""

            print(f"Sending TXT batch {idx}/{len(batches)} to Ollama for combining...")
            result = parse_with_ollama(prompt)
            if not result:
                print(f"Batch {idx} failed to combine; aborting.")
                return None
            # Keep the raw JSON string for next round to minimize re-serialization noise
            current_combined = extract_json_from_response(result)

        return current_combined
    except Exception as e:
        print(f"Error combining TXT data with Ollama: {e}")
        return None

def combine_with_ollama_from_json(json_dir="json_output", pages=None, batch_size=3):
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
            print("No JSON files found to combine")
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
                    print(f"Skipping {filename}: {e}")
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

            print(f"Sending JSON batch {idx}/{len(batches)} to Ollama for combining...")
            result = parse_with_ollama(prompt)
            if not result:
                print(f"Batch {idx} failed to combine; aborting.")
                return None
            current_combined = extract_json_from_response(result)

        return current_combined
    except Exception as e:
        print(f"Error combining JSON data with Ollama: {e}")
        return None

def combine_with_ollama_structured(json_dir="json_output", pages=None, batch_size=3):
    """
    Combine per-page JSON files using Ollama with structured outputs constrained by the CombinedSpec schema.
    Returns a JSON string conforming to CombinedSpec or None.
    """
    try:
        if ChatOllama is None or not hasattr(CombinedSpec, "model_json_schema"):
            print("langchain_ollama or Pydantic not available; falling back to unstructured combine.")
            return combine_with_ollama_from_json(json_dir=json_dir, pages=pages, batch_size=batch_size)

        # Build list of json files to include
        if pages:
            filenames = [f"page_{p}_data.json" for p in pages]
        else:
            filenames = [f for f in os.listdir(json_dir) if f.endswith('_data.json') or f.endswith('.json')]
        filenames = [f for f in sorted(filenames) if os.path.exists(os.path.join(json_dir, f))]
        if not filenames:
            print("No JSON files found to combine")
            return None

        print(f"Found {filenames} JSON files to combine.")

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
                print(f"Skipping {filename}: {e}")

        if not all_json_entries:
            print("No valid JSON entries found to combine")
            return None

        print(f"Combining {len(all_json_entries)} pages at once...")
        # print("All JSON entries:", all_json_entries)

        # Process all pages in a single API call
        llm = _get_chat_ollama(model="llama3.1")
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
        print(f"Error combining with Ollama structured outputs: {e}")
        return None

def call_gemini_generate(prompt, model="gemini-2.0-flash"):
    """
    Call Google Gemini API with a text prompt. Requires GOOGLE_API_KEY in environment.
    Returns raw text content or None.
    """
    try:
        api_key = os.getenv("GOOGLE_API_KEY")
        print(api_key)
        if not api_key:
            print("Error: GOOGLE_API_KEY not set in environment. Skipping Gemini combine.")
            return None
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        headers = {"Content-Type": "application/json"}
        params = {"key": api_key}
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt}
                    ]
                }
            ]
        }
        response = requests.post(url, headers=headers, params=params, json=payload, timeout=180)
        if response.status_code != 200:
            print(f"Gemini API error: {response.status_code} - {response.text}")
            return None
        data = response.json()
        candidates = data.get("candidates") or []
        if not candidates:
            print("Gemini: no candidates returned")
            return None
        parts = candidates[0].get("content", {}).get("parts") or []
        if not parts:
            print("Gemini: no content parts returned")
            return None
        return parts[0].get("text")
    except Exception as e:
        print(f"Error calling Gemini: {e}")
        return None

def combine_with_gemini(json_dir="json_output", model="gemini-2.0-flash", pages=None):
    """
    Read per-page JSON files and send them to Gemini for intelligent combining.
    If 'pages' is provided, only include those page numbers.
    """
    try:
        all_json_data = []
        selected_filenames = []
        if pages:
            # Only include files for the specified pages
            selected_filenames = [f"page_{p}_data.json" for p in pages]
            print(selected_filenames)
        else:
            # Fallback: include all json files in directory
            selected_filenames = [f for f in os.listdir(json_dir) if f.endswith('.json')]
        for filename in sorted(selected_filenames):
            json_filepath = os.path.join(json_dir, filename)
            if not os.path.exists(json_filepath):
                print(f"Skipping missing file: {json_filepath}")
                continue
            with open(json_filepath, 'r', encoding='utf-8') as jsonfile:
                page_data = json.load(jsonfile)
                all_json_data.append({
                    "filename": filename,
                    "data": page_data
                })
        if not all_json_data:
            print("No JSON files found to combine")
            return None

        # print(all_json_data)

        prompt = f"""Combine furniture spec data from multiple pages into one JSON.

Rules: Merge same product types, keep fractions as Unicode characters (¼, ½, ¾, ⅛, ⅜, ⅝, ⅞), remove duplicates.

Data: {json.dumps(all_json_data, indent=1)}

Combined JSON (output ONLY valid JSON):"""

        print("Sending selected pages to Gemini for intelligent combining...")
        raw = call_gemini_generate(prompt, model=model)
        if not raw:
            return None
        cleaned = extract_json_from_response(raw)
        cleaned = clean_json_fractions(cleaned)
        return cleaned
    except Exception as e:
        print(f"Error combining JSON data with Gemini: {e}")
        return None

def combine_with_gemini_structured(json_dir="json_output", model:"str"="gemini-2.0-flash", pages=None):
    """
    Combine JSON files using LangChain's structured outputs with Gemini and the CombinedSpec schema.
    Returns a JSON string conforming to CombinedSpec or None.
    """
    try:
        if ChatGoogleGenerativeAI is None or not hasattr(CombinedSpec, "model_validate"):
            print("LangChain GoogleGenAI or Pydantic not available; falling back to non-structured Gemini.")
            return combine_with_gemini(json_dir=json_dir, model=model, pages=pages)

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            print("Error: GOOGLE_API_KEY not set in environment. Skipping Gemini structured combine.")
            return None

        # Collect inputs
        all_json_data = []
        if pages:
            selected_filenames = [f"page_{p}_data.json" for p in pages]
        else:
            selected_filenames = [f for f in os.listdir(json_dir) if f.endswith('.json')]
        for filename in sorted(selected_filenames):
            json_filepath = os.path.join(json_dir, filename)
            if not os.path.exists(json_filepath):
                continue
            with open(json_filepath, 'r', encoding='utf-8') as jsonfile:
                page_data = json.load(jsonfile)
                all_json_data.append({"filename": filename, "data": page_data})
        if not all_json_data:
            print("No JSON files found to combine")
            return None

        base_model = ChatGoogleGenerativeAI(model=model, temperature=0)
        model_with_structure = base_model.with_structured_output(CombinedSpec)

        prompt = (
            "Combine furniture spec data from multiple pages into one CombinedSpec. "
            "Keep all details, deduplicate intelligently, preserve fractions as Unicode (¼, ½, ¾, ⅛, ⅜, ⅝, ⅞), "
            "and use the word inches instead of \". Return only the JSON for CombinedSpec.\n\n"
            f"This is schema: {CombinedSpec}\n\n"
            "NOTE: DO NOT put text in json key and append everything, make a proper key value structure with all details.\n\n"
            f"Data: {json.dumps(all_json_data, ensure_ascii=False)}"
        )

        result_obj: CombinedSpec = model_with_structure.invoke(prompt)
        return json.dumps(result_obj.model_dump(by_alias=True), ensure_ascii=False)
    except Exception as e:
        print(f"Error combining with Gemini structured outputs: {e}")
        return None

def save_final_data(combined_json, output_csv_path="combined_seating_specs.csv", output_json_path="final_combined_data.json"):
    """
    Save the final combined data to both JSON and CSV formats.
    """
    try:
        if not combined_json or not combined_json.strip():
            print("No combined data to save")
            return False
        
        print(f"Attempting to parse combined JSON...")
        print(f"JSON preview: {combined_json[:300]}...")
        
        try:
            parsed_data = json.loads(combined_json)
            print("✅ Successfully parsed combined JSON")
        except json.JSONDecodeError as e:
            print(f"❌ JSON parsing failed: {e}")
            clean_json = extract_json_from_response(combined_json)
            print(f"Trying with cleaned JSON: {clean_json[:200]}...")
            try:
                parsed_data = json.loads(clean_json)
                print("✅ Successfully parsed cleaned JSON")
            except json.JSONDecodeError as e2:
                print(f"❌ Still failed to parse JSON: {e2}")
                return False
        
        with open(output_json_path, 'w', encoding='utf-8') as jsonfile:
            json.dump(parsed_data, jsonfile, indent=2, ensure_ascii=False)
        print(f"Final combined JSON saved: {output_json_path}")
        
        json_to_csv(json.dumps(parsed_data), output_csv_path, append=False)
        print(f"Final combined CSV saved: {output_csv_path}")
        
        return True
    except Exception as e:
        print(f"Error saving final data: {e}")
        return False

def json_to_csv(json_data, output_csv_path, append=False):
    """
    Convert JSON data to CSV, handling dynamic JSON structures.
    """
    try:
        data = json.loads(json_data)
        flattened_data = flatten_json(data)
        headers = list(flattened_data.keys())
        
        mode = 'a' if append else 'w'
        with open(output_csv_path, mode, newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers, quoting=csv.QUOTE_MINIMAL)
            if not append or not os.path.exists(output_csv_path) or os.path.getsize(output_csv_path) == 0:
                writer.writeheader()
            writer.writerow(flattened_data)
        
        print(f"CSV file generated: {output_csv_path}")
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
    except Exception as e:
        print(f"Error generating CSV: {e}")

if __name__ == "__main__":
    pdf_file = "seating1.pdf"  # Path to your PDF
    # pages_to_extract = [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]  # Example pages
    pages_to_extract = [20, 21, 22, 23, 24]  # Example pages
    json_output_dir = "json_output"  # Directory for individual JSON files
    final_csv_path = "combined_seating_specs.csv"  # Final combined CSV file
    final_json_path = "final_combined_data.json"  # Final combined JSON file
    os.makedirs(json_output_dir, exist_ok=True)
    
    # Test clean_text function
    test_text = "23/4 inches, 31/2 turns, 91⁄4 width"
    print(f"Test clean_text: {clean_text(test_text)}")  # Debug test
    
    # Load progress (pages already extracted to structured text)
    processed_pages = load_progress()
    print(f"Found {len(processed_pages)} pages in progress file: {sorted(list(processed_pages))}")

    # We will process all pages in one run: ensure structured text exists (skip if in progress),
    # then convert to JSON for each page, and finally combine.
    pages_needing_pass1 = [p for p in pages_to_extract if p not in processed_pages]

    # Run pass 1 for pages not yet processed
    if pages_needing_pass1:
        print(f"Starting pass 1 for pages: {pages_needing_pass1}")
        extracted_pages_map = extract_pages(pdf_file, pages_needing_pass1)
        for page_num, text in extracted_pages_map.items():
            print(f"\n--- Page {page_num} (Cleaned Text Preview) ---\n{text[:500]}...\n")
            structured_text = extract_structured_text_pass1(text)
            if not structured_text:
                print(f"❌ Failed pass 1 extraction for page {page_num}")
            else:
                save_text_file(structured_text, page_num, json_output_dir)
                processed_pages.add(page_num)
                save_progress(processed_pages)
            if page_num < max(extracted_pages_map.keys()):
                print("Waiting 3 seconds before next page...")
                time.sleep(3)
        print(f"\n📊 Pass 1 complete. Total processed pages: {len(processed_pages)} / {len(pages_to_extract)}")

    # For every page in target list, ensure JSON is generated from its structured text
    print("\nStarting pass 2 (JSON conversion) for all target pages...")
    successful_json_pages = 0
    for page_num in pages_to_extract:
        txt_path = os.path.join(json_output_dir, f"page_{page_num}_extracted.txt")
        json_path = os.path.join(json_output_dir, f"page_{page_num}_data.json")
        # If JSON already exists, skip conversion
        if os.path.exists(json_path) and os.path.getsize(json_path) > 0:
            print(f"Skipping JSON conversion for page {page_num}: {json_path} already exists")
            successful_json_pages += 1
            continue
        if not os.path.exists(txt_path):
            print(f"Missing extracted text for page {page_num} at {txt_path}. Attempting pass 1 now...")
            # Try to run pass 1 ad-hoc
            extracted_map = extract_pages(pdf_file, [page_num])
            text = extracted_map.get(page_num, "")
            if not text:
                print(f"❌ Could not extract text for page {page_num}")
                continue
            structured_text = extract_structured_text_pass1(text)
            if not structured_text:
                print(f"❌ Failed to extract structured text for page {page_num}")
                continue
            save_text_file(structured_text, page_num, json_output_dir)
            processed_pages.add(page_num)
            save_progress(processed_pages)
        # At this point, structured text file exists; run conversion
        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                structured_text = f.read()
        except Exception as e:
            print(f"Error reading structured text for page {page_num}: {e}")
            continue
        if not structured_text.strip():
            print(f"Skipping page {page_num}: empty structured text")
            continue
        json_result = convert_structured_text_to_json_pass2(structured_text)
        if json_result:
            saved = save_json_file(json_result, page_num, json_output_dir)
            if saved:
                print(f"✅ JSON saved for page {page_num}")
                successful_json_pages += 1
        else:
            print(f"❌ Failed JSON conversion for page {page_num}")

    print(f"\n📊 JSON generated for {successful_json_pages} / {len(pages_to_extract)} pages")

    if successful_json_pages > 0:
        print("\n🔄 Now combining all data using Ollama (structured outputs where possible)...")

        combined_json = combine_with_ollama_structured(json_output_dir, pages=pages_to_extract, batch_size=3)
        # combined_json = combine_with_gemini_structured(json_dir=json_output_dir, model="gemini-2.0-flash", pages=pages_to_extract)

        if not combined_json:
            print("Falling back to non-structured Ollama combine...")
            combined_json = combine_with_ollama_from_json(json_output_dir, pages=pages_to_extract, batch_size=3)

        if not combined_json:
            print("Attempting Gemini structured combine as backup...")
            combined_json = combine_with_gemini_structured(json_dir=json_output_dir, model="gemini-2.0-flash", pages=pages_to_extract)
            if not combined_json:
                print("Falling back to non-structured Gemini combine...")
                combined_json = combine_with_gemini(json_dir=json_output_dir, model="gemini-2.0-flash", pages=pages_to_extract)

        if combined_json:
            print("END TIME: ", time.time() - startTime)
            print("✅ Successfully combined data")
            if save_final_data(combined_json, final_csv_path, final_json_path):
                print(f"\n🎉 Process completed successfully!")
                print(f"📁 Individual JSON files saved in: {json_output_dir}/")
                print(f"📊 Final combined CSV: {final_csv_path}")
                print(f"📋 Final combined JSON: {final_json_path}")
            else:
                print("❌ Error saving final combined data")
        else:
            print("❌ Failed to combine data with both Ollama and Gemini")