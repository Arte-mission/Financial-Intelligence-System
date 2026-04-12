import os
import requests
import json
import logging

logger = logging.getLogger(__name__)

def generate_ai_company_signal(text: str) -> dict:
    """
    Connects to the Gemini Developer API to return structured NLP impact summaries
    of the raw agenda text natively.
    Returns:
       {
          "label": str,
          "score": float,
          "impact_summary": str
       }
    Raises Exception if API fails or parsing fails.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not configured.")
        
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-3-flash-preview:generateContent"
    
    prompt = f"""
You are a financial analyst AI.

Analyze the following company signal:

"{text}"

STRICT RULES:
- Return ONLY valid JSON
- No markdown
- No explanations
- No extra text

Format:
{{
  "label": "Positive | Neutral | Negative | Uncertain",
  "score": number between -1 and 1,
  "impact_summary": "one short sentence"
}}
"""
    
    payload = {
        "contents": [{"parts": [{"text": prompt}]}]
    }
    
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key
    }
    
    logger.info("Calling Gemini API for company signal")
    response = requests.post(url, json=payload, headers=headers, timeout=10)
    response.raise_for_status()
    logger.info("Gemini response received")
    
    data = response.json()
    candidates = data.get("candidates", [])
    if not candidates:
        raise ValueError("No generative candidates returned from Gemini.")
        
    text_output = candidates[0].get("content", {}).get("parts", [])[0].get("text", "")
    
    # Prune possible markdown block bypasses 
    cleaned = text_output.strip().replace("```json", "").replace("```", "")
    result = json.loads(cleaned)
    
    # Strict key validation
    req_keys = {"label", "score", "impact_summary"}
    if not req_keys.issubset(result.keys()):
        raise ValueError("Gemini returned invalid or missing keys.")
        
    logger.info("Gemini JSON parsed successfully")
    return {
        "label": str(result["label"]),
        "score": float(result["score"]),
        "impact_summary": str(result["impact_summary"])
    }
