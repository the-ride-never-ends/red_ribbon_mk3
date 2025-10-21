"""
Response parser for LLM
"""
import logging
import json
from typing import Dict, Any, Union, List

logger = logging.getLogger(__name__)

class ResponseParser:
    """
    Parses responses from language models
    """
    
    def __init__(self):
        logger.info("ResponseParser initialized")
    
    def parse(self, response: str, output_format: str = "text") -> Union[str, Dict, List]:
        """
        Parse a response from a language model
        
        Args:
            response: Raw response from the model
            output_format: Desired output format (text, json, list)
            
        Returns:
            Parsed response in the specified format
        """
        logger.info(f"Parsing response to format: {output_format}")
        
        if output_format == "text":
            return response.strip()
            
        elif output_format == "json":
            # Try to extract JSON from the response
            try:
                # Look for JSON markers in the text
                start_idx = response.find('{')
                end_idx = response.rfind('}')
                
                if start_idx >= 0 and end_idx > start_idx:
                    json_str = response[start_idx:end_idx+1]
                    return json.loads(json_str)
                else:
                    logger.warning("No valid JSON found in response")
                    return {"error": "No valid JSON found", "raw": response}
                    
            except json.JSONDecodeError:
                logger.error("Failed to parse JSON from response")
                return {"error": "JSON parsing failed", "raw": response}
                
        elif output_format == "list":
            # Extract a list from the response
            lines = [line.strip() for line in response.split('\n') if line.strip()]
            # Remove numbering if present
            cleaned_lines = []
            for line in lines:
                if line[0].isdigit() and line[1:].startswith('. '):
                    cleaned_lines.append(line[line.find(' ')+1:])
                else:
                    cleaned_lines.append(line)
            return cleaned_lines
            
        else:
            logger.warning(f"Unknown output format: {output_format}")
            return response