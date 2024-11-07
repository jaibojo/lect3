from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import nltk
from nltk.tokenize import word_tokenize
import numpy as np
from transformers import BertTokenizer
import os
import json
import csv
import pandas as pd
import logging
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Download all required NLTK data
try:
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('maxent_ne_chunker')
    nltk.download('words')
    logger.info("Successfully downloaded NLTK data")
except Exception as e:
    logger.error(f"Error downloading NLTK data: {str(e)}")
    raise

app = FastAPI()

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
MAX_LENGTH = 512  # Maximum sequence length

def read_file_content(file_path: str) -> str:
    """Read different file formats and return text content"""
    try:
        file_extension = file_path.split('.')[-1].lower()
        logger.debug(f"Reading file: {file_path} with extension: {file_extension}")
        
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
        
        if file_extension == 'txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                logger.debug(f"Successfully read TXT file, content length: {len(content)}")
                return content
                
        elif file_extension == 'json':
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    content = json.dumps(data, indent=2)
                    logger.debug(f"Successfully read JSON file, content length: {len(content)}")
                    return content
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error: {str(e)}")
                raise HTTPException(status_code=400, detail=f"Invalid JSON file: {str(e)}")
                
        elif file_extension in ['csv', 'tsv']:
            try:
                df = pd.read_csv(file_path, sep=',' if file_extension == 'csv' else '\t')
                content = df.to_string()
                logger.debug(f"Successfully read CSV/TSV file, content length: {len(content)}")
                return content
            except pd.errors.EmptyDataError:
                return "Empty file"
            except Exception as e:
                logger.error(f"CSV/TSV read error: {str(e)}")
                raise HTTPException(status_code=400, detail=f"Error reading CSV/TSV file: {str(e)}")
            
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported file format: {file_extension}")
            
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error reading file: {str(e)}")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("static/index.html", "r") as f:
        return f.read()

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)) -> Dict[str, Any]:
    try:
        # Check file extension
        allowed_extensions = {'txt', 'csv', 'json', 'tsv'}
        file_extension = file.filename.split('.')[-1].lower()
        
        if file_extension not in allowed_extensions:
            raise HTTPException(status_code=400, detail="File format not supported")
        
        content = await file.read()
        
        # Save the file temporarily
        file_path = f"static/uploads/{file.filename}"
        os.makedirs("static/uploads", exist_ok=True)
        
        # Write content in binary mode
        with open(file_path, "wb") as f:
            f.write(content)
        
        logger.debug(f"File saved successfully: {file_path}")
        
        # Read the content based on file type
        text_content = read_file_content(file_path)
        
        return {"filename": file.filename, "content": text_content}
        
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/preprocess/{filename}")
async def preprocess_file(filename: str) -> Dict[str, Any]:
    try:
        file_path = f"static/uploads/{filename}"
        logger.debug(f"Preprocessing file: {file_path}")
        
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            raise HTTPException(status_code=404, detail=f"File not found: {filename}")
            
        try:
            text = read_file_content(file_path)
            lines = text.split('\n')
        except Exception as e:
            logger.error(f"Error reading file content: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error reading file: {str(e)}")
        
        try:
            processed_lines = []
            bert_processed_lines = []
            total_tokens = 0
            line_stats = []  # Store statistics for each line
            
            for i, line in enumerate(lines, 1):
                if not line.strip():
                    processed_lines.append('')
                    bert_processed_lines.append('')
                    line_stats.append({
                        "line_number": i,
                        "original_text": "",
                        "original_tokens": [],
                        "bert_tokens": [],
                        "token_count": {"original": 0, "bert": 0}
                    })
                    continue
                    
                # Original tokenization
                line_tokens = word_tokenize(line)
                processed_lines.append(' '.join(line_tokens))
                
                # BERT tokenization
                encoded = tokenizer.encode_plus(
                    line,
                    max_length=MAX_LENGTH,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                
                line_token_ids = encoded['input_ids'].tolist()[0]
                line_bert_tokens = tokenizer.convert_ids_to_tokens(line_token_ids)
                
                # Get actual BERT tokens (excluding special tokens and padding)
                actual_bert_tokens = [
                    token for token in line_bert_tokens 
                    if token not in ['[PAD]', '[CLS]', '[SEP]']
                ]
                
                bert_processed_lines.append(' '.join(actual_bert_tokens))
                
                # Store statistics for this line
                line_stats.append({
                    "line_number": i,
                    "original_text": line,
                    "original_tokens": line_tokens,  # Store actual tokens
                    "bert_tokens": actual_bert_tokens,  # Store actual BERT tokens
                    "token_count": {
                        "original": len(line_tokens),
                        "bert": len(actual_bert_tokens)
                    }
                })
                
                total_tokens += len(line_tokens)
            
            preprocessed_text = '\n'.join(processed_lines)
            bert_preprocessed_text = '\n'.join(bert_processed_lines)
            
            # Overall padding statistics
            original_length = len(tokenizer.encode(text))
            is_padded = original_length < MAX_LENGTH
            is_truncated = original_length > MAX_LENGTH
            padding_info = {
                "original_length": original_length,
                "max_length": MAX_LENGTH,
                "is_padded": is_padded,
                "is_truncated": is_truncated,
                "final_length": MAX_LENGTH if (is_padded or is_truncated) else original_length
            }
            
            logger.debug(f"Padding info: {padding_info}")
            
        except Exception as e:
            logger.error(f"Processing error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")
        
        return {
            "num_tokens": total_tokens,
            "preprocessed_text": preprocessed_text,
            "bert_preprocessed_text": bert_preprocessed_text,
            "padding_info": padding_info,
            "line_stats": line_stats
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected preprocessing error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

# Add cleanup endpoint to remove temporary files
@app.on_event("shutdown")
async def cleanup():
    try:
        import shutil
        shutil.rmtree("static/uploads")
        logger.info("Cleaned up upload directory")
    except Exception as e:
        logger.error(f"Cleanup error: {str(e)}")