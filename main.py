from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from nltk.tag import pos_tag
import numpy as np
from transformers import BertTokenizer, BertModel
import os
import json
import csv
import pandas as pd
import logging
from typing import Dict, Any, List
import torch
import random
import re
from docx import Document

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Download required NLTK data
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

# Download additional NLTK data for WordNet
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('omw-1.4')

app = FastAPI()

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
MAX_LENGTH = 512  # Maximum sequence length

# Initialize BERT model
model = BertModel.from_pretrained('bert-base-uncased')

def read_file_content(file_path: str) -> str:
    """Read different file formats and return text content"""
    try:
        file_extension = file_path.split('.')[-1].lower()
        logger.debug(f"Reading file: {file_path} with extension: {file_extension}")
        
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
        
        if file_extension == 'txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]
                content = '\n'.join(lines)
                logger.debug(f"Successfully read TXT file, content length: {len(content)}")
                return content
                
        elif file_extension == 'csv':
            try:
                # Read CSV file
                df = pd.read_csv(file_path)
                # Convert each row to string and join with newlines
                lines = []
                for _, row in df.iterrows():
                    # Join non-null values in the row
                    row_text = ' '.join(str(val) for val in row if pd.notna(val))
                    if row_text.strip():
                        lines.append(row_text)
                content = '\n'.join(lines)
                logger.debug(f"Successfully read CSV file, content length: {len(content)}")
                return content
            except Exception as e:
                logger.error(f"CSV read error: {str(e)}")
                raise HTTPException(status_code=400, detail=f"Error reading CSV file: {str(e)}")
                
        elif file_extension == 'docx':
            try:
                doc = Document(file_path)
                lines = []
                for paragraph in doc.paragraphs:
                    text = paragraph.text.strip()
                    if text:
                        # Split by comma and clean up each feedback
                        feedbacks = [feedback.strip() for feedback in text.split(',') if feedback.strip()]
                        # Add each feedback as a separate line
                        lines.extend(feedbacks)
                content = '\n'.join(lines)
                logger.debug(f"Successfully read DOCX file, content length: {len(content)}")
                return content
            except Exception as e:
                logger.error(f"DOCX read error: {str(e)}")
                raise HTTPException(status_code=400, detail=f"Error reading DOCX file: {str(e)}")
                
        elif file_extension == 'xlsx':
            try:
                # Read Excel file
                df = pd.read_excel(file_path)
                # Convert each row to string and join with newlines
                lines = []
                for _, row in df.iterrows():
                    # Join non-null values in the row
                    row_text = ' '.join(str(val) for val in row if pd.notna(val))
                    if row_text.strip():
                        lines.append(row_text)
                content = '\n'.join(lines)
                logger.debug(f"Successfully read Excel file, content length: {len(content)}")
                return content
            except Exception as e:
                logger.error(f"Excel read error: {str(e)}")
                raise HTTPException(status_code=400, detail=f"Error reading Excel file: {str(e)}")
                
        elif file_extension == 'json':
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Handle different JSON structures
                    if isinstance(data, list):
                        # If it's a list of items
                        lines = []
                        for item in data:
                            if isinstance(item, (str, int, float)):
                                lines.append(str(item))
                            else:
                                lines.append(json.dumps(item))
                        content = '\n'.join(lines)
                    else:
                        # If it's a single object
                        content = json.dumps(data, indent=2)
                    logger.debug(f"Successfully read JSON file, content length: {len(content)}")
                    return content
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error: {str(e)}")
                raise HTTPException(status_code=400, detail=f"Invalid JSON file: {str(e)}")
            
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
        allowed_extensions = {'txt', 'csv', 'json', 'docx', 'xlsx'}
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
            # Split by newline and filter out empty lines
            lines = [line for line in text.split('\n') if line.strip()]
        except Exception as e:
            logger.error(f"Error reading file content: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error reading file: {str(e)}")
        
        try:
            processed_lines = []
            bert_processed_lines = []
            total_tokens = 0
            line_stats = []
            
            # Process only non-empty lines without line numbers
            for line in lines:
                # Process non-empty lines
                nltk_tokens = word_tokenize(line)
                processed_lines.append(' '.join(nltk_tokens))
                
                # BERT tokenization and embeddings
                encoded = tokenizer.encode_plus(
                    line,
                    max_length=MAX_LENGTH,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt',
                    return_attention_mask=True
                )
                
                # Get token IDs and attention mask
                input_ids = encoded['input_ids']
                attention_mask = encoded['attention_mask']
                
                # Get BERT embeddings for this line
                with torch.no_grad():
                    outputs = model(input_ids, attention_mask=attention_mask)
                    # Get the last hidden state
                    last_hidden_state = outputs.last_hidden_state
                    
                # Convert embeddings to list and get only the actual tokens (exclude padding)
                embeddings = last_hidden_state[0].tolist()
                actual_embeddings = [
                    emb for emb, mask in zip(embeddings, attention_mask[0].tolist())
                    if mask == 1
                ]
                
                # Get BERT tokens
                line_token_ids = encoded['input_ids'].tolist()[0]
                line_bert_tokens = tokenizer.convert_ids_to_tokens(line_token_ids)
                actual_bert_tokens = [
                    token for token in line_bert_tokens 
                    if token not in ['[PAD]', '[CLS]', '[SEP]']
                ]
                bert_processed_lines.append(' '.join(actual_bert_tokens))
                
                # Store statistics without line number
                line_stats.append({
                    "original_text": line,
                    "nltk_tokens": nltk_tokens,
                    "bert_tokens": actual_bert_tokens,
                    "token_count": {
                        "nltk": len(nltk_tokens),
                        "bert": len(actual_bert_tokens)
                    },
                    "embeddings": {
                        "tokens": actual_bert_tokens[:5],
                        "vectors": [
                            [round(x, 4) for x in vector[:10]]
                            for vector in actual_embeddings[:5]
                        ]
                    }
                })
                
                total_tokens += len(nltk_tokens)
            
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

def get_wordnet_pos(tag: str) -> str:
    """Map POS tag to WordNet POS tag with more accurate mapping"""
    tag_dict = {
        'JJ': wordnet.ADJ,
        'JJR': wordnet.ADJ,
        'JJS': wordnet.ADJ,
        'NN': wordnet.NOUN,
        'NNS': wordnet.NOUN,
        'NNP': wordnet.NOUN,
        'NNPS': wordnet.NOUN,
        'RB': wordnet.ADV,
        'RBR': wordnet.ADV,
        'RBS': wordnet.ADV,
        'VB': wordnet.VERB,
        'VBD': wordnet.VERB,
        'VBG': wordnet.VERB,
        'VBN': wordnet.VERB,
        'VBP': wordnet.VERB,
        'VBZ': wordnet.VERB
    }
    return tag_dict.get(tag, None)  # Return None if POS tag not found

def get_synonyms(word: str, pos: str) -> List[str]:
    """Get better quality synonyms for a word with given POS tag"""
    if not pos:  # Skip words with unknown POS
        return []
        
    word = word.lower()
    synonyms = []
    
    # Get all synsets for the word
    synsets = wordnet.synsets(word, pos=pos)
    
    # Get synonyms from each synset
    for synset in synsets:
        # Get the definition and examples for context
        definition = synset.definition()
        examples = synset.examples()
        
        # Get lemmas (synonyms)
        for lemma in synset.lemmas():
            synonym = lemma.name()
            
            # Filter conditions for better quality
            if (synonym != word and  # Not the same word
                '_' not in synonym and  # No compound words
                len(synonym) > 2 and  # No very short words
                synonym.isalpha()):  # Only alphabetic characters
                
                synonyms.append(synonym)
    
    # Remove duplicates while preserving order
    seen = set()
    filtered_synonyms = []
    for syn in synonyms:
        if syn not in seen:
            seen.add(syn)
            filtered_synonyms.append(syn)
    
    return filtered_synonyms[:5]  # Limit to top 5 synonyms for better quality

def preserve_case(original_word: str, new_word: str) -> str:
    """Preserve the case pattern of the original word"""
    if original_word.isupper():
        return new_word.upper()
    elif original_word.istitle():
        return new_word.title()
    elif original_word.islower():
        return new_word.lower()
    return new_word

def get_random_word(pos: str) -> str:
    """Get a random word from WordNet based on POS"""
    synsets = list(wordnet.all_synsets(pos=pos))
    if synsets:
        synset = random.choice(synsets)
        return synset.lemmas()[0].name()
    return ""

def insert_random_word(words: List[str], tagged_words: List[tuple], num_insertions: int = 1) -> List[str]:
    """Insert random words that match the context"""
    new_words = words.copy()
    for _ in range(num_insertions):
        # Choose a random position
        insert_pos = random.randint(0, len(tagged_words))
        
        # Determine appropriate POS for insertion based on context
        if insert_pos > 0 and insert_pos < len(tagged_words):
            prev_pos = get_wordnet_pos(tagged_words[insert_pos-1][1])
            next_pos = get_wordnet_pos(tagged_words[insert_pos][1])
            pos_choices = [p for p in [prev_pos, next_pos] if p]
            if pos_choices:
                pos = random.choice(pos_choices)
                word = get_random_word(pos)
                if word:
                    new_words.insert(insert_pos, word)
    return new_words

def swap_words(words: List[str], num_swaps: int = 1) -> List[str]:
    """Randomly swap words that are near each other"""
    new_words = words.copy()
    for _ in range(num_swaps):
        if len(new_words) > 1:
            pos = random.randint(0, len(new_words)-2)
            new_words[pos], new_words[pos+1] = new_words[pos+1], new_words[pos]
    return new_words

def augment_text(text: str, num_augmentations: int = 3, replace_prob: float = 0.3) -> List[str]:
    """
    Enhanced text augmentation with multiple techniques:
    1. Synonym replacement
    2. Random word insertion
    3. Word swapping
    """
    words = word_tokenize(text)
    tagged_words = pos_tag(words)
    augmented_texts = []
    
    # Store word-synonym pairs for consistency
    word_synonyms = {}
    
    for _ in range(num_augmentations):
        # Choose augmentation technique randomly
        technique = random.choice(['synonym', 'insert', 'swap', 'combined'])
        new_words = words.copy()
        
        if technique in ['synonym', 'combined']:
            # Synonym replacement
            for i, (word, tag) in enumerate(tagged_words):
                if word.lower() in word_synonyms:
                    new_words[i] = preserve_case(word, word_synonyms[word.lower()])
                    continue
                    
                if (random.random() < replace_prob and 
                    len(word) > 2 and 
                    word.isalpha()):
                    
                    pos = get_wordnet_pos(tag)
                    if pos:
                        synonyms = get_synonyms(word, pos)
                        if synonyms:
                            synonym = random.choice(synonyms)
                            word_synonyms[word.lower()] = synonym
                            new_words[i] = preserve_case(word, synonym)
        
        if technique in ['insert', 'combined']:
            # Random word insertion
            new_words = insert_random_word(new_words, tagged_words, 
                                         num_insertions=random.randint(1, 2))
        
        if technique in ['swap', 'combined']:
            # Word swapping
            new_words = swap_words(new_words, 
                                 num_swaps=random.randint(1, 2))
        
        augmented_text = ' '.join(new_words)
        if augmented_text != text:  # Only add if different from original
            augmented_texts.append(augmented_text)
    
    # Ensure augmented texts are unique
    augmented_texts = list(set(augmented_texts))
    if len(augmented_texts) < num_augmentations:
        # If we don't have enough unique versions, add more using different techniques
        while len(augmented_texts) < num_augmentations:
            new_text = augment_text(text, 1, replace_prob)[0]
            if new_text not in augmented_texts:
                augmented_texts.append(new_text)
    
    return augmented_texts[:num_augmentations]

@app.post("/augment/{filename}")
async def augment_file(
    filename: str,
    num_augmentations: int = 3,
    replace_prob: float = 0.3
) -> Dict[str, Any]:
    try:
        file_path = f"static/uploads/{filename}"
        
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"File not found: {filename}")
        
        # Read the original text
        text = read_file_content(file_path)
        
        # Process text line by line
        original_lines = text.split('\n')
        augmented_versions = []
        
        for line in original_lines:
            if line.strip():
                # Augment each non-empty line
                augmented_lines = augment_text(
                    line,
                    num_augmentations=num_augmentations,
                    replace_prob=replace_prob
                )
                augmented_versions.append({
                    "original": line,
                    "augmented": augmented_lines
                })
            else:
                augmented_versions.append({
                    "original": line,
                    "augmented": [line] * num_augmentations
                })
        
        return {
            "num_augmentations": num_augmentations,
            "replace_probability": replace_prob,
            "augmented_versions": augmented_versions
        }
        
    except Exception as e:
        logger.error(f"Augmentation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Augmentation error: {str(e)}")

# Add cleanup endpoint to remove temporary files
@app.on_event("shutdown")
async def cleanup():
    try:
        import shutil
        shutil.rmtree("static/uploads")
        logger.info("Cleaned up upload directory")
    except Exception as e:
        logger.error(f"Cleanup error: {str(e)}")