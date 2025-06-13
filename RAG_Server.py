import os
import logging
from typing import List, Dict, Any
from datetime import datetime
from flask import Flask, request, jsonify
from langchain_unstructured import UnstructuredLoader
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import requests
import json
from tqdm import tqdm
import hashlib
import shutil
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from requests.exceptions import RequestException
import logging.handlers
import functools
from cachetools import TTLCache, cached
from langdetect import detect
import googletrans
from googletrans import Translator
import asyncio
import nest_asyncio
import re
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader, DirectoryLoader

# Konfigurasi
DOCS_FOLDER = "docsspero"
EMBEDDING_MODEL = "nomic-embed-text:latest"
OLLAMA_MODEL = "gemma3:latest"
OLLAMA_BASE_URL = "http://codex.petra.ac.id:11434"
MAX_RETRIES = 3
RETRY_DELAY = 1
MAX_TOKENS = 4096
TEMPERATURE = 0.7
TOP_K = 5
TOP_P = 0.95
REPEAT_PENALTY = 1.1
STOP_WORDS = ["</s>", "Human:", "Assistant:"]
CACHE_TTL = 3600  # Cache TTL in seconds (1 hour)
MAX_CACHE_SIZE = 1000  # Maximum number of cached items
TIMEOUT = (30, 300)  # (connect timeout, read timeout) in seconds

# Initialize translator
translator = Translator()

# Configure logging with more detail and rotation
handler = logging.handlers.RotatingFileHandler(
    'rag_api.log',
    maxBytes=10000000,  # 10MB
    backupCount=5
)
handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(levelname)s - %(message)s'
))
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(handler)
logger.addHandler(logging.StreamHandler())

# Configure retry strategy
retry_strategy = Retry(
    total=MAX_RETRIES,
    backoff_factor=1,  # Will wait 1, 2, 4, 8, 16 seconds between retries
    status_forcelist=[408, 429, 500, 502, 503, 504],
    allowed_methods=["HEAD", "GET", "PUT", "DELETE", "OPTIONS", "TRACE", "POST"]
)

# Create HTTP adapter with retry strategy
http_adapter = HTTPAdapter(max_retries=retry_strategy)

# Create session with retry configuration
def create_session():
    session = requests.Session()
    session.mount("http://", http_adapter)
    session.mount("https://", http_adapter)
    return session

def retry_with_backoff(func):
    """Decorator for retrying functions with exponential backoff."""
    def wrapper(*args, **kwargs):
        for attempt in range(MAX_RETRIES):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    raise
                wait_time = RETRY_DELAY * (2 ** attempt)  # exponential backoff
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
    return wrapper

class RetryEmbeddings:
    def __init__(self, base_embeddings):
        self.base = base_embeddings

    @retry_with_backoff
    def embed_documents(self, texts):
        return self.base.embed_documents(texts)

    @retry_with_backoff
    def embed_query(self, text):
        return self.base.embed_query(text)

# Create cache for embeddings
embedding_cache = TTLCache(maxsize=MAX_CACHE_SIZE, ttl=CACHE_TTL)

class DocumentProcessor:
    def __init__(self):
        self.embeddings = OllamaEmbeddings(
            model=EMBEDDING_MODEL,
            base_url=OLLAMA_BASE_URL
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # Smaller chunks for better precision
            chunk_overlap=200,  # More overlap to maintain context
            length_function=len,
            separators=[
                "\n\n",  # Double newline for major sections
                "\n",    # Single newline for paragraphs
                ". ",    # Sentences
                "! ",    # Exclamations
                "? ",    # Questions
                "; ",    # Semicolons
                ": ",    # Colons
                ", ",    # Commas
                " ",     # Words
                ""       # Characters
            ]
        )
        self.vector_store = None
        self.last_successful_response = None
        self.translator = translator
        # Process documents immediately on initialization
        self.process_documents(force_reindex=True)

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text content."""
        # Preserve numbered lists and bullet points
        text = re.sub(r'(\d+\.\s+)', r'\n\1', text)  # Add newline before numbered items
        text = re.sub(r'([•\-\*]\s+)', r'\n\1', text)  # Add newline before bullet points

        # Replace multiple newlines with a single newline
        text = re.sub(r'\n+', '\n', text)

        # Replace multiple spaces with a single space
        text = re.sub(r'\s+', ' ', text)

        # Remove any leading/trailing whitespace
        text = text.strip()

        return text

    @cached(embedding_cache)
    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding with caching."""
        return self.embeddings.embed_query(text)

    @retry_with_backoff
    def _query_llm_with_fallback(self, prompt: str) -> Dict:
        """Query LLM with fallback to last successful response."""
        session = create_session()
        try:
            response = session.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                headers={"Content-Type": "application/json"},
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=TIMEOUT
            )
            response.raise_for_status()
            result = response.json()
            self.last_successful_response = result  # Store successful response
            return result
        except Exception as e:
            logger.error(f"Error querying LLM: {str(e)}")
            if self.last_successful_response:
                logger.warning("Using last successful response as fallback")
                return self.last_successful_response
            raise
        finally:
            session.close()

    @retry_with_backoff
    def _embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Wrapper for embedding documents with retry."""
        try:
            session = create_session()
            self.embeddings.client.session = session
            result = self.embeddings.embed_documents(texts)
            return result
        finally:
            session.close()

    @retry_with_backoff
    def _embed_query(self, text: str) -> List[float]:
        """Wrapper for embedding query with retry."""
        try:
            session = create_session()
            self.embeddings.client.session = session
            result = self.embeddings.embed_query(text)
            return result
        finally:
            session.close()

    def process_documents(self, force_reindex: bool = False) -> bool:
        """Process all documents and store in memory."""
        try:
            if not os.path.exists(DOCS_FOLDER):
                logger.error(f"Documents folder not found: {DOCS_FOLDER}")
                return False

            all_files = [f for f in os.listdir(DOCS_FOLDER) if f.lower().endswith('.pdf')]
            if not all_files:
                logger.warning("No PDF documents found in docs folder")
                return False

            all_documents = []

            # Process all files
            for filename in tqdm(all_files, desc="Processing documents"):
                filepath = os.path.join(DOCS_FOLDER, filename)
                try:
                    logger.info(f"Processing file: {filepath}")
                    loader = UnstructuredLoader(filepath)
                    docs = loader.load()

                    # Enhanced metadata
                    file_stats = os.stat(filepath)

                    processed_docs = []
                    for doc in docs:
                        # Clean and preprocess the text
                        text = self._clean_text(doc.page_content)
                        if text:  # Only add non-empty documents
                            doc.page_content = text
                            doc.metadata = {
                                "source": os.path.basename(filepath),
                                "full_path": filepath,
                                "last_modified": datetime.fromtimestamp(file_stats.st_mtime).isoformat(),
                                "file_size": file_stats.st_size,
                                "processed_at": datetime.now().isoformat()
                            }
                            processed_docs.append(doc)

                    all_documents.extend(processed_docs)
                    logger.info(f"Successfully processed {len(processed_docs)} segments from {filepath}")
                except Exception as e:
                    logger.error(f"Error processing file {filepath}: {str(e)}")
                    continue

            if all_documents:
                logger.info(f"Total documents loaded: {len(all_documents)}")
                logger.info("Splitting documents into chunks...")
                docs_split = self.text_splitter.split_documents(all_documents)
                logger.info(f"Created {len(docs_split)} chunks")

                # Create in-memory Chroma instance with retry-enabled embedding
                logger.info("Creating in-memory vector store...")
                start_time = time.time()

                # Create a wrapper class for embeddings with retry
                retry_embeddings = RetryEmbeddings(self.embeddings)

                try:
                    # Process documents in smaller batches
                    batch_size = 50  # Smaller batch size for better processing
                    total_batches = (len(docs_split) + batch_size - 1) // batch_size

                    logger.info(f"Processing {total_batches} batches of {batch_size} documents each")

                    # Initialize empty Chroma
                    self.vector_store = Chroma(
                        embedding_function=retry_embeddings,
                        collection_metadata={"hnsw:space": "cosine"}
                    )

                    # Add documents in batches
                    for i in range(0, len(docs_split), batch_size):
                        batch = docs_split[i:i+batch_size]
                        logger.info(f"Processing batch {(i//batch_size)+1}/{total_batches}")
                        self.vector_store.add_documents(documents=batch)
                        elapsed = time.time() - start_time
                        logger.info(f"Batch processed in {elapsed:.2f} seconds")

                    total_time = time.time() - start_time
                    logger.info(f"Vector store creation completed in {total_time:.2f} seconds")
                    return True

                except Exception as e:
                    logger.error(f"Error creating vector store: {str(e)}")
                    return False

            return False
        except Exception as e:
            logger.error(f"Error in process_documents: {str(e)}")
            return False

    async def _translate_text(self, text: str, target_lang: str) -> str:
        """Helper method to perform the actual translation."""
        try:
            translation = await self.translator.translate(text, dest=target_lang)
            return translation.text
        except Exception as e:
            logger.warning(f"Translation failed in _translate_text: {str(e)}")
            return text

    async def _translate_text_multi(self, text: str, source_lang: str) -> Dict[str, str]:
        """Translate text to multiple target languages based on source language."""
        translations = {source_lang: text}  # Include original text

        try:
            # Define target languages based on source language
            if source_lang == 'en':
                target_langs = ['id', 'zh-cn']
            elif source_lang in ['id', 'zh-cn']:
                target_langs = ['en']
            else:
                target_langs = ['en', 'id', 'zh-cn']

            # Perform translations
            for lang in target_langs:
                if lang != source_lang:
                    try:
                        translation = await self.translator.translate(text, dest=lang)
                        translations[lang] = translation.text
                    except Exception as e:
                        logger.warning(f"Translation failed for language {lang}: {str(e)}")
                        translations[lang] = text

            return translations
        except Exception as e:
            logger.warning(f"Translation failed in _translate_text_multi: {str(e)}")
            return {source_lang: text}

    def _detect_language(self, text: str) -> str:
        """Detect language and limit to supported languages only."""
        try:
            # Pre-check for common Indonesian patterns
            id_patterns = ['apa itu', 'siapa', 'mengapa', 'bagaimana', 'kapan', 'dimana', 'kenapa']
            text_lower = text.lower()
            if any(pattern in text_lower for pattern in id_patterns):
                return 'id'

            detected = detect(text)
            # Only return supported languages, default to English for others
            if detected in ['en', 'id', 'zh-cn']:
                return detected

            # Additional Indonesian word check
            id_words = ['dengan', 'yang', 'untuk', 'dari', 'dalam', 'pada', 'ini', 'itu', 'juga', 'sudah', 'saya', 'akan']
            word_count = sum(1 for word in id_words if word in text_lower)
            if word_count >= 2:  # If 2 or more Indonesian words are found
                return 'id'

            return 'en'
        except:
            return 'en'

    def _translate_if_needed(self, text: str, target_lang: str) -> str:
        """Translate text if needed and only for supported languages."""
        try:
            if target_lang not in ['en', 'id', 'zh-cn']:
                return text  # Don't translate if target language is not supported

            source_lang = self._detect_language(text)
            if source_lang != target_lang:
                # Apply nest_asyncio to allow running async code in sync context
                nest_asyncio.apply()

                # Create event loop if it doesn't exist
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)

                # Run the translation
                return loop.run_until_complete(self._translate_text(text, target_lang))
            return text
        except Exception as e:
            logger.warning(f"Translation failed in _translate_if_needed: {str(e)}")
            return text

    def query_documents(self, question: str, k: int = 5) -> Dict[str, Any]:
        """Query documents with enhanced context retrieval."""
        if not self.vector_store:
            return {
                "error": "Vector store not initialized",
                "status": "error"
            }

        try:
            # Detect question language
            question_lang = self._detect_language(question)
            logger.info(f"Detected question language: {question_lang}")

            # Get relevant documents with MMR search for diversity
            k_multiplier = 6
            fetch_k = k * k_multiplier
            relevant_docs = self.vector_store.max_marginal_relevance_search(
                question,
                k=fetch_k,
                fetch_k=fetch_k * 2,
                lambda_mult=0.7
            )

            # Group documents by source
            docs_by_source = {}
            for doc in relevant_docs:
                source = doc.metadata['source']
                if source not in docs_by_source:
                    docs_by_source[source] = []
                docs_by_source[source].append(doc)

            # Calculate relevance score for each source
            source_scores = {}
            for source, docs in docs_by_source.items():
                first_position = relevant_docs.index(docs[0])
                source_scores[source] = {
                    'count': len(docs),
                    'first_position': first_position,
                    'score': (len(docs) * 0.4) + ((fetch_k - first_position) * 0.6)
                }

            # Select the most relevant source
            best_source = max(source_scores.items(), key=lambda x: x[1]['score'])[0]
            logger.info(f"Selected most relevant source: {best_source} (Score: {source_scores[best_source]})")

            # Create context using only documents from the best source
            context_parts = []
            seen_content = set()

            for doc in docs_by_source[best_source]:
                content = self._clean_text(doc.page_content)
                # Translate content if needed
                if question_lang in ['en', 'id', 'zh-cn']:
                    content_lang = self._detect_language(content)
                    if content_lang != question_lang:
                        content = self._translate_if_needed(content, question_lang)

                content_hash = hashlib.md5(content.encode()).hexdigest()
                if content_hash not in seen_content:
                    seen_content.add(content_hash)
                    if question_lang == 'zh-cn':
                        doc_label = self._translate_if_needed("Document:", question_lang)
                        context_parts.append(f"{doc_label} {best_source}\n{self._translate_if_needed('Content:', question_lang)} {content}")
                    else:
                        context_parts.append(f"Document: {best_source}\nContent: {content}")

            context = "\n\n".join(context_parts)

            # Prepare prompt based on language
            if question_lang == 'en':
                prompt = f"""Answer ONLY the specific question asked, using information from this context:

Context:
{context}

Question: {question}

Instructions:
1. Answer ONLY what was asked
2. Keep it brief and direct
3. Do not add extra information
4. Do not mention the source document
5. If the answer is not in the context, say "I cannot find relevant information."
"""
            elif question_lang == 'zh-cn':
                prompt = f"""只回答具体问题，使用以下上下文：

上下文：
{context}

问题：{question}

说明：
1. 只回答被问到的内容
2. 保持简短直接
3. 不要添加额外信息
4. 不要提及文档来源
5. 如果答案不在上下文中，请说"我找不到相关信息。"
"""
            else:  # Default to Indonesian
                prompt = f"""Jawab HANYA pertanyaan spesifik yang ditanyakan, menggunakan informasi dari konteks ini:

Context:
{context}

Question: {question}

Instructions:
1. Jawab HANYA yang ditanyakan
2. Tetap singkat dan langsung
3. Jangan tambahkan informasi tambahan
4. Jangan sebutkan dokumen sumber
5. Jika jawaban tidak ada dalam konteks, katakan "Saya tidak dapat menemukan informasi yang relevan."
"""

            # Query LLM
            result = self._query_llm_with_fallback(prompt)
            answer = result.get("response", "").strip()

            # Clean up the response in all supported languages
            cleanup_phrases = {
                'en': [
                    "Based on the context, ", "According to the document, ",
                    "According to the context, ", "Based on the document, ",
                    "In the document, ", "The document states that ",
                    "From the document, ", "The context shows that ",
                    "The information shows that ", "The document indicates that ",
                    "Based on the information provided, ", "According to the information, "
                ],
                'id': [
                    "Berdasarkan konteks, ", "Berdasarkan dokumen, ",
                    "Menurut dokumen, ", "Dalam dokumen, ",
                    "Dokumen menyatakan bahwa ", "Dari dokumen, ",
                    "Konteks menunjukkan bahwa ", "Informasi menunjukkan bahwa ",
                    "Dokumen menunjukkan bahwa ", "Berdasarkan informasi yang diberikan, ",
                    "Menurut informasi, "
                ],
                'zh-cn': [
                    "根据上下文，", "根据文档，", "文档显示，",
                    "根据内容，", "文档中说，", "从文档来看，",
                    "上下文表明，", "信息显示，", "文档指出，",
                    "根据提供的信息，", "根据信息，"
                ]
            }

            for phrases in cleanup_phrases.values():
                for phrase in phrases:
                    answer = answer.replace(phrase, "")

            # Remove source document mentions and extra information
            source_name = best_source.replace('.pdf', '').replace('_', ' ')
            answer = answer.replace(source_name, '')
            answer = answer.replace('.pdf', '')
            
            # Remove any text after the first complete sentence that seems unrelated
            sentences = answer.split('. ')
            if sentences:
                answer = sentences[0].strip() + '.'
            
            # Remove any parenthetical content
            answer = re.sub(r'\([^)]*\)', '', answer)
            
            # Remove any URLs
            answer = re.sub(r'http\S+', '', answer)
            
            # Clean up whitespace
            answer = ' '.join(answer.split())

            # Ensure answer is in the correct language
            if question_lang in ['en', 'id', 'zh-cn']:
                answer_lang = self._detect_language(answer)
                if answer_lang != question_lang:
                    answer = self._translate_if_needed(answer, question_lang)

                # For Chinese, ensure any remaining non-Chinese text is translated
                if question_lang == 'zh-cn':
                    parts = [p for p in re.split(r'([。，：；？！\n])', answer) if p.strip()]
                    translated_parts = []
                    for part in parts:
                        if not all('\u4e00' <= c <= '\u9fff' or c in '。，：；？！\n' for c in part):
                            part = self._translate_if_needed(part, 'zh-cn')
                        translated_parts.append(part)
                    answer = ''.join(translated_parts)

            # Final cleanup of any remaining document references
            answer = re.sub(r'(?i)(berdasarkan|menurut|dalam|dari)\s+([^,.]+\.pdf|dokumen|konteks)[,:]?\s*', '', answer)
            answer = re.sub(r'(?i)(based on|according to|in|from)\s+([^,.]+\.pdf|the document|context)[,:]?\s*', '', answer)
            answer = re.sub(r'(?i)(根据|按照|依据|从)\s*([^，。]+\.pdf|文档|上下文)[，。]?\s*', '', answer)

            return {
                "answer": answer,
                "sources": [best_source],  # Only return the best source
                "status": "success"
            }

        except Exception as e:
            logger.error(f"Error during query: {str(e)}")
            if self.last_successful_response:
                return {
                    "answer": self.last_successful_response.get("response", ""),
                    "sources": [],
                    "status": "success (fallback)"
                }
            return {
                "error": str(e),
                "status": "error"
            }

    def query_llm(self, prompt: str, lang: str = 'id') -> str:
        """Query LLM with retry logic and timeout."""
        for attempt in range(MAX_RETRIES):
            try:
                # Prepare the prompt with language instruction
                if lang == 'zh':
                    lang_instr = "请用中文回答，只需要回答中文，不要包含拼音和英文翻译。"
                elif lang == 'id':
                    lang_instr = "Jawab dalam Bahasa Indonesia."
                else:
                    lang_instr = "Answer in English."

                wrapped = f"{lang_instr}\n{prompt}"
                url = f"{OLLAMA_BASE_URL}/api/generate"
                payload = {
                    "model": OLLAMA_MODEL,
                    "prompt": wrapped,
                    "stream": False,
                    "options": {
                        "temperature": TEMPERATURE,
                        "top_k": TOP_K,
                        "top_p": TOP_P,
                        "repeat_penalty": REPEAT_PENALTY,
                        "stop": STOP_WORDS
                    }
                }

                logger.info(f"🌐 Requesting Ollama response with lang_instr...")
                lt = self.loading_animation("Waiting for Ollama")
                try:
                    res = requests.post(url, json=payload, timeout=TIMEOUT)
                    res.raise_for_status()
                    response = res.json().get("response", "")
                    # Clean Chinese response if needed
                    if lang == 'zh':
                        response = self.clean_chinese_text(response)
                    return response
                except Exception as e:
                    logger.error(f"[!] Ollama error: {e}")
                    return ''
                finally:
                    self.loading = False
                    lt.join()

            except Exception as e:
                logger.error(f"Error querying LLM: {str(e)}")
                if attempt < MAX_RETRIES - 1:
                    wait_time = RETRY_DELAY * (attempt + 1)
                    logger.warning(f"Attempt {attempt + 1} failed: {str(e)}. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    raise

        return ""

    def get_rag_response(self, prompt: str, lang: str) -> str:
        """Get response from RAG with improved language detection and fallback handling."""
        # Check if the prompt contains Chinese characters
        has_chinese = bool(re.search(r'[\u4e00-\u9fff]', prompt))
        if has_chinese:
            lang = 'zh'
            logger.info(f"🈶 Chinese characters detected in prompt, forcing Chinese response")

        # Language-specific instructions and fallback patterns
        lang_settings = {
            'zh': {
                'instruction': "请用中文回答，只需要回答中文，不要包含拼音和英文翻译。",
                'fallback_patterns': ["我不知道", "没有相关信息", "找不到相关信息", "无法找到", "没有找到", "抱歉", "对不起", "我不明白"],
                'fallback_response': "抱歉，我在文档中没有找到相关信息。让我为您查找其他信息。"
            },
            'id': {
                'instruction': "Jawab dalam Bahasa Indonesia.",
                'fallback_patterns': ["saya tidak tahu", "tidak ada informasi", "informasi tidak ditemukan", "tidak dapat menemukan", "tidak menemukan", "maaf", "maafkan saya", "saya tidak mengerti"],
                'fallback_response': "Maaf, saya tidak menemukan informasi yang relevan dalam dokumen. Mari saya cari informasi lainnya."
            },
            'en': {
                'instruction': "Answer in English.",
                'fallback_patterns': ["i don't know", "no information", "information not found", "cannot find", "no relevant", "not found in the documents", "sorry", "i don't understand", "i'm sorry"],
                'fallback_response': "I'm sorry, I couldn't find relevant information in the documents. Let me search for other information."
            }
        }

        # Get language-specific settings
        settings = lang_settings.get(lang, lang_settings['en'])
        lang_instr = settings['instruction']
        fallback_patterns = settings['fallback_patterns']
        fallback_response = settings['fallback_response']

        wrapped = f"{lang_instr}\n{prompt}"
        logger.info(f"🔗 Prompt to RAG server: {wrapped}")
        try:
            resp = requests.post(
                self.rag_url,
                json={"prompt": wrapped, "force_lang": lang},
                headers={"Content-Type": "application/json"},
                timeout=TIMEOUT
            )
            resp.raise_for_status()
            response = resp.json().get('response', '').lower()

            # Check for fallback patterns
            should_fallback = any(pattern in response.lower() for pattern in fallback_patterns)
            if should_fallback:
                logger.warning(f"⚠️ RAG response indicates no information: '{response}'")
                logger.info(f"🔄 Falling back to Ollama with message: {fallback_response}")
                return None

            return resp.json().get('response', '')
        except Exception as e:
            logger.error(f"RAG error: {str(e)}")
            return None

# Initialize Flask app
app = Flask(__name__)
processor = DocumentProcessor()

# Add request logging
@app.before_request
def log_request_info():
    logger.info(f"{request.remote_addr} - {request.method} {request.url}")
    if request.is_json:
        logger.info(f"Request JSON: {request.json}")

@app.route("/update", methods=["POST"])
def update_documents():
    """Endpoint to force update of document processing."""
    force_reindex = request.args.get('force', 'false').lower() == 'true'
    success = processor.process_documents(force_reindex=force_reindex)
    return jsonify({
        "status": "success" if success else "error",
        "message": "Documents processed successfully" if success else "No documents processed",
        "force_reindex": force_reindex
    })

@app.route("/query", methods=["POST"])
def query():
    """Endpoint to query the documents."""
    data = request.json
    if not data:
        logger.error("No data provided in request")
        return jsonify({
            "error": "No data provided",
            "status": "error"
        }), 400

    # Support both 'question' and 'prompt' parameters for compatibility
    question = data.get("question") or data.get("prompt")
    if not question:
        logger.error("No question/prompt provided in request")
        return jsonify({
            "error": "No question/prompt provided",
            "status": "error"
        }), 400

    k = data.get("k", 5)
    result = processor.query_documents(question, k)

    # Add response field for compatibility with new client
    if "answer" in result:
        result["response"] = result["answer"]

    logger.info(f"Query response: {result}")
    return jsonify(result)

@app.route("/documents", methods=["GET"])
def list_documents():
    """Endpoint to list all documents in the server."""
    try:
        if not os.path.exists(DOCS_FOLDER):
            return jsonify({
                "status": "error",
                "message": "Documents folder not found",
                "documents": []
            })

        documents = []
        for filename in os.listdir(DOCS_FOLDER):
            if filename.lower().endswith('.pdf'):
                filepath = os.path.join(DOCS_FOLDER, filename)
                file_stats = os.stat(filepath)
                documents.append({
                    "filename": filename,
                    "size_bytes": file_stats.st_size,
                    "last_modified": datetime.fromtimestamp(file_stats.st_mtime).isoformat(),
                    "path": filepath
                })

        return jsonify({
            "status": "success",
            "total_documents": len(documents),
            "documents": documents
        })
    except Exception as e:
        logger.error(f"Error listing documents: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e),
            "documents": []
        })

@app.route("/status", methods=["GET"])
def status():
    """Endpoint to check system status."""
    return jsonify({
        "status": "active",
        "docs_folder": DOCS_FOLDER,
        "embedding_model": EMBEDDING_MODEL,
        "llm_url": OLLAMA_BASE_URL,
        "llm_model": OLLAMA_MODEL
    })

@app.route('/add_document', methods=['POST'])
def add_document():
    try:
        # Cek apakah ada file yang diupload
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        # Simpan file
        filename = os.path.join('documents', file.filename)
        file.save(filename)

        # Load dan proses dokumen baru
        loader = TextLoader(filename)
        documents = loader.load()

        # Split dokumen
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        texts = text_splitter.split_documents(documents)

        # Tambahkan ke vector store
        global vector_store
        if vector_store is None:
            vector_store = FAISS.from_documents(texts, embeddings)
        else:
            vector_store.add_documents(texts)

        # Simpan vector store yang diupdate
        vector_store.save_local("faiss_index")

        return jsonify({
            "message": "Document added successfully",
            "filename": file.filename
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/add_text', methods=['POST'])
def add_text():
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({"error": "No text provided"}), 400

        text = data['text']
        title = data.get('title', 'untitled.txt')

        # Simpan text ke file
        filename = os.path.join('documents', title)
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(text)

        # Load dan proses dokumen baru
        loader = TextLoader(filename)
        documents = loader.load()

        # Split dokumen
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        texts = text_splitter.split_documents(documents)

        # Tambahkan ke vector store
        global vector_store
        if vector_store is None:
            vector_store = FAISS.from_documents(texts, embeddings)
        else:
            vector_store.add_documents(texts)

        # Simpan vector store yang diupdate
        vector_store.save_local("faiss_index")

        return jsonify({
            "message": "Text added successfully",
            "filename": title
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/list_documents', methods=['GET'])
def list_documents_txt():
    try:
        documents = []
        for filename in os.listdir('documents'):
            if filename.endswith('.txt'):
                filepath = os.path.join('documents', filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                documents.append({
                    "filename": filename,
                    "content": content[:200] + "..." if len(content) > 200 else content
                })
        return jsonify({"documents": documents})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Create docsspero directory if it doesn't exist
    if not os.path.exists(DOCS_FOLDER):
        os.makedirs(DOCS_FOLDER)
        logger.info(f"Created documents folder: {DOCS_FOLDER}")

    # Start Flask server
    logger.info("Starting Flask server...")
    app.run(host="0.0.0.0", port=50002, debug=False)
