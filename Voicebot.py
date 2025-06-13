import os
import re
import time
import itertools
import threading
import queue
import requests
import sounddevice as sd
import soundfile as sf
import torch
import whisper
import numpy as np
from langdetect import detect
from TTS.utils.manage import ModelManager
from TTS.utils.synthesizer import Synthesizer
from TTS.api import TTS
import json
import paho.mqtt.client as mqtt
import io
import wave
import pyaudio


class OllamaMultilingualTTS:
    def __init__(self, ollama_host, rag_host, wake_word="alexa", gpu=True, model_size="small"):
        print("\n🚀 Initializing Voice Assistant...")

        # Robot wake words and commands
        self.robot_wake_words = [
            "robot", "robo", "rob", "robots"
        ]

        self.robot_commands = {
            "stop"     : ["stop", "berhenti", "henti", "stop robot"],
            "start"    : ["start", "mulai", "jalan", "start robot"],
            "left"     : ["kiri", "belok kiri", "turn left"],
            "right"    : ["kanan", "belok kanan", "turn right"],
            "forward"  : ["maju", "jalan", "forward"],
            "waypoint" : ["checkpoint", "Waypoint","tujuan"],
            "backward" : ["mundur", "backward"]

        }

        # Initialize processing flags
        self.is_processing = False
        self.is_recording = False
        self.is_mode_selected = False

        # Initialize greeting variables
        self.last_greeting_time = 0
        self.greeting_cooldown = 30  # 30 seconds cooldown
        self.has_greeted = False

        # Audio settings
        self.sample_rate = 16000
        self.channels = 1
        self.input_device = None  # Will be set automatically

        # MQTT Configuration
        self.BROKER = "codex.petra.ac.id"
        self.PORT = 1883
        self.TOPIC = "test/topic"
        self.CLIENT_ID = "voice_assistant_client"

        # Initialize MQTT client
        self.mqtt_client = mqtt.Client(
            client_id=self.CLIENT_ID,
            protocol=mqtt.MQTTv5,
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2
        )

        # Connect to MQTT broker
        try:
            self.mqtt_client.connect(self.BROKER, self.PORT, 60)
            self.mqtt_client.loop_start()
            print("✅ Connected to MQTT broker!")
        except Exception as e:
            print(f"❌ MQTT Connection error: {str(e)}")

        # Set up RAG and Ollama hosts
        self.ollama_host = ollama_host
        self.rag_host = rag_host
        self.rag_url = f"http://{rag_host}/query"
        self.wake_word = wake_word.lower()
        self.gpu = gpu
        self.model_size = model_size
        self.device = "cuda" if torch.cuda.is_available() and gpu else "cpu"

        # List all available audio devices
        print("\n🎤 Available Audio Devices:")
        devices = sd.query_devices()
        for i, device in enumerate(devices):
            print(f"[{i}] {device['name']} (Max Input Channels: {device['max_input_channels']})")

        # Find USB Audio Device by name
        self.input_device = None
        for i, device in enumerate(devices):
            if "USB Audio Device" in device['name'] and device['max_input_channels'] > 0:
                self.input_device = i
                print(f"\n✅ Found USB Audio Device: [{i}] {device['name']}")
                break

        if self.input_device is None:
            print("\n⚠️ USB Audio Device not found, searching for any working input device...")
            for i, device in enumerate(devices):
                if device['max_input_channels'] > 0:
                    try:
                        sd.check_input_settings(
                            device=i,
                            channels=1,
                            samplerate=44100
                        )
                        print(f"✅ Found working device: [{i}] {device['name']}")
                        self.input_device = i
                        break
                    except:
                        continue

        if self.input_device is None:
            print("❌ No working input device found")
            self.sample_rate = 44100
            self.channels = 1
            return

        # Get audio device info and sample rate
        try:
            device_info = sd.query_devices(self.input_device)
            print(f"\n🎤 Using input device {self.input_device}:")
            print(f"Name: {device_info['name']}")
            print(f"Max Input Channels: {device_info['max_input_channels']}")
            print(f"Default Sample Rate: {device_info['default_samplerate']}Hz")

            if device_info['max_input_channels'] == 0:
                raise ValueError("Selected device has no input channels")

            # Always use mono for input
            self.channels = 1
            self.sample_rate = int(device_info['default_samplerate'])
            print(f"Using {self.channels} channel(s)")

            # Test the device configuration
            test_stream = sd.InputStream(
                device=self.input_device,
                channels=self.channels,
                samplerate=self.sample_rate
            )
            test_stream.start()
            test_stream.stop()
            test_stream.close()

        except Exception as e:
            print(f"⚠️ Error accessing device {self.input_device}: {str(e)}")
            print("Using default settings...")
            self.sample_rate = 44100
            self.channels = 1

        # Audio recording parameters
        self.silence_threshold = 0.01
        self.silence_window = 0.8
        self.min_phrase_duration = 1.0
        self.max_phrase_duration = 15.0

        self.silence_samples = int(self.silence_window * self.sample_rate)
        self.audio_buffer = np.zeros(self.silence_samples)

        print("\n📥 Loading TTS Models...")
        # Load all TTS models with optimized settings
        manager = ModelManager()

        # Monitor initial VRAM
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            initial_vram = torch.cuda.memory_allocated(0) / (1024**3)
            total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"\n💾 Initial VRAM Usage: {initial_vram:.2f} GB")
            print(f"💾 Total VRAM Available: {total_vram:.2f} GB")

        # Use Tacotron for Chinese TTS with optimized settings
        print("\n🈶 Loading Chinese TTS Model...")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            before_zh = torch.cuda.memory_allocated(0) / (1024**3)

        model_zh, config_zh, _ = manager.download_model("tts_models/zh-CN/baker/tacotron2-DDC-GST")
        self.synth_zh = Synthesizer(
            model_zh,
            config_zh,
            use_cuda=self.gpu
        )
        if hasattr(self.synth_zh, 'max_decoder_steps'):
            self.synth_zh.max_decoder_steps = 2000  # Increase max steps

        if torch.cuda.is_available():
            after_zh = torch.cuda.memory_allocated(0) / (1024**3)
            print(f"💾 Chinese TTS VRAM Usage: {after_zh - before_zh:.2f} GB")

        # Load English TTS
        print("\n🇬🇧 Loading English TTS Model...")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            before_en = torch.cuda.memory_allocated(0) / (1024**3)

        self.tts_en = TTS(model_name="tts_models/eng/fairseq/vits", gpu=self.gpu, progress_bar=False)

        if torch.cuda.is_available():
            after_en = torch.cuda.memory_allocated(0) / (1024**3)
            print(f"💾 English TTS VRAM Usage: {after_en - before_en:.2f} GB")

        # Load Indonesian TTS
        print("\n🇮🇩 Loading Indonesian TTS Model...")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            before_id = torch.cuda.memory_allocated(0) / (1024**3)

        self.tts_id = TTS(model_name="tts_models/ind/fairseq/vits", gpu=self.gpu, progress_bar=False)

        if torch.cuda.is_available():
            after_id = torch.cuda.memory_allocated(0) / (1024**3)
            print(f"💾 Indonesian TTS VRAM Usage: {after_id - before_id:.2f} GB")

            # Show total VRAM usage for all TTS models
            final_vram = torch.cuda.memory_allocated(0) / (1024**3)
            print(f"\n💾 Total VRAM Usage by TTS Models: {final_vram - initial_vram:.2f} GB")
            print(f"💾 VRAM Available after TTS load: {total_vram - final_vram:.2f} GB")

        # Initialize Whisper
        print(f"\n🎯 Loading Whisper Model ({model_size})...")
        try:
            print(f"Using device: {self.device}")
            print(f"CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"GPU: {torch.cuda.get_device_name(0)}")
                total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                allocated = torch.cuda.memory_allocated(0) / (1024**3)
                print(f"Total VRAM: {total_vram:.2f} GB")
                print(f"VRAM Allocated: {allocated:.2f} GB")
                print(f"VRAM Available: {total_vram - allocated:.2f} GB")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            self.whisper_model = whisper.load_model(
                model_size,
                device=self.device,
                download_root="./models",
                in_memory=True
            )

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                allocated_after = torch.cuda.memory_allocated(0) / (1024**3)
                print(f"\nVRAM Usage by Whisper: {allocated_after - allocated:.2f} GB")
                print(f"VRAM Available after load: {total_vram - allocated_after:.2f} GB")

            print("✅ Whisper model loaded successfully!")

        except Exception as e:
            print(f"⚠️ Error loading {model_size} model: {str(e)}")
            print("⚠️ Falling back to tiny model...")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            self.whisper_model = whisper.load_model(
                "tiny",
                device=self.device,
                download_root="./models",
                in_memory=True
            )

        # Generate wake word detection audio (English only)
        print("\n🗣️ Generating wake word response...")
        try:
            result = self.tts_en.tts("Wake word detected, please speak your question")
            if isinstance(result, tuple):
                wav = result[0]
                sr = self.tts_en.synthesizer.output_sample_rate
            else:
                wav = result
                sr = self.tts_en.synthesizer.output_sample_rate
            self.wake_detect_audio = (np.array(wav, dtype=np.float32), sr)
            print("✅ Wake word audio generated successfully!")
        except Exception as e:
            print(f"⚠️ Error generating wake word audio: {str(e)}")
            self.wake_detect_audio = None

        print("\n✅ Initialization complete! Ready to start.\n")
        print(f"💡 Using Whisper {model_size} model")
        print(f"💡 Wake word is set to '{wake_word}'")
        print(f"💡 GPU acceleration is {'enabled' if gpu else 'disabled'}\n")

    def loading_animation(self, message="Processing", delay=0.1):
        spinner = itertools.cycle(["⠋","⠙","⠹","⠸","⠼","⠴","⠦","⠧","⠇","⠏"])
        self.loading = True

        def animate():
            while self.loading:
                print(f"\r{message} {next(spinner)}", end="", flush=True)
                time.sleep(delay)
            print("\r" + " "*(len(message)+2), end="\r")

        thread = threading.Thread(target=animate)
        thread.start()
        return thread

    def detect_language(self, text):
        """Detect language with better accuracy and handle mixed language content"""
        try:
            # Count characters for each language type
            chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
            english_chars = len(re.findall(r'[a-zA-Z]', text))
            indo_chars = len(re.findall(r'[a-zA-Z]', text))  # Will be refined below
            total_chars = len(text.strip())

            # Common Indonesian question words
            indo_question_words = [
                'siapa', 'apa', 'bagaimana', 'mengapa', 'kenapa', 'kapan', 'dimana',
                'berapa', 'siapakah', 'apakah', 'bagaimanakah', 'mengapakah', 'kenapakah',
                'kapankah', 'dimanakah', 'berapakah'
            ]

            # Check for Indonesian question words first
            has_indo_question = any(word in text.lower() for word in indo_question_words)
            if has_indo_question:
                print("🇮🇩 Indonesian question word detected")
                return 'id'

            # Calculate percentages
            chinese_percent = chinese_chars / total_chars if total_chars > 0 else 0
            english_percent = english_chars / total_chars if total_chars > 0 else 0

            # Check for technical terms and common English words that shouldn't be translated
            technical_pattern = r'\b(API|CPU|GPU|RAM|USB|HTTP|SDK|AI|ML|IoT|ID)\b'
            has_technical_terms = bool(re.search(technical_pattern, text, re.IGNORECASE))

            # Common English words/names that shouldn't be translated
            common_english = r'\b(Windows|Linux|Python|Java|Android|iOS|Microsoft|Google|Apple|Amazon|Facebook)\b'
            has_common_english = bool(re.search(common_english, text))

            # If text contains significant Chinese characters (>15%)
            if chinese_percent > 0.15:
                return 'zh'

            # If text contains technical terms or common English words, treat as English
            if has_technical_terms or has_common_english:
                return 'en'

            # Common English patterns
            english_patterns = [
                r'\b(what|how|why|when|where|who|which)\b',
                r'\b(is|are|was|were|will|would|could|should)\b',
                r'\b(the|a|an|this|that|these|those)\b'
            ]

            # Common Indonesian patterns
            indo_patterns = [
                r'\b(apa|bagaimana|mengapa|kapan|dimana|siapa|yang|mana)\b',
                r'\b(adalah|ialah|merupakan|menjadi|akan)\b',
                r'\b(ini|itu|tersebut|tersebut)\b'
            ]

            # Count pattern matches
            english_pattern_matches = sum(len(re.findall(pattern, text.lower())) for pattern in english_patterns)
            indo_pattern_matches = sum(len(re.findall(pattern, text.lower())) for pattern in indo_patterns)

            # Strong indicators for English
            if english_pattern_matches >= 1 and indo_pattern_matches == 0:
                return 'en'

            # Strong indicators for Indonesian
            if indo_pattern_matches >= 1 and english_pattern_matches == 0:
                return 'id'

            # If significant English characters present (>30%)
            if english_percent > 0.3:
                return 'en'

            # Fallback to langdetect for ambiguous cases
            detected = detect(text)
            if detected in ['en', 'id', 'zh']:
                return detected

            return 'id'  # Default to Indonesian if uncertain

        except Exception as e:
            print(f"Language detection error: {str(e)}")
            return 'id'  # Default to Indonesian on error

    def clean_response(self, text):
        return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

    def get_rag_response(self, prompt, lang):
        """Get response from RAG server with better relevance and length control, and handle fallback if info is not relevant or not found"""
        try:
            # Get language-specific prompt
            lang_prompt = self.get_language_prompt(lang)

            # Prepare the full prompt - simplified to avoid over-filtering
            full_prompt = f"{prompt}"

            print(f"\n🔗 Menghubungi RAG server...")
            print(f"🔗 Prompt to RAG server: {full_prompt}")

            # Format URL properly with explicit http://
            rag_url = f"http://{self.rag_host}/query"
            print(f"🔗 RAG URL: {rag_url}")

            # Update request format to match working curl command
            headers = {
                "Content-Type": "application/json"
            }

            payload = {
                "question": full_prompt
            }

            print(f"🔗 Sending request to: {rag_url}")
            print(f"🔗 Headers: {headers}")
            print(f"🔗 Payload: {payload}")

            response = requests.post(
                rag_url,
                headers=headers,
                json=payload,
                timeout=30,
                verify=False
            )
            response.raise_for_status()

            result = response.json()
            if result.get("status") == "success" and "response" in result:
                # Clean and limit the response
                text = result["response"].strip()

                # Fallback patterns for each language
                fallback_patterns = {
                    'zh': ["我不知道", "没有相关信息", "找不到相关信息", "无法找到", "没有找到", "抱歉", "对不起", "我不明白"],
                    'id': ["saya tidak tahu", "tidak ada informasi", "informasi tidak ditemukan", "tidak dapat menemukan", "tidak menemukan", "maaf", "maafkan saya", "saya tidak mengerti"],
                    'en': ["i don't know", "no information", "information not found", "cannot find", "no relevant", "not found in the documents", "sorry", "i don't understand", "i'm sorry"]
                }

                patterns = fallback_patterns.get(lang, fallback_patterns['en'])

                # Check for fallback patterns
                if any(pattern in text.lower() for pattern in patterns):
                    print(f"⚠️ RAG response indicates no information: '{text}'")
                    print(f"🔄 Falling back to Ollama...")
                    return None

                # Remove any irrelevant content
                if "abstrak" in text.lower() or "kutipan" in text.lower() or "tugas akhir" in text.lower() or "penulisan" in text.lower():
                    print(f"⚠️ RAG response contains irrelevant content: '{text}'")
                    print(f"🔄 Falling back to Ollama...")
                    return None

                # Split into sentences and take first 2-3 sentences
                sentences = text.split('. ')
                if len(sentences) > 3:
                    text = '. '.join(sentences[:3]) + '.'

                print("\n==================================================")
                print("✅ RAG response received")
                print(f"💡 RAG: {text}")
                if "sources" in result:
                    print(f"📚 Sources: {', '.join(result['sources'])}")
                print("==================================================\n")

                return text
            return None

        except requests.exceptions.ConnectionError as e:
            print(f"⚠️ Error connecting to RAG server: {str(e)}")
            print("🔍 Please check if the RAG server is running and accessible")
            return None
        except requests.exceptions.Timeout as e:
            print(f"⚠️ RAG server request timed out: {str(e)}")
            return None
        except requests.exceptions.RequestException as e:
            print(f"⚠️ Error getting RAG response: {str(e)}")
            return None

    def get_ollama(self, prompt, lang):
        """Get response from Ollama with better control"""
        try:
            # Get language-specific prompt
            lang_prompt = self.get_language_prompt(lang)

            # Format URL properly with explicit http://
            ollama_url = f"http://{self.ollama_host}/api/generate"
            print(f"🔗 Ollama URL: {ollama_url}")

            # Prepare the full prompt
            full_prompt = f"{lang_prompt}\n{prompt}"

            # Set parameters for better response control
            params = {
                "model": "gemma3:latest",  # Using gemma3:latest instead of llama2
                "prompt": full_prompt,
                "stream": False,
                "max_tokens": 150,  # Limit response length
                "temperature": 0.7,  # Control randomness
                "top_p": 0.9,  # Nucleus sampling
                "top_k": 40,  # Top-k sampling
                "repeat_penalty": 1.1,  # Prevent repetition
                "stop": ["Human:", "Assistant:", "\n\n"]  # Stop sequences
            }

            print(f"🔗 Sending request to: {ollama_url}")
            print(f"🔗 Params: {params}")

            response = requests.post(
                ollama_url,
                json=params,
                timeout=30,
                verify=False
            )
            response.raise_for_status()

            result = response.json()
            if "response" in result:
                # Clean and limit the response
                text = result["response"].strip()

                # Split into sentences and take first 2-3 sentences
                sentences = text.split('. ')
                if len(sentences) > 3:
                    text = '. '.join(sentences[:3]) + '.'

                print("\n==================================================")
                print("✅ Ollama response received")
                print(f"💡 Ollama: {text}")
                print("==================================================\n")

                return text
            return ""

        except requests.exceptions.ConnectionError as e:
            print(f"⚠️ Error connecting to Ollama server: {str(e)}")
            print("🔍 Please check if the Ollama server is running and accessible")
            return ""
        except requests.exceptions.Timeout as e:
            print(f"⚠️ Ollama server request timed out: {str(e)}")
            return ""
        except requests.exceptions.RequestException as e:
            print(f"⚠️ Error getting Ollama response: {str(e)}")
            return ""

    def synthesize_to_array(self, text, lang):
        """Synthesize text to speech using detected language"""
        try:
            print(f"🔊 Using {lang.upper()} TTS")

            if lang == 'zh':
                wav = self.synth_zh.tts(text)
                return np.array(wav), self.synth_zh.output_sample_rate
            else:
                engine = self.tts_en if lang == 'en' else self.tts_id
                result = engine.tts(text)
                if isinstance(result, tuple):
                    wav = result[0]
                    sr = engine.synthesizer.output_sample_rate
                else:
                    wav = result
                    sr = engine.synthesizer.output_sample_rate
                return np.array(wav, dtype=np.float32), sr

        except Exception as e:
            print(f"⚠️ TTS error for {lang}: {str(e)}")
            if lang != 'en':
                print("Falling back to English TTS...")
                return self.synthesize_to_array(text, 'en')
            raise

    def is_valid_text(self, text):
        """Check if text contains valid content"""
        # Remove repeated characters
        cleaned = re.sub(r'(.)\1{4,}', r'\1\1\1', text)
        # Remove whitespace
        cleaned = cleaned.strip()
        # Check if text contains actual words/characters
        if not cleaned:
            return False
        # Check if text contains only valid unicode characters
        try:
            cleaned.encode('utf-8').decode('utf-8')
        except UnicodeError:
            return False
        # Check if text is not just punctuation or special characters
        if all(not c.isalnum() for c in cleaned):
            return False
        return True

    def detect_primary_language(self, text):
        """Detect if text is primarily English, Indonesian, or Chinese"""
        # Check for Chinese characters
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        if chinese_chars > 0:
            return 'zh'

        # Common English words/patterns
        eng_patterns = [
            r'\b(the|a|an|in|on|at|for|to|of|with|by|my|your|our|their)\b',
            r'\b(is|am|are|was|were|be|been|being|have|has|had|do|does|did)\b',
            r'\b(can|could|will|would|shall|should|may|might|must)\b',
            r'\b(this|that|these|those|here|there|what|where|when|who|why|how)\b'
        ]

        # Common Indonesian words/patterns
        indo_patterns = [
            r'\b(yang|dan|di|ke|dari|dengan|untuk|pada|dalam|ini|itu)\b',
            r'\b(saya|aku|kamu|dia|kami|kita|mereka|anda)\b',
            r'\b(adalah|ialah|merupakan|ada|sudah|telah|akan|bisa|harus)\b',
            r'\b(bagaimana|mengapa|kenapa|siapa|apa|dimana|kapan|berapa)\b'
        ]

        # Count matches
        eng_count = sum(len(re.findall(pattern, text.lower())) for pattern in eng_patterns)
        indo_count = sum(len(re.findall(pattern, text.lower())) for pattern in indo_patterns)

        # Return the language with more matches
        if eng_count > indo_count:
            return 'en'
        elif indo_count > eng_count:
            return 'id'
        else:
            return 'en'  # Default to English if unclear

    def transcribe(self, audio_path, detect_language=True):
        """Transcribe audio with multi-language support"""
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # First pass - detect language with more accurate settings
            print("\n🔄 Initial transcription to detect language...")
            initial_result = self.whisper_model.transcribe(
                audio_path,
                task="transcribe",  # Force transcription, not translation
                temperature=0.0,
                fp16=torch.cuda.is_available(),
                language=None,  # Let Whisper detect language
                condition_on_previous_text=False,
                initial_prompt="This is a transcription. Please transcribe exactly what is said without translation."  # Prevent auto-translation
            )

            initial_text = initial_result["text"].strip()
            if not initial_text:
                print("⚠️ No speech detected in audio")
                return ""

            if not detect_language:
                return initial_text

            # Only detect language if requested (after wake word)
            detected_lang = initial_result.get("language", "en")
            print(f"🎯 Whisper detected language: {detected_lang.upper()}")

            # Check for Chinese characters first (highest priority)
            if bool(re.search(r'[\u4e00-\u9fff]', initial_text)):
                detected_lang = 'zh'
                print("🈶 Chinese characters detected")
            # If Whisper detected a supported language, use it directly
            elif detected_lang in ['en', 'id', 'zh']:
                print(f"✅ Using Whisper's language detection: {detected_lang.upper()}")
            else:
                # Fallback to pattern matching
                print("⚠️ Whisper detected unsupported language, falling back to pattern matching...")

                # Check for Indonesian words
                indo_patterns = [
                    r'\b(yang|dan|di|ke|dari|dengan|untuk|pada|dalam|ini|itu)\b',
                    r'\b(saya|aku|kamu|dia|kami|kita|mereka|anda)\b',
                    r'\b(adalah|ialah|merupakan|ada|sudah|telah|akan|bisa|harus)\b',
                    r'\b(apa|bagaimana|mengapa|kenapa|siapa|dimana|kapan|berapa)\b',
                    r'\b(bisa|dapat|mau|ingin|perlu|harus|akan|sudah|telah|belum)\b'
                ]
                indo_count = sum(len(re.findall(pattern, initial_text.lower())) for pattern in indo_patterns)

                # Check for English words
                eng_patterns = [
                    r'\b(what|how|why|when|where|who|which)\b',
                    r'\b(is|are|was|were|will|would|could|should)\b',
                    r'\b(the|a|an|this|that|these|those)\b',
                    r'\b(can|will|would|should|could|must|may|might)\b',
                    r'\b(i|you|he|she|it|we|they|my|your|his|her|its|our|their)\b'
                ]
                eng_count = sum(len(re.findall(pattern, initial_text.lower())) for pattern in eng_patterns)

                # Determine language based on pattern matches
                if indo_count > eng_count:
                    detected_lang = 'id'
                    print("🇮🇩 Indonesian patterns detected")
                elif eng_count > 0:
                    detected_lang = 'en'
                    print("🇬🇧 English patterns detected")
                else:
                    # If no clear patterns, try to detect based on common words
                    common_indo = ['yang', 'dan', 'di', 'ke', 'dari', 'untuk', 'pada', 'dalam']
                    common_eng = ['the', 'and', 'in', 'to', 'from', 'for', 'on', 'in']

                    indo_words = sum(1 for word in initial_text.lower().split() if word in common_indo)
                    eng_words = sum(1 for word in initial_text.lower().split() if word in common_eng)

                    if indo_words > eng_words:
                        detected_lang = 'id'
                        print("🇮🇩 Indonesian words detected")
                    else:
                        detected_lang = 'en'
                        print("🇬🇧 English words detected")

            print(f"🎯 Final detected language: {detected_lang.upper()}")

            # Language-specific prompts to prevent translation
            prompts = {
                'en': "This is an English conversation. Please transcribe exactly what is said in English without translation.",
                'id': "Ini adalah percakapan dalam Bahasa Indonesia. Mohon transkripsikan persis apa yang dikatakan dalam Bahasa Indonesia tanpa terjemahan.",
                'zh': "这是一个中文对话。请准确转录所说的中文内容，不要翻译。"
            }

            # Second pass - transcribe with detected language and specific prompt
            print(f"🔄 Transcribing in detected language: {detected_lang.upper()}")
            result = self.whisper_model.transcribe(
                audio_path,
                task="transcribe",  # Force transcription, not translation
                language=detected_lang,
                temperature=0.0,
                fp16=torch.cuda.is_available(),
                initial_prompt=prompts.get(detected_lang, ""),
                condition_on_previous_text=False
            )

            text = result["text"].strip()
            if not text:
                print("⚠️ No text detected in transcription")
                return ""

            print(f"\n🎯 Raw Transcription ({detected_lang.upper()}):")
            print(f"Text: {text}")
            print("\n" + "="*50)
            print("📝 Question detected:")
            print(f"'{text}'")
            print("="*50 + "\n")

            return text

        except Exception as e:
            print(f"⚠️ Error in transcription: {str(e)}")
            return ""  # Return empty string instead of None

    def get_language_prompt(self, lang):
        """Get language-specific prompt to help with transcription accuracy"""
        prompts = {
            'en': "This is an English conversation.",
            'id': "Ini adalah percakapan dalam Bahasa Indonesia.",
            'zh': "这是一个中文对话。"
        }
        return prompts.get(lang, "")

    def clean_chinese_text(self, text):
        """Clean Chinese text to keep only Chinese characters and basic punctuation"""
        # Keep only Chinese characters and basic punctuation
        chinese_only = []
        for line in text.split('\n'):
            # Keep only Chinese characters and punctuation
            cleaned = ''.join(char for char in line if '\u4e00' <= char <= '\u9fff' or char in '，。！？；：')
            if cleaned:
                chinese_only.append(cleaned)
        return ''.join(chinese_only)

    def process(self, prompt):
        if not prompt or prompt.strip() == "":
            print("❌ Empty question detected, skipping processing")
            return

        print("\n" + "="*50)
        print("📝 Question detected:")
        print(f"'{prompt}'")
        print("="*50)

        # Use the language from the transcription
        detected_lang = self.detect_language(prompt)
        print(f"\n🌐 Using detected language: {detected_lang.upper()}")

        print("\n🔗 Menghubungi RAG server...")
        try:
            rag_ans = self.get_rag_response(prompt, detected_lang)
            if not rag_ans:
                print("\n" + "="*50)
                print("⚠️ RAG tidak menemukan informasi yang relevan")
                print("🔄 Menggunakan Ollama sebagai fallback...")
                print("="*50 + "\n")

                print("🌐 Menghubungi Ollama server...")
                resp = self.get_ollama(prompt, detected_lang)
                if not resp:
                    print("⚠️ Empty response from Ollama")
                    return
                clean = self.clean_response(resp)
            else:
                print("\n" + "="*50)
                print("✅ RAG response received")
                print(f"💡 RAG: {rag_ans}")
                print("="*50 + "\n")
                clean = self.clean_response(rag_ans)

            # For Chinese responses, clean to keep only Chinese characters
            if detected_lang == 'zh':
                clean = self.clean_chinese_text(clean)

            print("\n💬 Final response:")
            print("-"*30)
            print(clean)
            print("-"*30 + "\n")

            print("🔊 Generating speech...")
            print(f"[!] Using TTS for language: {detected_lang}")
            wav, sr = self.synthesize_to_array(clean, detected_lang)
            sf.write("temp.wav", wav, sr)
            sd.play(wav, sr)
            sd.wait()

        except Exception as e:
            print("\n" + "="*50)
            print(f"⚠️ Error processing request: {str(e)}")
            print("🔄 Falling back to Ollama...")
            print("="*50 + "\n")

            resp = self.get_ollama(prompt, detected_lang)
            if resp:
                clean = self.clean_response(resp)
                # For Chinese responses, clean to keep only Chinese characters
                if detected_lang == 'zh':
                    clean = self.clean_chinese_text(clean)

                print("\n💬 Ollama response:")
                print("-"*30)
                print(clean)
                print("-"*30 + "\n")

                print("🔊 Generating speech...")
                print(f"[!] Using TTS for language: {detected_lang}")
                wav, sr = self.synthesize_to_array(clean, detected_lang)
                sf.write("temp.wav", wav, sr)
                sd.play(wav, sr)
                sd.wait()
            else:
                print("❌ Failed to get response from both RAG and Ollama")

    def play_wake_word_detected(self, lang):
        """Play wake word detection message"""
        if self.wake_detect_audio is not None:
            try:
                wav, sr = self.wake_detect_audio
                sd.play(wav, sr)
                sd.wait()  # Wait for audio to finish playing
                print("🎤 Please speak your question...")
            except Exception as e:
                print(f"⚠️ Audio playback error: {str(e)}")

    def is_silent(self, audio_chunk):
        """
        Check if an audio chunk is silent using RMS energy
        """
        # Update rolling buffer
        chunk_samples = len(audio_chunk)
        self.audio_buffer = np.roll(self.audio_buffer, -chunk_samples)
        self.audio_buffer[-chunk_samples:] = audio_chunk.flatten()

        # Calculate RMS energy over the buffer window
        energy = np.sqrt(np.mean(self.audio_buffer**2))

        # Calculate zero crossing rate (helps detect consonant sounds)
        zero_crossings = np.sum(np.abs(np.diff(np.signbit(self.audio_buffer)))) / len(self.audio_buffer)

        is_quiet = energy < self.silence_threshold
        is_static = zero_crossings < 0.15

        return is_quiet and is_static

    def record_audio(self, duration=3):
        """Record audio for specified duration"""
        try:
            # Try to create a test stream first to verify settings
            try:
                test_stream = sd.InputStream(
                    samplerate=self.sample_rate,
                    channels=1,  # Always use mono
                    device=self.input_device,
                    latency='high'
                )
                test_stream.start()
                test_stream.stop()
                test_stream.close()
            except Exception as e:
                print(f"\n⚠️ Audio device test failed: {str(e)}")
                print("Trying to get supported sample rates...")
                try:
                    device_info = sd.query_devices(self.input_device)
                    print(f"Device supports sample rate: {device_info['default_samplerate']}Hz")
                    self.sample_rate = int(device_info['default_samplerate'])
                    print(f"Updated sample rate to: {self.sample_rate}Hz")
                except Exception as dev_e:
                    print(f"⚠️ Could not query device: {str(dev_e)}")
                    return None

            # Record audio
            print(f"🎤 Recording for {duration} seconds...")
            audio = sd.rec(
                int(duration * self.sample_rate),
                samplerate=self.sample_rate,
                channels=1,  # Always use mono
                device=self.input_device
            )
            sd.wait()

            # Convert to WAV format
            audio_data = (audio * 32767).astype(np.int16)
            wav_data = io.BytesIO()
            with wave.open(wav_data, 'wb') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 2 bytes for int16
                wav_file.setframerate(self.sample_rate)
                wav_file.writeframes(audio_data.tobytes())

            return wav_data

        except Exception as e:
            print(f"⚠️ Error recording audio: {str(e)}")
            return None

    def listen_loop(self):
        """Main loop for voice input"""
        try:
            print("\n🎤 Voice Assistant Ready!")
            print("1. Press 't' for Textual mode")
            print("2. Press 'v' for Voice mode")
            print("3. Say 'robot' followed by command for robot control")

            while True:
                try:
                    # Use select to handle both keyboard input and audio recording
                    import select
                    import sys

                    print("\n🎤 Listening for robot commands or mode selection...")
                    print("(Press 't' for Textual mode or 'v' for Voice mode)")

                    # Set up non-blocking input
                    import termios
                    import tty
                    old_settings = termios.tcgetattr(sys.stdin)
                    try:
                        tty.setcbreak(sys.stdin.fileno())

                        # Start recording in background
                        self.is_recording = True
                        audio = self.record_audio(duration=3)
                        self.is_recording = False

                        # Check for keyboard input
                        if select.select([sys.stdin], [], [], 0.1)[0]:
                            mode = sys.stdin.read(1).lower()
                            if mode in ['t', 'v']:
                                print(f"\nSelected mode: {mode}")
                            else:
                                print("\nInvalid input. Please press 't' or 'v'")
                                continue
                        else:
                            # Process audio if no keyboard input
                            if audio is not None:
                                # Save audio to file
                                audio_path = "temp_audio.wav"
                                with open(audio_path, "wb") as f:
                                    f.write(audio.getvalue())

                                # Transcribe audio
                                text = self.transcribe(audio_path)
                                if text:
                                    # Check if it's a robot command
                                    if self.check_robot_command(text):
                                        continue  # Skip mode selection if robot command was executed

                                    # Check if it's a mode selection
                                    if text.lower() in ['t', 'v']:
                                        mode = text.lower()
                                        print(f"\nSelected mode: {mode}")
                                    else:
                                        print("\nPlease select mode by saying 't' for Textual or 'v' for Voice")
                                        continue
                                else:
                                    print("\nPlease select mode by saying 't' for Textual or 'v' for Voice")
                                    continue
                            else:
                                print("\nNo audio detected. Please try again.")
                                continue
                    finally:
                        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

                    if mode == 't':
                        print("📝 Mode Tekstual dipilih. Ketik 'exit' untuk keluar.")
                        self.is_mode_selected = True
                        while True:
                            prompt = input("\nPertanyaan: ")
                            if prompt.lower() == 'exit':
                                print("Keluar. Sampai jumpa!")
                                break

                            # Check for robot command in text mode
                            if self.check_robot_command(prompt):
                                continue

                            self.process(prompt)
                    else:
                        print(f"\n🎤 Mode Suara dipilih. Katakan wake word ('{self.wake_word}') untuk mulai.")
                        self.is_mode_selected = True
                        while True:
                            try:
                                print("\n🎧 Listening...")
                                self.is_recording = True
                                audio = self.record_audio(duration=3)
                                self.is_recording = False

                                if audio is None:
                                    continue

                                # Save audio to file
                                audio_path = "temp_audio.wav"
                                with open(audio_path, "wb") as f:
                                    f.write(audio.getvalue())

                                # Transcribe audio
                                text = self.transcribe(audio_path)
                                if not text:
                                    continue

                                # Check for robot command first
                                if self.check_robot_command(text):
                                    continue

                                # Then check for wake word
                                if self.wake_word.lower() in text.lower():
                                    print(f"\n🔔 Wake word '{self.wake_word}' detected!")
                                    self.play_wake_word_detected(self.detect_language(text))

                                    # Record until silence is detected
                                    audio_data = self.record_until_silence()
                                    if audio_data is not None:
                                        sf.write("temp_query.wav", audio_data, self.sample_rate)
                                        print("\n🔄 Transcribing your question...")

                                        # Transcribe with language detection
                                        prompt = self.transcribe("temp_query.wav", detect_language=True)
                                        if prompt:
                                            self.process(prompt)
                                        else:
                                            print("❌ No question detected")
                                    else:
                                        print("❌ No audio recorded")
                                else:
                                    print(f"⏳ Wake word not detected. Keep trying...")

                            except KeyboardInterrupt:
                                break
                            except Exception as e:
                                print(f"⚠️ Error in listen loop: {str(e)}")
                                continue

                except KeyboardInterrupt:
                    break
                except Exception as e:
                    print(f"⚠️ Error in mode selection: {str(e)}")

        except Exception as e:
            print(f"⚠️ Error in listen loop: {str(e)}")

    def check_robot_command(self, text):
        """Check if text contains a robot command and execute it if found"""
        text_lower = text.lower()

        # Check for robot wake word
        is_robot_command = any(word in text_lower for word in self.robot_wake_words)

        if is_robot_command:
            print("🤖 Robot command detected!")

            # Check for stop command first
            if any(word in text_lower for word in self.robot_commands["stop"]):
                print("🛑 Executing stop command")
                self.send_mqtt_command("stop")
                return True

            # Check for start command
            if any(word in text_lower for word in self.robot_commands["start"]):
                print("▶️ Executing start command")
                self.send_mqtt_command("start")
                return True

            # Check for movement commands
            if any(word in text_lower for word in self.robot_commands["left"]):
                print("⬅️ Executing left command")
                self.send_mqtt_command("left")
                return True

            if any(word in text_lower for word in self.robot_commands["right"]):
                print("➡️ Executing right command")
                self.send_mqtt_command("right")
                return True

            if any(word in text_lower for word in self.robot_commands["forward"]):
                print("⬆️ Executing forward command")
                self.send_mqtt_command("forward")
                return True

            if any(word in text_lower for word in self.robot_commands["backward"]):
                print("⬇️ Executing backward command")
                self.send_mqtt_command("backward")
                return True

            print("❌ Unknown robot command")
            return True

        return False

    def string_similarity(self, a, b):
        """Calculate similarity ratio between two strings"""
        a = a.lower()
        b = b.lower()

        # Direct match
        if a == b:
            return 1.0

        # Check if one string contains the other
        if a in b or b in a:
            return 0.9

        # Calculate Levenshtein distance
        if len(a) < len(b):
            a, b = b, a

        if len(b) == 0:
            return 0.0

        distances = range(len(b) + 1)
        for i, ca in enumerate(a):
            distances_ = [i + 1]
            for j, cb in enumerate(b):
                if ca == cb:
                    distances_.append(distances[j])
                else:
                    distances_.append(1 + min((distances[j], distances[j + 1], distances_[-1])))
            distances = distances_

        # Convert distance to similarity ratio
        max_len = max(len(a), len(b))
        similarity = 1 - (distances[-1] / max_len)

        return similarity

    def resample_audio(self, audio, orig_sr, target_sr):
        """Resample audio to target sample rate"""
        if orig_sr == target_sr:
            return audio

        duration = len(audio) / orig_sr
        target_length = int(duration * target_sr)
        resampled = np.interp(
            np.linspace(0, len(audio), target_length, endpoint=False),
            np.arange(len(audio)),
            audio.flatten()
        )
        return resampled

    def generate_and_play(self, text, lang=None):
        """Generate and play TTS with improved performance"""
        if lang is None:
            lang = self.detect_language(text)
        print(f"[!] Generating speech in {lang}")

        try:
            # Process text in smaller chunks for better performance
            max_chunk_length = 100  # Maximum characters per chunk
            if len(text) > max_chunk_length:
                # Split into sentences first
                sentences = re.split(r'([。！？，.!?,])', text)
                chunks = []
                current_chunk = ""

                for i in range(0, len(sentences), 2):
                    sentence = sentences[i]
                    punct = sentences[i + 1] if i + 1 < len(sentences) else ""
                    if len(current_chunk) + len(sentence) + 1 <= max_chunk_length:
                        current_chunk += sentence + punct
                    else:
                        if current_chunk:
                            chunks.append(current_chunk)
                        current_chunk = sentence + punct

                if current_chunk:
                    chunks.append(current_chunk)

                # Process each chunk
                wav_parts = []
                sr = None
                for chunk in chunks:
                    part_wav, part_sr = self.synthesize_to_array(chunk, lang)
                    wav_parts.append(part_wav)
                    if sr is None:
                        sr = part_sr

                wav = np.concatenate(wav_parts)
            else:
                wav, sr = self.synthesize_to_array(text, lang)

            sd.play(wav, sr)
            sd.wait()
            return wav, sr

        except Exception as e:
            print(f"TTS error: {str(e)}")
            # Fallback to English TTS
            if lang != 'en':
                print("Falling back to English TTS...")
                return self.generate_and_play(text, 'en')
            raise

    def send_mqtt_command(self, command):
        """Send command to MQTT broker"""
        try:
            message = {
                "command": command
            }
            self.mqtt_client.publish(self.TOPIC, json.dumps(message))
            print(f"📤 Sent MQTT command: {command}")
        except Exception as e:
            print(f"❌ Error sending MQTT command: {str(e)}")

    def __del__(self):
        """Cleanup when object is destroyed"""
        try:
            if hasattr(self, 'mqtt_client'):
                self.mqtt_client.loop_stop()
                self.mqtt_client.disconnect()
                print("✅ MQTT connection closed")
        except Exception as e:
            print(f"❌ Error closing MQTT connection: {str(e)}")

    def record_until_silence(self):
        """Record audio until silence is detected"""
        print("\n" + "="*50)
        print("🎤 Recording... (speak your question)")

        q = queue.Queue()
        recording = []
        recording_start = time.time()
        is_recording = True
        has_speech = False
        stream = None
        silence_counter = 0
        required_silence_chunks = int(0.5 * self.sample_rate / 1024)  # 0.5 seconds of silence

        def audio_callback(indata, frames, time, status):
            if status:
                if status.input_overflow:
                    pass
                else:
                    print(f"⚠️ Audio status: {status}")
            # Always convert to mono
            mono_data = np.mean(indata, axis=1, keepdims=True) if indata.shape[1] > 1 else indata
            q.put(mono_data.copy())

        def process_audio():
            nonlocal has_speech
            while is_recording:
                try:
                    data = q.get(timeout=0.1)
                    recording.append(data)
                    if not has_speech and not self.is_silent(data):
                        has_speech = True
                except queue.Empty:
                    continue

        # Start processing thread
        processing_thread = threading.Thread(target=process_audio)
        processing_thread.start()

        try:
            # Create test stream first
            test_stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,  # Always use mono
                device=self.input_device,
                latency='high'
            )
            test_stream.start()
            test_stream.stop()
            test_stream.close()

            stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,  # Always use mono
                callback=audio_callback,
                blocksize=1024,
                device=self.input_device,
                latency='high'  # Use high latency for better stability
            )

            with stream:
                while True:
                    current_time = time.time()
                    duration = current_time - recording_start

                    # Check maximum duration
                    if duration >= self.max_phrase_duration:
                        print("\n⏰ Maximum recording duration reached")
                        break

                    # Get latest audio and check for silence
                    try:
                        data = q.get(timeout=0.1)
                        if self.is_silent(data):
                            silence_counter += 1
                            if silence_counter >= required_silence_chunks:
                                if has_speech and duration >= self.min_phrase_duration:
                                    print("\n🔇 Natural silence detected, stopping recording")
                                    break
                        else:
                            silence_counter = 0

                        # Visual indicator of audio level (use first channel if multi-channel)
                        energy = np.sqrt(np.mean(data[:, 0]**2) if self.channels > 1 else np.mean(data**2))
                        bars = min(40, int(energy / self.silence_threshold * 20))
                        print(f"\r⏱️ Recording: {duration:.1f}s [{'|' * bars}{' ' * (40-bars)}]", end="")

                    except queue.Empty:
                        continue

        except Exception as e:
            print(f"\n⚠️ Recording error: {str(e)}")
            return None
        finally:
            # Clean up resources
            is_recording = False

            # Clear the queue
            while not q.empty():
                try:
                    q.get_nowait()
                except queue.Empty:
                    break

            # Stop the processing thread
            if processing_thread.is_alive():
                processing_thread.join(timeout=1.0)

            # Close the stream if it's still open
            if stream is not None and stream.active:
                stream.stop()
                stream.close()

        # Combine all recorded chunks
        if recording and has_speech:
            try:
                full_recording = np.concatenate(recording)
                # Convert to mono if multi-channel
                if self.channels > 1:
                    full_recording = np.mean(full_recording, axis=1)
                print("\n✅ Recording completed successfully")
                print("="*50 + "\n")
                return full_recording
            except Exception as e:
                print(f"\n⚠️ Error processing recording: {str(e)}")
                return None

        print("\n❌ No speech detected in recording")
        print("="*50 + "\n")
        return None


if __name__ == "__main__":
    app = OllamaMultilingualTTS(
        ollama_host="codex.petra.ac.id:11434",
        rag_host="codex.petra.ac.id:50002",
        wake_word="alexa",
        gpu=True,
        model_size="small"
    )

    # Test MQTT first

    # Then start the main loop
    app.listen_loop()
