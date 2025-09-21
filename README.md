# Open Source AI Models Collection

A comprehensive catalog of popular open-source artificial intelligence models organized by category and use case.

## 📋 Table of Contents

- [Image Generation Models](#-image-generation-models)
- [Video Generation Models](#-video-generation-models)
- [Text-to-Speech (TTS) Models](#-text-to-speech-tts-models)
- [Speech-to-Text (STT) Models](#-speech-to-text-stt-models)
- [Large Language Models (LLMs)](#-large-language-models-llms)
- [Computer Vision Models](#-computer-vision-models)
- [Audio Processing Models](#-audio-processing-models)
- [Multimodal Models](#-multimodal-models)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎨 Image Generation Models

### Stable Diffusion Family
- **Stable Diffusion XL (SDXL)** - High-resolution image generation with improved quality
  - Repository: [stabilityai/stable-diffusion-xl-base-1.0](https://github.com/Stability-AI/stablediffusion)
  - Hugging Face: [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
  - License: CreativeML Open RAIL++-M
  - Resolution: 1024×1024
  - Use Cases: Art generation, concept art, photo-realistic images

- **Stable Diffusion v2.1** - Improved version with better prompt following
  - Repository: [stabilityai/stable-diffusion-2-1](https://github.com/Stability-AI/stablediffusion)
  - Hugging Face: [stabilityai/stable-diffusion-2-1](https://huggingface.co/stabilityai/stable-diffusion-2-1)
  - License: CreativeML Open RAIL++-M
  - Resolution: 512×512, 768×768
  - Use Cases: General image generation, artistic creation

- **Stable Diffusion v1.5** - Widely adopted baseline model
  - Repository: [runwayml/stable-diffusion-v1-5](https://github.com/runwayml/stable-diffusion)
  - Hugging Face: [runwayml/stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5)
  - License: CreativeML Open RAIL-M
  - Resolution: 512×512
  - Use Cases: Research, fine-tuning base

- **Stable Diffusion 3** - Latest version with improved architecture
  - Repository: [stabilityai/stable-diffusion-3-medium](https://github.com/Stability-AI/sd3-ref)
  - Hugging Face: [stabilityai/stable-diffusion-3-medium-diffusers](https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers)
  - License: Stability AI Community License
  - Resolution: Up to 1024×1024
  - Use Cases: High-quality image generation, commercial use

### FLUX Models
- **FLUX.1-dev** - Advanced diffusion model with superior prompt adherence
  - Repository: [black-forest-labs/flux](https://github.com/black-forest-labs/flux)
  - Hugging Face: [black-forest-labs/FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev)
  - License: FLUX.1-dev Non-Commercial License
  - Resolution: Up to 2048×2048
  - Use Cases: High-quality artistic generation, detailed scenes

- **FLUX.1-schnell** - Fast inference version of FLUX
  - Repository: [black-forest-labs/flux](https://github.com/black-forest-labs/flux)
  - Hugging Face: [black-forest-labs/FLUX.1-schnell](https://huggingface.co/black-forest-labs/FLUX.1-schnell)
  - License: Apache 2.0
  - Speed: 1-4 steps inference
  - Use Cases: Rapid prototyping, real-time applications

### Open Source Alternatives
- **DALL-E Mini / Craiyon** - Lightweight alternative to DALL-E
  - Repository: [borisdayma/dalle-mini](https://github.com/borisdayma/dalle-mini)
  - Hugging Face: [dalle-mini/dalle-mini](https://huggingface.co/dalle-mini/dalle-mini)
  - License: Apache 2.0
  - Use Cases: Quick image generation, educational purposes

- **DeepFloyd IF** - Multi-stage diffusion model with text rendering capabilities
  - Repository: [deep-floyd/IF](https://github.com/deep-floyd/IF)
  - Hugging Face: [DeepFloyd/IF-I-XL-v1.0](https://huggingface.co/DeepFloyd/IF-I-XL-v1.0)
  - License: DeepFloyd IF License
  - Use Cases: Text-in-image generation, high-quality outputs

- **Kandinsky 2.1/2.2** - Russian-developed diffusion model
  - Repository: [ai-forever/Kandinsky-2](https://github.com/ai-forever/Kandinsky-2)
  - Hugging Face: [kandinsky-community/kandinsky-2-2-decoder](https://huggingface.co/kandinsky-community/kandinsky-2-2-decoder)
  - License: Apache 2.0
  - Use Cases: Multilingual text-to-image, artistic generation

### Specialized Style Models
- **Waifu Diffusion** - Anime/manga style image generation
  - Repository: [harubaru/waifu-diffusion](https://github.com/harubaru/waifu-diffusion)
  - Hugging Face: [hakurei/waifu-diffusion](https://huggingface.co/hakurei/waifu-diffusion)
  - License: CreativeML Open RAIL-M
  - Style: Anime, manga artwork
  - Use Cases: Anime character generation, illustration

- **Anything V4/V5** - Versatile anime art generation
  - Hugging Face: [andite/anything-v4.0](https://huggingface.co/andite/anything-v4.0)
  - Civitai: [Anything V5](https://civitai.com/models/9409/anything-or-v5)
  - License: CreativeML Open RAIL-M
  - Style: Anime, versatile styles
  - Use Cases: Character design, anime artwork

- **Realistic Vision** - Photorealistic image generation
  - Hugging Face: [SG161222/Realistic_Vision_V4.0](https://huggingface.co/SG161222/Realistic_Vision_V4.0)
  - Civitai: [Realistic Vision V5.1](https://civitai.com/models/4201/realistic-vision-v60-b1)
  - License: CreativeML Open RAIL-M
  - Style: Photorealistic
  - Use Cases: Portrait generation, realistic scenes

- **DreamShaper** - Balanced artistic and realistic generation
  - Hugging Face: [Lykon/DreamShaper](https://huggingface.co/Lykon/DreamShaper)
  - Civitai: [DreamShaper](https://civitai.com/models/4384/dreamshaper)
  - License: CreativeML Open RAIL-M
  - Style: Balanced artistic/realistic
  - Use Cases: General purpose, versatile generation

### ControlNet Models
- **ControlNet** - Spatial control for Stable Diffusion
  - Repository: [lllyasviel/ControlNet](https://github.com/lllyasviel/ControlNet)
  - Hugging Face: [lllyasviel/sd-controlnet-canny](https://huggingface.co/lllyasviel/sd-controlnet-canny)
  - License: Apache 2.0
  - Controls: Canny, depth, pose, segmentation
  - Use Cases: Precise image control, pose-guided generation

- **T2I-Adapter** - Lightweight control for text-to-image
  - Repository: [TencentARC/T2I-Adapter](https://github.com/TencentARC/T2I-Adapter)
  - Hugging Face: [TencentARC/t2iadapter_canny_sd14v1](https://huggingface.co/TencentARC/t2iadapter_canny_sd14v1)
  - License: Apache 2.0
  - Use Cases: Efficient conditioning, plug-and-play control

### Image Upscaling Models
- **Real-ESRGAN** - Practical super-resolution for real-world images
  - Repository: [xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)
  - License: BSD-3-Clause
  - Upscale: 4x, 2x variants
  - Use Cases: Photo enhancement, anime upscaling

- **ESRGAN** - Enhanced Super-Resolution GAN
  - Repository: [xinntao/ESRGAN](https://github.com/xinntao/ESRGAN)
  - License: Apache 2.0
  - Upscale: 4x super-resolution
  - Use Cases: Image enhancement, texture improvement

- **SwinIR** - Transformer-based image restoration
  - Repository: [JingyunLiang/SwinIR](https://github.com/JingyunLiang/SwinIR)
  - License: Apache 2.0
  - Tasks: Super-resolution, denoising, JPEG compression artifact reduction
  - Use Cases: Image restoration, quality enhancement

---

## 🎬 Video Generation Models

### Text-to-Video
- **ModelScope Text-to-Video** - Chinese tech company's video generation model
  - Repository: [damo-vilab/text-to-video-ms-1.7b](https://github.com/damo-vilab/text-to-video-ms-1.7b)
  - Hugging Face: [damo-vilab/text-to-video-ms-1.7b](https://huggingface.co/damo-vilab/text-to-video-ms-1.7b)
  - License: Apache 2.0
  - Duration: 2-16 seconds
  - Resolution: 256×256
  - Use Cases: Short video clips, animations

- **Zeroscope** - High-quality text-to-video generation
  - Repository: [cerspense/zeroscope_v2_576w](https://github.com/ExponentialML/Text-To-Video-Finetuning)
  - Hugging Face: [cerspense/zeroscope_v2_576w](https://huggingface.co/cerspense/zeroscope_v2_576w)
  - License: CreativeML Open RAIL-M
  - Resolution: 576×320
  - Use Cases: Video content creation, storytelling

- **Text2Video-Zero** - Zero-shot video generation from text
  - Repository: [Picsart-AI-Research/Text2Video-Zero](https://github.com/Picsart-AI-Research/Text2Video-Zero)
  - License: Apache 2.0
  - Use Cases: Research, proof of concept

- **LaVie** - High-quality video generation with 3D awareness
  - Repository: [Vchitect/LaVie](https://github.com/Vchitect/LaVie)
  - License: Apache 2.0
  - Resolution: 512×320
  - Duration: 16 frames
  - Use Cases: Research, video synthesis

- **VideoLDM** - Latent Diffusion Models for Video Generation
  - Repository: [SHI-Labs/VidEdit](https://github.com/SHI-Labs/VidEdit)
  - License: Apache 2.0
  - Use Cases: Video editing, generation

### Image-to-Video
- **Stable Video Diffusion** - Stability AI's video generation model
  - Repository: [Stability-AI/generative-models](https://github.com/Stability-AI/generative-models)
  - Hugging Face: [stabilityai/stable-video-diffusion-img2vid](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid)
  - License: Stability AI Non-Commercial Research Community License
  - Input: Single image
  - Output: Short video sequences

- **AnimateDiff** - Animate still images using motion modules
  - Repository: [guoyww/AnimateDiff](https://github.com/guoyww/AnimateDiff)
  - License: Apache 2.0
  - Use Cases: Animation creation, motion synthesis

- **DynamiCrafter** - Animating open-domain images with video diffusion priors
  - Repository: [Doubiiu/DynamiCrafter](https://github.com/Doubiiu/DynamiCrafter)
  - License: Apache 2.0
  - Quality: 1024×576 resolution
  - Use Cases: High-quality image animation

- **I2VGen-XL** - High-quality image-to-video synthesis
  - Repository: [ali-vilab/i2vgen-xl](https://github.com/ali-vilab/i2vgen-xl)
  - License: Apache 2.0
  - Resolution: 1280×720
  - Use Cases: Commercial video production

### Video Editing and Processing
- **RIFE** - Real-Time Intermediate Flow Estimation for Video Interpolation
  - Repository: [megvii-research/ECCV2022-RIFE](https://github.com/megvii-research/ECCV2022-RIFE)
  - License: Apache 2.0
  - Use Cases: Frame interpolation, slow-motion effects

- **Real-Time-Video-Enhancement** - Video super-resolution and enhancement
  - Repository: [xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)
  - License: BSD-3-Clause
  - Use Cases: Video upscaling, quality improvement

- **CoDeF** - Content Deformation Fields for video editing
  - Repository: [qiuyu96/CoDeF](https://github.com/qiuyu96/CoDeF)
  - License: Apache 2.0
  - Use Cases: Video editing, object manipulation

- **TokenFlow** - Consistent diffusion features for consistent video editing
  - Repository: [omerbt/TokenFlow](https://github.com/omerbt/TokenFlow)
  - License: Apache 2.0
  - Use Cases: Consistent video style transfer

### Specialized Video Models
- **Make-A-Video** - PyTorch implementation
  - Repository: [lucidrains/make-a-video-pytorch](https://github.com/lucidrains/make-a-video-pytorch)
  - License: MIT
  - Use Cases: Research implementation

- **LVDM** - Latent Video Diffusion Models
  - Repository: [YingqingHe/LVDM](https://github.com/YingqingHe/LVDM)
  - License: Apache 2.0
  - Resolution: 256×256, 512×320
  - Use Cases: Long video generation

- **Tune-A-Video** - One-shot video tuning
  - Repository: [showlab/Tune-A-Video](https://github.com/showlab/Tune-A-Video)
  - License: Apache 2.0
  - Use Cases: Personalized video generation

- **Video-ChatGPT** - Detailed video understanding and generation
  - Repository: [mbzuai-oryx/Video-ChatGPT](https://github.com/mbzuai-oryx/Video-ChatGPT)
  - License: Apache 2.0
  - Use Cases: Video analysis, question answering

---

## 🗣️ Text-to-Speech (TTS) Models

### Neural TTS Models
- **Coqui TTS** - Deep learning toolkit for TTS
  - Repository: [coqui-ai/TTS](https://github.com/coqui-ai/TTS)
  - License: MPL 2.0
  - Languages: 100+ languages
  - Features: Voice cloning, multi-speaker, real-time synthesis

- **Tortoise TTS** - High-quality, slow TTS with voice cloning
  - Repository: [neonbjb/tortoise-tts](https://github.com/neonbjb/tortoise-tts)
  - License: Apache 2.0
  - Quality: Very high
  - Speed: Slow (high-quality mode)
  - Use Cases: Audiobooks, voice cloning

- **BARK** - Transformer-based TTS with non-verbal sounds
  - Repository: [suno-ai/bark](https://github.com/suno-ai/bark)
  - Hugging Face: [suno/bark](https://huggingface.co/suno/bark)
  - License: MIT (Non-commercial)
  - Features: Music, sound effects, emotions
  - Languages: Multiple supported

- **XTTS v2** - Coqui's multilingual voice cloning
  - Repository: [coqui-ai/TTS](https://github.com/coqui-ai/TTS)
  - Hugging Face: [coqui/XTTS-v2](https://huggingface.co/coqui/XTTS-v2)
  - License: Coqui Public Model License
  - Languages: 17 languages
  - Features: Few-shot voice cloning, cross-lingual synthesis

### Fast TTS Models
- **FastSpeech2** - Non-autoregressive neural vocoder
  - Repository: [ming024/FastSpeech2](https://github.com/ming024/FastSpeech2)
  - License: MIT
  - Speed: Real-time
  - Quality: High
  - Use Cases: Real-time applications

- **VITS** - Conditional Variational Autoencoder with Adversarial Learning
  - Repository: [jaywalnut310/vits](https://github.com/jaywalnut310/vits)
  - License: MIT
  - Features: End-to-end, single-stage training
  - Quality: High with fast inference

- **Tacotron 2** - Neural network architecture for speech synthesis
  - Repository: [NVIDIA/tacotron2](https://github.com/NVIDIA/tacotron2)
  - License: BSD-3-Clause
  - Quality: High natural speech
  - Use Cases: Research, baseline implementation

- **Mozilla TTS** - Deep learning based TTS by Mozilla
  - Repository: [mozilla/TTS](https://github.com/mozilla/TTS)
  - License: MPL 2.0
  - Features: Multi-speaker, multi-language
  - Use Cases: Open-source TTS development

### Voice Cloning and Conversion
- **Real-Time-Voice-Cloning** - Clone voices with minimal data
  - Repository: [CorentinJ/Real-Time-Voice-Cloning](https://github.com/CorentinJ/Real-Time-Voice-Cloning)
  - License: MIT
  - Requirements: 5-10 seconds of audio
  - Use Cases: Voice synthesis, personalization

- **So-VITS-SVC** - Singing voice conversion
  - Repository: [svc-develop-team/so-vits-svc](https://github.com/svc-develop-team/so-vits-svc)
  - License: AGPL-3.0
  - Use Cases: Singing voice synthesis, voice conversion

- **FreeVC** - High-quality voice conversion
  - Repository: [OlaWod/FreeVC](https://github.com/OlaWod/FreeVC)
  - License: MIT
  - Features: Any-to-any voice conversion
  - Use Cases: Voice conversion, speaker adaptation

- **RVC (Retrieval-based Voice Conversion)** - Easy voice conversion
  - Repository: [RVC-Project/Retrieval-based-Voice-Conversion-WebUI](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)
  - License: MIT
  - Features: Web UI, easy training
  - Use Cases: Voice conversion, content creation

### Multilingual TTS
- **MMS (Massively Multilingual Speech)** - Meta's multilingual model
  - Repository: [facebookresearch/fairseq](https://github.com/facebookresearch/fairseq/tree/main/examples/mms)
  - Hugging Face: [facebook/mms-tts](https://huggingface.co/facebook/mms-tts)
  - License: CC-BY-NC 4.0
  - Languages: 1100+ languages
  - Use Cases: Low-resource language synthesis

- **YourTTS** - Zero-shot multi-speaker TTS
  - Repository: [Edresson/YourTTS](https://github.com/Edresson/YourTTS)
  - License: MPL 2.0
  - Features: Zero-shot voice cloning, multilingual
  - Languages: English, Portuguese, French

- **NaturalSpeech** - Human-level quality TTS
  - Repository: [microsoft/NaturalSpeech](https://github.com/microsoft/NaturalSpeech)
  - License: MIT
  - Quality: Near human-level
  - Use Cases: High-quality speech synthesis

### Specialized TTS Models
- **EmotiVoice** - Multi-voice and prompt-controlled TTS
  - Repository: [netease-youdao/EmotiVoice](https://github.com/netease-youdao/EmotiVoice)
  - License: Apache 2.0
  - Features: Emotion control, multi-voice
  - Languages: English, Chinese

- **Fish Speech** - Few-shot voice cloning TTS
  - Repository: [fishaudio/fish-speech](https://github.com/fishaudio/fish-speech)
  - License: BSD-3-Clause
  - Features: Few-shot learning, voice cloning
  - Use Cases: Rapid voice adaptation

- **OpenVoice** - Instant voice cloning
  - Repository: [myshell-ai/OpenVoice](https://github.com/myshell-ai/OpenVoice)
  - License: MIT
  - Features: Cross-lingual voice cloning
  - Languages: Multiple supported

- **MeloTTS** - High-quality multi-language TTS
  - Repository: [myshell-ai/MeloTTS](https://github.com/myshell-ai/MeloTTS)
  - License: MIT
  - Features: Real-time synthesis, accent control
  - Languages: English, Chinese, Japanese, Korean

### Voice Conversion Specific
- **kNN-VC** - k-nearest neighbors voice conversion
  - Repository: [bshall/knn-vc](https://github.com/bshall/knn-vc)
  - License: MIT
  - Features: Non-parallel voice conversion
  - Use Cases: Voice style transfer

- **QuickVC** - Real-time voice conversion
  - Repository: [quickvc/QuickVC-VoiceConversion](https://github.com/quickvc/QuickVC-VoiceConversion)
  - License: Apache 2.0
  - Speed: Real-time conversion
  - Use Cases: Live voice modulation

---

## 🎤 Speech-to-Text (STT) Models

### General Purpose STT
- **OpenAI Whisper** - Robust speech recognition across languages
  - Repository: `openai/whisper`
  - License: MIT
  - Languages: 99 languages
  - Models: tiny, base, small, medium, large
  - Use Cases: Transcription, translation, language identification

- **wav2vec2** - Facebook's self-supervised speech representation
  - Repository: `pytorch/fairseq`
  - License: MIT
  - Languages: Multiple
  - Use Cases: Speech recognition, representation learning

- **SpeechT5** - Microsoft's unified speech and text model
  - Repository: `microsoft/SpeechT5`
  - License: MIT
  - Capabilities: STT, TTS, speech enhancement
  - Use Cases: Multi-task speech processing

### Specialized STT
- **Vosk** - Offline speech recognition
  - Repository: `alphacep/vosk-api`
  - License: Apache 2.0
  - Features: Offline, lightweight
  - Languages: 20+ languages
  - Use Cases: Edge computing, privacy-focused applications

- **DeepSpeech** - Mozilla's STT engine
  - Repository: `mozilla/DeepSpeech`
  - License: MPL 2.0
  - Languages: English, others available
  - Use Cases: Open-source alternative to commercial STT

---

## 🧠 Large Language Models (LLMs)

### Foundation Models
- **LLaMA 2** - Meta's large language model family
  - Repository: `facebookresearch/llama`
  - License: Custom (Commercial use allowed)
  - Sizes: 7B, 13B, 70B parameters
  - Use Cases: General text generation, chat, reasoning

- **Mistral 7B** - Efficient 7B parameter model
  - Repository: `mistralai/mistral-7b`
  - License: Apache 2.0
  - Performance: Competitive with larger models
  - Use Cases: Efficient deployment, fine-tuning

- **Code Llama** - Code-specialized version of LLaMA 2
  - Repository: `facebookresearch/codellama`
  - License: Custom
  - Sizes: 7B, 13B, 34B
  - Use Cases: Code generation, programming assistance

### Instruction-Tuned Models
- **Alpaca** - Stanford's instruction-following model
  - Repository: `tatsu-lab/stanford_alpaca`
  - License: Apache 2.0
  - Base: LLaMA
  - Use Cases: Instruction following, research

- **Vicuna** - Chatbot trained with user-shared conversations
  - Repository: `lm-sys/FastChat`
  - License: Apache 2.0
  - Performance: GPT-4 level on many tasks
  - Use Cases: Conversational AI, research

### Specialized LLMs
- **WizardCoder** - Code generation specialist
  - Repository: `nlpxucan/WizardLM`
  - License: Apache 2.0
  - Focus: Programming tasks
  - Use Cases: Code completion, generation

- **MedLLaMA** - Medical domain adaptation
  - Repository: Various medical fine-tunes
  - Focus: Healthcare applications
  - Use Cases: Medical Q&A, clinical assistance

---

## 👁️ Computer Vision Models

### Object Detection
- **YOLO (You Only Look Once)** - Real-time object detection
  - **YOLOv8**: `ultralytics/ultralytics`
  - **YOLOv5**: `ultralytics/yolov5`
  - License: AGPL-3.0
  - Use Cases: Real-time detection, surveillance, autonomous vehicles

- **DETR** - Detection Transformer
  - Repository: `facebookresearch/detr`
  - License: Apache 2.0
  - Approach: Transformer-based
  - Use Cases: Object detection research

### Image Classification
- **ResNet** - Deep residual networks
  - Repository: `pytorch/vision`
  - License: BSD-3-Clause
  - Variants: ResNet-18, 34, 50, 101, 152
  - Use Cases: Image classification baseline

- **Vision Transformer (ViT)** - Transformer for images
  - Repository: `google-research/vision_transformer`
  - License: Apache 2.0
  - Approach: Pure transformer
  - Use Cases: Image classification, foundation models

### Segmentation
- **Segment Anything Model (SAM)** - Universal image segmentation
  - Repository: `facebookresearch/segment-anything`
  - License: Apache 2.0
  - Capability: Zero-shot segmentation
  - Use Cases: Image editing, annotation tools

- **U-Net** - Biomedical image segmentation
  - Repository: Various implementations
  - License: MIT (typical)
  - Domain: Medical imaging
  - Use Cases: Medical image analysis

---

## 🎵 Audio Processing Models

### Music Generation
- **MusicGen** - Meta's music generation model
  - Repository: `facebookresearch/audiocraft`
  - License: MIT
  - Capability: Text-to-music generation
  - Use Cases: Music creation, sound design

- **Jukebox** - OpenAI's music generation model
  - Repository: `openai/jukebox`
  - License: MIT
  - Features: Artist style, genre control
  - Use Cases: Music research, generation

### Audio Enhancement
- **NVIDIA NeMo** - Conversational AI toolkit
  - Repository: `NVIDIA/NeMo`
  - License: Apache 2.0
  - Modules: ASR, TTS, NLP
  - Use Cases: Enterprise AI applications

- **Facebook Denoiser** - Real-time speech denoising
  - Repository: `facebookresearch/denoiser`
  - License: MIT
  - Application: Speech enhancement
  - Use Cases: Video calls, podcasting

### Sound Classification
- **PANNs** - Pre-trained Audio Neural Networks
  - Repository: `qiuqiangkong/audioset_tagging_cnn`
  - License: MIT
  - Dataset: AudioSet
  - Use Cases: Audio event detection, classification

---

## 🔄 Multimodal Models

### Vision-Language Models
- **CLIP** - Contrastive Language-Image Pre-training
  - Repository: `openai/CLIP`
  - License: MIT
  - Capability: Image-text understanding
  - Use Cases: Zero-shot classification, image search

- **BLIP/BLIP-2** - Bootstrapped vision-language pre-training
  - Repository: `salesforce/BLIP`
  - License: BSD-3-Clause
  - Tasks: Image captioning, VQA
  - Use Cases: Visual question answering

### Large Vision-Language Models
- **LLaVA** - Large Language and Vision Assistant
  - Repository: `haotian-liu/LLaVA`
  - License: Apache 2.0
  - Capability: Visual instruction following
  - Use Cases: Visual chatbot, multimodal assistance

- **MiniGPT-4** - Enhancing vision-language understanding
  - Repository: `Vision-CAIR/MiniGPT-4`
  - License: BSD-3-Clause
  - Features: Advanced visual reasoning
  - Use Cases: Image understanding, captioning

---

## 🛠️ Model Deployment and Serving

### Inference Engines
- **Ollama** - Run large language models locally
  - Repository: `jmorganca/ollama`
  - License: MIT
  - Features: Easy local deployment
  - Supported: LLaMA, Mistral, others

- **TensorRT-LLM** - NVIDIA's LLM optimization
  - Repository: `NVIDIA/TensorRT-LLM`
  - License: Apache 2.0
  - Focus: GPU acceleration
  - Use Cases: Production deployment

### Model Hubs and Tools
- **Hugging Face Transformers** - Model library and hub
  - Repository: `huggingface/transformers`
  - License: Apache 2.0
  - Models: Thousands of pre-trained models
  - Use Cases: Model discovery, fine-tuning

- **LangChain** - Framework for LLM applications
  - Repository: `hwchase17/langchain`
  - License: MIT
  - Purpose: LLM application development
  - Use Cases: RAG, agents, chatbots

---

## 📊 Model Performance and Benchmarks

### Evaluation Frameworks
- **OpenAI Evals** - Framework for evaluating LLMs
  - Repository: `openai/evals`
  - License: MIT
  - Purpose: Model evaluation
  - Use Cases: Benchmarking, testing

- **EleutherAI LM Evaluation Harness** - Language model evaluation
  - Repository: `EleutherAI/lm-evaluation-harness`
  - License: MIT
  - Coverage: Multiple tasks and metrics
  - Use Cases: Standardized evaluation

### Benchmarks
- **GLUE/SuperGLUE** - General Language Understanding
- **ImageNet** - Image classification benchmark
- **COCO** - Object detection and segmentation
- **LibriSpeech** - Speech recognition benchmark
- **BLEU/ROUGE** - Text generation metrics

---

## 🚀 Getting Started

### Prerequisites
```bash
# Python environment
python >= 3.8
pip or conda package manager

# Common dependencies
torch
transformers
diffusers
```

### Quick Setup Examples
```bash
# Install Hugging Face transformers
pip install transformers torch

# Install Stable Diffusion
pip install diffusers transformers accelerate

# Install Whisper
pip install openai-whisper

# Install TTS
pip install TTS
```

---

## 📈 Model Categories by Use Case

### Content Creation
- **Text**: GPT models, LLaMA, Mistral
- **Images**: Stable Diffusion, DALL-E alternatives
- **Videos**: ModelScope, Stable Video Diffusion
- **Audio**: MusicGen, Bark, Coqui TTS

### Business Applications
- **Customer Service**: Fine-tuned LLMs, STT/TTS
- **Content Moderation**: CLIP, specialized classifiers
- **Document Processing**: OCR + LLMs, multimodal models
- **Analytics**: Computer vision + NLP models

### Research and Education
- **Multimodal Research**: CLIP, BLIP, LLaVA
- **Domain-Specific**: Medical, legal, scientific models
- **Benchmarking**: Evaluation frameworks and datasets

---

## 🤝 Contributing

We welcome contributions to expand and improve this catalog! Please consider:

1. **Adding New Models**: Submit models with proper documentation
2. **Updating Information**: Keep model details current
3. **Performance Data**: Add benchmark results where available
4. **Use Case Examples**: Provide practical applications
5. **Category Organization**: Suggest better categorization

### Contribution Guidelines
- Ensure models are truly open source
- Include license information
- Provide repository links
- Add brief use case descriptions
- Maintain consistent formatting

---

## 📝 License

This documentation is provided under the MIT License. Individual models listed have their own licenses - please check each model's repository for specific terms.

---

## 🔗 Additional Resources

- [Hugging Face Model Hub](https://huggingface.co/models)
- [Papers with Code](https://paperswithcode.com/)
- [AI Model Zoo Collections](https://modelzoo.co/)
- [OpenAI Research](https://openai.com/research/)
- [Google AI Research](https://ai.google/research/)
- [Meta AI Research](https://ai.facebook.com/research/)

---

*This is a living document that will be regularly updated as new models are released and the AI landscape evolves.*
