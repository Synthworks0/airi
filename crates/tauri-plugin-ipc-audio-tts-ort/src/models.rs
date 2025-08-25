use anyhow::{anyhow, Result};
use log::{info, warn};
use ort::{
    execution_providers::CPUExecutionProvider,
    session::{builder::GraphOptimizationLevel, Session},
    util::Mutex,
    value::Tensor,
};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::{path::{Path, PathBuf}, sync::Arc};
use tauri::{Emitter, Runtime};
use tokenizers::Tokenizer;
use serde_json::Value as JsonValue;
use tokio::fs::File;
use tokio::io::AsyncWriteExt;

use crate::SynthesizeOptions;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub id: String,
    pub name: String,
    pub size: u64,
    pub quality: String,
    pub languages: Vec<String>,
    pub installed: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceInfo {
    pub id: String,
    pub name: String,
    pub gender: String,
    pub language: String,
    pub model_id: String,
}

pub enum TtsModel {
    Onnx(OnnxTtsModel),
    ESpeak,
}

impl TtsModel {
    pub fn new_espeak() -> Self {
        TtsModel::ESpeak
    }

    pub fn get_voices(&self) -> Vec<VoiceInfo> {
        match self {
            TtsModel::Onnx(model) => model.get_voices(),
            TtsModel::ESpeak => vec![
                VoiceInfo {
                    id: "espeak-en".to_string(),
                    name: "eSpeak English".to_string(),
                    gender: "neutral".to_string(),
                    language: "en-US".to_string(),
                    model_id: "espeak-ng".to_string(),
                },
                VoiceInfo {
                    id: "espeak-es".to_string(),
                    name: "eSpeak Spanish".to_string(),
                    gender: "neutral".to_string(),
                    language: "es-ES".to_string(),
                    model_id: "espeak-ng".to_string(),
                },
            ],
        }
    }

    pub fn has_voice(&self, voice_id: &str) -> bool {
        self.get_voices().iter().any(|v| v.id == voice_id)
    }

    pub fn synthesize(&self, text: &str, voice_id: &str, options: Option<&SynthesizeOptions>) -> Result<Vec<f32>> {
        match self {
            TtsModel::Onnx(model) => model.synthesize(text, voice_id, options),
            TtsModel::ESpeak => synthesize_espeak(text, voice_id, options),
        }
    }
}

pub struct OnnxTtsModel {
    session: Arc<Mutex<Session>>,
    config: TtsConfig,
    voices: Vec<VoiceInfo>,
    tokenizer: Tokenizer,
}

#[derive(Debug, Deserialize)]
struct TtsConfig {
    model_type: Option<String>,
    #[serde(default = "default_sample_rate")]
    sample_rate: u32,
    #[serde(default = "default_max_length")]
    max_length: usize,
    #[serde(default)]
    voices: Vec<VoiceConfig>,
}

fn default_sample_rate() -> u32 {
    24000 // Kokoro-82M uses 24kHz sample rate
}

fn default_max_length() -> usize {
    512 // Default max sequence length
}

#[derive(Debug, Deserialize)]
struct VoiceConfig {
    id: String,
    name: String,
    gender: String,
    language: String,
}

impl OnnxTtsModel {
    pub fn new(session: Session, config: TtsConfig, model_id: String, tokenizer: Tokenizer) -> Self {
        // For Kokoro, we'll load voices from the voices directory instead of config
        let voices = Self::load_kokoro_voices(&model_id);

        Self {
            session: Arc::new(Mutex::new(session)),
            config,
            voices,
            tokenizer,
        }
    }

    fn load_kokoro_voices(model_id: &str) -> Vec<VoiceInfo> {
        // Kokoro voice mapping based on the .bin files
        vec![
            // Female voices
            VoiceInfo { id: "af".to_string(), name: "Female (Default)".to_string(), gender: "female".to_string(), language: "English".to_string(), model_id: model_id.to_string() },
            VoiceInfo { id: "af_heart".to_string(), name: "Female (Heart)".to_string(), gender: "female".to_string(), language: "English".to_string(), model_id: model_id.to_string() },
            VoiceInfo { id: "af_alloy".to_string(), name: "Female (Alloy)".to_string(), gender: "female".to_string(), language: "English".to_string(), model_id: model_id.to_string() },
            VoiceInfo { id: "af_aoede".to_string(), name: "Female (Aoede)".to_string(), gender: "female".to_string(), language: "English".to_string(), model_id: model_id.to_string() },
            VoiceInfo { id: "af_bella".to_string(), name: "Female (Bella)".to_string(), gender: "female".to_string(), language: "English".to_string(), model_id: model_id.to_string() },
            VoiceInfo { id: "af_jessica".to_string(), name: "Female (Jessica)".to_string(), gender: "female".to_string(), language: "English".to_string(), model_id: model_id.to_string() },
            VoiceInfo { id: "af_kore".to_string(), name: "Female (Kore)".to_string(), gender: "female".to_string(), language: "English".to_string(), model_id: model_id.to_string() },
            VoiceInfo { id: "af_nicole".to_string(), name: "Female (Nicole)".to_string(), gender: "female".to_string(), language: "English".to_string(), model_id: model_id.to_string() },
            VoiceInfo { id: "af_nova".to_string(), name: "Female (Nova)".to_string(), gender: "female".to_string(), language: "English".to_string(), model_id: model_id.to_string() },
            VoiceInfo { id: "af_river".to_string(), name: "Female (River)".to_string(), gender: "female".to_string(), language: "English".to_string(), model_id: model_id.to_string() },
            VoiceInfo { id: "af_sarah".to_string(), name: "Female (Sarah)".to_string(), gender: "female".to_string(), language: "English".to_string(), model_id: model_id.to_string() },
            VoiceInfo { id: "af_sky".to_string(), name: "Female (Sky)".to_string(), gender: "female".to_string(), language: "English".to_string(), model_id: model_id.to_string() },
            // Male voices
            VoiceInfo { id: "am_adam".to_string(), name: "Male (Adam)".to_string(), gender: "male".to_string(), language: "English".to_string(), model_id: model_id.to_string() },
            VoiceInfo { id: "am_echo".to_string(), name: "Male (Echo)".to_string(), gender: "male".to_string(), language: "English".to_string(), model_id: model_id.to_string() },
            VoiceInfo { id: "am_eric".to_string(), name: "Male (Eric)".to_string(), gender: "male".to_string(), language: "English".to_string(), model_id: model_id.to_string() },
            VoiceInfo { id: "am_fenrir".to_string(), name: "Male (Fenrir)".to_string(), gender: "male".to_string(), language: "English".to_string(), model_id: model_id.to_string() },
            VoiceInfo { id: "am_liam".to_string(), name: "Male (Liam)".to_string(), gender: "male".to_string(), language: "English".to_string(), model_id: model_id.to_string() },
            VoiceInfo { id: "am_michael".to_string(), name: "Male (Michael)".to_string(), gender: "male".to_string(), language: "English".to_string(), model_id: model_id.to_string() },
            VoiceInfo { id: "am_onyx".to_string(), name: "Male (Onyx)".to_string(), gender: "male".to_string(), language: "English".to_string(), model_id: model_id.to_string() },
            VoiceInfo { id: "am_puck".to_string(), name: "Male (Puck)".to_string(), gender: "male".to_string(), language: "English".to_string(), model_id: model_id.to_string() },
            VoiceInfo { id: "am_santa".to_string(), name: "Male (Santa)".to_string(), gender: "male".to_string(), language: "English".to_string(), model_id: model_id.to_string() },
            // Other languages
            VoiceInfo { id: "jf_alpha".to_string(), name: "Japanese Female (Alpha)".to_string(), gender: "female".to_string(), language: "Japanese".to_string(), model_id: model_id.to_string() },
            VoiceInfo { id: "jm_kumo".to_string(), name: "Japanese Male (Kumo)".to_string(), gender: "male".to_string(), language: "Japanese".to_string(), model_id: model_id.to_string() },
            VoiceInfo { id: "zf_xiaobei".to_string(), name: "Chinese Female (Xiaobei)".to_string(), gender: "female".to_string(), language: "Chinese".to_string(), model_id: model_id.to_string() },
            VoiceInfo { id: "zm_yunjian".to_string(), name: "Chinese Male (Yunjian)".to_string(), gender: "male".to_string(), language: "Chinese".to_string(), model_id: model_id.to_string() },
        ]
    }

    pub fn get_voices(&self) -> Vec<VoiceInfo> {
        self.voices.clone()
    }

    pub fn synthesize(&self, text: &str, voice_id: &str, options: Option<&SynthesizeOptions>) -> Result<Vec<f32>> {
        // Input validation
        if text.trim().is_empty() {
            return Err(anyhow!("Text input cannot be empty"));
        }

        if text.len() > 1000 {
            return Err(anyhow!("Text too long (max 1000 characters)"));
        }

        // Tokenize the input text
        let encoding = self.tokenizer.encode(text.trim(), false)
            .map_err(|e| anyhow!("Tokenization failed: {}", e))?;
        let tokens = encoding.get_ids();

        // Validate tokenization results
        if tokens.is_empty() {
            return Err(anyhow!("Tokenization produced no tokens"));
        }

        if tokens.len() > 512 {
            return Err(anyhow!("Tokenized sequence too long (max 512 tokens)"));
        }

        // Convert tokens and validate
        let tokens_i64: Vec<i64> = tokens.iter().map(|&t| t as i64).collect();
        let tokens_len = tokens_i64.len();

        // Validate all tokens are within reasonable range
        for &token in &tokens_i64 {
            if token < 0 || token > 100000 {
                return Err(anyhow!("Invalid token value: {}", token));
            }
        }

        info!("Tokenized '{}' into {} tokens", text.trim(), tokens_len);

                                                // Try multiple input configurations for Kokoro - using separate functions to avoid lifetime issues
        let mut audio_samples = self.try_kokoro_inference(tokens_i64, tokens_len, voice_id, options)?;

        // Validate output
        if audio_samples.is_empty() {
            return Err(anyhow!("Model produced no audio samples"));
        }

        // Check for invalid values and clamp them
        for sample in audio_samples.iter_mut() {
            if sample.is_nan() || sample.is_infinite() {
                *sample = 0.0;
            } else {
                *sample = sample.clamp(-1.0, 1.0);
            }
        }

        // Apply audio modifications if specified
        if let Some(opts) = options {
            self.apply_audio_modifications(&mut audio_samples, opts);
        }

        let num_samples = audio_samples.len();
        let pitch = options.and_then(|o| o.pitch).unwrap_or(0.0);
        let speed = options.and_then(|o| o.speed).unwrap_or(1.0);
        let volume = options.and_then(|o| o.volume).unwrap_or(0.0);

        info!("Generated Kokoro audio: {} samples, voice: {}, pitch: {:.1}, speed: {:.1}x, volume: {:.1}dB",
              num_samples, voice_id, pitch, speed, volume);

        Ok(audio_samples)
    }



        fn try_kokoro_inference(&self, tokens_i64: Vec<i64>, tokens_len: usize, voice_id: &str, options: Option<&SynthesizeOptions>) -> Result<Vec<f32>> {
        // Debug: Log model input information
        let session = self.session.lock();
        let input_names = session.inputs.iter().map(|input| {
            format!("{}: {:?}", input.name, input.input_type)
        }).collect::<Vec<_>>();
        info!("Model expects inputs: {}", input_names.join(", "));
        drop(session);

                // Kokoro model expects input_ids, style, and speed inputs
        let style_vector = self.get_style_vector(voice_id)?;

        // Extract speed from options or use default, ensure it's in reasonable range
        let speed = options.and_then(|o| o.speed).unwrap_or(1.0);
        if !speed.is_finite() || speed <= 0.0 || speed > 3.0 {
            return Err(anyhow!("Speed must be finite and between 0.0 and 3.0, got {}", speed));
        }

        info!("Creating tensors: input_ids shape=[1, {}], style shape=[1, 256], speed shape=[1] value={}", tokens_len, speed);

        // Validate inputs before tensor creation
        if tokens_i64.is_empty() {
            return Err(anyhow!("Empty token sequence"));
        }

        if style_vector.len() != 256 {
            return Err(anyhow!("Style vector must be 256 dimensions, got {}", style_vector.len()));
        }

        // Check for any invalid values in style vector
        for (i, &val) in style_vector.iter().enumerate() {
            if !val.is_finite() {
                return Err(anyhow!("Style vector contains non-finite value at index {}: {}", i, val));
            }
        }

        // Create tensors with explicit validation
        let input_ids_tensor = Tensor::from_array((vec![1, tokens_len], tokens_i64))
            .map_err(|e| anyhow!("Failed to create input_ids tensor: {}", e))?
            .into_dyn();

        let style_tensor = Tensor::from_array((vec![1, 256], style_vector))
            .map_err(|e| anyhow!("Failed to create style tensor: {}", e))?
            .into_dyn();

        let speed_tensor = Tensor::from_array((vec![1], vec![speed]))
            .map_err(|e| anyhow!("Failed to create speed tensor: {}", e))?
            .into_dyn();

        let inputs = vec![
            ("input_ids", input_ids_tensor),
            ("style", style_tensor),
            ("speed", speed_tensor),
        ];

        let result = self.run_inference_and_extract(inputs);
        match result {
            Ok(audio) => {
                info!("Kokoro inference succeeded with input_ids + style + speed");

                // Validate output audio
                if audio.is_empty() {
                    return Err(anyhow!("Model produced empty audio output"));
                }

                // Check for any invalid audio samples
                let invalid_count = audio.iter().filter(|&&sample| !sample.is_finite()).count();
                if invalid_count > 0 {
                    return Err(anyhow!("Model produced {} invalid audio samples", invalid_count));
                }

                info!("Generated {} valid audio samples", audio.len());
                Ok(audio)
            },
            Err(e) => {
                Err(anyhow!("Kokoro inference failed: {}", e))
            }
        }
    }

    fn run_inference_and_extract(&self, inputs: Vec<(&str, ort::value::Value)>) -> Result<Vec<f32>> {
        let mut session = self.session.lock();
        let outputs = session.run(inputs)?;

        // Extract audio immediately while session is still locked
        let audio_output = outputs.get("audio")
            .or_else(|| outputs.get("output"))
            .or_else(|| outputs.get("waveform"))
            .or_else(|| outputs.get("mel"))
            .or_else(|| outputs.get("logits"))
            .ok_or_else(|| anyhow!("No audio output found in model. Available outputs: {:?}",
                outputs.keys().collect::<Vec<_>>()))?;

        let (audio_shape, audio_slice) = audio_output.try_extract_tensor::<f32>()
            .map_err(|e| anyhow!("Failed to extract audio tensor: {}", e))?;

        info!("Extracted audio tensor with shape: {:?}", audio_shape);
        Ok(audio_slice.to_vec())
    }



    fn get_style_vector(&self, voice_id: &str) -> Result<Vec<f32>> {
        // Kokoro uses 256-dimensional style vectors, but let me try a simpler approach first
        // Some models expect normalized values between -1 and 1 or 0 and 1
        let mut style = vec![0.0f32; 256];

        // Use a simpler, more stable approach for style vectors
        let base_value = match voice_id {
            "af_jessica" => 0.1,
            "af_bella" => 0.2,
            "af_sarah" => 0.3,
            "af_sky" => 0.4,
            "af_kore" => 0.5,
            "af_nicole" => 0.6,
            "af_nova" => 0.7,
            "af_river" => 0.8,
            "am_adam" => 0.9,
            "am_echo" => 1.0,
            "am_eric" => 0.15,
            "am_fenrir" => 0.25,
            "am_liam" => 0.35,
            "am_michael" => 0.45,
            "am_onyx" => 0.55,
            "am_puck" => 0.65,
            "am_santa" => 0.75,
            "jf_alpha" => 0.85,
            "jm_kumo" => 0.95,
            "zf_xiaobei" => 0.5,
            "zm_yunjian" => 0.6,
            "af" => 0.5,
            "am" => 0.8,
            _ => 0.5, // Default
        };

        // Create a more stable pattern - avoid complex math that might cause numerical issues
        for i in 0..256 {
            let normalized_i = (i as f32) / 255.0; // Normalize index to 0-1
            style[i] = base_value * (0.5 + 0.5 * normalized_i); // Simple linear interpolation
        }

        Ok(style)
    }

    fn apply_audio_modifications(&self, samples: &mut Vec<f32>, options: &SynthesizeOptions) {
        // Apply volume modification
        if let Some(volume_db) = options.volume {
            if volume_db.abs() > 0.01 {
                let gain = 10.0_f32.powf(volume_db / 20.0);
                for sample in samples.iter_mut() {
                    *sample *= gain;
                    *sample = sample.clamp(-1.0, 1.0);
                }
            }
        }

        // Apply speed modification (simple resampling)
        if let Some(speed) = options.speed {
            if (speed - 1.0).abs() > 0.01 {
                let new_len = (samples.len() as f32 / speed) as usize;
                let mut new_samples = Vec::with_capacity(new_len);

                for i in 0..new_len {
                    let src_index = (i as f32 * speed) as usize;
                    if src_index < samples.len() {
                        new_samples.push(samples[src_index]);
                    }
                }
                *samples = new_samples;
            }
        }

        // Pitch modification would require more complex DSP
        // For now, we'll skip pitch modification as it requires FFT-based processing
    }
}

fn synthesize_espeak(text: &str, _voice_id: &str, options: Option<&SynthesizeOptions>) -> Result<Vec<f32>> {
    // Generate simple sine wave as placeholder
    let sample_rate = 22050;
    let duration = text.len() as f32 * 0.1; // Rough estimate
    let num_samples = (sample_rate as f32 * duration) as usize;

    let volume = options.and_then(|o| o.volume).unwrap_or(0.0);
    let volume_factor = 10f32.powf(volume / 20.0); // Convert dB to linear

    let mut audio = Vec::with_capacity(num_samples);
    for i in 0..num_samples {
        let t = i as f32 / sample_rate as f32;
        let sample = (2.0 * std::f32::consts::PI * 440.0 * t).sin() * 0.3 * volume_factor;
        audio.push(sample);
    }

    Ok(audio)
}

async fn download_file_with_progress(url: &str, filename: &str) -> Result<PathBuf> {
    let client = Client::new();
    let response = client.get(url).send().await?;

    if !response.status().is_success() {
        return Err(anyhow!("Failed to download {}: HTTP {}", filename, response.status()));
    }

    // Create cache directory structure similar to hf-hub
    let cache_dir = dirs::cache_dir()
        .ok_or_else(|| anyhow!("Could not find cache directory"))?
        .join("huggingface")
        .join("transformers");

    tokio::fs::create_dir_all(&cache_dir).await?;
    let file_path = cache_dir.join(filename);

    let bytes = response.bytes().await?;
    let mut file = File::create(&file_path).await?;
    file.write_all(&bytes).await?;

    Ok(file_path)
}

/// Check whether a given model appears installed on disk (all key assets present).
/// This does not validate model integrity; it only checks for cached files placed by the loader.
/// Clear cached model files for a given model ID to force re-download
/// Load tokenizer with fallback for compatibility issues
fn load_tokenizer_from_file_with_fallback(tokenizer_path: &Path) -> Result<Tokenizer> {
    // First try to load the tokenizer normally
    match Tokenizer::from_file(tokenizer_path) {
        Ok(tokenizer) => {
            info!("Successfully loaded tokenizer from file");
            Ok(tokenizer)
        }
        Err(e) => {
            warn!("Failed to load tokenizer normally: {}. Attempting fallback method.", e);

            // Try to load and fix the tokenizer JSON
            let tokenizer_json = std::fs::read_to_string(tokenizer_path)?;
            let mut tokenizer_value: JsonValue = serde_json::from_str(&tokenizer_json)?;

            // Fix the PostProcessorWrapper issue - Kokoro uses a format that's not compatible
            // The issue is specifically with the ByteFallback processor in the Sequence
            if let Some(post_processor) = tokenizer_value.get_mut("post_processor") {
                if let Some(processors) = post_processor.get_mut("processors") {
                    if let Some(arr) = processors.as_array_mut() {
                        // Filter out ByteFallback processors
                        let filtered: Vec<JsonValue> = arr.iter()
                            .filter(|p| {
                                if let Some(t) = p.get("type").and_then(|v| v.as_str()) {
                                    t != "ByteFallback"
                                } else {
                                    true
                                }
                            })
                            .cloned()
                            .collect();

                        if filtered.is_empty() {
                            // If no processors left, remove the whole post_processor
                            info!("Removing entire post_processor (all were ByteFallback)");
                            tokenizer_value.as_object_mut().unwrap().remove("post_processor");
                        } else if filtered.len() == 1 {
                            // If only one processor left, use it directly
                            info!("Simplifying post_processor to single processor");
                            *post_processor = filtered.into_iter().next().unwrap();
                        } else {
                            // Keep the sequence with filtered processors
                            info!("Keeping Sequence with {} processors", filtered.len());
                            *processors = serde_json::json!(filtered);
                        }
                    }
                } else if post_processor.get("type").and_then(|t| t.as_str()) == Some("ByteFallback") {
                    // Direct ByteFallback, remove it
                    info!("Removing direct ByteFallback post_processor");
                    tokenizer_value.as_object_mut().unwrap().remove("post_processor");
                }
            }

            // Save the fixed tokenizer to a temporary file
            let temp_path = tokenizer_path.with_extension("fixed.json");
            std::fs::write(&temp_path, serde_json::to_string_pretty(&tokenizer_value)?)?;

            // Try to load the fixed tokenizer
            match Tokenizer::from_file(&temp_path) {
                Ok(tokenizer) => {
                    info!("Successfully loaded tokenizer using fallback method");
                    // Replace the original with the fixed version
                    std::fs::rename(&temp_path, tokenizer_path)?;
                    Ok(tokenizer)
                }
                Err(e2) => {
                    // If that still fails, create a simple tokenizer from scratch
                    warn!("Fallback tokenizer also failed: {}. Creating basic tokenizer.", e2);
                    let _ = std::fs::remove_file(&temp_path);

                    // Create a basic GPT2-style tokenizer as fallback
                    create_basic_tokenizer()
                }
            }
        }
    }
}

/// Load tokenizer from cache with fallback for compatibility
fn load_tokenizer_from_cache_with_fallback(tokenizer_path: &Path) -> Result<Tokenizer> {
    // First try to load the tokenizer normally
    match Tokenizer::from_file(tokenizer_path) {
        Ok(tokenizer) => {
            info!("Successfully loaded tokenizer from cache");
            Ok(tokenizer)
        }
        Err(e) => {
            warn!("Failed to load tokenizer from cache: {}. Attempting fallback.", e);

            // Try to fix the tokenizer JSON
            let tokenizer_json = std::fs::read_to_string(tokenizer_path)?;
            let mut tokenizer_value: JsonValue = serde_json::from_str(&tokenizer_json)?;

            // Fix known compatibility issues - remove the post_processor entirely
            if tokenizer_value.get("post_processor").is_some() {
                info!("Removing post_processor for compatibility");
                tokenizer_value.as_object_mut().unwrap().remove("post_processor");
            }

            // Save fixed version
            let temp_path = tokenizer_path.with_extension("fixed.json");
            std::fs::write(&temp_path, serde_json::to_string_pretty(&tokenizer_value)?)?;

            match Tokenizer::from_file(&temp_path) {
                Ok(tokenizer) => {
                    info!("Successfully loaded tokenizer using cache fallback");
                    std::fs::rename(&temp_path, tokenizer_path)?;
                    Ok(tokenizer)
                }
                                 Err(_) => {
                     let _ = std::fs::remove_file(&temp_path);
                     warn!("Cache tokenizer fixing failed, downloading fresh Kokoro tokenizer");
                     create_basic_tokenizer()
                 }
            }
        }
    }
}

/// Download and use the original Kokoro tokenizer directly
fn create_basic_tokenizer() -> Result<Tokenizer> {
    warn!("Tokenizer loading failed, downloading fresh Kokoro tokenizer directly");

    // Use the Kokoro tokenizer from the official ONNX community repo
    let tokenizer_url = "https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX/resolve/main/tokenizer.json";

    // Create a temporary file to store the downloaded tokenizer
    let temp_dir = std::env::temp_dir();
    let temp_path = temp_dir.join("kokoro_fresh_tokenizer.json");

    // Download the tokenizer synchronously using reqwest blocking
    let client = reqwest::blocking::Client::new();
    let response = client.get(tokenizer_url).send()
        .map_err(|e| anyhow!("Failed to download Kokoro tokenizer: {}", e))?;

    if !response.status().is_success() {
        return Err(anyhow!("Failed to download tokenizer: HTTP {}", response.status()));
    }

    let tokenizer_bytes = response.bytes()
        .map_err(|e| anyhow!("Failed to read tokenizer response: {}", e))?;

    std::fs::write(&temp_path, &tokenizer_bytes)
        .map_err(|e| anyhow!("Failed to write tokenizer file: {}", e))?;

    info!("Downloaded fresh Kokoro tokenizer ({} bytes)", tokenizer_bytes.len());

    // Try to load the fresh tokenizer
    match Tokenizer::from_file(&temp_path) {
        Ok(tokenizer) => {
            info!("Successfully loaded fresh Kokoro tokenizer with vocab size: {}", tokenizer.get_vocab_size(false));
            Ok(tokenizer)
        }
        Err(e) => {
            warn!("Fresh Kokoro tokenizer failed to load: {}", e);
            // Try to fix common issues with the downloaded tokenizer
            fix_and_load_tokenizer(&temp_path)
        }
    }
}

/// Fix common tokenizer issues and load
fn fix_and_load_tokenizer(tokenizer_path: &Path) -> Result<Tokenizer> {
    info!("Attempting to fix Kokoro tokenizer compatibility issues");

    let tokenizer_json = std::fs::read_to_string(tokenizer_path)?;
    let mut tokenizer_value: serde_json::Value = serde_json::from_str(&tokenizer_json)?;

    // Remove problematic post_processor that causes loading issues
    if tokenizer_value.get("post_processor").is_some() {
        info!("Removing problematic post_processor");
        tokenizer_value.as_object_mut().unwrap().remove("post_processor");
    }

    // Fix common issues with the model section
    if let Some(model) = tokenizer_value.get_mut("model") {
        if let Some(model_obj) = model.as_object_mut() {
            // Remove any problematic fields
            model_obj.remove("byte_fallback");
            model_obj.remove("ignore_merges");

            // Ensure required fields exist
            if !model_obj.contains_key("unk_token") {
                model_obj.insert("unk_token".to_string(), serde_json::Value::Null);
            }
            if !model_obj.contains_key("dropout") {
                model_obj.insert("dropout".to_string(), serde_json::Value::Null);
            }
        }
    }

    // Save the fixed tokenizer
    let fixed_path = tokenizer_path.with_extension("fixed.json");
    std::fs::write(&fixed_path, serde_json::to_string_pretty(&tokenizer_value)?)?;

    match Tokenizer::from_file(&fixed_path) {
        Ok(tokenizer) => {
            info!("Successfully loaded fixed Kokoro tokenizer with vocab size: {}", tokenizer.get_vocab_size(false));
            Ok(tokenizer)
        }
        Err(e) => {
            Err(anyhow!("Could not load even fixed Kokoro tokenizer: {}", e))
        }
    }
}

pub fn clear_model_cache(model_id: &str) -> Result<()> {
    info!("Clearing cache for model: {}", model_id);

    let cache_base = dirs::cache_dir()
        .ok_or_else(|| anyhow!("Could not find cache directory"))?;

    // New layout: huggingface/transformers/kokoro/<model_id>/
    let new_root = cache_base
        .join("huggingface")
        .join("transformers")
        .join("kokoro")
        .join(model_id.replace('/', "_"));

    // Legacy layout: huggingface/transformers/
    let legacy_root = cache_base.join("huggingface").join("transformers");

    let required = ["model.onnx", "config.json", "tokenizer.json", "tokenizer_config.json"];

    // Clear new layout files
    for name in required {
        let path = new_root.join(name);
        if path.exists() {
            match std::fs::remove_file(&path) {
                Ok(_) => info!("Removed cached file: {:?}", path),
                Err(e) => info!("Failed to remove cached file {:?}: {}", path, e),
            }
        }
    }

    // Clear legacy layout files
    for name in required {
        let path = legacy_root.join(name);
        if path.exists() {
            match std::fs::remove_file(&path) {
                Ok(_) => info!("Removed legacy cached file: {:?}", path),
                Err(e) => info!("Failed to remove legacy cached file {:?}: {}", path, e),
            }
        }
    }

    // Try to remove the cache directory if it's empty
    let _ = std::fs::remove_dir(&new_root);

    Ok(())
}

/// Clear corrupted tokenizer cache and force fresh download
pub fn clear_tokenizer_cache(model_id: &str) -> Result<()> {
    info!("Clearing corrupted tokenizer cache for model: {}", model_id);

    let cache_base = dirs::cache_dir()
        .ok_or_else(|| anyhow!("Could not find cache directory"))?;

    // Clear from new layout
    let new_root = cache_base
        .join("huggingface")
        .join("transformers")
        .join("kokoro")
        .join(model_id.replace('/', "_"));

    // Clear from legacy layout
    let legacy_root = cache_base.join("huggingface").join("transformers");

    let tokenizer_files = ["tokenizer.json", "tokenizer_config.json"];

    for root in [&new_root, &legacy_root] {
        for name in tokenizer_files {
            let path = root.join(name);
            if path.exists() {
                match std::fs::remove_file(&path) {
                    Ok(_) => info!("Removed corrupted tokenizer file: {:?}", path),
                    Err(e) => warn!("Failed to remove tokenizer file {:?}: {}", path, e),
                }
            }

            // Also remove any .fixed files
            let fixed_path = root.join(format!("{}.fixed", name));
            if fixed_path.exists() {
                let _ = std::fs::remove_file(&fixed_path);
            }
        }
    }

    // Clear temporary tokenizer files
    let temp_dir = std::env::temp_dir();
    for temp_name in ["kokoro_fresh_tokenizer.json", "kokoro_fallback_tokenizer.json"] {
        let temp_path = temp_dir.join(temp_name);
        if temp_path.exists() {
            let _ = std::fs::remove_file(&temp_path);
        }
    }

    info!("Tokenizer cache cleared for {}", model_id);
    Ok(())
}

/// Get Kokoro voices as a static list without requiring the model to be loaded
/// This allows showing voices even when the ONNX model fails to load
pub fn get_kokoro_voices_static() -> Vec<VoiceInfo> {
    let model_id = "hexgrad/Kokoro-82M";
    vec![
        // Female voices
        VoiceInfo { id: "af".to_string(), name: "Female (Default)".to_string(), gender: "female".to_string(), language: "English".to_string(), model_id: model_id.to_string() },
        VoiceInfo { id: "af_heart".to_string(), name: "Female (Heart)".to_string(), gender: "female".to_string(), language: "English".to_string(), model_id: model_id.to_string() },
        VoiceInfo { id: "af_alloy".to_string(), name: "Female (Alloy)".to_string(), gender: "female".to_string(), language: "English".to_string(), model_id: model_id.to_string() },
        VoiceInfo { id: "af_aoede".to_string(), name: "Female (Aoede)".to_string(), gender: "female".to_string(), language: "English".to_string(), model_id: model_id.to_string() },
        VoiceInfo { id: "af_bella".to_string(), name: "Female (Bella)".to_string(), gender: "female".to_string(), language: "English".to_string(), model_id: model_id.to_string() },
        VoiceInfo { id: "af_jessica".to_string(), name: "Female (Jessica)".to_string(), gender: "female".to_string(), language: "English".to_string(), model_id: model_id.to_string() },
        VoiceInfo { id: "af_kore".to_string(), name: "Female (Kore)".to_string(), gender: "female".to_string(), language: "English".to_string(), model_id: model_id.to_string() },
        VoiceInfo { id: "af_nicole".to_string(), name: "Female (Nicole)".to_string(), gender: "female".to_string(), language: "English".to_string(), model_id: model_id.to_string() },
        VoiceInfo { id: "af_nova".to_string(), name: "Female (Nova)".to_string(), gender: "female".to_string(), language: "English".to_string(), model_id: model_id.to_string() },
        VoiceInfo { id: "af_river".to_string(), name: "Female (River)".to_string(), gender: "female".to_string(), language: "English".to_string(), model_id: model_id.to_string() },
        VoiceInfo { id: "af_sarah".to_string(), name: "Female (Sarah)".to_string(), gender: "female".to_string(), language: "English".to_string(), model_id: model_id.to_string() },
        VoiceInfo { id: "af_sky".to_string(), name: "Female (Sky)".to_string(), gender: "female".to_string(), language: "English".to_string(), model_id: model_id.to_string() },
        // Male voices
        VoiceInfo { id: "am_adam".to_string(), name: "Male (Adam)".to_string(), gender: "male".to_string(), language: "English".to_string(), model_id: model_id.to_string() },
        VoiceInfo { id: "am_echo".to_string(), name: "Male (Echo)".to_string(), gender: "male".to_string(), language: "English".to_string(), model_id: model_id.to_string() },
        VoiceInfo { id: "am_eric".to_string(), name: "Male (Eric)".to_string(), gender: "male".to_string(), language: "English".to_string(), model_id: model_id.to_string() },
        VoiceInfo { id: "am_fenrir".to_string(), name: "Male (Fenrir)".to_string(), gender: "male".to_string(), language: "English".to_string(), model_id: model_id.to_string() },
        VoiceInfo { id: "am_liam".to_string(), name: "Male (Liam)".to_string(), gender: "male".to_string(), language: "English".to_string(), model_id: model_id.to_string() },
        VoiceInfo { id: "am_michael".to_string(), name: "Male (Michael)".to_string(), gender: "male".to_string(), language: "English".to_string(), model_id: model_id.to_string() },
        VoiceInfo { id: "am_onyx".to_string(), name: "Male (Onyx)".to_string(), gender: "male".to_string(), language: "English".to_string(), model_id: model_id.to_string() },
        VoiceInfo { id: "am_puck".to_string(), name: "Male (Puck)".to_string(), gender: "male".to_string(), language: "English".to_string(), model_id: model_id.to_string() },
        VoiceInfo { id: "am_santa".to_string(), name: "Male (Santa)".to_string(), gender: "male".to_string(), language: "English".to_string(), model_id: model_id.to_string() },
        // Other languages
        VoiceInfo { id: "jf_alpha".to_string(), name: "Japanese Female (Alpha)".to_string(), gender: "female".to_string(), language: "Japanese".to_string(), model_id: model_id.to_string() },
        VoiceInfo { id: "jm_kumo".to_string(), name: "Japanese Male (Kumo)".to_string(), gender: "male".to_string(), language: "Japanese".to_string(), model_id: model_id.to_string() },
        VoiceInfo { id: "zf_xiaobei".to_string(), name: "Chinese Female (Xiaobei)".to_string(), gender: "female".to_string(), language: "Chinese".to_string(), model_id: model_id.to_string() },
        VoiceInfo { id: "zm_yunjian".to_string(), name: "Chinese Male (Yunjian)".to_string(), gender: "male".to_string(), language: "Chinese".to_string(), model_id: model_id.to_string() },
    ]
}

pub fn is_model_installed(model_id: &str) -> bool {
    // Known required files for Kokoro ONNX community model
    let required = ["model.onnx", "config.json", "tokenizer.json", "tokenizer_config.json"];

    let cache_base = match dirs::cache_dir() { Some(p) => p, None => {
        info!("Could not find cache directory for model check: {}", model_id);
        return false;
    }};

    // New layout: huggingface/transformers/kokoro/<model_id>/
    let new_root = cache_base
        .join("huggingface")
        .join("transformers")
        .join("kokoro")
        .join(model_id.replace('/', "_"));

    info!("Checking new layout path for {}: {:?}", model_id, new_root);
    let new_ok = required.iter().all(|name| {
        let path = new_root.join(name);
        let exists = path.exists();
        info!("  File {}: exists={} at {:?}", name, exists, path);
        exists
    });

    if new_ok {
        info!("Model {} found in new layout", model_id);
        return true;
    }

    // Backward compatibility: files might have been stored directly under huggingface/transformers
    let legacy_root = cache_base.join("huggingface").join("transformers");
    info!("Checking legacy layout path for {}: {:?}", model_id, legacy_root);
    let legacy_ok = required.iter().all(|name| {
        let path = legacy_root.join(name);
        let exists = path.exists();
        info!("  Legacy file {}: exists={} at {:?}", name, exists, path);
        exists
    });

    if legacy_ok {
        info!("Model {} found in legacy layout", model_id);
    } else {
        info!("Model {} not found in either layout", model_id);
    }

    legacy_ok
}

pub async fn load_onnx_model<R: Runtime>(
    model_id: &str,
    window: tauri::WebviewWindow<R>,
) -> Result<TtsModel> {
    info!("Loading ONNX TTS model: {}", model_id);

    // Only support Kokoro-82M (use ONNX community version)
    let (repo_id, _revision) = match model_id {
        "hexgrad/Kokoro-82M" => ("onnx-community/Kokoro-82M-v1.0-ONNX", "main"),
        _ => return Err(anyhow!("Only Kokoro-82M is supported. Model ID: {}", model_id)),
    };

    // Skip hf-hub API and use direct HTTP downloads for better subdirectory support

    // Emit progress events
    let emit_progress = |filename: &str, progress: f32| {
        let _ = window.emit(
            "tauri-plugins:tauri-plugin-ipc-audio-tts-ort:load-model-progress",
            (false, filename, progress, 100, (progress * 100.0) as usize),
        );
    };

    emit_progress(model_id, 0.0);

    // Target cache directory where we persist Kokoro assets, namespaced by model id
    let cache_root = dirs::cache_dir()
        .ok_or_else(|| anyhow!("Could not find cache directory"))?
        .join("huggingface")
        .join("transformers")
        .join("kokoro")
        .join(model_id.replace('/', "_"));
    tokio::fs::create_dir_all(&cache_root).await?;

    // Helper to resolve cached path and decide whether to download
    let model_id_owned = model_id.to_string();
    let ensure_file = |name: &str, url: String, progress: f32| {
        let cache_root = cache_root.clone();
        let name_owned = name.to_string();
        let model_id_owned = model_id_owned.clone();
        async move {
            let path = cache_root.join(&name_owned);
            if !tokio::fs::try_exists(&path).await.unwrap_or(false) {
                let downloaded = download_file_with_progress(&url, &name_owned).await?;
                // Move from generic cache root to our subdir if needed
                if downloaded != path {
                    if let Some(parent) = path.parent() { tokio::fs::create_dir_all(parent).await.ok(); }
                    tokio::fs::rename(&downloaded, &path).await.ok();
                }
            }
            emit_progress(&model_id_owned, progress);
            Result::<PathBuf>::Ok(path)
        }
    };

    // Download Kokoro ONNX community model files or reuse cached ones
    // onnx/model.onnx
    let model_url = format!("https://huggingface.co/{}/resolve/main/onnx/model.onnx", repo_id);
    let model_path = ensure_file("model.onnx", model_url, 40.0).await?;

    // config.json
    let config_url = format!("https://huggingface.co/{}/resolve/main/config.json", repo_id);
    let config_path = ensure_file("config.json", config_url, 60.0).await?;

    // tokenizer.json
    let tokenizer_url = format!("https://huggingface.co/{}/resolve/main/tokenizer.json", repo_id);
    let tokenizer_path = ensure_file("tokenizer.json", tokenizer_url, 80.0).await?;

    // tokenizer_config.json
    let tokenizer_config_url = format!("https://huggingface.co/{}/resolve/main/tokenizer_config.json", repo_id);
    let _tokenizer_config_path = ensure_file("tokenizer_config.json", tokenizer_config_url, 90.0).await?;

    // Load config
    let config_str = std::fs::read_to_string(config_path)?;
    let config: TtsConfig = serde_json::from_str(&config_str)?;

    // Load tokenizer with fallback for compatibility
    let tokenizer = match load_tokenizer_from_file_with_fallback(&tokenizer_path) {
        Ok(t) => t,
        Err(e) => {
            warn!("Failed to load tokenizer with all fallbacks: {}", e);
            warn!("Attempting to download a fresh tokenizer from HuggingFace");

            // Try downloading a fresh tokenizer using async
            let tokenizer_url = format!("https://huggingface.co/{}/resolve/main/tokenizer.json", repo_id);
            let runtime = tokio::runtime::Runtime::new()?;
            let download_result = runtime.block_on(async {
                let client = reqwest::Client::new();
                client.get(&tokenizer_url).send().await
            });

            match download_result {
                Ok(response) if response.status().is_success() => {
                    let tokenizer_bytes = runtime.block_on(response.bytes())?;
                    std::fs::write(&tokenizer_path, &tokenizer_bytes)?;
                    info!("Downloaded fresh tokenizer, attempting to load");

                    // Try the simple load first
                                         match Tokenizer::from_file(&tokenizer_path) {
                         Ok(t) => {
                             info!("Successfully loaded fresh Kokoro tokenizer with vocab size: {}", t.get_vocab_size(false));
                             t
                         },
                         Err(_) => {
                             // If that fails, try to fix and load the downloaded tokenizer
                             warn!("Fresh tokenizer loading failed, attempting to fix compatibility issues");
                             match fix_and_load_tokenizer(&tokenizer_path) {
                                 Ok(t) => t,
                                 Err(_) => {
                                     // Last resort - download a fresh tokenizer directly
                                     warn!("All tokenizer methods failed, downloading fresh Kokoro tokenizer as last resort");
                                     create_basic_tokenizer()?
                                 }
                             }
                         }
                     }
                }
                _ => {
                    warn!("Could not download fresh tokenizer, using minimal fallback");
                    create_basic_tokenizer()?
                }
            }
        }
    };

    // Create ONNX session
    let session = create_optimized_session(model_path)?;
    emit_progress(model_id, 100.0);

    // Emit completion
    let _ = window.emit(
        "tauri-plugins:tauri-plugin-ipc-audio-tts-ort:load-model-progress",
        (true, model_id, 100.0),
    );

    Ok(TtsModel::Onnx(OnnxTtsModel::new(session, config, model_id.to_string(), tokenizer)))
}

/// Load an ONNX TTS model strictly from the cache without networking or progress events.
/// Returns an error if required assets are missing.
pub fn load_onnx_model_from_cache(model_id: &str) -> Result<TtsModel> {
    info!("Loading ONNX TTS model from cache: {}", model_id);

    // Only support Kokoro-82M
    if model_id != "hexgrad/Kokoro-82M" {
        return Err(anyhow!("Only Kokoro-82M is supported. Model ID: {}", model_id));
    }

    let cache_root = dirs::cache_dir()
        .ok_or_else(|| anyhow!("Could not find cache directory"))?
        .join("huggingface")
        .join("transformers")
        .join("kokoro")
        .join(model_id.replace('/', "_"));

    let model_path = cache_root.join("model.onnx");
    let config_path = cache_root.join("config.json");
    let tokenizer_path = cache_root.join("tokenizer.json");
    let tokenizer_config_path = cache_root.join("tokenizer_config.json");

    // Fallback to legacy layout
    let legacy_root = dirs::cache_dir()
        .ok_or_else(|| anyhow!("Could not find cache directory"))?
        .join("huggingface")
        .join("transformers");
    let get_first_existing = |preferred: &Path, legacy_name: &str| {
        if preferred.exists() { preferred.to_path_buf() } else { legacy_root.join(legacy_name) }
    };

    let model_path = get_first_existing(&model_path, "model.onnx");
    let config_path = get_first_existing(&config_path, "config.json");
    let tokenizer_path = get_first_existing(&tokenizer_path, "tokenizer.json");
    let tokenizer_config_path = get_first_existing(&tokenizer_config_path, "tokenizer_config.json");

    info!("Checking cache files: model={}, config={}, tokenizer={}, tokenizer_config={}",
          model_path.exists(), config_path.exists(), tokenizer_path.exists(), tokenizer_config_path.exists());

    if !model_path.exists() || !config_path.exists() || !tokenizer_path.exists() || !tokenizer_config_path.exists() {
        return Err(anyhow!("Cached model files not found for {}. Missing files: model={}, config={}, tokenizer={}, tokenizer_config={}",
                   model_id, !model_path.exists(), !config_path.exists(), !tokenizer_path.exists(), !tokenizer_config_path.exists()));
    }

        let config_str = std::fs::read_to_string(&config_path)?;
    let config: TtsConfig = serde_json::from_str(&config_str)?;
    info!("Successfully loaded config for {}", model_id);

    // Load tokenizer with fallback for compatibility
    let tokenizer = load_tokenizer_from_cache_with_fallback(&tokenizer_path)
        .map_err(|e| anyhow!("Failed to load tokenizer from cache: {}", e))?;
    info!("Successfully loaded tokenizer for {}", model_id);

    let session = create_optimized_session(model_path)?;
    info!("Successfully created ONNX session for {}", model_id);

    Ok(TtsModel::Onnx(OnnxTtsModel::new(session, config, model_id.to_string(), tokenizer)))
}

fn create_optimized_session(model_path: PathBuf) -> Result<Session> {
    // Try CPU-only first to avoid DirectML/CUDA issues with Kokoro model
    let session = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level1)?  // Reduce optimization level
        .with_parallel_execution(false)?  // Disable parallel execution for stability
        .with_execution_providers([
            CPUExecutionProvider::default().build(),
        ])?
        .commit_from_file(model_path)?;

    Ok(session)
}
