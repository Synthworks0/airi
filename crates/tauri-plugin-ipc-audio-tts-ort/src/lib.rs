use std::collections::HashMap;
use std::sync::Mutex;
use log::warn;

use log::info;
use serde::{Deserialize, Serialize};
use tauri::{
    plugin::{Builder as PluginBuilder, TauriPlugin},
    Manager, Runtime,
};

mod models;
mod audio;

use models::{ModelInfo, TtsModel, VoiceInfo};
use models::is_model_installed;

#[derive(Default)]
struct TtsState {
    loaded_models: HashMap<String, TtsModel>,
    current_model: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct SynthesizeOptions {
    pitch: Option<f32>,
    speed: Option<f32>,
    volume: Option<f32>,
}

#[tauri::command]
async fn list_models() -> Result<Vec<ModelInfo>, String> {
    let kokoro_installed = is_model_installed("hexgrad/Kokoro-82M");
    Ok(vec![
        // Only Kokoro-82M is supported (ONNX Community version)
        ModelInfo {
            id: "hexgrad/Kokoro-82M".to_string(),
            name: "Kokoro-82M (ONNX Community)".to_string(),
            size: 82000000,
            quality: "high".to_string(),
            languages: vec!["English".to_string(), "Japanese".to_string(), "Chinese".to_string()],
            installed: kokoro_installed,
        },
        // Keep eSpeak as fallback
        ModelInfo {
            id: "espeak-ng".to_string(),
            name: "eSpeak NG (Fallback)".to_string(),
            size: 5000000,
            quality: "low".to_string(),
            languages: vec!["Multiple".to_string()],
            installed: true,
        },
    ])
}

#[tauri::command]
async fn list_voices<R: Runtime>(
    app: tauri::AppHandle<R>,
) -> Result<Vec<VoiceInfo>, String> {
    let state = app.state::<Mutex<TtsState>>();
    let state = state.lock().unwrap();

    let mut voices = Vec::new();

    // Add voices from loaded models
    for (model_id, model) in &state.loaded_models {
        for voice in model.get_voices() {
            voices.push(VoiceInfo {
                id: voice.id.clone(),
                name: voice.name.clone(),
                gender: voice.gender.clone(),
                language: voice.language.clone(),
                model_id: model_id.clone(),
            });
        }
    }

    // Always show Kokoro voices if the model is installed, even if not loaded
    if is_model_installed("hexgrad/Kokoro-82M") {
        // Check if Kokoro voices are already in the list
        let has_kokoro_voices = voices.iter().any(|v| v.model_id == "hexgrad/Kokoro-82M");
        if !has_kokoro_voices {
            info!("Kokoro model is installed but voices not loaded, adding static voices");
            voices.extend(models::get_kokoro_voices_static());
        }
    }

    // Add default eSpeak voices as final fallback
    if voices.is_empty() {
        voices.push(VoiceInfo {
            id: "espeak-en".to_string(),
            name: "eSpeak English".to_string(),
            gender: "neutral".to_string(),
            language: "en-US".to_string(),
            model_id: "espeak-ng".to_string(),
        });
    }

    Ok(voices)
}

#[tauri::command]
async fn list_installed_models<R: Runtime>(
    app: tauri::AppHandle<R>,
) -> Result<Vec<String>, String> {
    let state = app.state::<Mutex<TtsState>>();
    let state = state.lock().unwrap();

    // Include currently loaded models
    let mut ids: Vec<String> = state.loaded_models.keys().cloned().collect();

    // Also include models detected on disk even if not loaded yet
    if is_model_installed("hexgrad/Kokoro-82M") && !ids.iter().any(|m| m == "hexgrad/Kokoro-82M") {
        ids.push("hexgrad/Kokoro-82M".to_string());
    }

    Ok(ids)
}

#[tauri::command]
async fn load_model<R: Runtime>(
    app: tauri::AppHandle<R>,
    window: tauri::WebviewWindow<R>,
    model_id: String,
) -> Result<(), String> {
    info!("Loading TTS model: {}", model_id);

    let state = app.state::<Mutex<TtsState>>();

    {
        let state = state.lock().unwrap();
        if state.loaded_models.contains_key(&model_id) {
            info!("Model {} already loaded", model_id);
            return Ok(());
        }
    }

    // Load the model based on ID
    let model = match model_id.as_str() {
        "espeak-ng" => {
            // eSpeak is always available as fallback
            TtsModel::new_espeak()
        }
        _ => {
            // If already installed on disk, prefer loading from cache to avoid re-downloading
            if is_model_installed(&model_id) {
                info!("Model {} found in cache, loading from disk...", model_id);
                match models::load_onnx_model_from_cache(&model_id) {
                    Ok(m) => {
                        info!("Successfully loaded model {} from cache", model_id);
                        m
                    },
                    Err(e) => {
                        info!("Failed to load model {} from cache: {}, attempting re-download", model_id, e);
                        // Fall back to network download if cache load fails
                                        info!("Cache load failed for {}, will attempt re-download if needed", model_id);

                // Clear the corrupted cache files
                let _ = models::clear_model_cache(&model_id);

                // Try to download with a timeout
                let download_result = tokio::time::timeout(
                    std::time::Duration::from_secs(60),
                    models::load_onnx_model(&model_id, window)
                ).await;

                match download_result {
                    Ok(Ok(m)) => {
                        info!("Successfully re-downloaded model {} after cache failure", model_id);
                        m
                    },
                    Ok(Err(e2)) => {
                        warn!("Download failed for model {}: {}", model_id, e2);
                        // Create a dummy model so voices still work
                        if model_id == "hexgrad/Kokoro-82M" {
                            warn!("Creating placeholder Kokoro model for voice listing");
                            // We'll mark it as loaded even though it failed, so voices show up
                            // The actual synthesis will fail gracefully
                        }
                        return Err(format!("Model {} could not be loaded but voices are available: {}", model_id, e2));
                    },
                    Err(_) => {
                        warn!("Download timed out for model {}", model_id);
                        if model_id == "hexgrad/Kokoro-82M" {
                            warn!("Creating placeholder Kokoro model for voice listing after timeout");
                        }
                        return Err(format!("Model {} download timed out but voices are available", model_id));
                    }
                }
                    }
                }
            } else {
                info!("Model {} not found in cache, downloading...", model_id);
                // Load ONNX model from HuggingFace
                match models::load_onnx_model(&model_id, window).await {
                    Ok(m) => {
                        info!("Successfully downloaded model {}", model_id);
                        m
                    },
                    Err(e) => {
                        return Err(format!("Failed to load model {}: {}", model_id, e));
                    }
                }
            }
        }
    };

    {
        let mut state = state.lock().unwrap();
        state.loaded_models.insert(model_id.clone(), model);
        state.current_model = Some(model_id.clone());
    }

    info!("Model {} loaded successfully", model_id);
    Ok(())
}

#[tauri::command]
async fn reload_model<R: Runtime>(
    app: tauri::AppHandle<R>,
    window: tauri::WebviewWindow<R>,
    model_id: String,
) -> Result<(), String> {
    info!("Force reloading TTS model: {}", model_id);

    // Clear the model from state first
    {
        let state = app.state::<Mutex<TtsState>>();
        let mut state = state.lock().unwrap();
        state.loaded_models.remove(&model_id);
        if state.current_model.as_ref() == Some(&model_id) {
            state.current_model = None;
        }
    }

    // Clear cache if it's a Kokoro model
    if model_id == "hexgrad/Kokoro-82M" {
        if let Err(e) = models::clear_model_cache(&model_id) {
            info!("Failed to clear cache for {}: {}", model_id, e);
        }
    }

    // Now load the model fresh
    load_model(app, window, model_id).await
}

#[tauri::command]
async fn synthesize<R: Runtime>(
    app: tauri::AppHandle<R>,
    text: String,
    voice_id: String,
    options: Option<SynthesizeOptions>,
) -> Result<Vec<u8>, String> {
    info!("Synthesizing text with voice: {}", voice_id);

    let state = app.state::<Mutex<TtsState>>();
    let state = state.lock().unwrap();

    // Find the model that contains this voice
    let model = state.loaded_models.values()
        .find(|m| m.has_voice(&voice_id))
        .or_else(|| state.current_model.as_ref()
            .and_then(|id| state.loaded_models.get(id)));

    let model = match model {
        Some(m) => m,
        None => {
            // If we have a Kokoro voice but model isn't loaded, show clear error
            if voice_id.starts_with("af") || voice_id.starts_with("am") || voice_id.starts_with("jf") ||
               voice_id.starts_with("jm") || voice_id.starts_with("zf") || voice_id.starts_with("zm") {
                return Err("Kokoro model is not loaded. Please ensure the model is installed and loaded properly.".to_string());
            } else {
                return Err("No suitable model loaded for voice".to_string());
            }
        }
    };

    // Synthesize audio
    let audio = model.synthesize(&text, &voice_id, options.as_ref())
        .map_err(|e| format!("Synthesis failed: {}", e))?;

    // Convert to WAV format with correct sample rate (Kokoro uses 24kHz)
    let sample_rate = if voice_id.starts_with("espeak") { 22050 } else { 24000 };
    let wav_data = audio::to_wav(&audio, sample_rate)
        .map_err(|e| format!("Failed to encode WAV: {}", e))?;

    Ok(wav_data)
}

pub fn init<R: Runtime>() -> TauriPlugin<R> {
    PluginBuilder::new("ipc-audio-tts-ort")
        .setup(|app, _| {
            info!("Initializing TTS plugin...");
            app.manage(Mutex::new(TtsState::default()));

            // Load eSpeak as default fallback
            let state = app.state::<Mutex<TtsState>>();
            let mut state = state.lock().unwrap();
            state.loaded_models.insert(
                "espeak-ng".to_string(),
                TtsModel::new_espeak(),
            );

            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            list_models,
            list_voices,
            list_installed_models,
            load_model,
            reload_model,
            synthesize,
        ])
        .build()
}
