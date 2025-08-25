import { invoke } from '@tauri-apps/api/core'

export interface SttModelInfo {
  id: string
  name: string
  size: number
  accuracy: 'high' | 'medium' | 'low'
  speed: 'fast' | 'medium' | 'slow'
  installed: boolean
}

export async function listModels(): Promise<SttModelInfo[]> {
  return await invoke('plugin:ipc-audio-transcription-ort|list_models')
}

export async function listInstalledModels(): Promise<string[]> {
  return await invoke('plugin:ipc-audio-transcription-ort|list_installed_models')
}

export async function loadWhisperModel(modelType: string): Promise<void> {
  return await invoke('plugin:ipc-audio-transcription-ort|load_ort_model_whisper', {
    modelType,
  })
}

export async function transcribe(
  chunk: Float32Array,
  language?: string
): Promise<string> {
  return await invoke('plugin:ipc-audio-transcription-ort|ipc_audio_transcription', {
    chunk: Array.from(chunk),
    language: language || 'en',
  })
}
