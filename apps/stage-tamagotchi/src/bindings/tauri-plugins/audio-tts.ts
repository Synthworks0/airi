import { invoke } from '@tauri-apps/api/core'

export interface TtsModelInfo {
  id: string
  name: string
  size: number
  quality: 'high' | 'medium' | 'low'
  languages: string[]
  installed: boolean
}

export interface TtsVoiceInfo {
  id: string
  name: string
  gender: string
  language: string
  modelId: string
}

export interface SynthesizeOptions {
  pitch?: number
  speed?: number
  volume?: number
}

export async function listModels(): Promise<TtsModelInfo[]> {
  return await invoke('plugin:ipc-audio-tts-ort|list_models')
}

export async function listVoices(): Promise<TtsVoiceInfo[]> {
  return await invoke('plugin:ipc-audio-tts-ort|list_voices')
}

export async function listInstalledModels(): Promise<string[]> {
  return await invoke('plugin:ipc-audio-tts-ort|list_installed_models')
}

export async function loadModel(modelId: string): Promise<void> {
  return await invoke('plugin:ipc-audio-tts-ort|load_model', { modelId })
}

export async function synthesize(
  text: string,
  voiceId: string,
  options?: SynthesizeOptions
): Promise<Uint8Array> {
  const result = await invoke('plugin:ipc-audio-tts-ort|synthesize', {
    text,
    voiceId,
    options,
  }) as number[]

  return new Uint8Array(result)
}
