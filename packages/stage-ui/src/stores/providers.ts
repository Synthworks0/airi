import type {
  ChatProvider,
  ChatProviderWithExtraOptions,
  EmbedProvider,
  EmbedProviderWithExtraOptions,
  SpeechProvider,
  SpeechProviderWithExtraOptions,
  TranscriptionProvider,
  TranscriptionProviderWithExtraOptions,
} from '@xsai-ext/shared-providers'
import type { ProgressInfo } from '@xsai-transformers/shared/types'
import type {
  UnAlibabaCloudOptions,
  UnElevenLabsOptions,
  UnMicrosoftOptions,
  UnVolcengineOptions,
  VoiceProviderWithExtraOptions,
} from 'unspeech'

import { useLocalStorage } from '@vueuse/core'
import {
  createAnthropic,
  createAzure,
  createDeepSeek,
  createFireworks,
  createGoogleGenerativeAI,
  createMistral,
  createMoonshot,
  createNovita,
  createOpenAI,
  createOpenRouter,
  createPerplexity,
  createTogetherAI,
  createWorkersAI,
  createXAI,
} from '@xsai-ext/providers-cloud'
import { createOllama, createPlayer2 } from '@xsai-ext/providers-local'
import { listModels } from '@xsai/model'
import { isWebGPUSupported } from 'gpuu/webgpu'
import { defineStore } from 'pinia'
import {
  createUnAlibabaCloud,
  createUnElevenLabs,
  createUnMicrosoft,
  createUnVolcengine,
  listVoices,
} from 'unspeech'
import { computed, ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'

import { isAbsoluteUrl } from '../utils/string'
import { models as elevenLabsModels } from './providers/elevenlabs/list-models'

// Add at top after imports
declare module '@tauri-apps/api/core' {
  export function invoke(cmd: string, args?: Record<string, unknown>): Promise<any>
}

declare module '@tauri-apps/api/event' {
  export function listen(event: string, handler: (event: any) => void): Promise<() => void>
}

export interface ProviderMetadata {
  id: string
  order?: number
  category: 'chat' | 'embed' | 'speech' | 'transcription'
  tasks: string[]
  nameKey: string // i18n key for provider name
  name: string // Default name (fallback)
  localizedName?: string
  descriptionKey: string // i18n key for description
  description: string // Default description (fallback)
  localizedDescription?: string
  configured?: boolean
  /**
   * Indicates whether the provider is available.
   * If not specified, the provider is always available.
   *
   * May be specified when any of the following criteria is required:
   *
   * Platform requirements:
   *
   * - app-* providers are only available on desktop, this is responsible for Tauri runtime checks
   * - web-* providers are only available on web, this means Node.js and Tauri should not be imported or used
   *
   * System spec requirements:
   *
   * - may requires WebGPU / NVIDIA / other types of GPU,
   *   on Web, WebGPU will automatically compiled to use targeting GPU hardware
   * - may requires significant amount of GPU memory to run, especially for
   *   using of small language models within browser or Tauri app
   * - may requires significant amount of memory to run, especially for those
   *   non-WebGPU supported environments.
   */
  isAvailableBy?: () => Promise<boolean> | boolean
  /**
   * Iconify JSON icon name for the provider.
   *
   * Icons are available for most of the AI provides under @proj-airi/lobe-icons.
   */
  icon?: string
  iconColor?: string
  /**
   * In case of having image instead of icon, you can specify the image URL here.
   */
  iconImage?: string
  defaultOptions?: () => Record<string, unknown>
  createProvider: (
    config: Record<string, unknown>
  ) =>
    | ChatProvider
    | ChatProviderWithExtraOptions
    | EmbedProvider
    | EmbedProviderWithExtraOptions
    | SpeechProvider
    | SpeechProviderWithExtraOptions
    | TranscriptionProvider
    | TranscriptionProviderWithExtraOptions
    | Promise<ChatProvider>
    | Promise<ChatProviderWithExtraOptions>
    | Promise<EmbedProvider>
    | Promise<EmbedProviderWithExtraOptions>
    | Promise<SpeechProvider>
    | Promise<SpeechProviderWithExtraOptions>
    | Promise<TranscriptionProvider>
    | Promise<TranscriptionProviderWithExtraOptions>
  capabilities: {
    listModels?: (config: Record<string, unknown>) => Promise<ModelInfo[]>
    listVoices?: (config: Record<string, unknown>) => Promise<VoiceInfo[]>
    loadModel?: (config: Record<string, unknown>, hooks?: { onProgress?: (progress: ProgressInfo) => Promise<void> | void }) => Promise<void>
  }
  validators: {
    validateProviderConfig: (config: Record<string, unknown>) => Promise<{
      errors: unknown[]
      reason: string
      valid: boolean
    }> | {
      errors: unknown[]
      reason: string
      valid: boolean
    }
  }
}

export interface ModelInfo {
  id: string
  name: string
  provider: string
  description?: string
  capabilities?: string[]
  contextLength?: number
  deprecated?: boolean
}

export interface VoiceInfo {
  id: string
  name: string
  provider: string
  description?: string
  gender?: string
  deprecated?: boolean
  previewURL?: string
  modelId?: string
  languages: {
    code: string
    title: string
  }[]
}

export const useProvidersStore = defineStore('providers', () => {
  const providerCredentials = useLocalStorage<Record<string, Record<string, unknown>>>('settings/credentials/providers', {})
  const { t } = useI18n()
  const notBaseUrlError = computed(() => ({
    errors: [new Error('Base URL is not absolute')],
    reason: 'Base URL is not absolute. Check your input.',
    valid: false,
  }))

  // Helper function to fetch OpenRouter models manually
  async function fetchOpenRouterModels(config: Record<string, unknown>): Promise<ModelInfo[]> {
    try {
      const response = await fetch('https://openrouter.ai/api/v1/models', {
        headers: {
          'Authorization': `Bearer ${(config.apiKey as string).trim()}`,
          'Content-Type': 'application/json',
        },
      })

      if (!response.ok) {
        throw new Error(`Failed to fetch OpenRouter models: ${response.statusText}`)
      }

      const data = await response.json()
      return data.data.map((model: any) => ({
        id: model.id,
        name: model.name || model.id,
        provider: 'openrouter-ai',
        description: model.description || '',
        contextLength: model.context_length,
        deprecated: false,
      }))
    }
    catch (error) {
      console.error('Error fetching OpenRouter models:', error)
      throw error
    }
  }

  // Centralized provider metadata with provider factory functions
  const providerMetadata: Record<string, ProviderMetadata> = {
    'openrouter-ai': {
      id: 'openrouter-ai',
      category: 'chat',
      tasks: ['text-generation'],
      nameKey: 'settings.pages.providers.provider.openrouter.title',
      name: 'OpenRouter',
      descriptionKey: 'settings.pages.providers.provider.openrouter.description',
      description: 'openrouter.ai',
      icon: 'i-lobe-icons:openrouter',
      defaultOptions: () => ({
        baseUrl: 'https://openrouter.ai/api/v1/',
      }),
      createProvider: async config => createOpenRouter((config.apiKey as string).trim(), (config.baseUrl as string).trim()),
      capabilities: {
        listModels: async (config) => {
          return fetchOpenRouterModels(config)
        },
      },
      validators: {
        validateProviderConfig: (config) => {
          const errors = [
            !config.apiKey && new Error('API key is required'),
            !config.baseUrl && new Error('Base URL is required'),
          ].filter(Boolean)

          if (!!config.baseUrl && !isAbsoluteUrl(config.baseUrl as string)) {
            return notBaseUrlError.value
          }

          return {
            errors,
            reason: errors.filter(e => e).map(e => String(e)).join(', ') || '',
            valid: !!config.apiKey && !!config.baseUrl,
          }
        },
      },
    },
    'app-local-audio-speech': {
      id: 'app-local-audio-speech',
      category: 'speech',
      tasks: ['text-to-speech', 'tts'],
      isAvailableBy: async () => {
        // Detect Tauri runtime (v1 or v2) reliably
        if (typeof window !== 'undefined') {
          const w = window as unknown as Record<string, unknown>
          if (w.__TAURI__ != null || w.__TAURI_INTERNALS__ != null || w.__TAURI_METADATA__ != null)
            return true
        }

        // Fallback: try dynamic import which only succeeds inside Tauri/WebView
        try {
          const mod = await import('@tauri-apps/api/core')
          return typeof (mod as any)?.invoke === 'function'
        }
        catch {
          return false
        }
      },
      nameKey: 'settings.pages.providers.provider.app-local-audio-speech.title',
      name: 'Local TTS',
      descriptionKey: 'settings.pages.providers.provider.app-local-audio-speech.description',
      description: 'High-quality offline text-to-speech with ONNX models',
      icon: 'i-solar:microphone-3-bold-duotone',
      defaultOptions: () => ({
        model: 'hexgrad/Kokoro-82M',
        voice: 'af',
        voiceSettings: {
          pitch: 0,
          speed: 1.0,
          volume: 0,
        },
      }),
      // @ts-expect-error - returning SpeechProvider instead of expected union type
      createProvider: async () => {
        return {
          speech: (model: string, options: Record<string, any>) => ({
            model,
            generateSpeech: async ({ input, voice }: { input: string, voice: string }) => {
              try {
                // Validate inputs at provider level
                if (!input || typeof input !== 'string') {
                  throw new Error('Invalid or missing input text')
                }
                if (!voice || typeof voice !== 'string') {
                  throw new Error('Invalid or missing voice ID')
                }

                // Defer Tauri import to call site; if not available, surface a clear error
                let invoke: (cmd: string, args?: Record<string, unknown>) => Promise<any>
                try {
                  invoke = (await import('@tauri-apps/api/core')).invoke
                }
                catch {
                  throw new Error('Local TTS provider requires Tauri runtime')
                }

                const synthesisOptions = {
                  pitch: options?.voiceSettings?.pitch ?? options?.pitch ?? 0,
                  speed: options?.voiceSettings?.speed ?? options?.speed ?? 1.0,
                  volume: options?.voiceSettings?.volume ?? options?.volume ?? 0,
                }

                const result = await invoke('plugin:ipc-audio-tts-ort|synthesize', {
                  text: input,
                  voiceId: voice,
                  options: synthesisOptions,
                }) as number[]

                // Validate result
                if (!result || !Array.isArray(result) || result.length === 0) {
                  throw new Error('TTS plugin returned invalid or empty audio data')
                }

                // Convert to ArrayBuffer
                const uint8Array = new Uint8Array(result)
                return uint8Array.buffer
              }
              catch (error) {
                // Ensure we always throw a proper Error object with a message
                if (error instanceof Error) {
                  throw error
                }

                const errorMessage = (() => {
                  if (error === null || error === undefined) {
                    return 'TTS synthesis failed with unknown error'
                  }
                  if (typeof error === 'string') {
                    return error
                  }
                  try {
                    return String(error)
                  }
                  catch {
                    return 'TTS synthesis failed with unhandleable error'
                  }
                })()

                throw new Error(errorMessage)
              }
            },
          }),
        }
      },
      capabilities: {
        listModels: async () => {
          try {
            const { invoke } = await import('@tauri-apps/api/core')
            const models = await invoke('plugin:ipc-audio-tts-ort|list_models') as Array<{
              id: string
              name: string
              size: number
              quality: 'high' | 'medium' | 'low'
              languages: string[]
              installed: boolean
            }>
            return models.map(m => ({
              id: m.id,
              name: m.name,
              provider: 'app-local-audio-speech',
              description: `${m.quality} quality, ${m.languages.join(', ')}`,
              deprecated: false,
            }))
          }
          catch {
            return [
              { id: 'piper-amy-en', name: 'Piper Amy (English)', provider: 'app-local-audio-speech', description: 'High quality English voice' },
              { id: 'piper-ryan-en', name: 'Piper Ryan (English)', provider: 'app-local-audio-speech', description: 'High quality English voice' },
              { id: 'mimic3-en_US-vctk', name: 'Mimic 3 VCTK', provider: 'app-local-audio-speech', description: 'Multi-speaker English voices' },
            ]
          }
        },
        listVoices: async () => {
          try {
            const { invoke } = await import('@tauri-apps/api/core')
            const voices = await invoke('plugin:ipc-audio-tts-ort|list_voices') as Array<{
              id: string
              name: string
              gender: string
              language: string
              modelId: string
            }>
            return voices.map(v => ({
              id: v.id,
              name: v.name,
              provider: 'app-local-audio-speech',
              gender: v.gender,
              modelId: v.modelId,
              languages: [{ code: (v.language || 'en').split('-')[0], title: v.language || 'English' }],
            }))
          }
          catch {
            return [
              { id: 'amy', name: 'Amy', provider: 'app-local-audio-speech', gender: 'female', languages: [{ code: 'en', title: 'English' }] },
              { id: 'ryan', name: 'Ryan', provider: 'app-local-audio-speech', gender: 'male', languages: [{ code: 'en', title: 'English' }] },
            ]
          }
        },
        loadModel: async (config, hooks) => {
          try {
            const { invoke } = await import('@tauri-apps/api/core')
            const { listen } = await import('@tauri-apps/api/event')

            const unlisten = await listen('tauri-plugins:tauri-plugin-ipc-audio-tts-ort:load-model-progress', (event: any) => {
              const [done, _filename, progress, totalSize, currentSize] = event.payload as [boolean, string, number, number, number]
              if (hooks?.onProgress) {
                hooks.onProgress({
                  loaded: currentSize,
                  total: totalSize,
                  progress: progress / 100,
                  done,
                } as any)
              }
            })

            try {
              await invoke('plugin:ipc-audio-tts-ort|load_model', {
                modelId: config.modelId,
              })
            }
            finally {
              unlisten()
            }
          }
          catch {
            // Non-Tauri context; nothing to load
          }
        },
      },
      validators: {
        validateProviderConfig: () => {
          return {
            errors: [],
            reason: '',
            valid: true,
          }
        },
      },
    },
    'app-local-audio-transcription': {
      id: 'app-local-audio-transcription',
      category: 'transcription',
      tasks: ['speech-to-text', 'automatic-speech-recognition', 'asr', 'stt'],
      isAvailableBy: async () => {
        // Detect Tauri runtime (v1 or v2) reliably
        if (typeof window !== 'undefined') {
          const w = window as unknown as Record<string, unknown>
          if (w.__TAURI__ != null || w.__TAURI_INTERNALS__ != null || w.__TAURI_METADATA__ != null)
            return true
        }

        // Fallback: try dynamic import
        try {
          const mod = await import('@tauri-apps/api/core')
          return typeof (mod as any)?.invoke === 'function'
        }
        catch {
          return false
        }
      },
      nameKey: 'settings.pages.providers.provider.app-local-audio-transcription.title',
      name: 'Local STT',
      descriptionKey: 'settings.pages.providers.provider.app-local-audio-transcription.description',
      description: 'Fast offline speech recognition with Whisper ONNX models',
      icon: 'i-solar:microphone-3-bold-duotone',
      defaultOptions: () => ({
        model: 'base',
      }),
      // @ts-expect-error - returning TranscriptionProvider instead of expected union type
      createProvider: async () => {
        // Only import Tauri APIs in Tauri context
        if (typeof window === 'undefined' || !('__TAURI__' in window)) {
          throw new Error('Local STT provider requires Tauri runtime')
        }
        const { invoke } = await import('@tauri-apps/api/core')
        return {
          transcription: (model: string) => ({
            model,
            transcribe: async ({ audio, language }: { audio: File | Blob, language?: string }) => {
              const arrayBuffer = await audio.arrayBuffer()
              const float32Array = new Float32Array(arrayBuffer)

              const result = await invoke('plugin:ipc-audio-transcription-ort|ipc_audio_transcription', {
                chunk: Array.from(float32Array),
                language: language || 'en',
              }) as string

              return { text: result }
            },
          }),
        }
      },
      capabilities: {
        listModels: async () => {
          if (typeof window === 'undefined' || !('__TAURI__' in window))
            return []
          const { invoke } = await import('@tauri-apps/api/core')
          try {
            const models = await invoke('plugin:ipc-audio-transcription-ort|list_models') as Array<{
              id: string
              name: string
              size: number
              accuracy: 'high' | 'medium' | 'low'
              speed: 'fast' | 'medium' | 'slow'
              installed: boolean
            }>
            return models.map(m => ({
              id: m.id,
              name: m.name,
              provider: 'app-local-audio-transcription',
              description: `${m.accuracy} accuracy, ${m.speed} speed`,
              deprecated: false,
            }))
          }
          catch {
            return [
              { id: 'whisper-large-v3-turbo', name: 'Whisper Large v3 Turbo', provider: 'app-local-audio-transcription', description: 'Best quality, GPU recommended' },
              { id: 'whisper-medium', name: 'Whisper Medium', provider: 'app-local-audio-transcription', description: 'Balanced quality and speed' },
              { id: 'whisper-base', name: 'Whisper Base', provider: 'app-local-audio-transcription', description: 'Fast, lower accuracy' },
              { id: 'whisper-tiny', name: 'Whisper Tiny', provider: 'app-local-audio-transcription', description: 'Fastest, basic accuracy' },
            ]
          }
        },
        loadModel: async (config, hooks) => {
          if (typeof window === 'undefined' || !('__TAURI__' in window))
            return
          const { invoke } = await import('@tauri-apps/api/core')
          const { listen } = await import('@tauri-apps/api/event')

          const unlisten = await listen('tauri-plugins:tauri-plugin-ipc-audio-transcription-ort:load-model-whisper-progress', (event: any) => {
            const [done, progress, totalSize, currentSize] = event.payload as [boolean, number, number, number] // remove filename
            if (hooks?.onProgress) {
              hooks.onProgress({
                loaded: currentSize,
                total: totalSize,
                progress: progress / 100,
                done,
              } as any)
            }
          })

          try {
            const modelType = (config.modelId as string | undefined)?.replace('whisper-', '').replace('-', '') || 'base'
            await invoke('plugin:ipc-audio-transcription-ort|load_ort_model_whisper', {
              modelType,
            })
          }
          finally {
            unlisten()
          }
        },
      },
      validators: {
        validateProviderConfig: () => {
          return {
            errors: [],
            reason: '',
            valid: true,
          }
        },
      },
    },
    'browser-local-audio-speech': {
      id: 'browser-local-audio-speech',
      category: 'speech',
      tasks: ['text-to-speech', 'tts'],
      isAvailableBy: async () => {
        const webGPUAvailable = await isWebGPUSupported()
        if (webGPUAvailable) {
          return true
        }

        if ('navigator' in globalThis && globalThis.navigator != null && 'deviceMemory' in globalThis.navigator && typeof globalThis.navigator.deviceMemory === 'number') {
          const memory = globalThis.navigator.deviceMemory
          // Check if the device has at least 8GB of RAM
          if (memory >= 8) {
            return true
          }
        }

        return false
      },
      nameKey: 'settings.pages.providers.provider.browser-local-audio-speech.title',
      name: 'Browser (Local)',
      descriptionKey: 'settings.pages.providers.provider.browser-local-audio-speech.description',
      description: 'https://github.com/moeru-ai/xsai-transformers',
      icon: 'i-lobe-icons:huggingface',
      defaultOptions: () => ({}),
      createProvider: async config => createOpenAI((config.baseUrl as string).trim()),
      capabilities: {
        listModels: async (config) => {
          return (await listModels({
            ...createOpenAI((config.baseUrl as string).trim()).model(),
          })).map((model) => {
            return {
              id: model.id,
              name: model.id,
              provider: 'browser-local-transformers',
              description: '',
              contextLength: 0,
              deprecated: false,
            } satisfies ModelInfo
          })
        },
      },
      validators: {
        validateProviderConfig: (config) => {
          if (!config.baseUrl) {
            return {
              errors: [new Error('Base URL is required.')],
              reason: 'Base URL is required. This is likely a bug, report to developers on https://github.com/moeru-ai/airi/issues.',
              valid: false,
            }
          }

          return {
            errors: [],
            reason: '',
            valid: true,
          }
        },
      },
    },
    'browser-local-audio-transcription': {
      id: 'browser-local-audio-transcription',
      category: 'transcription',
      tasks: ['speech-to-text', 'automatic-speech-recognition', 'asr', 'stt'],
      isAvailableBy: async () => {
        const webGPUAvailable = await isWebGPUSupported()
        if (webGPUAvailable) {
          return true
        }

        if ('navigator' in globalThis && globalThis.navigator != null && 'deviceMemory' in globalThis.navigator && typeof globalThis.navigator.deviceMemory === 'number') {
          const memory = globalThis.navigator.deviceMemory
          // Check if the device has at least 8GB of RAM
          if (memory >= 8) {
            return true
          }
        }

        return false
      },
      nameKey: 'settings.pages.providers.provider.browser-local-audio-transcription.title',
      name: 'Browser (Local)',
      descriptionKey: 'settings.pages.providers.provider.browser-local-audio-transcription.description',
      description: 'https://github.com/moeru-ai/xsai-transformers',
      icon: 'i-lobe-icons:huggingface',
      defaultOptions: () => ({}),
      createProvider: async config => createOpenAI((config.baseUrl as string).trim()),
      capabilities: {
        listModels: async (config) => {
          return (await listModels({
            ...createOpenAI((config.baseUrl as string).trim()).model(),
          })).map((model) => {
            return {
              id: model.id,
              name: model.id,
              provider: 'browser-local-transformers',
              description: '',
              contextLength: 0,
              deprecated: false,
            } satisfies ModelInfo
          })
        },
      },
      validators: {
        validateProviderConfig: (config) => {
          if (!config.baseUrl) {
            return {
              errors: [new Error('Base URL is required.')],
              reason: 'Base URL is required. This is likely a bug, report to developers on https://github.com/moeru-ai/airi/issues.',
              valid: false,
            }
          }

          return {
            errors: [],
            reason: '',
            valid: true,
          }
        },
      },
    },
    'ollama': {
      id: 'ollama',
      category: 'chat',
      tasks: ['text-generation'],
      nameKey: 'settings.pages.providers.provider.ollama.title',
      name: 'Ollama',
      descriptionKey: 'settings.pages.providers.provider.ollama.description',
      description: 'ollama.com',
      icon: 'i-lobe-icons:ollama',
      defaultOptions: () => ({
        baseUrl: 'http://localhost:11434/v1/',
      }),
      createProvider: async config => createOllama((config.baseUrl as string).trim()),
      capabilities: {
        listModels: async (config) => {
          return (await listModels({
            ...createOllama((config.baseUrl as string).trim()).model(),
          })).map((model) => {
            return {
              id: model.id,
              name: model.id,
              provider: 'ollama',
              description: '',
              contextLength: 0,
              deprecated: false,
            } satisfies ModelInfo
          })
        },
      },
      validators: {
        validateProviderConfig: (config) => {
          if (!config.baseUrl) {
            return {
              errors: [new Error('Base URL is required.')],
              reason: 'Base URL is required. Default to http://localhost:11434/v1/ for Ollama.',
              valid: false,
            }
          }

          if (!isAbsoluteUrl(config.baseUrl as string)) {
            return notBaseUrlError.value
          }

          // Check if the Ollama server is reachable
          return fetch(`${(config.baseUrl as string).trim()}models`, { headers: (config.headers as HeadersInit) || undefined })
            .then((response) => {
              const errors = [
                !response.ok && new Error(`Ollama server returned non-ok status code: ${response.statusText}`),
              ].filter(Boolean)

              return {
                errors,
                reason: errors.filter(e => e).map(e => String(e)).join(', ') || '',
                valid: response.ok,
              }
            })
            .catch((err) => {
              return {
                errors: [err],
                reason: `Failed to reach Ollama server, error: ${String(err)} occurred.\n\nIf you are using Ollama locally, this is likely the CORS (Cross-Origin Resource Sharing) security issue, where you will need to set OLLAMA_ORIGINS=* or OLLAMA_ORIGINS=https://airi.moeru.ai,http://localhost environment variable before launching Ollama server to make this work.`,
                valid: false,
              }
            })
        },
      },
    },
    'ollama-embedding': {
      id: 'ollama-embedding',
      category: 'embed',
      tasks: ['text-feature-extraction'],
      nameKey: 'settings.pages.providers.provider.ollama.title',
      name: 'Ollama',
      descriptionKey: 'settings.pages.providers.provider.ollama.description',
      description: 'ollama.com',
      icon: 'i-lobe-icons:ollama',
      defaultOptions: () => ({
        baseUrl: 'http://localhost:11434/v1/',
      }),
      createProvider: async config => createOllama((config.baseUrl as string).trim()),
      capabilities: {
        listModels: async (config) => {
          return (await listModels({
            ...createOllama((config.baseUrl as string).trim()).model(),
          })).map((model) => {
            return {
              id: model.id,
              name: model.id,
              provider: 'ollama',
              description: '',
              contextLength: 0,
              deprecated: false,
            } satisfies ModelInfo
          })
        },
      },
      validators: {
        validateProviderConfig: (config) => {
          if (!config.baseUrl) {
            return {
              errors: [new Error('Base URL is required.')],
              reason: 'Base URL is required. Default to http://localhost:11434/v1/ for Ollama.',
              valid: false,
            }
          }

          if (!isAbsoluteUrl(config.baseUrl as string)) {
            return notBaseUrlError.value
          }

          // Check if the Ollama server is reachable
          return fetch(`${(config.baseUrl as string).trim()}models`, { headers: (config.headers as HeadersInit) || undefined })
            .then((response) => {
              const errors = [
                !response.ok && new Error(`Ollama server returned non-ok status code: ${response.statusText}`),
              ].filter(Boolean)

              return {
                errors,
                reason: errors.filter(e => e).map(e => String(e)).join(', ') || '',
                valid: response.ok,
              }
            })
            .catch((err) => {
              return {
                errors: [err],
                reason: `Failed to reach Ollama server, error: ${String(err)} occurred.\n\nIf you are using Ollama locally, this is likely the CORS (Cross-Origin Resource Sharing) security issue, where you will need to set OLLAMA_ORIGINS=* or OLLAMA_ORIGINS=https://airi.moeru.ai,http://localhost environment variable before launching Ollama server to make this work.`,
                valid: false,
              }
            })
        },
      },
    },
    'vllm': {
      id: 'vllm',
      category: 'chat',
      tasks: ['text-generation'],
      nameKey: 'settings.pages.providers.provider.vllm.title',
      name: 'vLLM',
      descriptionKey: 'settings.pages.providers.provider.vllm.description',
      description: 'vllm.ai',
      iconColor: 'i-lobe-icons:vllm',
      createProvider: async config => createOllama((config.baseUrl as string).trim()),
      capabilities: {
        listModels: async () => {
          return [
            {
              id: 'llama-2-7b',
              name: 'Llama 2 (7B)',
              provider: 'vllm',
              description: 'Meta\'s Llama 2 7B parameter model',
              contextLength: 4096,
            },
            {
              id: 'llama-2-13b',
              name: 'Llama 2 (13B)',
              provider: 'vllm',
              description: 'Meta\'s Llama 2 13B parameter model',
              contextLength: 4096,
            },
            {
              id: 'llama-2-70b',
              name: 'Llama 2 (70B)',
              provider: 'vllm',
              description: 'Meta\'s Llama 2 70B parameter model',
              contextLength: 4096,
            },
            {
              id: 'mistral-7b',
              name: 'Mistral (7B)',
              provider: 'vllm',
              description: 'Mistral AI\'s 7B parameter model',
              contextLength: 8192,
            },
            {
              id: 'mixtral-8x7b',
              name: 'Mixtral (8x7B)',
              provider: 'vllm',
              description: 'Mistral AI\'s Mixtral 8x7B MoE model',
              contextLength: 32768,
            },
            {
              id: 'custom',
              name: 'Custom Model',
              provider: 'vllm',
              description: 'Specify a custom model name',
              contextLength: 0,
            },
          ]
        },
      },
      validators: {
        validateProviderConfig: (config) => {
          if (!config.baseUrl) {
            return {
              errors: [new Error('Base URL is required.')],
              reason: 'Base URL is required. Default to http://localhost:8000/v1/ for vLLM.',
              valid: false,
            }
          }

          if (!isAbsoluteUrl(config.baseUrl as string)) {
            return notBaseUrlError.value
          }

          // Check if the vLLM is reachable
          return fetch(`${(config.baseUrl as string).trim()}models`, { headers: (config.headers as HeadersInit) || undefined })
            .then((response) => {
              const errors = [
                !response.ok && new Error(`vLLM returned non-ok status code: ${response.statusText}`),
              ].filter(Boolean)

              return {
                errors,
                reason: errors.filter(e => e).map(e => String(e)).join(', ') || '',
                valid: response.ok,
              }
            })
            .catch((err) => {
              return {
                errors: [err],
                reason: `Failed to reach vLLM, error: ${String(err)} occurred.`,
                valid: false,
              }
            })
        },
      },
    },
    'lm-studio': {
      id: 'lm-studio',
      category: 'chat',
      tasks: ['text-generation'],
      nameKey: 'settings.pages.providers.provider.lm-studio.title',
      name: 'LM Studio',
      descriptionKey: 'settings.pages.providers.provider.lm-studio.description',
      description: 'lmstudio.ai',
      icon: 'i-lobe-icons:lmstudio',
      defaultOptions: () => ({
        baseUrl: 'http://localhost:1234/v1/',
      }),
      createProvider: async config => createOpenAI('', (config.baseUrl as string).trim()),
      capabilities: {
        listModels: async (config) => {
          try {
            const response = await fetch(`${(config.baseUrl as string).trim()}models`, {
              headers: (config.headers as HeadersInit) || undefined,
            })

            if (!response.ok) {
              throw new Error(`LM Studio server returned non-ok status code: ${response.statusText}`)
            }

            const data = await response.json()
            return data.data.map((model: any) => ({
              id: model.id,
              name: model.id,
              provider: 'lm-studio',
              description: model.description || '',
              contextLength: model.context_length || 0,
              deprecated: false,
            })) satisfies ModelInfo[]
          }
          catch (error) {
            console.error('Error fetching LM Studio models:', error)
            return []
          }
        },
      },
      validators: {
        validateProviderConfig: (config) => {
          if (!config.baseUrl) {
            return {
              errors: [new Error('Base URL is required.')],
              reason: 'Base URL is required. Default to http://localhost:1234/v1/ for LM Studio.',
              valid: false,
            }
          }

          // Check if the LM Studio server is reachable
          return fetch(`${(config.baseUrl as string).trim()}models`, { headers: (config.headers as HeadersInit) || undefined })
            .then((response) => {
              const errors = [
                !response.ok && new Error(`LM Studio server returned non-ok status code: ${response.statusText}`),
              ].filter(Boolean)

              return {
                errors,
                reason: errors.filter(e => e).map(e => String(e)).join(', ') || '',
                valid: response.ok,
              }
            })
            .catch((err) => {
              return {
                errors: [err],
                reason: `Failed to reach LM Studio server, error: ${String(err)} occurred.\n\nMake sure LM Studio is running and the local server is started. You can start the local server in LM Studio by going to the 'Local Server' tab and clicking 'Start Server'.`,
                valid: false,
              }
            })
        },
      },
    },
    'openai': {
      id: 'openai',
      category: 'chat',
      tasks: ['text-generation'],
      nameKey: 'settings.pages.providers.provider.openai.title',
      name: 'OpenAI',
      descriptionKey: 'settings.pages.providers.provider.openai.description',
      description: 'openai.com',
      icon: 'i-lobe-icons:openai',
      defaultOptions: () => ({
        baseUrl: 'https://api.openai.com/v1/',
      }),
      createProvider: async config => createOpenAI((config.apiKey as string).trim(), (config.baseUrl as string).trim()),
      capabilities: {
        listModels: async (config) => {
          return (await listModels({
            ...createOpenAI((config.apiKey as string).trim(), (config.baseUrl as string).trim()).model(),
          })).map((model) => {
            return {
              id: model.id,
              name: model.id,
              provider: 'openai',
              description: '',
              contextLength: 0,
              deprecated: false,
            } satisfies ModelInfo
          })
        },
      },
      validators: {
        validateProviderConfig: (config) => {
          const errors = [
            !config.baseUrl && new Error('Base URL is required. Default to https://api.openai.com/v1/ for official OpenAI API.'),
          ].filter(Boolean)

          if (!!config.baseUrl && !isAbsoluteUrl(config.baseUrl as string)) {
            return notBaseUrlError.value
          }

          return {
            errors,
            reason: errors.filter(e => e).map(e => String(e)).join(', ') || '',
            valid: !!config.baseUrl,
          }
        },
      },
    },
    'openai-compatible': {
      id: 'openai-compatible',
      category: 'chat',
      tasks: ['text-generation'],
      nameKey: 'settings.pages.providers.provider.openai-compatible.title',
      name: 'OpenAI Compatible',
      descriptionKey: 'settings.pages.providers.provider.openai-compatible.description',
      description: 'Connect to any API that follows the OpenAI specification.',
      icon: 'i-lobe-icons:openai',
      defaultOptions: () => ({
        baseUrl: '',
      }),
      createProvider: async config => createOpenAI((config.apiKey as string).trim(), (config.baseUrl as string).trim()),
      capabilities: {
        listModels: async (config) => {
          return (await listModels({
            ...createOpenAI((config.apiKey as string).trim(), (config.baseUrl as string).trim()).model(),
          })).map((model) => {
            return {
              id: model.id,
              name: model.id,
              provider: 'openai-compatible',
              description: '',
              contextLength: 0,
              deprecated: false,
            } satisfies ModelInfo
          })
        },
      },
      validators: {
        validateProviderConfig: (config) => {
          const errors = [
            !config.apiKey && new Error('API key is required'),
            !config.baseUrl && new Error('Base URL is required'),
          ].filter(Boolean)

          if (!!config.baseUrl && !isAbsoluteUrl(config.baseUrl as string)) {
            return notBaseUrlError.value
          }

          return {
            errors,
            reason: errors.filter(e => e).map(e => String(e)).join(', ') || '',
            valid: !!config.apiKey && !!config.baseUrl,
          }
        },
      },
    },
    'openai-audio-speech': {
      id: 'openai-audio-speech',
      category: 'speech',
      tasks: ['text-to-speech'],
      nameKey: 'settings.pages.providers.provider.openai.title',
      name: 'OpenAI',
      descriptionKey: 'settings.pages.providers.provider.openai.description',
      description: 'openai.com',
      icon: 'i-lobe-icons:openai',
      defaultOptions: () => ({
        baseUrl: 'https://api.openai.com/v1/',
      }),
      createProvider: async config => createOpenAI((config.apiKey as string).trim(), (config.baseUrl as string).trim()),
      capabilities: {
        listModels: async (config) => {
          return (await listModels({
            ...createOpenAI((config.apiKey as string).trim(), (config.baseUrl as string).trim()).model(),
          })).map((model) => {
            return {
              id: model.id,
              name: model.id,
              provider: 'openai',
              description: '',
              contextLength: 0,
              deprecated: false,
            } satisfies ModelInfo
          })
        },
        listVoices: async () => {
          return [
            {
              id: 'alloy',
              name: 'Alloy',
              provider: 'openai-audio-speech',
              languages: [],
            },
            {
              id: 'ash',
              name: 'Ash',
              provider: 'openai-audio-speech',
              languages: [],
            },
            {
              id: 'ballad',
              name: 'Ballad',
              provider: 'openai-audio-speech',
              languages: [],
            },
            {
              id: 'coral',
              name: 'Coral',
              provider: 'openai-audio-speech',
              languages: [],
            },
            {
              id: 'echo',
              name: 'Echo',
              provider: 'openai-audio-speech',
              languages: [],
            },
            {
              id: 'fable',
              name: 'Fable',
              provider: 'openai-audio-speech',
              languages: [],
            },
            {
              id: 'onyx',
              name: 'Onyx',
              provider: 'openai-audio-speech',
              languages: [],
            },
            {
              id: 'nova',
              name: 'Nova',
              provider: 'openai-audio-speech',
              languages: [],
            },
            {
              id: 'sage',
              name: 'Sage',
              provider: 'openai-audio-speech',
              languages: [],
            },
            {
              id: 'shimmer',
              name: 'Shimmer',
              provider: 'openai-audio-speech',
              languages: [],
            },
            {
              id: 'verse',
              name: 'Verse',
              provider: 'openai-audio-speech',
              languages: [],
            },
          ] satisfies VoiceInfo[]
        },
      },
      validators: {
        validateProviderConfig: (config) => {
          const errors = [
            !config.baseUrl && new Error('Base URL is required. Default to https://api.openai.com/v1/ for official OpenAI API.'),
          ].filter(Boolean)

          if (!!config.baseUrl && !isAbsoluteUrl(config.baseUrl as string)) {
            return notBaseUrlError.value
          }

          return {
            errors,
            reason: errors.filter(e => e).map(e => String(e)).join(', ') || '',
            valid: !!config.baseUrl,
          }
        },
      },
    },
    'openai-compatible-audio-speech': {
      id: 'openai-compatible-audio-speech',
      category: 'speech',
      tasks: ['text-to-speech'],
      nameKey: 'settings.pages.providers.provider.openai-compatible.title',
      name: 'OpenAI Compatible',
      descriptionKey: 'settings.pages.providers.provider.openai-compatible.description',
      description: 'Connect to any API that follows the OpenAI specification.',
      icon: 'i-lobe-icons:openai',
      defaultOptions: () => ({
        baseUrl: '',
      }),
      createProvider: async config => createOpenAI((config.apiKey as string).trim(), (config.baseUrl as string).trim()),
      capabilities: {
        listModels: async (config) => {
          return (await listModels({
            ...createOpenAI((config.apiKey as string).trim(), (config.baseUrl as string).trim()).model(),
          })).map((model) => {
            return {
              id: model.id,
              name: model.id,
              provider: 'openai-compatible-audio-speech',
              description: '',
              contextLength: 0,
              deprecated: false,
            } satisfies ModelInfo
          })
        },
        listVoices: async () => {
          return []
        },
      },
      validators: {
        validateProviderConfig: (config) => {
          const errors = [
            !config.apiKey && new Error('API key is required'),
            !config.baseUrl && new Error('Base URL is required'),
          ].filter(Boolean)

          if (!!config.baseUrl && !isAbsoluteUrl(config.baseUrl as string)) {
            return notBaseUrlError.value
          }

          return {
            errors,
            reason: errors.filter(e => e).map(e => String(e)).join(', ') || '',
            valid: !!config.apiKey && !!config.baseUrl,
          }
        },
      },
    },
    'openai-audio-transcription': {
      id: 'openai-audio-transcription',
      category: 'transcription',
      tasks: ['speech-to-text', 'automatic-speech-recognition', 'asr', 'stt'],
      nameKey: 'settings.pages.providers.provider.openai.title',
      name: 'OpenAI',
      descriptionKey: 'settings.pages.providers.provider.openai.description',
      description: 'openai.com',
      icon: 'i-lobe-icons:openai',
      defaultOptions: () => ({
        baseUrl: 'https://api.openai.com/v1/',
      }),
      createProvider: async config => createOpenAI((config.apiKey as string).trim(), (config.baseUrl as string).trim()),
      capabilities: {
        listModels: async (config) => {
          return (await listModels({
            ...createOpenAI((config.apiKey as string).trim(), (config.baseUrl as string).trim()).model(),
          })).map((model) => {
            return {
              id: model.id,
              name: model.id,
              provider: 'openai',
              description: '',
              contextLength: 0,
              deprecated: false,
            } satisfies ModelInfo
          })
        },
      },
      validators: {
        validateProviderConfig: (config) => {
          const errors = [
            !config.baseUrl && new Error('Base URL is required. Default to https://api.openai.com/v1/ for official OpenAI API.'),
          ].filter(Boolean)

          if (!!config.baseUrl && !isAbsoluteUrl(config.baseUrl as string)) {
            return notBaseUrlError.value
          }

          return {
            errors,
            reason: errors.filter(e => e).map(e => String(e)).join(', ') || '',
            valid: !!config.baseUrl,
          }
        },
      },
    },
    'openai-compatible-audio-transcription': {
      id: 'openai-compatible-audio-transcription',
      category: 'transcription',
      tasks: ['speech-to-text', 'automatic-speech-recognition', 'asr', 'stt'],
      nameKey: 'settings.pages.providers.provider.openai-compatible.title',
      name: 'OpenAI Compatible',
      descriptionKey: 'settings.pages.providers.provider.openai-compatible.description',
      description: 'Connect to any API that follows the OpenAI specification.',
      icon: 'i-lobe-icons:openai',
      defaultOptions: () => ({
        baseUrl: '',
      }),
      createProvider: async config => createOpenAI((config.apiKey as string).trim(), (config.baseUrl as string).trim()),
      capabilities: {
        listModels: async (config) => {
          return (await listModels({
            ...createOpenAI((config.apiKey as string).trim(), (config.baseUrl as string).trim()).model(),
          })).map((model) => {
            return {
              id: model.id,
              name: model.id,
              provider: 'openai-compatible-audio-transcription',
              description: '',
              contextLength: 0,
              deprecated: false,
            } satisfies ModelInfo
          })
        },
      },
      validators: {
        validateProviderConfig: (config) => {
          const errors = [
            !config.apiKey && new Error('API key is required'),
            !config.baseUrl && new Error('Base URL is required'),
          ].filter(Boolean)

          if (!!config.baseUrl && !isAbsoluteUrl(config.baseUrl as string)) {
            return notBaseUrlError.value
          }

          return {
            errors,
            reason: errors.filter(e => e).map(e => String(e)).join(', ') || '',
            valid: !!config.apiKey && !!config.baseUrl,
          }
        },
      },
    },
    'azure-ai-foundry': {
      id: 'azure-ai-foundry',
      category: 'chat',
      tasks: ['text-generation'],
      nameKey: 'settings.pages.providers.provider.azure_ai_foundry.title',
      name: 'Azure AI Foundry',
      descriptionKey: 'settings.pages.providers.provider.azure_ai_foundry.description',
      description: 'azure.com',
      icon: 'i-lobe-icons:microsoft',
      defaultOptions: () => ({}),
      createProvider: async (config) => {
        return await createAzure({
          apiKey: async () => (config.apiKey as string).trim(),
          resourceName: config.resourceName as string,
          apiVersion: config.apiVersion as string,
        })
      },
      capabilities: {
        listModels: async (config) => {
          return [{ id: config.modelId }].map((model) => {
            return {
              id: model.id as string,
              name: model.id as string,
              provider: 'azure-ai-foundry',
              description: '',
              contextLength: 0,
              deprecated: false,
            } satisfies ModelInfo
          })
        },
      },
      validators: {
        validateProviderConfig: (config) => {
          // return !!config.apiKey && !!config.resourceName && !!config.modelId

          const errors = [
            !config.apiKey && new Error('API key is required'),
            !config.resourceName && new Error('Resource name is required'),
            !config.modelId && new Error('Model ID is required'),
          ]

          return {
            errors,
            reason: errors.filter(e => e).map(e => String(e)).join(', ') || '',
            valid: !!config.apiKey && !!config.resourceName && !!config.modelId,
          }
        },
      },
    },
    'anthropic': {
      id: 'anthropic',
      category: 'chat',
      tasks: ['text-generation'],
      nameKey: 'settings.pages.providers.provider.anthropic.title',
      name: 'Anthropic',
      descriptionKey: 'settings.pages.providers.provider.anthropic.description',
      description: 'anthropic.com',
      icon: 'i-lobe-icons:anthropic',
      defaultOptions: () => ({
        baseUrl: 'https://api.anthropic.com/v1/',
      }),
      createProvider: async config => createAnthropic((config.apiKey as string).trim(), (config.baseUrl as string).trim()),
      capabilities: {
        listModels: async () => {
          return [
            {
              id: 'claude-3-7-sonnet-20250219',
              name: 'Claude 3.7 Sonnet',
              provider: 'anthropic',
              description: '',
              contextLength: 0,
              deprecated: false,
            },
            {
              id: 'claude-3-5-sonnet-20241022',
              name: 'Claude 3.5 Sonnet (New)',
              provider: 'anthropic',
              description: '',
              contextLength: 0,
              deprecated: false,
            },
            {
              id: 'claude-3-5-haiku-20241022',
              name: 'Claude 3.5 Haiku',
              provider: 'anthropic',
              description: '',
              contextLength: 0,
              deprecated: false,
            },
            {
              id: 'claude-3-5-sonnet-20240620',
              name: 'Claude 3.5 Sonnet (Old)',
              provider: 'anthropic',
              description: '',
              contextLength: 0,
              deprecated: false,
            },
            {
              id: 'claude-3-haiku-20240307',
              name: 'Claude 3 Haiku',
              provider: 'anthropic',
              description: '',
              contextLength: 0,
              deprecated: false,
            },
            {
              id: 'claude-3-opus-20240229',
              name: 'Claude 3 Opus',
              provider: 'anthropic',
              description: '',
              contextLength: 0,
              deprecated: false,
            },
          ] satisfies ModelInfo[]
        },
      },
      validators: {
        validateProviderConfig: (config) => {
          const errors = [
            !config.apiKey && new Error('API key is required.'),
            !config.baseUrl && new Error('Base URL is required. Default to https://api.anthropic.com/v1/ for official Claude API with OpenAI compatibility.'),
          ].filter(Boolean)

          if (!!config.baseUrl && !isAbsoluteUrl(config.baseUrl as string)) {
            return notBaseUrlError.value
          }

          return {
            errors,
            reason: errors.filter(e => e).map(e => String(e)).join(', ') || '',
            valid: !!config.apiKey && !!config.baseUrl,
          }
        },
      },
    },
    'google-generative-ai': {
      id: 'google-generative-ai',
      category: 'chat',
      tasks: ['text-generation'],
      nameKey: 'settings.pages.providers.provider.google-generative-ai.title',
      name: 'Google Gemini',
      descriptionKey: 'settings.pages.providers.provider.google-generative-ai.description',
      description: 'ai.google.dev',
      icon: 'i-lobe-icons:gemini',
      defaultOptions: () => ({
        baseUrl: 'https://generativelanguage.googleapis.com/v1beta/openai/',
      }),
      createProvider: async config => createGoogleGenerativeAI((config.apiKey as string).trim(), (config.baseUrl as string).trim()),
      capabilities: {
        listModels: async (config) => {
          return (await listModels({
            ...createGoogleGenerativeAI((config.apiKey as string).trim(), (config.baseUrl as string).trim()).model(),
          })).map((model) => {
            return {
              id: model.id,
              name: model.id,
              provider: 'google-generative-ai',
              description: '',
              contextLength: 0,
              deprecated: false,
            } satisfies ModelInfo
          })
        },
      },
      validators: {
        validateProviderConfig: (config) => {
          const errors = [
            !config.apiKey && new Error('API key is required.'),
            !config.baseUrl && new Error('Base URL is required. Default to https://generativelanguage.googleapis.com/v1beta/openai/ for official Google Gemini API with OpenAI compatibility.'),
          ].filter(Boolean)

          if (!!config.baseUrl && !isAbsoluteUrl(config.baseUrl as string)) {
            return notBaseUrlError.value
          }

          return {
            errors,
            reason: errors.filter(e => e).map(e => String(e)).join(', ') || '',
            valid: !!config.apiKey && !!config.baseUrl,
          }
        },
      },
    },
    'xai': {
      id: 'xai',
      category: 'chat',
      tasks: ['text-generation'],
      nameKey: 'settings.pages.providers.provider.xai.title',
      name: 'xAI',
      descriptionKey: 'settings.pages.providers.provider.xai.description',
      description: 'x.ai',
      icon: 'i-lobe-icons:xai',
      createProvider: async config => createXAI((config.apiKey as string).trim(), (config.baseUrl as string).trim()),
      capabilities: {
        listModels: async (config) => {
          return (await listModels({
            ...createXAI((config.apiKey as string).trim(), (config.baseUrl as string).trim()).model(),
          })).map((model) => {
            return {
              id: model.id,
              name: model.id,
              provider: 'xai',
              description: '',
              contextLength: 0,
              deprecated: false,
            } satisfies ModelInfo
          })
        },
      },
      validators: {
        validateProviderConfig: (config) => {
          const errors = [
            !config.apiKey && new Error('API key is required.'),
            !config.baseUrl && new Error('Base URL is required.'),
          ].filter(Boolean)

          return {
            errors,
            reason: errors.filter(e => e).map(e => String(e)).join(', ') || '',
            valid: !!config.apiKey && !!config.baseUrl,
          }
        },
      },
    },
    'deepseek': {
      id: 'deepseek',
      category: 'chat',
      tasks: ['text-generation'],
      nameKey: 'settings.pages.providers.provider.deepseek.title',
      name: 'DeepSeek',
      descriptionKey: 'settings.pages.providers.provider.deepseek.description',
      description: 'deepseek.com',
      iconColor: 'i-lobe-icons:deepseek',
      defaultOptions: () => ({
        baseUrl: 'https://api.deepseek.com/',
      }),
      createProvider: async config => createDeepSeek((config.apiKey as string).trim(), (config.baseUrl as string).trim()),
      capabilities: {
        listModels: async (config) => {
          return (await listModels({
            ...createDeepSeek((config.apiKey as string).trim(), (config.baseUrl as string).trim()).model(),
          })).map((model) => {
            return {
              id: model.id,
              name: model.id,
              provider: 'deepseek',
              description: '',
              contextLength: 0,
              deprecated: false,
            } satisfies ModelInfo
          })
        },
      },
      validators: {
        validateProviderConfig: (config) => {
          const errors = [
            !config.apiKey && new Error('API key is required.'),
            !config.baseUrl && new Error('Base URL is required.'),
          ].filter(Boolean)

          if (!!config.baseUrl && !isAbsoluteUrl(config.baseUrl as string)) {
            return notBaseUrlError.value
          }

          return {
            errors,
            reason: errors.filter(e => e).map(e => String(e)).join(', ') || '',
            valid: !!config.apiKey && !!config.baseUrl,
          }
        },
      },
    },
    'elevenlabs': {
      id: 'elevenlabs',
      category: 'speech',
      tasks: ['text-to-speech'],
      nameKey: 'settings.pages.providers.provider.elevenlabs.title',
      name: 'ElevenLabs',
      descriptionKey: 'settings.pages.providers.provider.elevenlabs.description',
      description: 'elevenlabs.io',
      icon: 'i-simple-icons:elevenlabs',
      defaultOptions: () => ({
        baseUrl: 'https://unspeech.hyp3r.link/v1/',
        voiceSettings: {
          similarityBoost: 0.75,
          stability: 0.5,
        },
      }),
      createProvider: async config => createUnElevenLabs((config.apiKey as string).trim(), (config.baseUrl as string).trim()) as SpeechProviderWithExtraOptions<string, UnElevenLabsOptions>,
      capabilities: {
        listModels: async () => {
          return elevenLabsModels.map((model) => {
            return {
              id: model.model_id,
              name: model.name,
              provider: 'elevenlabs',
              description: model.description,
              contextLength: 0,
              deprecated: false,
            } satisfies ModelInfo
          })
        },
        listVoices: async (config) => {
          const provider = createUnElevenLabs((config.apiKey as string).trim(), (config.baseUrl as string).trim()) as VoiceProviderWithExtraOptions<UnElevenLabsOptions>

          const voices = await listVoices({
            ...provider.voice(),
          })

          // Find indices of Aria and Bill
          const ariaIndex = voices.findIndex(voice => voice.name.includes('Aria'))
          const billIndex = voices.findIndex(voice => voice.name.includes('Bill'))

          // Determine the range to move (ensure valid indices and proper order)
          const startIndex = ariaIndex !== -1 ? ariaIndex : 0
          const endIndex = billIndex !== -1 ? billIndex : voices.length - 1
          const lowerIndex = Math.min(startIndex, endIndex)
          const higherIndex = Math.max(startIndex, endIndex)

          // Rearrange voices: voices outside the range first, then voices within the range
          const rearrangedVoices = [
            ...voices.slice(0, lowerIndex),
            ...voices.slice(higherIndex + 1),
            ...voices.slice(lowerIndex, higherIndex + 1),
          ]

          return rearrangedVoices.map((voice) => {
            return {
              id: voice.id,
              name: voice.name,
              provider: 'elevenlabs',
              previewURL: voice.preview_audio_url,
              languages: voice.languages,
            }
          })
        },
      },
      validators: {
        validateProviderConfig: (config) => {
          const errors = [
            !config.apiKey && new Error('API key is required.'),
            !config.baseUrl && new Error('Base URL is required.'),
          ].filter(Boolean)

          if (!!config.baseUrl && !isAbsoluteUrl(config.baseUrl as string)) {
            return notBaseUrlError.value
          }

          return {
            errors,
            reason: errors.filter(e => e).map(e => String(e)).join(', ') || '',
            valid: !!config.apiKey && !!config.baseUrl,
          }
        },
      },
    },
    'microsoft-speech': {
      id: 'microsoft-speech',
      category: 'speech',
      tasks: ['text-to-speech'],
      nameKey: 'settings.pages.providers.provider.microsoft-speech.title',
      name: 'Microsoft / Azure Speech',
      descriptionKey: 'settings.pages.providers.provider.microsoft-speech.description',
      description: 'speech.microsoft.com',
      iconColor: 'i-lobe-icons:microsoft',
      defaultOptions: () => ({
        baseUrl: 'https://unspeech.hyp3r.link/v1/',
      }),
      createProvider: async config => createUnMicrosoft((config.apiKey as string).trim(), (config.baseUrl as string).trim()) as SpeechProviderWithExtraOptions<string, UnMicrosoftOptions>,
      capabilities: {
        listModels: async () => {
          return [
            {
              id: 'v1',
              name: 'v1',
              provider: 'microsoft-speech',
              description: '',
              contextLength: 0,
              deprecated: false,
            },
          ]
        },
        listVoices: async (config) => {
          const provider = createUnMicrosoft((config.apiKey as string).trim(), (config.baseUrl as string).trim()) as VoiceProviderWithExtraOptions<UnMicrosoftOptions>

          const voices = await listVoices({
            ...provider.voice({ region: config.region as string }),
          })

          return voices.map((voice) => {
            return {
              id: voice.id,
              name: voice.name,
              provider: 'microsoft-speech',
              previewURL: voice.preview_audio_url,
              languages: voice.languages,
              gender: voice.labels?.gender,
            }
          })
        },
      },
      validators: {
        validateProviderConfig: (config) => {
          const errors = [
            !config.apiKey && new Error('API key is required.'),
            !config.baseUrl && new Error('Base URL is required.'),
          ].filter(Boolean)

          if (!!config.baseUrl && !isAbsoluteUrl(config.baseUrl as string)) {
            return notBaseUrlError.value
          }

          return {
            errors,
            reason: errors.filter(e => e).map(e => String(e)).join(', ') || '',
            valid: !!config.apiKey && !!config.baseUrl,
          }
        },
      },
    },
    'index-tts-vllm': {
      id: 'index-tts-vllm',
      category: 'speech',
      tasks: ['text-to-speech'],
      nameKey: 'settings.pages.providers.provider.index-tts-vllm.title',
      name: 'Index-TTS by Bilibili',
      descriptionKey: 'settings.pages.providers.provider.index-tts-vllm.description',
      description: 'index-tts.github.io',
      iconColor: 'i-lobe-icons:bilibiliindex',
      defaultOptions: () => ({
        baseUrl: 'http://localhost:11996/tts',
      }),
      createProvider: async (config) => {
        const provider: SpeechProvider = {
          speech: () => {
            const req = {
              baseURL: config.baseUrl as string,
              model: 'IndexTTS-1.5',
            }
            return req
          },
        }
        return provider
      },
      capabilities: {
        listVoices: async (config) => {
          const voicesUrl = config.baseUrl as string
          const response = await fetch(`${voicesUrl}/audio/voices`)
          if (!response.ok) {
            throw new Error(`Failed to fetch voices: ${response.statusText}`)
          }
          const voices = await response.json()
          return Object.keys(voices).map((voice: any) => {
            return {
              id: voice,
              name: voice,
              provider: 'index-tts-vllm',
              // previewURL: voice.preview_audio_url,
              languages: [{ code: 'cn', title: 'Chinese' }, { code: 'en', title: 'English' }],
            }
          })
        },
      },
      validators: {
        validateProviderConfig: (config) => {
          const errors = [
            !config.baseUrl && new Error('Base URL is required. Default to http://localhost:11996/tts for Index-TTS.'),
          ].filter(Boolean)

          if (!!config.baseUrl && !isAbsoluteUrl(config.baseUrl as string)) {
            return notBaseUrlError.value
          }

          return {
            errors,
            reason: errors.filter(e => e).map(e => String(e)).join(', ') || '',
            valid: !!config.baseUrl,
          }
        },
      },
    },
    'alibaba-cloud-model-studio': {
      id: 'alibaba-cloud-model-studio',
      category: 'speech',
      tasks: ['text-to-speech'],
      nameKey: 'settings.pages.providers.provider.alibaba-cloud-model-studio.title',
      name: 'Alibaba Cloud Model Studio',
      descriptionKey: 'settings.pages.providers.provider.alibaba-cloud-model-studio.description',
      description: 'bailian.console.aliyun.com',
      iconColor: 'i-lobe-icons:alibabacloud',
      defaultOptions: () => ({
        baseUrl: 'https://unspeech.hyp3r.link/v1/',
      }),
      createProvider: async config => createUnAlibabaCloud((config.apiKey as string).trim(), (config.baseUrl as string).trim()),
      capabilities: {
        listVoices: async (config) => {
          const provider = createUnAlibabaCloud((config.apiKey as string).trim(), (config.baseUrl as string).trim()) as VoiceProviderWithExtraOptions<UnAlibabaCloudOptions>

          const voices = await listVoices({
            ...provider.voice(),
          })

          return voices.map((voice) => {
            return {
              id: voice.id,
              name: voice.name,
              provider: 'alibaba-cloud-model-studio',
              previewURL: voice.preview_audio_url,
              languages: voice.languages,
              gender: voice.labels?.gender,
            }
          })
        },
        listModels: async () => {
          return [
            {
              id: 'cozyvoice-v1',
              name: 'CozyVoice',
              provider: 'alibaba-cloud-model-studio',
              description: '',
              contextLength: 0,
              deprecated: false,
            },
            {
              id: 'cozyvoice-v2',
              name: 'CozyVoice (New)',
              provider: 'alibaba-cloud-model-studio',
              description: '',
              contextLength: 0,
              deprecated: false,
            },
          ]
        },
      },
      validators: {
        validateProviderConfig: (config) => {
          const errors = [
            !config.apiKey && new Error('API key is required.'),
            !config.baseUrl && new Error('Base URL is required.'),
          ].filter(Boolean)

          if (!!config.baseUrl && !isAbsoluteUrl(config.baseUrl as string)) {
            return notBaseUrlError.value
          }

          return {
            errors,
            reason: errors.filter(e => e).map(e => String(e)).join(', ') || '',
            valid: !!config.apiKey && !!config.baseUrl,
          }
        },
      },
    },
    'volcengine': {
      id: 'volcengine',
      category: 'speech',
      tasks: ['text-to-speech'],
      nameKey: 'settings.pages.providers.provider.volcengine.title',
      name: 'settings.pages.providers.provider.volcengine.title',
      descriptionKey: 'settings.pages.providers.provider.volcengine.description',
      description: 'volcengine.com',
      iconColor: 'i-lobe-icons:volcengine',
      defaultOptions: () => ({
        baseUrl: 'https://unspeech.hyp3r.link/v1/',
      }),
      createProvider: async config => createUnVolcengine((config.apiKey as string).trim(), (config.baseUrl as string).trim()),
      capabilities: {
        listVoices: async (config) => {
          const provider = createUnVolcengine((config.apiKey as string).trim(), (config.baseUrl as string).trim()) as VoiceProviderWithExtraOptions<UnVolcengineOptions>

          const voices = await listVoices({
            ...provider.voice(),
          })

          return voices.map((voice) => {
            return {
              id: voice.id,
              name: voice.name,
              provider: 'volcano-engine',
              previewURL: voice.preview_audio_url,
              languages: voice.languages,
              gender: voice.labels?.gender,
            }
          })
        },
        listModels: async () => {
          return [
            {
              id: 'v1',
              name: 'v1',
              provider: 'volcano-engine',
              description: '',
              contextLength: 0,
              deprecated: false,
            },
          ]
        },
      },
      validators: {
        validateProviderConfig: (config) => {
          const errors = [
            !config.apiKey && new Error('API key is required.'),
            !config.baseUrl && new Error('Base URL is required.'),
            !((config.app as any)?.appId) && new Error('App ID is required.'),
          ].filter(Boolean)

          if (!!config.baseUrl && !isAbsoluteUrl(config.baseUrl as string)) {
            return notBaseUrlError.value
          }

          return {
            errors,
            reason: errors.filter(e => e).map(e => String(e)).join(', ') || '',
            valid: !!config.apiKey && !!config.baseUrl && !!config.app && !!(config.app as any).appId,
          }
        },
      },
    },
    'together-ai': {
      id: 'together-ai',
      category: 'chat',
      tasks: ['text-generation'],
      nameKey: 'settings.pages.providers.provider.together.title',
      name: 'Together.ai',
      descriptionKey: 'settings.pages.providers.provider.together.description',
      description: 'together.ai',
      iconColor: 'i-lobe-icons:together',
      createProvider: async config => createTogetherAI((config.apiKey as string).trim(), (config.baseUrl as string).trim()),
      capabilities: {
        listModels: async (config) => {
          return (await listModels({
            ...createTogetherAI((config.apiKey as string).trim(), (config.baseUrl as string).trim()).model(),
          })).map((model) => {
            return {
              id: model.id,
              name: model.id,
              provider: 'together-ai',
              description: '',
              contextLength: 0,
              deprecated: false,
            } satisfies ModelInfo
          })
        },
      },
      validators: {
        validateProviderConfig: (config) => {
          const errors = [
            !config.apiKey && new Error('API key is required.'),
            !config.baseUrl && new Error('Base URL is required.'),
          ].filter(Boolean)

          return {
            errors,
            reason: errors.filter(e => e).map(e => String(e)).join(', ') || '',
            valid: !!config.apiKey && !!config.baseUrl,
          }
        },
      },
    },
    'novita-ai': {
      id: 'novita-ai',
      category: 'chat',
      tasks: ['text-generation'],
      nameKey: 'settings.pages.providers.provider.novita.title',
      name: 'Novita',
      descriptionKey: 'settings.pages.providers.provider.novita.description',
      description: 'novita.ai',
      iconColor: 'i-lobe-icons:novita',
      createProvider: async config => createNovita((config.apiKey as string).trim(), (config.baseUrl as string).trim()),
      capabilities: {
        listModels: async (config) => {
          return (await listModels({
            ...createNovita((config.apiKey as string).trim(), (config.baseUrl as string).trim()).model(),
          })).map((model) => {
            return {
              id: model.id,
              name: model.id,
              provider: 'novita-ai',
              description: '',
              contextLength: 0,
              deprecated: false,
            } satisfies ModelInfo
          })
        },
      },
      validators: {
        validateProviderConfig: (config) => {
          const errors = [
            !config.apiKey && new Error('API key is required.'),
            !config.baseUrl && new Error('Base URL is required.'),
          ].filter(Boolean)

          return {
            errors,
            reason: errors.filter(e => e).map(e => String(e)).join(', ') || '',
            valid: !!config.apiKey && !!config.baseUrl,
          }
        },
      },
    },
    'fireworks-ai': {
      id: 'fireworks-ai',
      category: 'chat',
      tasks: ['text-generation'],
      nameKey: 'settings.pages.providers.provider.fireworks.title',
      name: 'Fireworks.ai',
      descriptionKey: 'settings.pages.providers.provider.fireworks.description',
      description: 'fireworks.ai',
      icon: 'i-lobe-icons:fireworks',
      createProvider: async config => createFireworks((config.apiKey as string).trim(), (config.baseUrl as string).trim()),
      capabilities: {
        listModels: async (config) => {
          return (await listModels({
            ...createFireworks((config.apiKey as string).trim(), (config.baseUrl as string).trim()).model(),
          })).map((model) => {
            return {
              id: model.id,
              name: model.id,
              provider: 'fireworks-ai',
              description: '',
              contextLength: 0,
              deprecated: false,
            } satisfies ModelInfo
          })
        },
      },
      validators: {
        validateProviderConfig: (config) => {
          const errors = [
            !config.apiKey && new Error('API key is required.'),
            !config.baseUrl && new Error('Base URL is required.'),
          ].filter(Boolean)

          return {
            errors,
            reason: errors.filter(e => e).map(e => String(e)).join(', ') || '',
            valid: !!config.apiKey && !!config.baseUrl,
          }
        },
      },
    },
    'featherless-ai': {
      id: 'featherless-ai',
      category: 'chat',
      tasks: ['text-generation'],
      nameKey: 'settings.pages.providers.provider.featherless.title',
      name: 'Featherless.ai',
      descriptionKey: 'settings.pages.providers.provider.featherless.description',
      description: 'featherless.ai',
      icon: 'i-lobe-icons:featherless-ai',
      defaultOptions: () => ({
        baseUrl: 'https://api.featherless.ai/v1/',
      }),
      createProvider: async config => createOpenAI((config.apiKey as string).trim(), (config.baseUrl as string).trim()),
      capabilities: {
        listModels: async (config) => {
          return (await listModels({
            ...createOpenAI((config.apiKey as string).trim(), (config.baseUrl as string).trim()).model(),
          })).map((model) => {
            return {
              id: model.id,
              name: model.id,
              provider: 'featherless-ai',
              description: '',
              contextLength: 0,
              deprecated: false,
            } satisfies ModelInfo
          })
        },
      },
      validators: {
        validateProviderConfig: (config) => {
          const errors = [
            !config.apiKey && new Error('API key is required.'),
            !config.baseUrl && new Error('Base URL is required.'),
          ].filter(Boolean)

          if (!!config.baseUrl && !isAbsoluteUrl(config.baseUrl as string)) {
            return notBaseUrlError.value
          }

          return {
            errors,
            reason: errors.filter(e => e).map(e => String(e)).join(', ') || '',
            valid: !!config.apiKey && !!config.baseUrl,
          }
        },
      },
    },
    'cloudflare-workers-ai': {
      id: 'cloudflare-workers-ai',
      category: 'chat',
      tasks: ['text-generation'],
      nameKey: 'settings.pages.providers.provider.cloudflare-workers-ai.title',
      name: 'Cloudflare Workers AI',
      descriptionKey: 'settings.pages.providers.provider.cloudflare-workers-ai.description',
      description: 'cloudflare.com',
      iconColor: 'i-lobe-icons:cloudflare',
      createProvider: async config => createWorkersAI((config.apiKey as string).trim(), config.accountId as string),
      capabilities: {
        listModels: async () => {
          return []
        },
      },
      validators: {
        validateProviderConfig: (config) => {
          const errors = [
            !config.apiKey && new Error('API key is required.'),
            !config.accountId && new Error('Account ID is required.'),
          ].filter(Boolean)

          return {
            errors,
            reason: errors.filter(e => e).map(e => String(e)).join(', ') || '',
            valid: !!config.apiKey && !!config.accountId,
          }
        },
      },
    },
    'perplexity-ai': {
      id: 'perplexity-ai',
      category: 'chat',
      tasks: ['text-generation'],
      nameKey: 'settings.pages.providers.provider.perplexity.title',
      name: 'Perplexity',
      descriptionKey: 'settings.pages.providers.provider.perplexity.description',
      description: 'perplexity.ai',
      icon: 'i-lobe-icons:perplexity',
      defaultOptions: () => ({
        baseUrl: 'https://api.perplexity.ai',
      }),
      createProvider: async config => createPerplexity((config.apiKey as string).trim(), (config.baseUrl as string).trim()),
      capabilities: {
        listModels: async () => {
          return [
            {
              id: 'sonar-small-online',
              name: 'Sonar Small (Online)',
              provider: 'perplexity-ai',
              description: 'Efficient model with online search capabilities',
              contextLength: 12000,
            },
            {
              id: 'sonar-medium-online',
              name: 'Sonar Medium (Online)',
              provider: 'perplexity-ai',
              description: 'Balanced model with online search capabilities',
              contextLength: 12000,
            },
            {
              id: 'sonar-large-online',
              name: 'Sonar Large (Online)',
              provider: 'perplexity-ai',
              description: 'Powerful model with online search capabilities',
              contextLength: 12000,
            },
            {
              id: 'codey-small',
              name: 'Codey Small',
              provider: 'perplexity-ai',
              description: 'Specialized for code generation and understanding',
              contextLength: 12000,
            },
            {
              id: 'codey-large',
              name: 'Codey Large',
              provider: 'perplexity-ai',
              description: 'Advanced code generation and understanding',
              contextLength: 12000,
            },
          ]
        },
      },
      validators: {
        validateProviderConfig: (config) => {
          const errors = [
            !config.apiKey && new Error('API key is required.'),
            !config.baseUrl && new Error('Base URL is required.'),
          ].filter(Boolean)

          if (!!config.baseUrl && !isAbsoluteUrl(config.baseUrl as string)) {
            return notBaseUrlError.value
          }

          return {
            errors,
            reason: errors.filter(e => e).map(e => String(e)).join(', ') || '',
            valid: !!config.apiKey && !!config.baseUrl,
          }
        },
      },
    },
    'mistral-ai': {
      id: 'mistral-ai',
      category: 'chat',
      tasks: ['text-generation'],
      nameKey: 'settings.pages.providers.provider.mistral.title',
      name: 'Mistral',
      descriptionKey: 'settings.pages.providers.provider.mistral.description',
      description: 'mistral.ai',
      iconColor: 'i-lobe-icons:mistral',
      createProvider: async config => createMistral((config.apiKey as string).trim(), (config.baseUrl as string).trim()),
      capabilities: {
        listModels: async (config) => {
          return (await listModels({
            ...createMistral((config.apiKey as string).trim(), (config.baseUrl as string).trim()).model(),
          })).map((model) => {
            return {
              id: model.id,
              name: model.id,
              provider: 'mistral-ai',
              description: '',
              contextLength: 0,
              deprecated: false,
            } satisfies ModelInfo
          })
        },
      },
      validators: {
        validateProviderConfig: (config) => {
          const errors = [
            !config.apiKey && new Error('API key is required.'),
            !config.baseUrl && new Error('Base URL is required.'),
          ].filter(Boolean)

          return {
            errors,
            reason: errors.filter(e => e).map(e => String(e)).join(', ') || '',
            valid: !!config.apiKey && !!config.baseUrl,
          }
        },
      },
    },
    'moonshot-ai': {
      id: 'moonshot-ai',
      category: 'chat',
      tasks: ['text-generation'],
      nameKey: 'settings.pages.providers.provider.moonshot.title',
      name: 'Moonshot AI',
      descriptionKey: 'settings.pages.providers.provider.moonshot.description',
      description: 'moonshot.ai',
      icon: 'i-lobe-icons:moonshot',
      createProvider: async config => createMoonshot((config.apiKey as string).trim(), (config.baseUrl as string).trim()),
      capabilities: {
        listModels: async (config) => {
          return (await listModels({
            ...createMoonshot((config.apiKey as string).trim(), (config.baseUrl as string).trim()).model(),
          })).map((model) => {
            return {
              id: model.id,
              name: model.id,
              provider: 'moonshot-ai',
              description: '',
              contextLength: 0,
              deprecated: false,
            } satisfies ModelInfo
          })
        },
      },
      validators: {
        validateProviderConfig: (config) => {
          const errors = [
            !config.apiKey && new Error('API key is required.'),
            !config.baseUrl && new Error('Base URL is required.'),
          ].filter(Boolean)

          return {
            errors,
            reason: errors.filter(e => e).map(e => String(e)).join(', ') || '',
            valid: !!config.apiKey && !!config.baseUrl,
          }
        },
      },
    },
    'player2': {
      id: 'player2',
      category: 'chat',
      tasks: ['text-generation'],
      nameKey: 'settings.pages.providers.provider.player2.title',
      name: 'Player2',
      descriptionKey: 'settings.pages.providers.provider.player2.description',
      description: 'player2.game',
      icon: 'i-lobe-icons:player2',
      defaultOptions: () => ({
        baseUrl: 'http://localhost:4315/v1/',
      }),
      createProvider: (config) => {
        return createPlayer2((config.baseUrl as string).trim())
      },
      capabilities: {
        listModels: async () => [
          {
            id: 'player2-model',
            name: 'Player2 Model',
            provider: 'player2',
          },
        ],
      },
      validators: {
        validateProviderConfig: async (config) => {
          if (!config.baseUrl) {
            return {
              errors: [new Error('Base URL is required.')],
              reason: 'Base URL is required. Default to http://localhost:4315/v1/',
              valid: false,
            }
          }

          if (!isAbsoluteUrl(config.baseUrl as string)) {
            return notBaseUrlError.value
          }

          // Check if the local running Player 2 is reachable
          return await fetch(`${(config.baseUrl as string).endsWith('/') ? (config.baseUrl as string).slice(0, -1) : config.baseUrl}/health`, {
            method: 'GET',
            headers: {
              'player2-game-key': 'airi',
            },
          })
            .then((response) => {
              const errors = [
                !response.ok && new Error(`Player 2 returned non-ok status code: ${response.statusText}`),
              ].filter(Boolean)

              return {
                errors,
                reason: errors.filter(e => e).map(e => String(e)).join(', ') || '',
                valid: response.ok,
              }
            })
            .catch((err) => {
              return {
                errors: [err],
                reason: `Failed to reach Player 2, error: ${String(err)} occurred. If you do not have Player 2 running, please start it and try again.`,
                valid: false,
              }
            })
        },
      },
    },
    'player2-speech': {
      id: 'player2-speech',
      category: 'speech',
      tasks: ['text-to-speech'],
      nameKey: 'settings.pages.providers.provider.player2.title',
      name: 'Player2 Speech',
      descriptionKey: 'settings.pages.providers.provider.player2.description',
      description: 'player2.game',
      icon: 'i-lobe-icons:player2',
      defaultOptions: () => ({
        baseUrl: 'http://localhost:4315/v1/',
      }),
      createProvider: async config => createPlayer2((config.baseUrl as string).trim(), 'airi'),
      capabilities: {
        listVoices: async () => {
          return await fetch('http://localhost:4315/v1/tts/voices').then(res => res.json()).then(({ voices }) => (voices as { id: string, language: 'american_english' | 'british_english' | 'japanese' | 'mandarin_chinese' | 'spanish' | 'french' | 'hindi' | 'italian' | 'brazilian_portuguese', name: string, gender: string }[]).map(({ id, language, name, gender }) => (
            {

              id,
              name,
              provider: 'player2-speech',
              gender,
              languages: [{
                american_english: {
                  code: 'en',
                  title: 'English',
                },
                british_english: {
                  code: 'en',
                  title: 'English',
                },
                japanese: {
                  code: 'ja',
                  title: 'Japanese',
                },
                mandarin_chinese: {
                  code: 'zh',
                  title: 'Chinese',
                },
                spanish: {
                  code: 'es',
                  title: 'Spanish',
                },
                french: {
                  code: 'fr',
                  title: 'French',
                },
                hindi: {
                  code: 'hi',
                  title: 'Hindi',
                },

                italian: {
                  code: 'it',
                  title: 'Italian',
                },
                brazilian_portuguese:
                {
                  code: 'pt',
                  title: 'Portuguese',
                },

              }[language]],
            }
          )))
        },
      },
      validators: {
        validateProviderConfig: (config: any) => {
          if (!config.baseUrl) {
            return {
              errors: [new Error('Base URL is required.')],
              reason: 'Base URL is required. Default to http://localhost:4315/v1/',
              valid: false,
            }
          }

          if (!isAbsoluteUrl(config.baseUrl as string)) {
            return notBaseUrlError.value
          }

          return {
            errors: [],
            reason: '',
            valid: true,
          }
        },
      },
    },
  }

  // Configuration validation functions
  async function validateProvider(providerId: string): Promise<boolean> {
    const config = providerCredentials.value[providerId]
    if (!config)
      return false

    const metadata = providerMetadata[providerId]
    if (!metadata)
      return false

    const validationResult = await metadata.validators.validateProviderConfig(config)
    if (!validationResult.valid) {
      throw new Error(validationResult.reason)
    }

    return validationResult.valid
  }

  // Create computed properties for each provider's configuration status
  const configuredProviders = ref<Record<string, boolean>>({})

  // Initialize provider configurations
  function initializeProvider(providerId: string) {
    if (!providerCredentials.value[providerId]) {
      const metadata = providerMetadata[providerId]
      const defaultOptions = metadata.defaultOptions?.() || {}
      providerCredentials.value[providerId] = {
        baseUrl: defaultOptions.baseUrl || '',
      }
    }
  }

  // Initialize all providers
  Object.keys(providerMetadata).forEach(initializeProvider)

  // Track if we're currently updating to prevent infinite loops
  const isUpdatingConfiguration = ref(false)

  // Update configuration status for all providers
  async function updateConfigurationStatus() {
    try {
      const newStatus: Record<string, boolean> = {}

      // Process providers sequentially to avoid overwhelming the system
      for (const providerId of Object.keys(providerMetadata)) {
        newStatus[providerId] = await validateProvider(providerId)
      }

      // Only update if there are actual changes
      const hasChanges = Object.keys(newStatus).some(
        key => configuredProviders.value[key] !== newStatus[key],
      )

      if (hasChanges) {
        Object.assign(configuredProviders.value, newStatus)
      }
    }
    finally {
      isUpdatingConfiguration.value = false
    }
  }

  // Debounced version to prevent excessive calls
  let updateTimeout: NodeJS.Timeout | null = null
  function debouncedUpdateConfigurationStatus() {
    if (updateTimeout)
      clearTimeout(updateTimeout)
    updateTimeout = setTimeout(updateConfigurationStatus, 100)
  }

  // Call initially
  updateConfigurationStatus()

  // Watch for changes (not immediate)
  watch(providerCredentials, debouncedUpdateConfigurationStatus, { deep: true })

  // Available providers (only those that are properly configured or don't require configuration)
  const availableProviders = computed(() => {
    return Object.keys(providerMetadata).filter((providerId) => {
      if (providerId.startsWith('app-local-audio') || providerId.startsWith('browser-local')) {
        return true
      }

      return configuredProviders.value[providerId]
    })
  })

  // Store available models for each provider
  const availableModels = ref<Record<string, ModelInfo[]>>({})
  const isLoadingModels = ref<Record<string, boolean>>({})
  const modelLoadError = ref<Record<string, string | null>>({})

  // Function to fetch models for a specific provider
  async function fetchModelsForProvider(providerId: string) {
    const config = providerCredentials.value[providerId]
    if (!config)
      return []

    const metadata = providerMetadata[providerId]
    if (!metadata)
      return []

    isLoadingModels.value[providerId] = true
    modelLoadError.value[providerId] = null

    try {
      const models = metadata.capabilities.listModels ? await metadata.capabilities.listModels(config) : []

      // Transform and store the models
      availableModels.value[providerId] = models.map(model => ({
        id: model.id,
        name: model.name,
        description: model.description,
        contextLength: model.contextLength,
        deprecated: model.deprecated,
        provider: providerId,
      }))

      return availableModels.value[providerId]
    }
    catch (error) {
      console.error(`Error fetching models for ${providerId}:`, error)
      modelLoadError.value[providerId] = error instanceof Error ? error.message : 'Unknown error'
      return []
    }
    finally {
      isLoadingModels.value[providerId] = false
    }
  }

  // Get models for a specific provider
  function getModelsForProvider(providerId: string) {
    return availableModels.value[providerId] || []
  }

  // Get all available models across all configured providers
  const allAvailableModels = computed(() => {
    const models: ModelInfo[] = []
    for (const providerId of availableProviders.value) {
      models.push(...(availableModels.value[providerId] || []))
    }
    return models
  })

  // Load models for all configured providers
  async function loadModelsForConfiguredProviders() {
    for (const providerId of availableProviders.value) {
      if (providerMetadata[providerId].capabilities.listModels) {
        await fetchModelsForProvider(providerId)
      }
    }
  }

  // Function to get localized provider metadata
  function getProviderMetadata(providerId: string) {
    const metadata = providerMetadata[providerId]

    if (!metadata)
      throw new Error(`Provider metadata for ${providerId} not found`)

    return {
      ...metadata,
      localizedName: t(metadata.nameKey, metadata.name),
      localizedDescription: t(metadata.descriptionKey, metadata.description),
    }
  }

  // Get all providers metadata (for settings page)
  const allProvidersMetadata = computed(() => {
    return Object.values(providerMetadata).map((metadata) => {
      let configured = configuredProviders.value[metadata.id] || false

      // Local providers don't need configuration, they're considered configured if available
      if (metadata.id.startsWith('app-local-audio') || metadata.id.startsWith('browser-local')) {
        configured = true // They'll be filtered by availability check instead
      }

      return {
        ...metadata,
        localizedName: t(metadata.nameKey, metadata.name),
        localizedDescription: t(metadata.descriptionKey, metadata.description),
        configured,
      }
    })
  })

  // Function to get provider object by provider id
  async function getProviderInstance<R extends
  | ChatProvider
  | ChatProviderWithExtraOptions
  | EmbedProvider
  | EmbedProviderWithExtraOptions
  | SpeechProvider
  | SpeechProviderWithExtraOptions
  | TranscriptionProvider
  | TranscriptionProviderWithExtraOptions,
  >(providerId: string): Promise<R> {
    const config = providerCredentials.value[providerId]
    if (!config)
      throw new Error(`Provider credentials for ${providerId} not found`)

    const metadata = providerMetadata[providerId]
    if (!metadata)
      throw new Error(`Provider metadata for ${providerId} not found`)

    try {
      return await metadata.createProvider(config) as R
    }
    catch (error) {
      console.error(`Error creating provider instance for ${providerId}:`, error)
      throw error
    }
  }

  // Available providers metadata (filtered by availability)
  const availableProvidersMetadata = ref<ReturnType<typeof getProviderMetadata>[]>([])

  async function updateAvailableProvidersMetadata() {
    const providers = []
    for (const provider of allProvidersMetadata.value) {
      const p = getProviderMetadata(provider.id)
      const isAvailableBy = p.isAvailableBy || (() => true)
      try {
        const isAvailable = await isAvailableBy()
        if (isAvailable) {
          providers.push(provider)
        }
      }
      catch (error) {
        console.warn(`Availability check failed for ${provider.id}:`, error)
      }
    }
    availableProvidersMetadata.value = providers
  }

  // Call initially
  updateAvailableProvidersMetadata()

  // And watch for changes in allProvidersMetadata
  watch(allProvidersMetadata, () => updateAvailableProvidersMetadata(), { deep: true })

  // Then the category-specific ones
  const allChatProvidersMetadata = computed(() => {
    return availableProvidersMetadata.value.filter(metadata => metadata.category === 'chat')
  })

  const allAudioSpeechProvidersMetadata = computed(() => {
    return availableProvidersMetadata.value.filter(metadata => metadata.category === 'speech')
  })

  const allAudioTranscriptionProvidersMetadata = computed(() => {
    return availableProvidersMetadata.value.filter(metadata => metadata.category === 'transcription')
  })

  const configuredChatProvidersMetadata = computed(() => {
    return allChatProvidersMetadata.value.filter(metadata => configuredProviders.value[metadata.id])
  })

  const configuredSpeechProvidersMetadata = computed(() => {
    return allAudioSpeechProvidersMetadata.value.filter((metadata) => {
      // Local providers are considered configured if they're available
      if (metadata.id.startsWith('app-local-audio') || metadata.id.startsWith('browser-local')) {
        return true
      }
      return configuredProviders.value[metadata.id]
    })
  })

  const configuredTranscriptionProvidersMetadata = computed(() => {
    return allAudioTranscriptionProvidersMetadata.value.filter((metadata) => {
      // Local providers are considered configured if they're available
      if (metadata.id.startsWith('app-local-audio') || metadata.id.startsWith('browser-local')) {
        return true
      }
      return configuredProviders.value[metadata.id]
    })
  })

  function getProviderConfig(providerId: string) {
    return providerCredentials.value[providerId]
  }

  return {
    providers: providerCredentials,
    getProviderConfig,
    availableProviders,
    configuredProviders,
    providerMetadata,
    getProviderMetadata,
    allProvidersMetadata,
    initializeProvider,
    validateProvider,
    availableModels,
    isLoadingModels,
    modelLoadError,
    fetchModelsForProvider,
    getModelsForProvider,
    allAvailableModels,
    loadModelsForConfiguredProviders,
    getProviderInstance,
    availableProvidersMetadata,
    allChatProvidersMetadata,
    allAudioSpeechProvidersMetadata,
    allAudioTranscriptionProvidersMetadata,
    configuredChatProvidersMetadata,
    configuredSpeechProvidersMetadata,
    configuredTranscriptionProvidersMetadata,
  }
})
