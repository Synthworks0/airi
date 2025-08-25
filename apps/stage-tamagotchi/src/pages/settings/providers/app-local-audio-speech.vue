<script setup lang="ts">
import type { SpeechProvider } from '@xsai-ext/shared-providers'

import {
  Button,
  SpeechProviderSettings,
} from '@proj-airi/stage-ui/components'
import { useProvidersStore, useSpeechStore } from '@proj-airi/stage-ui/stores'
import { FieldRange } from '@proj-airi/ui'
import { computed, onMounted, ref, watch } from 'vue'

const providerId = 'app-local-audio-speech'
const defaultModel = 'hexgrad/Kokoro-82M'

const speechStore = useSpeechStore()
const providersStore = useProvidersStore()

const availableVoices = computed(() => {
  return speechStore.availableVoices[providerId] || []
})

const models = ref<Array<{
  id: string
  name: string
  size: number
  quality: 'high' | 'medium' | 'low'
  languages: string[]
  installed: boolean
  recommended?: boolean
}>>([])

const loadingModels = ref<Set<string>>(new Set())
const modelProgress = ref<Record<string, number>>({})
const showAdvanced = ref(false)
const searchQuery = ref('')

// Playground state
const testText = ref('Hello! This is a test of the local text-to-speech system.')
const selectedVoice = ref('')
const isGenerating = ref(false)
const audioUrl = ref('')
const errorMessage = ref('')
const audioPlayer = ref<HTMLAudioElement | null>(null)

const recommendedModels = computed(() =>
  models.value.filter(m => m.recommended).slice(0, 3),
)

const filteredModels = computed(() => {
  if (!searchQuery.value)
    return models.value
  const query = searchQuery.value.toLowerCase()
  return models.value.filter(m =>
    m.name.toLowerCase().includes(query)
    || m.id.toLowerCase().includes(query)
    || m.languages.some(l => l.toLowerCase().includes(query)),
  )
})

async function loadModels() {
  try {
    // Load models from the backend first
    if (typeof window !== 'undefined' && '__TAURI__' in window) {
      try {
        const { invoke } = await import('@tauri-apps/api/core')
        const backendModels = await invoke('plugin:ipc-audio-tts-ort|list_models') as Array<{
          id: string
          name: string
          size: number
          quality: 'high' | 'medium' | 'low'
          languages: string[]
          installed: boolean
        }>

        models.value = backendModels.map(m => ({
          ...m,
          // Recommend Kokoro and eSpeak by default in local TTS context
          recommended: ['hexgrad/Kokoro-82M', 'espeak-ng'].includes(m.id),
        }))
        return
      }
      catch (error) {
        console.warn('Failed to load models from backend, using fallback:', error)
      }
    }

    // Only Kokoro-82M is supported
    models.value = [
      {
        id: 'hexgrad/Kokoro-82M',
        name: 'Kokoro-82M (ONNX Community)',
        size: 82000000,
        quality: 'high',
        languages: ['English', 'Japanese', 'Chinese'],
        installed: false,
        recommended: true,
      },
      // Keep eSpeak as fallback
      {
        id: 'espeak-ng',
        name: 'eSpeak NG (Fallback)',
        size: 5000000,
        quality: 'low',
        languages: ['Multiple'],
        installed: true,
        recommended: false,
      },
    ]
  }
  catch (error) {
    console.error('Failed to load models:', error)
  }
}

async function installModel(modelId: string) {
  if (loadingModels.value.has(modelId))
    return

  loadingModels.value.add(modelId)
  modelProgress.value[modelId] = 0

  try {
    const { invoke } = await import('@tauri-apps/api/core')
    const { listen } = await import('@tauri-apps/api/event')

    // Listen for real progress events from the backend
    const unlisten = await listen('tauri-plugins:tauri-plugin-ipc-audio-tts-ort:load-model-progress', (event: any) => {
      const [done, filename, progress, _totalSize, _currentSize] = event.payload as [boolean, string, number, number, number]

      if (!filename || filename === modelId || filename.includes(modelId)) {
        modelProgress.value[modelId] = progress

        if (done) {
          modelProgress.value[modelId] = 100
          const model = models.value.find(m => m.id === modelId)
          if (model)
            model.installed = true

          // Refresh voices now that the model is installed
          speechStore.loadVoicesForProvider(providerId).then(() => {
            if (!selectedVoice.value && availableVoices.value.length > 0)
              selectedVoice.value = availableVoices.value[0].id
          })

          setTimeout(() => {
            delete modelProgress.value[modelId]
          }, 500)
        }
      }
    })

    // Simulate progress if no events come through
    const progressInterval = setInterval(() => {
      if (!modelProgress.value[modelId] || modelProgress.value[modelId] < 90) {
        modelProgress.value[modelId] = Math.min((modelProgress.value[modelId] || 0) + 5, 90)
      }
    }, 500)

    try {
      await invoke('plugin:ipc-audio-tts-ort|load_model', { modelId })

      // Model loaded successfully
      clearInterval(progressInterval)
      modelProgress.value[modelId] = 100

      const model = models.value.find(m => m.id === modelId)
      if (model)
        model.installed = true

      // Reload voices to surface newly available Kokoro voices and choose one
      await speechStore.loadVoicesForProvider(providerId)
      if (!selectedVoice.value && availableVoices.value.length > 0)
        selectedVoice.value = availableVoices.value[0].id

      setTimeout(() => {
        delete modelProgress.value[modelId]
      }, 500)
    }
    catch (error) {
      clearInterval(progressInterval)

      // Check if it's a permission error (robust error message extraction)
      const errorStr = (() => {
        try {
          return typeof error === 'string' ? error : (error && typeof error.toString === 'function') ? error.toString() : String(error || 'Unknown error')
        }
        catch {
          return 'Unknown error'
        }
      })()

      if (errorStr.includes('not allowed')) {
        errorMessage.value = 'Permission denied. Please restart the application for the new permissions to take effect.'
        // Show error message instead of alert
        console.error('Permission denied. Please restart the application for the new permissions to take effect.')
      }
      else {
        errorMessage.value = `Failed to install ${modelId}: ${errorStr}`
      }
      throw error
    }
    finally {
      unlisten()
    }
  }
  catch (error) {
    console.error(`Failed to install model ${modelId}:`, error)
    delete modelProgress.value[modelId]
  }
  finally {
    loadingModels.value.delete(modelId)
  }
}

function formatSize(bytes: number): string {
  const mb = bytes / (1024 * 1024)
  return mb < 1000 ? `${mb.toFixed(0)} MB` : `${(mb / 1024).toFixed(1)} GB`
}

onMounted(async () => {
  // Initialize provider if not already done
  providersStore.initializeProvider(providerId)

  await loadModels()
  await speechStore.loadVoicesForProvider(providerId)

  // Auto-select first available voice on initial load
  if (!selectedVoice.value && availableVoices.value.length > 0) {
    selectedVoice.value = availableVoices.value[0].id
  }

  // Check if any models are already installed
  try {
    const { invoke } = await import('@tauri-apps/api/core')
    const installedModels = await invoke('plugin:ipc-audio-tts-ort|list_installed_models') as string[]
    models.value.forEach((model) => {
      if (installedModels.includes(model.id))
        model.installed = true
    })

    // If Kokoro is installed, ensure it is loaded so voices appear immediately
    if (installedModels.includes(defaultModel)) {
      try {
        console.warn(`Loading default model: ${defaultModel}`)
        await invoke('plugin:ipc-audio-tts-ort|load_model', { modelId: defaultModel })
        console.warn('Model loaded, refreshing voices...')

        // Force reload voices with a small delay to ensure model is fully loaded
        await new Promise(resolve => setTimeout(resolve, 100))
        await speechStore.loadVoicesForProvider(providerId)

        console.warn('Available voices after model load:', availableVoices.value)
        if (!selectedVoice.value && availableVoices.value.length > 0) {
          selectedVoice.value = availableVoices.value[0].id
          console.warn('Auto-selected voice:', selectedVoice.value)
        }
      }
      catch (err) {
        console.warn('Kokoro model was installed but failed to load automatically:', err)
      }
    }
  }
  catch (error) {
    console.warn('Failed to check installed models:', error)
  }
})

// When voices change, keep a sensible selection so the Test button is enabled
watch(availableVoices, (voices) => {
  if (!selectedVoice.value && voices.length > 0)
    selectedVoice.value = voices[0].id
})

async function handleGenerateSpeech(input: string, voiceId: string) {
  // Validate inputs
  if (!input?.trim()) {
    throw new Error('Text input is required')
  }
  if (!voiceId?.trim()) {
    throw new Error('Voice selection is required')
  }

  let provider
  try {
    provider = await providersStore.getProviderInstance(providerId) as SpeechProvider
  }
  catch (providerError) {
    console.error('Provider creation failed:', providerError)
    throw new Error(`Failed to initialize speech provider: ${providerError}`)
  }

  if (!provider) {
    throw new Error('Failed to initialize speech provider')
  }

  const providerConfig = providersStore.getProviderConfig(providerId) || {}
  const model = (providerConfig.model as string | undefined) || defaultModel

  const finalConfig = {
    ...providerConfig,
    // Ensure voice settings have safe defaults
    voiceSettings: {
      pitch: 0,
      speed: 1.0,
      volume: 0,
      ...providerConfig.voiceSettings,
    },
  }

  try {
    const result = await speechStore.speech(
      provider,
      model,
      input,
      voiceId,
      finalConfig,
    )
    return result
  }
  catch (speechError) {
    console.error('Speech store failed:', speechError)
    throw speechError
  }
}

async function handleTestVoice() {
  if (!testText.value.trim() || !selectedVoice.value)
    return

  // Additional validation to ensure we have proper values
  if (!availableVoices.value.length) {
    errorMessage.value = 'No voices available. Please install a TTS model first.'
    return
  }

  const selectedVoiceObj = availableVoices.value.find(v => v.id === selectedVoice.value)
  if (!selectedVoiceObj) {
    errorMessage.value = 'Selected voice not found. Please select a different voice.'
    return
  }

  isGenerating.value = true
  errorMessage.value = ''

  try {
    // Stop any currently playing audio
    if (audioUrl.value) {
      stopTestAudio()
    }

    const response = await handleGenerateSpeech(testText.value, selectedVoice.value)

    // Convert the response to a blob and create an object URL
    audioUrl.value = URL.createObjectURL(new Blob([response]))

    // Play the audio
    setTimeout(() => {
      if (audioPlayer.value) {
        audioPlayer.value.play()
      }
    }, 100)
  }
  catch (error) {
    console.error('Error generating speech:', error)
    // Ultra-robust error message extraction - handle any possible error type
    let errorStr = 'Unknown error'
    try {
      if (error === null || error === undefined) {
        errorStr = 'Unknown error (null/undefined)'
      }
      else if (typeof error === 'string') {
        errorStr = error
      }
      else if (typeof error === 'object') {
        if (error instanceof Error && error.message) {
          errorStr = error.message
        }
        else if (error.message && typeof error.message === 'string') {
          errorStr = error.message
        }
        else if (typeof error.toString === 'function') {
          try {
            errorStr = error.toString()
          }
          catch {
            errorStr = 'Error object exists but toString failed'
          }
        }
        else {
          try {
            errorStr = JSON.stringify(error)
          }
          catch {
            errorStr = 'Error object exists but cannot be serialized'
          }
        }
      }
      else {
        try {
          errorStr = String(error)
        }
        catch {
          errorStr = `Error of type ${typeof error} cannot be converted to string`
        }
      }
    }
    catch (conversionError) {
      console.error('Failed to extract error message:', conversionError)
      errorStr = 'Error occurred but message extraction failed'
    }
    errorMessage.value = `Error generating speech: ${errorStr}`
  }
  finally {
    isGenerating.value = false
  }
}

function stopTestAudio() {
  if (audioPlayer.value) {
    audioPlayer.value.pause()
    audioPlayer.value.currentTime = 0
  }

  if (audioUrl.value) {
    URL.revokeObjectURL(audioUrl.value)
    audioUrl.value = ''
  }
}
</script>

<template>
  <SpeechProviderSettings :provider-id="providerId" :default-model="defaultModel">
    <template #basic-settings>
      <!-- Model Management moved to basic settings -->
      <div flex="~ col gap-4">
        <h3 text-lg font-medium>
          Model Management
        </h3>

        <div v-if="recommendedModels.length > 0" flex="~ col gap-3">
          <h4 text-sm text-neutral-600 dark:text-neutral-400>
            Recommended Models
          </h4>
          <div
            v-for="model in recommendedModels" :key="model.id"
            flex="~ items-center justify-between" rounded-lg p-3
            bg="neutral-100 dark:neutral-800"
          >
            <div flex="~ col gap-1">
              <span font-medium>{{ model.name }}</span>
              <span text-xs text-neutral-500>
                {{ model.quality }} quality • {{ formatSize(model.size) }} • {{ model.languages.join(', ') }}
              </span>
            </div>
            <div flex="~ items-center gap-2">
              <div
                v-if="modelProgress[model.id] !== undefined"
                flex="~ items-center gap-2"
              >
                <div bg="neutral-200 dark:neutral-700" h-2 w-24 overflow-hidden rounded-full>
                  <div
                    h-full bg-primary-500 transition-all duration-300
                    :style="{ width: `${modelProgress[model.id]}%` }"
                  />
                </div>
                <span text-xs>{{ Math.round(modelProgress[model.id]) }}%</span>
              </div>
              <Button
                v-else-if="!model.installed"
                size="sm"
                :disabled="loadingModels.has(model.id)"
                @click="installModel(model.id)"
              >
                Install
              </Button>
              <span v-else text-xs text-green-600 font-medium dark:text-green-400>
                Installed
              </span>
            </div>
          </div>
        </div>

        <div mt-2>
          <button
            flex="~ items-center gap-2"
            text-sm text-primary-500 hover:text-primary-600 @click="showAdvanced = !showAdvanced"
          >
            <span>{{ showAdvanced ? 'Hide' : 'Show' }} Advanced Models</span>
            <div :class="showAdvanced ? 'i-solar:alt-arrow-up-bold' : 'i-solar:alt-arrow-down-bold'" />
          </button>
        </div>

        <div v-if="showAdvanced" flex="~ col gap-3" mt-2>
          <input
            v-model="searchQuery"
            type="text"
            placeholder="Search models..."
            rounded-lg px-3 py-2 border="1 neutral-300 dark:neutral-600"
            bg="white dark:neutral-800"
          >

          <div flex="~ col gap-2" max-h-64 overflow-y-auto>
            <div
              v-for="model in filteredModels" :key="model.id"
              flex="~ items-center justify-between" rounded p-2
              hover:bg="neutral-100 dark:neutral-800"
            >
              <div flex="~ col">
                <span text-sm>{{ model.name }}</span>
                <span text-xs text-neutral-500>
                  {{ formatSize(model.size) }} • {{ model.languages.join(', ') }}
                </span>
              </div>
              <Button
                v-if="!model.installed"
                size="xs"
                variant="secondary"
                :disabled="loadingModels.has(model.id)"
                @click="installModel(model.id)"
              >
                Install
              </Button>
              <span v-else text-xs text-neutral-500>Installed</span>
            </div>
          </div>
        </div>
      </div>
    </template>

    <template #voice-settings>
      <div flex="~ col gap-4">
        <FieldRange
          :model-value="0"
          label="Pitch"
          description="Adjust voice pitch"
          :min="-10"
          :max="10"
          :step="1"
          :format-value="v => v > 0 ? `+${v}` : String(v)"
        />
        <FieldRange
          :model-value="1.0"
          label="Speed"
          description="Adjust speaking speed"
          :min="0.5"
          :max="2.0"
          :step="0.1"
          :format-value="v => `${v}x`"
        />
        <FieldRange
          :model-value="0"
          label="Volume"
          description="Adjust voice volume"
          :min="-20"
          :max="20"
          :step="1"
          :format-value="v => `${v > 0 ? '+' : ''}${v} dB`"
        />
      </div>
    </template>

    <template #advanced-settings>
      <!-- Advanced settings can be empty or contain other advanced options -->
    </template>

    <template #playground>
      <div w-full rounded-xl>
        <h2 class="mb-4 text-lg text-neutral-500 md:text-2xl dark:text-neutral-400" w-full>
          <div class="inline-flex items-center gap-4">
            <div i-solar:microphone-3-bold-duotone />
            <div>Voice Playground</div>
          </div>
        </h2>
        <div flex="~ col gap-4">
          <textarea
            v-model="testText"
            placeholder="Hello! This is a test of the local text-to-speech system."
            border="neutral-100 dark:neutral-800 solid 2 focus:neutral-200 dark:focus:neutral-700"
            transition="all duration-250 ease-in-out"
            bg="neutral-100 dark:neutral-800 focus:neutral-50 dark:focus:neutral-900"
            h-24 w-full rounded-lg px-3 py-2 text-sm outline-none
          />

          <div v-if="availableVoices.length > 0" flex="~ items-center gap-4">
            <select
              v-model="selectedVoice"
              border="neutral-100 dark:neutral-800 solid 2"
              bg="neutral-100 dark:neutral-800"
              w-full rounded-lg px-3 py-2 text-sm outline-none
            >
              <option value="">
                Select a voice
              </option>
              <option v-for="voice in availableVoices" :key="voice.id" :value="voice.id">
                {{ voice.name }}
              </option>
            </select>
          </div>

          <div v-else class="text-sm text-neutral-500">
            Please select a voice
          </div>

          <div flex="~ row gap-4">
            <button
              border="neutral-800 dark:neutral-200 solid 2"
              transition="border duration-250 ease-in-out"
              rounded-lg px-4 py-2 text-sm
              bg="neutral-700 dark:neutral-300"
              text="neutral-100 dark:neutral-900"
              :disabled="isGenerating || !testText.trim() || !selectedVoice"
              :class="{ 'opacity-50 cursor-not-allowed': isGenerating || !testText.trim() || !selectedVoice }"
              @click="handleTestVoice"
            >
              <div flex="~ row items-center gap-2">
                <div i-solar:play-circle-bold-duotone />
                <span>{{ isGenerating ? 'Generating...' : 'Test Voice' }}</span>
              </div>
            </button>
          </div>

          <div v-if="errorMessage" class="text-sm text-red-500">
            {{ errorMessage }}
          </div>

          <audio v-if="audioUrl" ref="audioPlayer" :src="audioUrl" controls class="w-full" />
        </div>
      </div>
    </template>
  </SpeechProviderSettings>
</template>

<route lang="yaml">
meta:
  layout: settings
  stageTransition:
    name: slide
</route>
