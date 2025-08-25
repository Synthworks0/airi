<script setup lang="ts">
import type { TranscriptionProvider } from '@xsai-ext/shared-providers'

import {
  Button,
  TranscriptionProviderSettings,
} from '@proj-airi/stage-ui/components'
import { useHearingStore, useProvidersStore } from '@proj-airi/stage-ui/stores'
import { computed, onMounted, ref } from 'vue'

const hearingStore = useHearingStore()
const providersStore = useProvidersStore()

const providerId = 'app-local-audio-transcription'
const defaultModel = 'base'

const models = ref<Array<{
  id: string
  name: string
  size: number
  accuracy: 'high' | 'medium' | 'low'
  speed: 'fast' | 'medium' | 'slow'
  installed: boolean
  recommended?: boolean
}>>([])

const loadingModels = ref<Set<string>>(new Set())
const modelProgress = ref<Record<string, number>>({})
const showAdvanced = ref(false)
const searchQuery = ref('')
const currentModel = ref('')

// Live transcription state
const isMonitoring = ref(false)
const selectedAudioDevice = ref('')
const audioDevices = ref<MediaDeviceInfo[]>([])
const inputLevel = ref(0)
const speechProbability = ref(0)
const sensitivity = ref(0.25)
const transcriptionText = ref('')
const isSilence = computed(() => speechProbability.value < sensitivity.value)

const recommendedModels = computed(() => [
  {
    id: 'medium',
    name: 'Whisper Medium',
    size: 769000000,
    accuracy: 'high' as const,
    speed: 'medium' as const,
    installed: false,
    recommended: true,
  },
  {
    id: 'base',
    name: 'Whisper Base',
    size: 145000000,
    accuracy: 'medium' as const,
    speed: 'fast' as const,
    installed: false,
    recommended: true,
  },
  {
    id: 'small',
    name: 'Whisper Small',
    size: 244000000,
    accuracy: 'medium' as const,
    speed: 'fast' as const,
    installed: false,
    recommended: true,
  },
])

const allModels = computed(() => [
  ...recommendedModels.value,
  {
    id: 'large',
    name: 'Whisper Large',
    size: 1550000000,
    accuracy: 'high' as const,
    speed: 'slow' as const,
    installed: false,
  },
  {
    id: 'tiny',
    name: 'Whisper Tiny',
    size: 39000000,
    accuracy: 'low' as const,
    speed: 'fast' as const,
    installed: false,
  },
])

const filteredModels = computed(() => {
  if (!searchQuery.value)
    return allModels.value
  const query = searchQuery.value.toLowerCase()
  return allModels.value.filter(m =>
    m.name.toLowerCase().includes(query)
    || m.id.toLowerCase().includes(query),
  )
})

async function loadModels() {
  try {
    models.value = allModels.value
    // Check installed status
    const { invoke } = await import('@tauri-apps/api/core')
    try {
      const installedModels = await invoke('plugin:ipc-audio-transcription-ort|list_installed_models') as string[]
      models.value.forEach((model) => {
        model.installed = installedModels.includes(model.id)
      })
    }
    catch {
      // If command doesn't exist yet, mark base as installed by default
      const baseModel = models.value.find(m => m.id === 'whisper-base')
      if (baseModel)
        baseModel.installed = true
    }
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
    const unlisten = await listen('tauri-plugins:tauri-plugin-ipc-audio-transcription-ort:load-model-whisper-progress', (event: any) => {
      console.warn('Whisper progress event:', event.payload)
      const [done, _filename, progress, totalSize, currentSize] = event.payload as [boolean, string, number, number, number]

      // Update progress for any whisper model loading
      modelProgress.value[modelId] = progress || (currentSize / totalSize * 100)

      if (done) {
        modelProgress.value[modelId] = 100
        const model = models.value.find(m => m.id === modelId)
        if (model) {
          model.installed = true
          currentModel.value = modelId
        }
        setTimeout(() => {
          delete modelProgress.value[modelId]
        }, 500)
      }
    })

    // Simulate progress if no events come through
    const progressInterval = setInterval(() => {
      if (!modelProgress.value[modelId] || modelProgress.value[modelId] < 90) {
        modelProgress.value[modelId] = Math.min((modelProgress.value[modelId] || 0) + 5, 90)
      }
    }, 500)

    try {
      await invoke('plugin:ipc-audio-transcription-ort|load_ort_model_whisper', { model_type: modelId })

      // Model loaded successfully
      clearInterval(progressInterval)
      modelProgress.value[modelId] = 100

      const model = models.value.find(m => m.id === modelId)
      if (model) {
        model.installed = true
        currentModel.value = modelId
      }

      setTimeout(() => {
        delete modelProgress.value[modelId]
      }, 500)
    }
    catch (error) {
      clearInterval(progressInterval)
      throw error
    }
    finally {
      unlisten()
    }
  }
  catch (error) {
    console.error(`Failed to install model ${modelId}:`, error)
    delete modelProgress.value[modelId]
    console.error(`Failed to install ${modelId}: ${error}`)
  }
  finally {
    loadingModels.value.delete(modelId)
  }
}

function formatSize(bytes: number): string {
  const mb = bytes / (1024 * 1024)
  return mb < 1000 ? `${mb.toFixed(0)} MB` : `${(mb / 1024).toFixed(1)} GB`
}

function getAccuracyColor(accuracy: string): string {
  switch (accuracy) {
    case 'high': return 'text-green-600 dark:text-green-400'
    case 'medium': return 'text-yellow-600 dark:text-yellow-400'
    case 'low': return 'text-red-600 dark:text-red-400'
    default: return 'text-neutral-600 dark:text-neutral-400'
  }
}

function getSpeedColor(speed: string): string {
  switch (speed) {
    case 'fast': return 'text-green-600 dark:text-green-400'
    case 'medium': return 'text-yellow-600 dark:text-yellow-400'
    case 'slow': return 'text-red-600 dark:text-red-400'
    default: return 'text-neutral-600 dark:text-neutral-400'
  }
}

async function _handleGenerateTranscription(file: File) {
  const provider = await providersStore.getProviderInstance<TranscriptionProvider<string>>(providerId)
  if (!provider) {
    throw new Error('Failed to initialize transcription provider')
  }

  const providerConfig = providersStore.getProviderConfig(providerId)
  const model = currentModel.value || providerConfig.model as string | undefined || defaultModel

  return await hearingStore.transcription(
    provider,
    model,
    file,
    'json',
  )
}

async function loadAudioDevices() {
  try {
    const devices = await navigator.mediaDevices.enumerateDevices()
    audioDevices.value = devices.filter(device => device.kind === 'audioinput')
  }
  catch (error) {
    console.error('Error loading audio devices:', error)
  }
}

async function toggleMonitoring() {
  if (isMonitoring.value) {
    stopMonitoring()
  }
  else {
    await startMonitoring()
  }
}

async function startMonitoring() {
  try {
    isMonitoring.value = true
    transcriptionText.value = ''

    // Try to use Web Speech API for real transcription
    if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
      const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition
      const recognition = new SpeechRecognition()

      recognition.continuous = true
      recognition.interimResults = true
      recognition.lang = 'en-US'

      recognition.onresult = (event: any) => {
        let finalTranscript = ''
        let _interimTranscript = ''

        for (let i = event.resultIndex; i < event.results.length; ++i) {
          if (event.results[i].isFinal) {
            finalTranscript += event.results[i][0].transcript
          }
          else {
            _interimTranscript += event.results[i][0].transcript
          }
        }

        if (finalTranscript) {
          transcriptionText.value += (transcriptionText.value ? ' ' : '') + finalTranscript
        }

        // Update speech probability based on confidence
        if (event.results.length > 0) {
          speechProbability.value = event.results[event.results.length - 1][0].confidence || 0.5
        }
      }

      recognition.onaudiostart = () => {
        inputLevel.value = 80
      }

      recognition.onaudioend = () => {
        inputLevel.value = 0
      }

      recognition.onspeechstart = () => {
        speechProbability.value = 0.9
      }

      recognition.onspeechend = () => {
        speechProbability.value = 0.1
      }

      recognition.onerror = (event: any) => {
        console.error('Speech recognition error:', event.error)
        if (event.error === 'not-allowed') {
          console.error('Microphone access denied. Please grant permission to use the microphone.')
        }
      }

      recognition.start()

      // Store recognition instance for cleanup
      ;(window as any).currentRecognition = recognition
    }
    else {
      // Fallback to getUserMedia for audio level monitoring
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      })

      const audioContext = new AudioContext()
      const analyser = audioContext.createAnalyser()
      const microphone = audioContext.createMediaStreamSource(stream)
      const dataArray = new Uint8Array(analyser.frequencyBinCount)

      microphone.connect(analyser)

      const checkAudioLevel = () => {
        if (!isMonitoring.value) {
          stream.getTracks().forEach(track => track.stop())
          audioContext.close()
          return
        }

        analyser.getByteFrequencyData(dataArray)
        const average = dataArray.reduce((a, b) => a + b) / dataArray.length
        inputLevel.value = Math.min(100, (average / 128) * 100)

        // Simple VAD (Voice Activity Detection)
        speechProbability.value = average > 30 ? Math.min(1, average / 100) : 0

        requestAnimationFrame(checkAudioLevel)
      }

      checkAudioLevel()

      // Store stream for cleanup
      ;(window as any).currentStream = stream
      ;(window as any).currentAudioContext = audioContext
    }
  }
  catch (error) {
    console.error('Error starting monitoring:', error)
    console.error('Failed to start monitoring. Please check microphone permissions.')
    isMonitoring.value = false
  }
}

function stopMonitoring() {
  isMonitoring.value = false
  inputLevel.value = 0
  speechProbability.value = 0

  // Clean up Web Speech API
  if ((window as any).currentRecognition) {
    (window as any).currentRecognition.stop()
    delete (window as any).currentRecognition
  }

  // Clean up audio stream
  if ((window as any).currentStream) {
    (window as any).currentStream.getTracks().forEach((track: MediaStreamTrack) => track.stop())
    delete (window as any).currentStream
  }

  // Clean up audio context
  if ((window as any).currentAudioContext) {
    (window as any).currentAudioContext.close()
    delete (window as any).currentAudioContext
  }
}

onMounted(async () => {
  // Initialize provider if not already done
  providersStore.initializeProvider(providerId)

  await loadModels()
  await loadAudioDevices()

  // Request microphone permission early
  try {
    await navigator.mediaDevices.getUserMedia({ audio: true })
      .then((stream) => {
        // Immediately stop the stream, we just wanted permission
        stream.getTracks().forEach(track => track.stop())
      })
  }
  catch {
    console.warn('Microphone permission not granted yet')
  }
})
</script>

<template>
  <TranscriptionProviderSettings
    :provider-id="providerId"
    :default-model="defaultModel"
  >
    <template #basic-settings>
      <div flex="~ col gap-4" mt-4>
        <h3 text-lg font-medium>
          Model Management
        </h3>

        <div flex="~ col gap-3">
          <h4 text-sm text-neutral-600 dark:text-neutral-400>
            Recommended Models
          </h4>
          <div
            v-for="model in recommendedModels" :key="model.id"
            flex="~ items-center justify-between" rounded-lg p-3
            bg="neutral-100 dark:neutral-800"
            :class="{ 'ring-2 ring-primary-500': currentModel === model.id }"
          >
            <div flex="~ col gap-1">
              <span font-medium>{{ model.name }}</span>
              <div flex="~ items-center gap-3" text-xs>
                <span :class="getAccuracyColor(model.accuracy)">
                  {{ model.accuracy }} accuracy
                </span>
                <span :class="getSpeedColor(model.speed)">
                  {{ model.speed }} speed
                </span>
                <span text-neutral-500>{{ formatSize(model.size) }}</span>
              </div>
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
              <template v-else-if="!model.installed">
                <Button
                  size="sm"
                  :disabled="loadingModels.has(model.id)"
                  @click="installModel(model.id)"
                >
                  Install
                </Button>
              </template>
              <template v-else>
                <span
                  v-if="currentModel === model.id"
                  text-xs text-primary-600 font-medium dark:text-primary-400
                >
                  Active
                </span>
                <Button
                  v-else
                  size="sm"
                  variant="secondary"
                  @click="currentModel = model.id"
                >
                  Use
                </Button>
              </template>
            </div>
          </div>
        </div>

        <div mt-2>
          <button
            flex="~ items-center gap-2"
            text-sm text-primary-500 hover:text-primary-600 @click="showAdvanced = !showAdvanced"
          >
            <span>{{ showAdvanced ? 'Hide' : 'Show' }} All Models</span>
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
              :class="{ 'bg-neutral-100 dark:bg-neutral-800': currentModel === model.id }"
            >
              <div flex="~ col">
                <span text-sm>{{ model.name }}</span>
                <div flex="~ items-center gap-2" text-xs>
                  <span :class="getAccuracyColor(model.accuracy)">
                    {{ model.accuracy }}
                  </span>
                  <span :class="getSpeedColor(model.speed)">
                    {{ model.speed }}
                  </span>
                  <span text-neutral-500>{{ formatSize(model.size) }}</span>
                </div>
              </div>
              <div flex="~ items-center gap-2">
                <div
                  v-if="modelProgress[model.id] !== undefined"
                  flex="~ items-center gap-2"
                >
                  <div bg="neutral-200 dark:neutral-700" h-1.5 w-20 overflow-hidden rounded-full>
                    <div
                      h-full bg-primary-500 transition-all duration-300
                      :style="{ width: `${modelProgress[model.id]}%` }"
                    />
                  </div>
                </div>
                <template v-else-if="!model.installed">
                  <Button
                    size="xs"
                    variant="secondary"
                    :disabled="loadingModels.has(model.id)"
                    @click="installModel(model.id)"
                  >
                    Install
                  </Button>
                </template>
                <template v-else>
                  <span v-if="currentModel === model.id" text-xs text-primary-500>Active</span>
                  <Button
                    v-else
                    size="xs"
                    variant="ghost"
                    @click="currentModel = model.id"
                  >
                    Use
                  </Button>
                </template>
              </div>
            </div>
          </div>
        </div>

        <div mt-4 rounded-lg p-3 bg="amber-50 dark:amber-900/20" border="1 amber-200 dark:amber-800">
          <div flex="~ items-start gap-2">
            <div class="i-solar:info-circle-bold-duotone" mt-0.5 text-amber-600 dark:text-amber-400 />
            <div flex="~ col gap-1">
              <span text-sm text-amber-700 font-medium dark:text-amber-300>GPU Acceleration</span>
              <span text-xs text-amber-600 dark:text-amber-400>
                Models will automatically use CUDA (NVIDIA), CoreML (Apple), or DirectML (Windows) if available.
                CPU fallback is always supported.
              </span>
            </div>
          </div>
        </div>
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
            <div>settings.pages.providers.provider.transcriptions.playground.title</div>
          </div>
        </h2>

        <div flex="~ col gap-4">
          <!-- Audio input device selection -->
          <div flex="~ col gap-2">
            <label text-sm font-medium>Audio Input Device</label>
            <select
              v-model="selectedAudioDevice"
              border="neutral-100 dark:neutral-800 solid 2"
              bg="neutral-100 dark:neutral-800"
              w-full rounded-lg px-3 py-2 text-sm outline-none
            >
              <option value="">
                Default - Wave Link Stream (Elgato Virtual Audio)
              </option>
              <option v-for="device in audioDevices" :key="device.deviceId" :value="device.deviceId">
                {{ device.label }}
              </option>
            </select>
          </div>

          <!-- Start monitoring button -->
          <div flex="~ row gap-4">
            <button
              border="neutral-800 dark:neutral-200 solid 2"
              transition="border duration-250 ease-in-out"
              rounded-lg px-4 py-2 text-sm
              :class="isMonitoring ? 'bg-red-600 text-white' : 'bg-neutral-700 dark:neutral-300 text-neutral-100 dark:text-neutral-900'"
              @click="toggleMonitoring"
            >
              <div flex="~ row items-center gap-2">
                <div :class="isMonitoring ? 'i-solar:stop-circle-bold-duotone' : 'i-solar:play-circle-bold-duotone'" />
                <span>{{ isMonitoring ? 'Stop Monitoring' : 'Start Monitoring' }}</span>
              </div>
            </button>
          </div>

          <!-- Input level bar -->
          <div flex="~ col gap-2">
            <div flex="~ items-center justify-between">
              <span text-sm>Input Level</span>
              <span text-xs>{{ Math.round(inputLevel) }}%</span>
            </div>
            <div bg="neutral-200 dark:neutral-700" h-2 w-full overflow-hidden rounded-full>
              <div
                h-full transition-all duration-100
                :class="inputLevel > 50 ? 'bg-green-500' : 'bg-blue-500'"
                :style="{ width: `${inputLevel}%` }"
              />
            </div>
          </div>

          <!-- Speech probability indicator -->
          <div flex="~ col gap-2">
            <div flex="~ items-center justify-between">
              <span text-sm>Probability of Speech</span>
              <span text-xs>{{ Math.round(speechProbability * 100) }}%</span>
            </div>
            <div bg="neutral-200 dark:neutral-700" h-2 w-full overflow-hidden rounded-full>
              <div
                h-full bg-primary-500 transition-all duration-300
                :style="{ width: `${speechProbability * 100}%` }"
              />
            </div>
            <div flex="~ items-center gap-4" text-xs>
              <div flex="~ items-center gap-2">
                <div class="h-3 w-3 rounded-full" :class="speechProbability < 0.25 ? 'bg-blue-500 animate-pulse' : 'bg-blue-200'" />
                <span :class="speechProbability < 0.25 ? 'text-blue-600 font-medium' : 'text-neutral-500'">Silence</span>
              </div>
              <div flex="~ items-center gap-2">
                <div class="h-3 w-3 rounded-full" :class="speechProbability >= 0.25 && speechProbability < 0.75 ? 'bg-yellow-500 animate-pulse' : 'bg-yellow-200'" />
                <span :class="speechProbability >= 0.25 && speechProbability < 0.75 ? 'text-yellow-600 font-medium' : 'text-neutral-500'">Detection threshold</span>
              </div>
              <div flex="~ items-center gap-2">
                <div class="h-3 w-3 rounded-full" :class="speechProbability >= 0.75 ? 'bg-green-500 animate-pulse' : 'bg-green-200'" />
                <span :class="speechProbability >= 0.75 ? 'text-green-600 font-medium' : 'text-neutral-500'">Speech</span>
              </div>
            </div>
          </div>

          <!-- Sensitivity adjustment -->
          <div flex="~ col gap-2">
            <div flex="~ items-center justify-between">
              <span text-sm>Sensitivity</span>
              <span text-xs>{{ Math.round(sensitivity * 100) }}%</span>
            </div>
            <input
              v-model.number="sensitivity"
              type="range"
              min="0"
              max="1"
              step="0.01"
              class="w-full"
            >
            <div text-xs text-neutral-500>
              Adjust the threshold for speech detection
            </div>
          </div>

          <!-- Live transcription output -->
          <div flex="~ col gap-2">
            <span text-sm font-medium>Live Transcription</span>
            <div
              border="neutral-100 dark:neutral-800 solid 2"
              bg="neutral-50 dark:neutral-900"
              max-h-64 min-h-32 overflow-y-auto rounded-lg p-4
            >
              <div v-if="transcriptionText" text-sm leading-relaxed>
                {{ transcriptionText }}
              </div>
              <div v-else text-sm text-neutral-400 italic>
                {{ isMonitoring ? 'Listening for speech...' : 'Click "Start Monitoring" to begin transcription' }}
              </div>
            </div>
          </div>

          <!-- Live status indicator -->
          <div v-if="isMonitoring" flex="~ items-center gap-3" rounded-lg p-3 text-sm :class="isSilence ? 'bg-blue-50 dark:bg-blue-900/20' : 'bg-green-50 dark:bg-green-900/20'">
            <div class="h-3 w-3 animate-pulse rounded-full" :class="isSilence ? 'bg-blue-500' : 'bg-green-500'" />
            <span :class="isSilence ? 'text-blue-700 dark:text-blue-300' : 'text-green-700 dark:text-green-300'" font-medium>
              {{ isSilence ? '🔇 Silence detected' : '🎙️ Speech detected' }}
            </span>
            <div text-xs text-neutral-500>
              {{ Math.round(speechProbability * 100) }}% confidence
            </div>
          </div>
        </div>
      </div>
    </template>
  </TranscriptionProviderSettings>
</template>

<route lang="yaml">
meta:
  layout: settings
  stageTransition:
    name: slide
</route>
