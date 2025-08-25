<script setup lang="ts">
import type { TranscriptionProvider } from '@xsai-ext/shared-providers'

import {
  Button,
  TranscriptionPlayground,
  TranscriptionProviderSettings,
} from '@proj-airi/stage-ui/components'
import { useHearingStore, useProvidersStore } from '@proj-airi/stage-ui/stores'
import { computed, onMounted, ref } from 'vue'

const hearingStore = useHearingStore()
const providersStore = useProvidersStore()

const providerId = 'app-local-audio-transcription'
const defaultModel = 'whisper-base'

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

const recommendedModels = computed(() => [
  {
    id: 'whisper-large-v3-turbo',
    name: 'Whisper Large v3 Turbo',
    size: 1550000000,
    accuracy: 'high' as const,
    speed: 'medium' as const,
    installed: false,
    recommended: true,
  },
  {
    id: 'whisper-medium',
    name: 'Whisper Medium',
    size: 769000000,
    accuracy: 'medium' as const,
    speed: 'medium' as const,
    installed: false,
    recommended: true,
  },
  {
    id: 'whisper-base',
    name: 'Whisper Base',
    size: 145000000,
    accuracy: 'medium' as const,
    speed: 'fast' as const,
    installed: false,
    recommended: true,
  },
])

const allModels = computed(() => [
  ...recommendedModels.value,
  {
    id: 'whisper-small',
    name: 'Whisper Small',
    size: 244000000,
    accuracy: 'medium' as const,
    speed: 'fast' as const,
    installed: false,
  },
  {
    id: 'whisper-tiny',
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
    const { invoke } = await import('@tauri-apps/api/core')
    try {
      const installedModels = await invoke('plugin:ipc-audio-transcription-ort|list_installed_models') as string[]
      models.value.forEach((model) => {
        model.installed = installedModels.includes(model.id)
      })
    }
    catch {
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
    await providersStore.getProviderMetadata(providerId).capabilities.loadModel?.(
      { modelId },
      {
        onProgress: (info) => {
          modelProgress.value[modelId] = info.progress * 100
          if (info.done) {
            const model = models.value.find(m => m.id === modelId)
            if (model) {
              model.installed = true
              currentModel.value = modelId
            }
            delete modelProgress.value[modelId]
          }
        },
      },
    )
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

async function handleGenerateTranscription(file: File) {
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

onMounted(async () => {
  await loadModels()
})
</script>

<template>
  <TranscriptionProviderSettings
    :provider-id="providerId"
    :default-model="defaultModel"
  >
    <template #advanced-settings>
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

    <template #playground>
      <TranscriptionPlayground
        :generate-transcription="handleGenerateTranscription"
        :api-key-configured="true"
      />
    </template>
  </TranscriptionProviderSettings>
</template>

<route lang="yaml">
meta:
  layout: settings
  stageTransition:
    name: slide
</route>
