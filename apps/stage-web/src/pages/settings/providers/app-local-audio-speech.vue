<script setup lang="ts">
import type { SpeechProvider } from '@xsai-ext/shared-providers'

import {
  Button,
  SpeechPlayground,
  SpeechProviderSettings,
} from '@proj-airi/stage-ui/components'
import { useProvidersStore, useSpeechStore } from '@proj-airi/stage-ui/stores'
import { FieldRange } from '@proj-airi/ui'
import { computed, onMounted, ref, watch } from 'vue'

const providerId = 'app-local-audio-speech'
const defaultModel = 'piper-amy-en'

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
    const _modelList = await providersStore.getProviderMetadata(providerId).capabilities.listModels?.({}) || []
    models.value = [
      {
        id: 'piper-amy-en',
        name: 'Piper Amy (English)',
        size: 63000000,
        quality: 'high',
        languages: ['English'],
        installed: false,
        recommended: true,
      },
      {
        id: 'piper-ryan-en',
        name: 'Piper Ryan (English)',
        size: 63000000,
        quality: 'high',
        languages: ['English'],
        installed: false,
        recommended: true,
      },
      {
        id: 'mimic3-en_US-vctk',
        name: 'Mimic 3 VCTK',
        size: 180000000,
        quality: 'high',
        languages: ['English'],
        installed: false,
        recommended: true,
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
    await providersStore.getProviderMetadata(providerId).capabilities.loadModel?.(
      { modelId },
      {
        onProgress: (info) => {
          modelProgress.value[modelId] = info.progress * 100
          if (info.done) {
            const model = models.value.find(m => m.id === modelId)
            if (model)
              model.installed = true
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

onMounted(async () => {
  await loadModels()
  await speechStore.loadVoicesForProvider(providerId)
})

watch(availableVoices, () => {
  models.value.forEach((model) => {
    model.installed = availableVoices.value.some(v => v.id.includes(model.id.split('-')[1]))
  })
})

async function handleGenerateSpeech(input: string, voiceId: string) {
  const provider = await providersStore.getProviderInstance(providerId) as SpeechProvider
  if (!provider) {
    throw new Error('Failed to initialize speech provider')
  }

  const providerConfig = providersStore.getProviderConfig(providerId)
  const model = providerConfig.model as string | undefined || defaultModel

  return await speechStore.speech(
    provider,
    model,
    input,
    voiceId,
    providerConfig,
  )
}
</script>

<template>
  <SpeechProviderSettings :provider-id="providerId" :default-model="defaultModel">
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
      <div flex="~ col gap-4" mt-4>
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

    <template #playground>
      <SpeechPlayground
        :available-voices="availableVoices"
        :generate-speech="handleGenerateSpeech"
        :api-key-configured="true"
        :use-ssml="false"
        default-text="Hello! This is a test of the local text-to-speech system."
      />
    </template>
  </SpeechProviderSettings>
</template>

<route lang="yaml">
meta:
  layout: settings
  stageTransition:
    name: slide
</route>
