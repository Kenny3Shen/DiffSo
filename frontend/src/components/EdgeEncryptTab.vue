<script setup>
import { onBeforeUnmount, ref } from 'vue'
import share1Img from '../share1.png'
import share2Img from '../share2.png'

const threshold = ref(128)
const dither = ref('Floyd-Steinberg')
const encrypted = ref(false)
const originalImageUrl = ref('')

const latencyMs = '1.62'

function revokeObjectUrlIfNeeded() {
  if (originalImageUrl.value && originalImageUrl.value.startsWith('blob:')) {
    URL.revokeObjectURL(originalImageUrl.value)
  }
}

function handleImageUpload(event) {
  const file = event.target.files?.[0]
  if (!file) return
  revokeObjectUrlIfNeeded()
  originalImageUrl.value = URL.createObjectURL(file)
}

function runEncryption() {
  encrypted.value = true
}

function resetPanel() {
  encrypted.value = false
}

onBeforeUnmount(() => {
  revokeObjectUrlIfNeeded()
})
</script>

<template>
  <div class="grid desktop-grid" style="grid-template-columns: 1.2fr 1fr;">
    <section class="card panel">
      <h2 class="section-title">图像与参数配置</h2>

      <div class="grid" style="grid-template-columns: repeat(2, minmax(0, 1fr)); margin-top: 16px;">
        <div class="field" style="grid-column: span 2;">
          <label>输入图像</label>
          <input type="file" accept="image/*" @change="handleImageUpload" />
        </div>

        <div class="field" style="grid-column: span 2;">
          <label>原始图像预览</label>
          <div class="original-preview">
            <img v-if="originalImageUrl" :src="originalImageUrl" alt="原始图像" />
            <span v-else>请先上传图像</span>
          </div>
        </div>

        <div class="field">
          <label>重映射阈值</label>
          <input v-model.number="threshold" type="range" min="0" max="255" />
          <small>{{ threshold }}</small>
        </div>

        <div class="field">
          <label>扩散算法</label>
          <select v-model="dither">
            <option>Floyd-Steinberg</option>
            <option>Ostromoukhov</option>
            <option>Jarvis-Judice-Ninke</option>
          </select>
        </div>
      </div>

      <div style="display: flex; gap: 10px; margin-top: 14px;">
        <button class="btn" @click="runEncryption">加密</button>
        <button class="btn secondary" @click="resetPanel">重置</button>
      </div>
    </section>

    <section class="card panel">
      <h2 class="section-title">分享可视化</h2>

      <div class="share-grid">
        <div class="share-box">
          <h4>Share 1</h4>
          <img class="share-image" :class="{ active: encrypted }" :src="share1Img" alt="Share 1" />
        </div>
        <div class="share-box">
          <h4>Share 2</h4>
          <img class="share-image" :class="{ active: encrypted }" :src="share2Img" alt="Share 2" />
        </div>
      </div>

      <div class="metrics">
        <div>
          <span>加密耗时</span>
          <strong>{{ latencyMs }} ms</strong>
        </div>
        <div>
          <span>算法</span>
          <strong>{{ dither }}</strong>
        </div>
      </div>
    </section>
  </div>
</template>

<style scoped>
.panel {
  padding: 16px;
}

.original-preview {
  margin-top: 2px;
  border: 1px dashed #c7d3e6;
  border-radius: 12px;
  min-height: 160px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: #f8fbff;
  overflow: hidden;
}

.original-preview img {
  width: 100%;
  max-height: 260px;
  object-fit: contain;
}

.original-preview span {
  color: #64748b;
  font-size: 14px;
}

.share-grid {
  margin-top: 14px;
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 12px;
}

.share-box {
  border: 1px dashed #c7d3e6;
  border-radius: 12px;
  padding: 10px;
}

.share-box h4 {
  margin-bottom: 8px;
  color: #334155;
}

.share-image {
  width: 100%;
  aspect-ratio: 1 / 1;
  border-radius: 8px;
  object-fit: cover;
  border: 1px solid #d3ddeb;
  filter: grayscale(0.2);
}

.share-image.active {
  filter: grayscale(0);
}

.metrics {
  margin-top: 12px;
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 10px;
}

.metrics > div {
  background: #f8fbff;
  border: 1px solid #dbe5f5;
  border-radius: 10px;
  padding: 10px;
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.metrics span {
  color: #64748b;
  font-size: 13px;
}
</style>
