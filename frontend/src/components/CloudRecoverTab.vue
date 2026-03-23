<script setup>
import { ref } from 'vue'
import decryptedImagePath from '../assets/datasets/drdd-evcs.jpg'
import restoredImagePath from '../assets/datasets/drdd-re.jpg'

const step = ref(5)
const model = ref('FaceNet')
const sobelWeight = ref(0.5)
const restored = ref(false)
const evaluated = ref(false)

function runRestore() {
  restored.value = true
  evaluated.value = false
}

function runEval() {
  if (!restored.value) return
  evaluated.value = true
}
</script>

<template>
  <div class="grid" style="gap: 14px;">
    <section class="card panel">
      <h2 class="section-title">恢复流程配置</h2>

      <div class="grid desktop-grid" style="grid-template-columns: repeat(4, minmax(0, 1fr)); margin-top: 14px;">
        <div class="field">
          <label>扩散步数</label>
          <select v-model.number="step">
            <option :value="5">5（推荐）</option>
            <option :value="20">20</option>
            <option :value="50">50</option>
          </select>
        </div>

        <div class="field">
          <label>Sobel 条件权重</label>
          <input v-model.number="sobelWeight" min="0" max="1" step="0.05" type="number" />
        </div>

        <div class="field">
          <label>下游模型</label>
          <select v-model="model">
            <option>FaceNet</option>
            <option>ResNet18-DRDD</option>
          </select>
        </div>

        <div style="display: flex; align-items: end; gap: 8px;">
          <button class="btn" @click="runRestore">恢复</button>
          <button class="btn secondary" @click="runEval">识别</button>
        </div>
      </div>
    </section>

    <section class="card panel grid desktop-grid" style="grid-template-columns: repeat(2, minmax(0, 1fr));">
      <div>
        <h4>叠加解密图 </h4>
        <div class="img-frame">
          <img :src="decryptedImagePath" alt="drdd-evcs" class="img-real" />
        </div>
      </div>
      <div>
        <h4>DiffSo 恢复图</h4>
        <div class="img-frame">
          <img :src="restoredImagePath" alt="drdd-re" class="img-real" :class="{ on: restored }" />
        </div>
      </div>
    </section>

    <section class="card panel">
      <h3 class="section-title">识别结果</h3>
      <div class="result-box">
        <div class="img-frame result-preview">
          <img :src="restoredImagePath" alt="drdd-re" class="img-real" :class="{ on: restored }" />
        </div>

        <h4 class="result-title">识别类型</h4>
        <p class="result-name" :class="{ 'disease-text': evaluated }">
          {{ evaluated ? 'DR (糖尿病视网膜病变)' : '--' }}
        </p>
      </div>

      <p class="section-desc" style="margin-top: 8px;">
        <template v-if="evaluated">
          模型 {{ model }} 识别完成，识别结果为：
          <span class="disease-text">DR (糖尿病视网膜病变)</span>
        </template>
        <template v-else>点击“识别”生成结果面板。</template>
      </p>
    </section>
  </div>
</template>

<style scoped>
.panel {
  padding: 16px;
}

.img-frame {
  margin-top: 8px;
  width: 256px;
  height: 256px;
  border-radius: 12px;
  border: 1px solid #d6e0f0;
  overflow: hidden;
  background: #eef2f7;
}

.img-real {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.img-real.on {
  filter: saturate(1.08) contrast(1.03);
}

.result-box {
  margin-top: 12px;
  max-width: 360px;
}

.result-preview {
  margin-top: 8px;
  height: 256px;
}

.result-title {
  margin-top: 12px;
  color: #334155;
}

.result-name {
  margin-top: 6px;
  padding: 10px 12px;
  border-radius: 10px;
  border: 1px solid #dbe5f5;
  background: #f8fbff;
  font-weight: 700;
  color: #1e293b;
}

.disease-text {
  color: #dc2626;
  font-weight: 700;
}
</style>
