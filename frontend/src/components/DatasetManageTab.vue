<script setup>
import { ref } from 'vue'
import lfw1 from '../assets/datasets/Aaron_Eckhart_0001.jpg'
import lfw2 from '../assets/datasets/Aaron_Guiel_0001.jpg'
import lfw3 from '../assets/datasets/Aaron_Patterson_0001.jpg'

import drdd1 from '../assets/datasets/drdd1.jpg'
import drdd2 from '../assets/datasets/drdd2.jpg'
import drdd3 from '../assets/datasets/drdd3.jpg'

const datasets = ref([
  {
    name: 'LFW数据集',
    modality: '人脸',
    count: '13200+',
    uploadDate: '2025-10-12',
    desc: '用于验证加密-恢复后的人脸识别可用性。',
    images: [lfw1, lfw2, lfw3]
  },
  {
    name: 'DRDD糖尿病视网膜数据集',
    modality: '医学病灶',
    count: '3800+',
    uploadDate: '2025-11-20',
    desc: '用于医学图像恢复与分类精度评估。',
    images: [drdd1, drdd2, drdd3]
  }
])

const datasetName = ref('')
const datasetDesc = ref('')

function addDataset() {
  if (!datasetName.value.trim()) return
  datasets.value.unshift({
    name: datasetName.value.trim(),
    modality: '自定义',
    count: '待统计',
    uploadDate: '2026-03-20',
    desc: datasetDesc.value || '用户上传数据集',
    images: [lfw1, drdd1, lfw2]
  })
  datasetName.value = ''
  datasetDesc.value = ''
}

function removeDataset(name) {
  datasets.value = datasets.value.filter((item) => item.name !== name)
}
</script>

<template>
  <div class="grid" style="gap: 14px;">
    <section class="card panel">
      <h2 class="section-title">数据集上传通道</h2>
      <p class="section-desc">支持压缩包解析、自动解压与标签化管理，便于闭环评测。</p>

      <div class="grid desktop-grid" style="grid-template-columns: 1fr 1fr 1.2fr auto; margin-top: 14px; align-items: end;">
        <div class="field">
          <label>数据集标签名</label>
          <input v-model="datasetName" placeholder="例如：FVC2000-Finger" />
        </div>

        <div class="field">
          <label>上传资源</label>
          <input type="text" placeholder="选择 zip / tar.gz（示意）" />
        </div>

        <div class="field">
          <label>简介</label>
          <input v-model="datasetDesc" placeholder="描述数据来源与用途" />
        </div>

        <button class="btn" @click="addDataset">上传并解析</button>
      </div>
    </section>

    <section class="grid desktop-grid" style="grid-template-columns: repeat(2, minmax(0, 1fr));">
      <article v-for="item in datasets" :key="item.name" class="card card-item">
        <div class="thumbs">
          <img v-for="(img, idx) in item.images" :key="`${item.name}-${idx}`" :src="img" :alt="`${item.name}-${idx}`" />
        </div>

        <h3>{{ item.name }}</h3>
        <p class="meta">模态：{{ item.modality }} ｜ 样本量：{{ item.count }}</p>
        <p class="meta">上传日期：{{ item.uploadDate }}</p>
        <p class="desc">{{ item.desc }}</p>

        <div class="actions">
          <button class="btn secondary">查看详情</button>
          <button class="btn danger" @click="removeDataset(item.name)">删除</button>
        </div>
      </article>
    </section>
  </div>
</template>

<style scoped>
.panel {
  padding: 16px;
}

.card-item {
  padding: 14px;
}

.thumbs {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 6px;
  margin-bottom: 10px;
}

.thumbs img {
  height: 72px;
  width: 100%;
  border-radius: 10px;
  border: 1px solid #d9e4f5;
  object-fit: cover;
}

h3 {
  font-size: 17px;
  color: #1f2937;
}

.meta {
  margin-top: 6px;
  font-size: 13px;
  color: #64748b;
}

.desc {
  margin-top: 8px;
  font-size: 14px;
  color: #334155;
  line-height: 1.6;
}

.actions {
  margin-top: 12px;
  display: flex;
  gap: 8px;
}

.btn.danger {
  background: #dc2626;
}

.btn.danger:hover {
  background: #b91c1c;
}
</style>
