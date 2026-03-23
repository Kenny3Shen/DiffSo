<script setup>
import { computed, ref } from 'vue'
import EdgeEncryptTab from './components/EdgeEncryptTab.vue'
import CloudRecoverTab from './components/CloudRecoverTab.vue'
import DatasetManageTab from './components/DatasetManageTab.vue'
import ModelManageTab from './components/ModelManageTab.vue'

const tabs = [
  { key: 'edge', label: '边缘轻量级加密', component: EdgeEncryptTab },
  { key: 'cloud', label: '云端恢复与识别', component: CloudRecoverTab },
  { key: 'dataset', label: '数据集管理', component: DatasetManageTab },
  { key: 'model', label: '模型管理', component: ModelManageTab }
]

const activeTab = ref('edge')
const activeComponent = computed(() => tabs.find((tab) => tab.key === activeTab.value)?.component)
</script>

<template>
  <div class="app-wrapper">
    <header class="header card">
      <div>
        <h1>轻量级安全图像传输系统</h1>
      </div>

    </header>

    <nav class="tabs card">
      <button
        v-for="item in tabs"
        :key="item.key"
        :class="['tab-btn', { active: item.key === activeTab }]"
        @click="activeTab = item.key"
      >
        {{ item.label }}
      </button>
    </nav>

    <main class="content card">
      <component :is="activeComponent" />
    </main>
  </div>
</template>

<style scoped>
.app-wrapper {
  width: min(1280px, 94vw);
  margin: 24px auto;
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.header {
  padding: 18px 22px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 16px;
}

.header h1 {
  font-size: 26px;
  color: #0f172a;
}

.header p {
  margin-top: 8px;
  color: #64748b;
  font-size: 14px;
}

.tabs {
  padding: 10px;
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

.tab-btn {
  border: 1px solid #dbe5f4;
  border-radius: 10px;
  background: #fff;
  color: #334155;
  padding: 9px 14px;
  cursor: pointer;
  font-weight: 600;
}

.tab-btn.active {
  background: linear-gradient(135deg, #2563eb, #1d4ed8);
  border-color: #1d4ed8;
  color: white;
}

.content {
  padding: 18px;
  min-height: 680px;
}
</style>
