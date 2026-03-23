<script setup>
import { reactive, ref } from 'vue'

const form = reactive({
  name: '',
  domain: '人脸',
  format: '.pth'
})

const models = ref([
  { name: 'FaceNet', backbone: 'Inception-ResNet', domain: '人脸', date: '2026-03-01', size: '89 MB', enabled: true },
  { name: 'ResNet18-DRDD', backbone: 'ResNet18', domain: '医学病灶', date: '2026-02-18', size: '10 MB', enabled: false }
])

function addModel() {
  if (!form.name.trim()) return
  models.value.unshift({
    name: form.name.trim(),
    backbone: 'Custom Backbone',
    domain: form.domain,
    date: '2026-03-20',
    size: form.format === '.onnx' ? '61 MB' : '53 MB',
    enabled: false
  })
  form.name = ''
}

function removeModel(name) {
  models.value = models.value.filter((item) => item.name !== name)
}
</script>

<template>
  <div class="grid desktop-grid" style="grid-template-columns: 1fr 1.5fr;">
    <section class="card panel">
      <h2 class="section-title">模型导入</h2>
      <p class="section-desc">支持上传 .pth / .onnx，绑定适配模态并加入评测流程。</p>

      <div class="field" style="margin-top: 14px;">
        <label>模型名称</label>
        <input v-model="form.name" placeholder="例如：FaceNet 通用版" />
      </div>

      <div class="field" style="margin-top: 10px;">
        <label>权重格式</label>
        <select v-model="form.format">
          <option>.pth</option>
          <option>.onnx</option>
        </select>
      </div>

      <div style="display: flex; gap: 8px; margin-top: 12px;">
        <button class="btn" @click="addModel">上传资源</button>
        <button class="btn secondary">选择模型权重</button>
      </div>
    </section>

    <section class="card panel">
      <h2 class="section-title">模型列表维护</h2>
      <p class="section-desc">可启停与删除模型，支持横向对比恢复图像在不同网络上的泛化表现。</p>

      <table class="model-table">
        <thead>
          <tr>
            <th>名称</th>
            <th>架构</th>
            <th>领域</th>
            <th>上传日期</th>
            <th>大小</th>
            <th>状态</th>
            <th>操作</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="item in models" :key="item.name">
            <td>{{ item.name }}</td>
            <td>{{ item.backbone }}</td>
            <td>{{ item.domain }}</td>
            <td>{{ item.date }}</td>
            <td>{{ item.size }}</td>
            <td>
              <label class="switch">
                <input v-model="item.enabled" type="checkbox" />
                <span>{{ item.enabled ? '启用' : '停用' }}</span>
              </label>
            </td>
            <td>
              <button class="mini danger" @click="removeModel(item.name)">删除</button>
            </td>
          </tr>
        </tbody>
      </table>
    </section>
  </div>
</template>

<style scoped>
.panel {
  padding: 16px;
}

.model-table {
  margin-top: 12px;
  width: 100%;
  border-collapse: collapse;
}

.model-table th,
.model-table td {
  border-bottom: 1px solid #e2e8f0;
  text-align: left;
  padding: 9px 8px;
  font-size: 13px;
}

.model-table th {
  color: #475569;
  font-weight: 700;
}

.switch {
  display: inline-flex;
  gap: 8px;
  align-items: center;
}

.mini {
  border: 0;
  border-radius: 8px;
  padding: 6px 10px;
  cursor: pointer;
  font-size: 12px;
}

.mini.danger {
  background: #fee2e2;
  color: #b91c1c;
}
</style>
