import '@fontsource-variable/noto-sans'
import '@fontsource-variable/noto-sans-mono'
import './styles/main.css'

import { createApp } from 'vue'
import { createRouter, createWebHashHistory } from 'vue-router'
import WorkerApp from './components/worker/WorkerApp.vue'

const router = createRouter({
  history: createWebHashHistory(),
  routes: [
    {
      path: '/',
      component: () => import('./components/worker/WorkerStatusPage.vue'),
    },
    {
      path: '/task/:taskId/threads',
      component: () => import('./components/controller/ThreadDump.vue'),
      props: true,
      meta: { rpc: 'worker' },
    },
    {
      path: '/task/:taskId',
      component: () => import('./components/worker/WorkerTaskDetail.vue'),
      props: true,
    },
  ],
})

const app = createApp(WorkerApp)
app.use(router)
app.mount('#app')
