import { createRouter, createWebHashHistory } from 'vue-router'

const routes = [
  {
    path: '/',
    component: () => import('./components/controller/JobsTab.vue'),
  },
  {
    path: '/backends',
    component: () => import('./components/controller/BackendsTab.vue'),
  },
  {
    path: '/tasks',
    component: () => import('./components/controller/TasksTab.vue'),
  },
  {
    path: '/nodes',
    component: () => import('./components/controller/NodesTab.vue'),
  },
  {
    path: '/capacity',
    component: () => import('./components/controller/CapacityTab.vue'),
  },
  {
    path: '/endpoints',
    component: () => import('./components/controller/EndpointsTab.vue'),
  },
  {
    path: '/logs',
    component: () => import('./components/controller/LogsTab.vue'),
  },
  {
    path: '/status',
    component: () => import('./components/controller/StatusTab.vue'),
  },
  {
    path: '/account',
    component: () => import('./components/controller/AccountTab.vue'),
  },
  {
    path: '/system/controller/threads',
    component: () => import('./components/controller/ThreadDump.vue'),
  },
  {
    path: '/task/:clusterId/:taskId(.+)',
    name: 'task-detail',
    component: () => import('./components/controller/TaskDetail.vue'),
    props: true,
  },
  {
    path: '/job/:clusterId/:jobId(.+)',
    name: 'job-detail',
    component: () => import('./components/controller/JobDetail.vue'),
    props: true,
  },
  {
    path: '/node/:clusterId/:backendId/:nodeUid/:nodeId(.+)',
    name: 'node-detail',
    component: () => import('./components/controller/NodeDetail.vue'),
    props: true,
  },
  {
    path: '/:pathMatch(.*)*',
    redirect: '/',
  },
]

export const router = createRouter({
  history: createWebHashHistory(),
  routes,
})

export default router
