import { createRouter, createWebHashHistory } from 'vue-router'

const routes = [
  {
    path: '/',
    component: () => import('./components/controller/JobsTab.vue'),
  },
  {
    path: '/fleet',
    component: () => import('./components/controller/FleetTab.vue'),
  },
  {
    path: '/autoscaler',
    component: () => import('./components/controller/AutoscalerTab.vue'),
  },
  {
    path: '/endpoints',
    component: () => import('./components/controller/EndpointsTab.vue'),
  },
  {
    path: '/status',
    component: () => import('./components/controller/StatusTab.vue'),
  },
  {
    path: '/transactions',
    component: () => import('./components/controller/TransactionsTab.vue'),
  },
  {
    path: '/users',
    component: () => import('./components/controller/UsersTab.vue'),
  },
  {
    path: '/threads',
    component: () => import('./components/controller/ThreadDump.vue'),
  },
  {
    path: '/job/:jobId(.+)/task/:taskId(.+)',
    component: () => import('./components/controller/TaskDetail.vue'),
    props: true,
  },
  {
    path: '/job/:jobId(.+)',
    component: () => import('./components/controller/JobDetail.vue'),
    props: true,
  },
  {
    path: '/worker/:workerId(.+)',
    component: () => import('./components/controller/WorkerDetail.vue'),
    props: true,
  },
]

export const router = createRouter({
  history: createWebHashHistory(),
  routes,
})

export default router
