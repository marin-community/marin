import { createRouter, createWebHistory, type RouteRecordRaw } from 'vue-router'
import { RUN_STATUS } from '@/types/api'

const routes: RouteRecordRaw[] = [
  // Saved panel links carry their cohort or benchmark selection in the root URL.
  { path: '/', redirect: (to) => ({ name: 'cohort' in to.query || 'benchmarks' in to.query ? 'panel' : 'runs', query: to.query }) },
  { path: '/benchmarks', name: 'benchmarks', component: () => import('@/pages/BenchmarksPage.vue') },
  { path: '/benchmarks/all', name: 'panel', component: () => import('@/pages/PanelPage.vue') },
  { path: '/models', name: 'models', component: () => import('@/pages/ModelsIndexPage.vue') },
  {
    path: '/models/:model',
    name: 'model',
    component: () => import('@/pages/ModelDetailPage.vue'),
    props: true,
  },
  { path: '/compare', name: 'compare', component: () => import('@/pages/ComparePage.vue') },
  { path: '/runs', name: 'runs', component: () => import('@/pages/RunsPage.vue') },
  {
    path: '/runs/:runId',
    name: 'run',
    component: () => import('@/pages/RunDetailPage.vue'),
    props: true,
  },
  {
    path: '/runs/:runId/samples',
    name: 'samples',
    component: () => import('@/pages/SampleViewerPage.vue'),
    props: true,
  },
  { path: '/reports', name: 'reports', component: () => import('@/pages/ReportsPage.vue') },
  {
    path: '/reports/:reportId',
    name: 'report',
    component: () => import('@/pages/ReportDetailPage.vue'),
    props: true,
  },
  { path: '/inspect', redirect: to => ({ name: 'runs', query: { ...to.query, view: 'runs', status: RUN_STATUS.SUCCEEDED } }) },
  { path: '/debug', name: 'debug', component: () => import('@/pages/DebugPage.vue') },
]

// The kernel serves this app under /evaldash/ and answers any path below it with index.html,
// so a deep link and a reload both land back here.
export const router = createRouter({
  history: createWebHistory('/evaldash/'),
  routes,
})
