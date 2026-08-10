import { createRouter, createWebHistory, type RouteRecordRaw } from 'vue-router'

const routes: RouteRecordRaw[] = [
  { path: '/', name: 'leaderboard', component: () => import('@/pages/LeaderboardPage.vue') },
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
  { path: '/inspect', name: 'inspect', component: () => import('@/pages/InspectPage.vue') },
  { path: '/debug', name: 'debug', component: () => import('@/pages/DebugPage.vue') },
]

export const router = createRouter({
  history: createWebHistory(),
  routes,
})
