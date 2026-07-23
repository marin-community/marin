import { createRouter, createWebHistory, type RouteRecordRaw } from 'vue-router'

const routes: RouteRecordRaw[] = [
  { path: '/', name: 'leaderboard', component: () => import('@/pages/LeaderboardPage.vue') },
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
  { path: '/status', name: 'status', component: () => import('@/pages/StatusPage.vue') },
]

export const router = createRouter({
  history: createWebHistory(),
  routes,
})
