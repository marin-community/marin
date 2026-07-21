import { createRouter, createWebHistory, type RouteRecordRaw } from 'vue-router'

const routes: RouteRecordRaw[] = [
  { path: '/', component: () => import('@/pages/InboxPage.vue') },
  { path: '/diagnostics', component: () => import('@/pages/DiagnosticsPage.vue') },
  { path: '/cases/:caseId', component: () => import('@/pages/CasePage.vue'), props: true },
  { path: '/ask', component: () => import('@/pages/AskPage.vue') },
]

export const router = createRouter({ history: createWebHistory(), routes })
