import { createRouter, createWebHistory, type RouteRecordRaw } from 'vue-router'

const routes: RouteRecordRaw[] = [
  { path: '/', name: 'namespaces', component: () => import('@/pages/NamespacesPage.vue') },
  {
    path: '/ns/:name',
    name: 'namespace',
    component: () => import('@/pages/NamespaceDetailPage.vue'),
    props: true,
  },
  { path: '/query', name: 'query', component: () => import('@/pages/QueryPage.vue') },
  { path: '/logs', name: 'logs', component: () => import('@/pages/LogsPage.vue') },
]

export const router = createRouter({
  history: createWebHistory(),
  routes,
})
