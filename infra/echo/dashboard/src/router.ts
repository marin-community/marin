// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

import { createRouter, createWebHistory } from 'vue-router'
import ChunkView from './views/ChunkView.vue'
import SearchView from './views/SearchView.vue'
import WikiEntryView from './views/WikiEntryView.vue'
import WikiListView from './views/WikiListView.vue'

export const router = createRouter({
  history: createWebHistory(),
  routes: [
    { path: '/', name: 'search', component: SearchView },
    { path: '/wiki', name: 'wiki-list', component: WikiListView },
    { path: '/wiki/:id', name: 'wiki-entry', component: WikiEntryView, props: true },
    { path: '/chunk/:id', name: 'chunk', component: ChunkView, props: true },
  ],
})
