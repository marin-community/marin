// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

import { createRouter, createWebHistory } from 'vue-router'
import ChunkView from './views/ChunkView.vue'
import ConversationView from './views/ConversationView.vue'
import FeedbackView from './views/FeedbackView.vue'
import SearchView from './views/SearchView.vue'
import WikiEntryView from './views/WikiEntryView.vue'
import WikiListView from './views/WikiListView.vue'

export const router = createRouter({
  history: createWebHistory(),
  routes: [
    { path: '/', name: 'search', component: SearchView },
    { path: '/wiki', name: 'wiki-list', component: WikiListView },
    { path: '/conversation', name: 'conversation', component: ConversationView },
    { path: '/feedback', name: 'feedback', component: FeedbackView },
    { path: '/wiki/:id', name: 'wiki-entry', component: WikiEntryView, props: true },
    { path: '/chunk/:id', name: 'chunk', component: ChunkView, props: true },
  ],
})
