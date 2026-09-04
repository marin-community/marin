import { createApp } from 'vue'
// The kit's tokens carry the shell's colours, its reset and its phone rules; the instrument
// palette in main.css is this app's own layer on top of them, so it is imported second.
import '@marina/tokens.css'
import App from './App.vue'
import { router } from './router'
import './styles/main.css'

createApp(App).use(router).mount('#app')
