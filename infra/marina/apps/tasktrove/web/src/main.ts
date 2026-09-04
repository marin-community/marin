import { createApp } from 'vue'
import '@marina/tokens.css'
import '@marina/parts.css'
import './style.css'
import App from './App.vue'
import { router } from './routes'

createApp(App).use(router).mount('#app')
