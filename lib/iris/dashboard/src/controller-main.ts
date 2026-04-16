import '@fontsource-variable/noto-sans'
import '@fontsource-variable/noto-sans-mono'
import './styles/main.css'

import { createApp } from 'vue'
import App from './App.vue'
import router from './router'

const app = createApp(App)
app.use(router)
app.mount('#app')
