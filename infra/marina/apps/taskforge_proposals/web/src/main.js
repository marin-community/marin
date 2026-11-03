import { createApp } from "vue";

import App from "./App.vue";
import "./style.css";

try {
  const saved = localStorage.getItem("tf-theme");
  if (saved) document.documentElement.dataset.theme = saved;
} catch { /* storage unavailable */ }

createApp(App).mount("#app");
