import { createApp } from "vue";

import "@marina/tokens.css";
import "./style.css";
import App from "./App.vue";
import { router } from "./routes.js";

createApp(App).use(router).mount("#app");
