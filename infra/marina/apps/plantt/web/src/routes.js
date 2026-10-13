import { createRouter, createWebHistory } from "vue-router";

import Workspace from "./views/Workspace.vue";

export const router = createRouter({
  history: createWebHistory("/plantt/"),
  routes: [
    { path: "/", component: Workspace },
    { path: "/charts/:id", component: Workspace },
  ],
});
