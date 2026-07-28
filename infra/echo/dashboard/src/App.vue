<script setup lang="ts"></script>

<template>
  <div class="min-h-screen">
    <header class="border-b border-line/80 bg-cream/75 backdrop-blur">
      <div class="mx-auto flex max-w-6xl items-center justify-between px-5 py-5 sm:px-8">
        <router-link class="flex items-center gap-3" to="/">
          <div class="grid size-10 place-items-center rounded-xl bg-moss text-lg font-semibold text-white shadow-card">E</div>
          <div>
            <p class="text-lg font-semibold tracking-tight">Echo</p>
            <p class="text-xs text-ink/55">Marin activity and wiki search</p>
          </div>
        </router-link>
        <div class="flex items-center gap-5">
          <router-link class="text-sm font-medium text-moss hover:text-fern" to="/wiki">Wiki</router-link>
          <a class="text-sm font-medium text-moss hover:text-fern" href="/docs">API docs</a>
        </div>
      </div>
    </header>

    <main class="mx-auto max-w-6xl px-5 pb-16 pt-12 sm:px-8 sm:pt-16">
      <router-view />
    </main>
  </div>
</template>
