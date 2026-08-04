<template>
  <div class="min-h-screen">
    <header class="border-b border-line bg-cream/90">
      <div class="mx-auto flex max-w-6xl flex-col gap-3 px-5 py-4 sm:flex-row sm:items-center sm:justify-between sm:px-8">
        <router-link class="flex items-baseline gap-2" to="/">
          <span class="text-xl font-semibold tracking-tight">Echo</span>
          <span class="text-xs text-ink/45">Marin search</span>
        </router-link>
        <nav class="flex items-center gap-1 overflow-x-auto text-sm" aria-label="Echo areas">
          <router-link class="quick-tab" to="/">Search</router-link>
          <a class="quick-tab" href="/?domain=file">Files</a>
          <router-link class="quick-tab" to="/wiki">Wiki</router-link>
          <a class="quick-tab" href="/?domain=pr&domain=issue">GitHub</a>
          <a class="quick-tab" href="/?domain=discord">Discord</a>
          <a class="quick-tab" href="/docs">API</a>
        </nav>
      </div>
    </header>

    <main class="mx-auto max-w-6xl px-5 pb-16 pt-8 sm:px-8 sm:pt-10">
      <router-view />
    </main>
  </div>
</template>
