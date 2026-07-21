<template>
  <div class="min-h-screen bg-[#f5f6f2] text-[#1d2420]">
    <header class="border-b border-[#d7dbd2] bg-[#17251e] text-white">
      <div class="mx-auto flex h-16 max-w-[1500px] items-center justify-between px-6">
        <RouterLink to="/" class="flex items-center gap-3">
          <span class="grid h-9 w-9 place-items-center rounded-lg bg-[#b9ef84] font-mono text-sm font-bold text-[#17251e]">OA</span>
          <div>
            <h1 class="text-[15px] font-semibold tracking-wide">Marin Operations</h1>
            <p class="text-[11px] text-[#aebbb3]">Grafana-triggered investigations</p>
          </div>
        </RouterLink>
        <nav class="flex items-center gap-1 text-sm">
          <RouterLink to="/" class="rounded-md px-3 py-2 hover:bg-white/10">Inbox</RouterLink>
          <RouterLink to="/diagnostics" class="rounded-md px-3 py-2 hover:bg-white/10">Diagnostics</RouterLink>
          <RouterLink to="/ask" class="rounded-md bg-[#b9ef84] px-3 py-2 font-medium text-[#17251e] hover:bg-[#cdf8a3]">Ask ops agent</RouterLink>
        </nav>
      </div>
    </header>
    <main class="mx-auto max-w-[1500px] px-6 py-7">
      <RouterView />
    </main>
  </div>
</template>
