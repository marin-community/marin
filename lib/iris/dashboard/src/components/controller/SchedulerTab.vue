<script setup lang="ts">
import { computed, onMounted } from 'vue'
import { useControllerRpc } from '@/composables/useRpc'
import { useAutoRefresh } from '@/composables/useAutoRefresh'
import type {
  GetSchedulerStateResponse,
  SchedulerBandGroup,
  SchedulerTaskEntry,
  SchedulerUserBudget,
  SchedulerRunningTask,
} from '@/types/rpc'
import EmptyState from '@/components/shared/EmptyState.vue'

const { data, loading, error, refresh } = useControllerRpc<GetSchedulerStateResponse>('GetSchedulerState')

useAutoRefresh(refresh, 15_000)
onMounted(refresh)

const pendingQueue = computed<SchedulerBandGroup[]>(() => data.value?.pendingQueue ?? [])
const userBudgets = computed<SchedulerUserBudget[]>(() => data.value?.userBudgets ?? [])
const runningTasks = computed<SchedulerRunningTask[]>(() => data.value?.runningTasks ?? [])
const totalPending = computed(() => data.value?.totalPending ?? 0)
const totalRunning = computed(() => data.value?.totalRunning ?? 0)

function bandDisplayName(band: string | undefined): string {
  if (!band) return 'Unknown'
  const name = band.replace(/^PRIORITY_BAND_/, '')
  return name.charAt(0) + name.slice(1).toLowerCase()
}

function isDowngraded(entry: SchedulerTaskEntry): boolean {
  return entry.originalBand !== entry.effectiveBand
}

function bandColor(band: string | undefined): string {
  if (!band) return 'text-text-muted'
  const name = band.replace(/^PRIORITY_BAND_/, '')
  if (name === 'PRODUCTION') return 'text-status-danger'
  if (name === 'INTERACTIVE') return 'text-accent'
  if (name === 'BATCH') return 'text-text-muted'
  return 'text-text-muted'
}

function utilizationColor(pct: number): string {
  if (pct >= 100) return 'text-status-danger font-semibold'
  if (pct >= 75) return 'text-status-warning'
  return 'text-text-primary'
}

function preemptibleByNames(bands: string[]): string {
  if (!bands || bands.length === 0) return '-'
  return bands.map(bandDisplayName).join(', ')
}
</script>

<template>
  <!-- Error -->
  <div
    v-if="error"
    class="mb-4 px-4 py-3 text-sm text-status-danger bg-status-danger-bg rounded-lg border border-status-danger-border"
  >
    {{ error }}
  </div>

  <!-- Loading -->
  <div v-if="loading && !data" class="flex items-center justify-center py-12 text-text-muted text-sm">
    <svg class="animate-spin -ml-1 mr-2 h-4 w-4" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
      <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" />
      <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
    </svg>
    Loading...
  </div>

  <div v-else-if="data" class="space-y-8">
    <!-- Pending Queue -->
    <section>
      <h2 class="text-lg font-semibold mb-3">Pending Queue ({{ totalPending }})</h2>
      <EmptyState v-if="pendingQueue.length === 0" message="No pending tasks" />
      <div v-else class="space-y-4">
        <div v-for="group in pendingQueue" :key="group.band">
          <h3 class="text-sm font-semibold mb-1" :class="bandColor(group.band)">
            {{ bandDisplayName(group.band) }}
            <span class="text-text-muted font-normal">({{ group.totalInBand }} task{{ group.totalInBand !== 1 ? 's' : '' }})</span>
          </h3>
          <div class="overflow-x-auto">
            <table class="w-full border-collapse">
              <thead>
                <tr class="border-b border-surface-border">
                  <th class="px-3 py-2 text-left text-xs font-semibold uppercase tracking-wider text-text-secondary">#</th>
                  <th class="px-3 py-2 text-left text-xs font-semibold uppercase tracking-wider text-text-secondary">Task</th>
                  <th class="px-3 py-2 text-left text-xs font-semibold uppercase tracking-wider text-text-secondary">Job</th>
                  <th class="px-3 py-2 text-left text-xs font-semibold uppercase tracking-wider text-text-secondary">User</th>
                  <th class="px-3 py-2 text-left text-xs font-semibold uppercase tracking-wider text-text-secondary">Band</th>
                  <th class="px-3 py-2 text-right text-xs font-semibold uppercase tracking-wider text-text-secondary">Resource Value</th>
                </tr>
              </thead>
              <tbody>
                <tr
                  v-for="task in group.tasks"
                  :key="task.taskId"
                  class="border-b border-surface-border-subtle hover:bg-surface-raised transition-colors"
                >
                  <td class="px-3 py-2 text-[13px] tabular-nums text-text-muted">{{ task.queuePosition }}</td>
                  <td class="px-3 py-2 text-[13px] font-mono">{{ task.taskId }}</td>
                  <td class="px-3 py-2 text-[13px] font-mono">{{ task.jobId }}</td>
                  <td class="px-3 py-2 text-[13px]">{{ task.userId }}</td>
                  <td class="px-3 py-2 text-[13px]">
                    <span :class="bandColor(task.effectiveBand)">{{ bandDisplayName(task.effectiveBand) }}</span>
                    <span
                      v-if="isDowngraded(task)"
                      class="ml-1 text-xs text-status-warning"
                      :title="'Originally ' + bandDisplayName(task.originalBand)"
                    >
                      (was {{ bandDisplayName(task.originalBand) }})
                    </span>
                  </td>
                  <td class="px-3 py-2 text-[13px] text-right tabular-nums">{{ task.resourceValue }}</td>
                </tr>
              </tbody>
            </table>
            <div v-if="group.totalInBand > group.tasks.length" class="px-3 py-2 text-xs text-text-secondary border-t border-surface-border">
              Showing {{ group.tasks.length }} of {{ group.totalInBand }} tasks
            </div>
          </div>
        </div>
      </div>
    </section>

    <!-- User Budgets -->
    <section>
      <h2 class="text-lg font-semibold mb-3">User Budgets</h2>
      <EmptyState v-if="userBudgets.length === 0" message="No budget configurations" />
      <div v-else class="overflow-x-auto">
        <table class="w-full border-collapse">
          <thead>
            <tr class="border-b border-surface-border">
              <th class="px-3 py-2 text-left text-xs font-semibold uppercase tracking-wider text-text-secondary">User</th>
              <th class="px-3 py-2 text-right text-xs font-semibold uppercase tracking-wider text-text-secondary">Spent</th>
              <th class="px-3 py-2 text-right text-xs font-semibold uppercase tracking-wider text-text-secondary">Limit</th>
              <th class="px-3 py-2 text-right text-xs font-semibold uppercase tracking-wider text-text-secondary">Utilization</th>
              <th class="px-3 py-2 text-left text-xs font-semibold uppercase tracking-wider text-text-secondary">Max Band</th>
              <th class="px-3 py-2 text-left text-xs font-semibold uppercase tracking-wider text-text-secondary">Effective Band</th>
            </tr>
          </thead>
          <tbody>
            <tr
              v-for="budget in userBudgets"
              :key="budget.userId"
              class="border-b border-surface-border-subtle hover:bg-surface-raised transition-colors"
            >
              <td class="px-3 py-2 text-[13px] font-mono">{{ budget.userId }}</td>
              <td class="px-3 py-2 text-[13px] text-right tabular-nums">{{ budget.budgetSpent }}</td>
              <td class="px-3 py-2 text-[13px] text-right tabular-nums">
                {{ budget.budgetLimit === '0' ? 'Unlimited' : budget.budgetLimit }}
              </td>
              <td class="px-3 py-2 text-[13px] text-right tabular-nums" :class="utilizationColor(budget.utilizationPercent)">
                {{ budget.budgetLimit === '0' ? '-' : budget.utilizationPercent.toFixed(1) + '%' }}
              </td>
              <td class="px-3 py-2 text-[13px]">
                <span :class="bandColor(budget.maxBand)">{{ bandDisplayName(budget.maxBand) }}</span>
              </td>
              <td class="px-3 py-2 text-[13px]">
                <span :class="bandColor(budget.effectiveBand)">{{ bandDisplayName(budget.effectiveBand) }}</span>
                <span
                  v-if="budget.maxBand !== budget.effectiveBand"
                  class="ml-1 text-xs text-status-warning"
                >
                  (downgraded)
                </span>
              </td>
            </tr>
          </tbody>
        </table>
        <div class="px-3 py-2 text-xs text-text-secondary border-t border-surface-border">
          {{ userBudgets.length }} user{{ userBudgets.length !== 1 ? 's' : '' }}
        </div>
      </div>
    </section>

    <!-- Running Tasks -->
    <section>
      <h2 class="text-lg font-semibold mb-3">Running Tasks ({{ totalRunning }})</h2>
      <EmptyState v-if="runningTasks.length === 0" message="No running tasks" />
      <div v-else class="overflow-x-auto">
        <table class="w-full border-collapse">
          <thead>
            <tr class="border-b border-surface-border">
              <th class="px-3 py-2 text-left text-xs font-semibold uppercase tracking-wider text-text-secondary">Task</th>
              <th class="px-3 py-2 text-left text-xs font-semibold uppercase tracking-wider text-text-secondary">Job</th>
              <th class="px-3 py-2 text-left text-xs font-semibold uppercase tracking-wider text-text-secondary">User</th>
              <th class="px-3 py-2 text-left text-xs font-semibold uppercase tracking-wider text-text-secondary">Worker</th>
              <th class="px-3 py-2 text-left text-xs font-semibold uppercase tracking-wider text-text-secondary">Eff. Band</th>
              <th class="px-3 py-2 text-right text-xs font-semibold uppercase tracking-wider text-text-secondary">Resource Value</th>
              <th class="px-3 py-2 text-left text-xs font-semibold uppercase tracking-wider text-text-secondary">Preemptible</th>
              <th class="px-3 py-2 text-left text-xs font-semibold uppercase tracking-wider text-text-secondary">Preemptible By</th>
              <th class="px-3 py-2 text-left text-xs font-semibold uppercase tracking-wider text-text-secondary">Cosched.</th>
            </tr>
          </thead>
          <tbody>
            <tr
              v-for="task in runningTasks"
              :key="task.taskId"
              class="border-b border-surface-border-subtle hover:bg-surface-raised transition-colors"
            >
              <td class="px-3 py-2 text-[13px] font-mono">{{ task.taskId }}</td>
              <td class="px-3 py-2 text-[13px] font-mono">{{ task.jobId }}</td>
              <td class="px-3 py-2 text-[13px]">{{ task.userId }}</td>
              <td class="px-3 py-2 text-[13px] font-mono">{{ task.workerId }}</td>
              <td class="px-3 py-2 text-[13px]">
                <span :class="bandColor(task.effectiveBand)">{{ bandDisplayName(task.effectiveBand) }}</span>
              </td>
              <td class="px-3 py-2 text-[13px] text-right tabular-nums">{{ task.resourceValue }}</td>
              <td class="px-3 py-2 text-[13px]">
                <span v-if="task.preemptible" class="text-status-warning">Yes</span>
                <span v-else class="text-text-muted">No</span>
              </td>
              <td class="px-3 py-2 text-[13px]">{{ preemptibleByNames(task.preemptibleBy) }}</td>
              <td class="px-3 py-2 text-[13px]">
                <span v-if="task.isCoscheduled" class="text-status-purple">Yes</span>
                <span v-else class="text-text-muted">No</span>
              </td>
            </tr>
          </tbody>
        </table>
        <div class="px-3 py-2 text-xs text-text-secondary border-t border-surface-border">
          {{ totalRunning }} task{{ totalRunning !== 1 ? 's' : '' }}
        </div>
      </div>
    </section>
  </div>
</template>
