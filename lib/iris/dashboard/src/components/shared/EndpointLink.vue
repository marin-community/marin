<script setup lang="ts">
import { computed } from 'vue'
import { canProxyEndpoint, proxyPathForEndpoint, endpointLabel } from '@/utils/endpoints'

// A registered endpoint rendered as a link through the controller's reverse
// proxy (e.g. /tunix/inference/server -> /proxy/tunix.inference.server). Names
// the proxy cannot encode (they contain a literal dot) fall back to muted
// text. Sizing/font come from a fallthrough `class` on the rendered element.
defineOptions({ inheritAttrs: false })

const props = defineProps<{
  name: string
  /** Show only the last path segment instead of the full name. */
  short?: boolean
}>()

const text = computed(() => (props.short ? endpointLabel(props.name) : props.name))
</script>

<template>
  <a
    v-if="canProxyEndpoint(name)"
    v-bind="$attrs"
    :href="proxyPathForEndpoint(name)"
    target="_blank"
    rel="noopener"
    class="text-accent hover:underline inline-flex items-center gap-1"
    :title="`Open ${name} via proxy`"
  >
    <span aria-hidden="true">↗</span>{{ text }}
  </a>
  <span
    v-else
    v-bind="$attrs"
    class="text-text-muted"
    :title="`Not proxyable: ${name} contains a dot`"
  >
    {{ text }}
  </span>
</template>
