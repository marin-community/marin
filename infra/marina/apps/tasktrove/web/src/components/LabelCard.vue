<script setup lang="ts">
// What the audit made of one sampled task.
import type { Label } from '../corpus'
import Score from './Score.vue'
import Verdict from './Verdict.vue'

defineProps<{ label: Label }>()
</script>

<template>
  <div class="panel label" :data-verdict="label.shellsim_now">
    <p class="summary">{{ label.summary }}</p>
    <div class="tags">
      <span class="chip">{{ label.task_kind }}</span>
      <span class="chip">verifier: {{ label.verifier_mechanism }}</span>
      <Verdict :value="label.shellsim_now" prefix="shellsim: " />
      <span v-if="label.shellsim_with !== 'none'" class="chip">needs {{ label.shellsim_with }}</span>
      <span v-for="need in label.agent_needs" :key="need" class="chip">{{ need }}</span>
    </div>
    <dl class="facts">
      <dt>Interesting</dt>
      <dd><Score :value="label.interesting" /> {{ label.interesting }}</dd>
      <dt>Well defined</dt>
      <dd><Score :value="label.well_defined" /> {{ label.well_defined }}</dd>
      <dt>Hack risk</dt>
      <dd><Score :value="label.hack_risk" :warn="label.hack_risk >= 4" /> {{ label.hack_risk }}</dd>
      <dt v-if="label.hack_vector">Hack vector</dt>
      <dd v-if="label.hack_vector">{{ label.hack_vector }}</dd>
      <dt v-if="label.defects">Defects</dt>
      <dd v-if="label.defects">{{ label.defects }}</dd>
    </dl>
  </div>
</template>
