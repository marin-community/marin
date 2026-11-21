<script setup>
defineProps({ images: { type: Array, required: true } });
</script>

<template>
  <div class="image-gallery">
    <slot />
    <figure v-for="image in images" :key="image.path">
      <img :src="image.previewUrl" :alt="image.path" />
      <figcaption>
        <a :href="image.url" :download="image.path">{{ image.path }} ↓</a>
      </figcaption>
    </figure>
  </div>
</template>
