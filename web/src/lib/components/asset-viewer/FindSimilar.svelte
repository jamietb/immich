<script lang="ts">
  const { asset } = $props<{ asset: { id: string } }>();
  let photos = $state(true);
  let videos = $state(true);
  const noneSelected = $derived(!photos && !videos);

  function openSimilar() {
    if (noneSelected) return; // hard block
    const ts: string[] = [];
    if (photos) ts.push('image');
    if (videos) ts.push('video');
    const url = `/similar/${asset.id}?types=${encodeURIComponent(ts.join(','))}`;
    window.open(url, '_blank', 'noopener,noreferrer');
  }
</script>

<div class="flex flex-col gap-3 pt-4 border-t border-gray-200 dark:border-gray-700">
  <div class="text-sm font-semibold text-gray-700 dark:text-gray-200">Find Similar</div>
  <div class="flex items-center gap-4">
    <label class="inline-flex items-center gap-2 text-sm">
      <input type="checkbox" bind:checked={photos} class="h-4 w-4" />
      <span>Photos</span>
    </label>
    <label class="inline-flex items-center gap-2 text-sm">
      <input type="checkbox" bind:checked={videos} class="h-4 w-4" />
      <span>Videos</span>
    </label>
  </div>
  <div>
    <button class="btn btn-primary" on:click={openSimilar} disabled={noneSelected}>Find Similar</button>
  </div>
</div>
