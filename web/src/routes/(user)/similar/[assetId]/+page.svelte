<script lang="ts">
  import { page } from '$app/state';
  import { onMount } from 'svelte';

  // Route params and initial query
  const params = $derived(page.params);
  const search = $derived(page.url.searchParams);

  const assetId: string = $derived(params.assetId);

  // Filters (UI state)
  let types = $state(search.get('types') ?? 'image,video'); // 'image,video' | 'image' | 'video'
  let stacksOnly = $state(true);
  let minDate = $state<string | null>(null);   // ISO datetime-local (e.g., 2025-08-24T09:00)
  let maxDate = $state<string | null>(null);
  let albumId = $state<string | null>(null);
  let cameraMake = $state<string | null>(null);
  let cameraModel = $state<string | null>(null);

  // Grouping
  let groupByType = $state(true);

  // Data state
  let loading = $state(true);
  let error: string | null = $state(null);
  let items = $state<Array<{ id: string; type: 'IMAGE' | 'VIDEO'; distance?: number }>>([]);
  let offset = $state(0);
  const limit = 48;
  let hasMore = $state(true);

  // Save-as-album state
  let saving = $state(false);
  let albumName = $state('');

  // Derived helpers
  const photosItems = $derived(items.filter((a) => a.type === 'IMAGE'));
  const videosItems = $derived(items.filter((a) => a.type === 'VIDEO'));

  function score(distance?: number) {
    if (distance == null) return null;
    // map cosine distance [0..2] to similarity [100..0]
    const sim = Math.max(0, 1 - distance / 2);
    return Math.round(sim * 100);
  }

  function buildQuery(extra?: Record<string, string>) {
    const qp = new URLSearchParams();
    if (types) qp.set('types', types);
    qp.set('limit', String(limit));
    qp.set('offset', String(offset));
    qp.set('stacksOnly', String(stacksOnly));
    if (minDate) qp.set('minDate', minDate);
    if (maxDate) qp.set('maxDate', maxDate);
    if (albumId) qp.set('albumId', albumId);
    if (cameraMake) qp.set('cameraMake', cameraMake);
    if (cameraModel) qp.set('cameraModel', cameraModel);
    if (extra) for (const [k, v] of Object.entries(extra)) qp.set(k, v);
    return qp.toString();
  }

  async function fetchPage(reset = false) {
    try {
      if (reset) {
        offset = 0;
        items = [];
        hasMore = true;
      }
      loading = true;
      error = null;

      // block if no types selected
      const selected = types.split(',').map((t) => t.trim()).filter(Boolean);
      const noneSelected = !selected.includes('image') && !selected.includes('video');
      if (noneSelected) {
        items = [];
        hasMore = false;
        return;
      }

      const res = await fetch(`/api/assets/${assetId}/similar?` + buildQuery(), {
        credentials: 'include',
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);

      const data = await res.json(); // { items: [...] }
      const batch = (data?.items ?? []) as Array<{ id: string; type: 'IMAGE' | 'VIDEO'; distance?: number }>;
      items = items.concat(batch);
      hasMore = batch.length === limit;
      offset += batch.length;
    } catch (e: any) {
      error = e?.message ?? 'Failed to load similar assets';
    } finally {
      loading = false;
    }
  }

  async function saveAsAlbum() {
    try {
      if (!items.length || !albumName.trim()) return;
      saving = true;
      const assetIds = items.map((a) => a.id);
      // Immich albums API (v1.139.x)
      const res = await fetch('/api/albums', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify({
          albumName: albumName.trim(),
          assetIds,
        }),
      });
      if (!res.ok) throw new Error(`Album create failed: HTTP ${res.status}`);
      albumName = '';
      alert('Album created from similar results!');
    } catch (e: any) {
      alert(e?.message ?? 'Failed to create album');
    } finally {
      saving = false;
    }
  }

  function onFilterChange() {
    fetchPage(true);
  }

  onMount(() => {
    fetchPage(true);
  });
</script>

<div class="px-4 py-4 sm:px-6">
  <a id="top"></a>

  <h1 class="text-xl font-semibold text-gray-800 dark:text-gray-100 mb-4">
    Similar assets
  </h1>

  <!-- Controls -->
  <div class="flex flex-wrap gap-3 items-end mb-4 text-sm">
    <div class="flex items-center gap-2">
      <label>Types:</label>
      <select bind:value={types} class="border rounded px-2 py-1 bg-transparent" on:change={onFilterChange}>
        <option value="image,video">Photos & Videos</option>
        <option value="image">Photos only</option>
        <option value="video">Videos only</option>
      </select>
    </div>

    <div class="flex items-center gap-2">
      <label>Stacks primary only:</label>
      <input type="checkbox" bind:checked={stacksOnly} on:change={onFilterChange} />
    </div>

    <div class="flex items-center gap-2">
      <label>From:</label>
      <input type="datetime-local" bind:value={minDate} class="border rounded px-2 py-1 bg-transparent" on:change={onFilterChange} />
    </div>

    <div class="flex items-center gap-2">
      <label>To:</label>
      <input type="datetime-local" bind:value={maxDate} class="border rounded px-2 py-1 bg-transparent" on:change={onFilterChange} />
    </div>

    <div class="flex items-center gap-2">
      <label>Album ID:</label>
      <input type="text" bind:value={albumId} placeholder="optional" class="border rounded px-2 py-1 bg-transparent" />
      <button class="btn btn-secondary" on:click={() => fetchPage(true)}>Apply</button>
    </div>

    <div class="flex items-center gap-2">
      <label>Camera:</label>
      <input type="text" bind:value={cameraMake} placeholder="make" class="border rounded px-2 py-1 bg-transparent" />
      <input type="text" bind:value={cameraModel} placeholder="model" class="border rounded px-2 py-1 bg-transparent" />
      <button class="btn btn-secondary" on:click={() => fetchPage(true)}>Apply</button>
    </div>

    <div class="flex items-center gap-2">
      <label>Group by:</label>
      <select bind:value={groupByType} class="border rounded px-2 py-1 bg-transparent">
        <option value={true}>Type (Photos / Videos)</option>
        <option value={false}>None</option>
      </select>
    </div>
  </div>

  <!-- Anchors for grouped view -->
  {#if groupByType}
    <nav class="flex items-center gap-3 text-xs text-gray-600 dark:text-gray-300 mb-2">
      {#if photosItems.length}
        <a href="#photos" class="hover:underline">Photos</a>
      {/if}
      {#if videosItems.length}
        <a href="#videos" class="hover:underline">Videos</a>
      {/if}
      <span class="mx-1 text-gray-400">•</span>
      <a href="#top" class="hover:underline">Top</a>
    </nav>
  {/if}

  <!-- Status -->
  {#if error}
    <div class="text-sm text-red-600 dark:text-red-400 mb-2">{error}</div>
  {/if}
  {#if !items.length && !loading && !error}
    <div class="text-sm text-gray-600 dark:text-gray-300">No similar assets found.</div>
  {/if}

  <!-- Results -->
  {#if !groupByType}
    <!-- Flat grid -->
    <div class="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-2">
      {#each items as a}
        <a href={`/photos/${a.id}`} class="relative block group">
          <img
            src={`/api/assets/${a.id}/thumbnail?size=preview`}
            alt=""
            loading="lazy"
            class="w-full h-auto rounded-md"
          />
          {#if a.distance != null}
            <div class="absolute top-1 right-1 rounded bg-black/60 text-white text-[10px] px-1.5 py-0.5">
              {score(a.distance)}%
            </div>
          {/if}
        </a>
      {/each}
    </div>
  {:else}
    <!-- Grouped: Photos -->
    {#if photosItems.length}
      <h2 id="photos" class="mt-2 mb-2 text-base font-semibold text-gray-800 dark:text-gray-100">
        Photos <span class="text-gray-500 dark:text-gray-400">({photosItems.length})</span>
      </h2>
      <div class="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-2">
        {#each photosItems as a}
          <a href={`/photos/${a.id}`} class="relative block group">
            <img
              src={`/api/assets/${a.id}/thumbnail?size=preview`}
              alt=""
              loading="lazy"
              class="w-full h-auto rounded-md"
            />
            {#if a.distance != null}
              <div class="absolute top-1 right-1 rounded bg-black/60 text-white text-[10px] px-1.5 py-0.5">
                {score(a.distance)}%
              </div>
            {/if}
          </a>
        {/each}
      </div>
    {/if}

    <!-- Grouped: Videos -->
    {#if videosItems.length}
      <h2 id="videos" class="mt-4 mb-2 text-base font-semibold text-gray-800 dark:text-gray-100">
        Videos <span class="text-gray-500 dark:text-gray-400">({videosItems.length})</span>
      </h2>
      <div class="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-2">
        {#each videosItems as a}
          <a href={`/photos/${a.id}`} class="relative block group">
            <img
              src={`/api/assets/${a.id}/thumbnail?size=preview`}
              alt=""
              loading="lazy"
              class="w-full h-auto rounded-md"
            />
            {#if a.distance != null}
              <div class="absolute top-1 right-1 rounded bg-black/60 text-white text-[10px] px-1.5 py-0.5">
                {score(a.distance)}%
              </div>
            {/if}
          </a>
        {/each}
      </div>
    {/if}
  {/if}

  <!-- Pager -->
  <div class="mt-4">
    {#if hasMore}
      <button class="btn btn-secondary" on:click={() => fetchPage(false)} disabled={loading}>
        {loading ? 'Loading…' : 'Load more'}
      </button>
    {/if}
  </div>

  <!-- Save as album -->
  <div class="flex items-center gap-2 mt-4">
    <input
      type="text"
      bind:value={albumName}
      placeholder="Album name…"
      class="border rounded px-2 py-1 bg-transparent"
    />
    <button class="btn btn-primary" disabled={!items.length || !albumName.trim() || saving} on:click={saveAsAlbum}>
      {saving ? 'Saving…' : 'Save results as album'}
    </button>
  </div>
</div>
