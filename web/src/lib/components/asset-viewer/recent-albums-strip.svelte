<script lang="ts">
  const { asset } = $props<{ asset: { id: string } }>();
  let albums = $state<Array<{ id: string; albumName: string; albumThumbnailAssetId?: string | null }>>([]);
  let loading = $state(true);

async function load() {
  loading = true;

  // Try a few common query param styles first (some builds support them)
  const candidateUrls = [
    // try explicit take/order/direction
    '/api/albums?take=4&order=createdAt&direction=DESC',
    // try limit with CSV order
    '/api/albums?limit=4&order=createdAt,DESC',
    // plain (fallback to client-side sort/slice)
    '/api/albums',
  ];

  let data: any[] | null = null;

  for (const url of candidateUrls) {
    try {
      const res = await fetch(url, { credentials: 'include' });
      if (!res.ok) continue;

      const json = await res.json();
      // Some builds wrap results; normalize to array
      const arr = Array.isArray(json) ? json : Array.isArray(json?.items) ? json.items : null;
      if (!arr) continue;

      data = arr;
      // If this call already returned ≤ 4, we’re done
      if (arr.length <= 4 && url !== '/api/albums') {
        albums = arr;
        loading = false;
        return;
      }
      // otherwise let the client-side sort/slice run below
      break;
    } catch {
      // try next candidate
    }
  }

  if (!data) data = [];

  // ✅ Client-side: newest first by createdAt, keep 4
  data.sort((a: any, b: any) => {
    const da = new Date(a.createdAt ?? 0).getTime();
    const db = new Date(b.createdAt ?? 0).getTime();
    return db - da;
  });

  albums = data.slice(0, 4);
  loading = false;
}

  async function addToAlbum(id: string) {
    await fetch(`/api/albums/${id}/assets`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      credentials: 'include',
      body: JSON.stringify({ assetIds: [asset.id] }),
    });
  }

  $effect(load);
</script>

<div class="pt-4 border-t border-gray-200 dark:border-gray-700">
  <div class="text-sm font-semibold text-gray-700 dark:text-gray-200 mb-2">Quick add to album</div>

  {#if loading}
    <div class="text-xs text-gray-500 dark:text-gray-400">Loading…</div>
  {:else if !albums.length}
    <div class="text-xs text-gray-500 dark:text-gray-400">No albums yet.</div>
  {:else}
    <div class="grid grid-cols-2 gap-2">
      {#each albums as al}
        <button class="flex items-center gap-2 p-2 rounded hover:bg-gray-100 dark:hover:bg-gray-800 text-left"
                on:click={() => addToAlbum(al.id)}>
          {#if al.albumThumbnailAssetId}
            <img
              src={`/api/assets/${al.albumThumbnailAssetId}/thumbnail?size=preview`}
              class="w-12 h-12 object-cover rounded"
              alt=""
            />
          {/if}
          <span class="text-sm">{al.albumName}</span>
        </button>
      {/each}
    </div>
  {/if}
</div>
