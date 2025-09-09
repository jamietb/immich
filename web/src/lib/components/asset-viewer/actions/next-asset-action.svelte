<script lang="ts">
  import { goto } from '$app/navigation';
  import { page } from '$app/state';
  import { step, hasSimilarSession } from '$lib/stores/similar-session';
  import { shortcuts } from '$lib/actions/shortcut';
  import Icon from '$lib/components/elements/icon.svelte';
  import { mdiChevronRight } from '@mdi/js';
  import { t } from 'svelte-i18n';
  import NavigationArea from '../navigation-area.svelte';

  interface Props {
    onNextAsset: () => void;
  }

  let { onNextAsset: parentNext }: Props = $props();

  function onNextAsset() {
    const query = page.url.searchParams;
    const useSimilar = query.get('similar') === '1';

    if (useSimilar && hasSimilarSession()) {
      const id = step(1);
      if (id) {
        goto(`/photos/${id}?similar=1`, { keepfocus: true });
        return;
      }
    }

    // fallback
    parentNext();
  }
</script>

<svelte:document
  use:shortcuts={[
    { shortcut: { key: 'ArrowRight' }, onShortcut: onNextAsset },
    { shortcut: { key: 'd' }, onShortcut: onNextAsset },
  ]}
/>

<NavigationArea onClick={onNextAsset} label={$t('view_next_asset')}>
  <Icon path={mdiChevronRight} size="36" ariaHidden />
</NavigationArea>
