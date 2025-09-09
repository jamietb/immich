<script lang="ts">
  import { goto } from '$app/navigation';
  import { page } from '$app/state';
  import { step, hasSimilarSession } from '$lib/stores/similar-session';
  import { shortcuts } from '$lib/actions/shortcut';
  import Icon from '$lib/components/elements/icon.svelte';
  import { mdiChevronLeft } from '@mdi/js';
  import { t } from 'svelte-i18n';
  import NavigationArea from '../navigation-area.svelte';

  interface Props {
    onPreviousAsset: () => void;
  }

  // rename incoming prop so we can wrap it
  let { onPreviousAsset: parentPrev }: Props = $props();

  function onPreviousAsset() {
    const query = page.url.searchParams;
    const useSimilar = query.get('similar') === '1';

    if (useSimilar && hasSimilarSession()) {
      const id = step(-1);
      if (id) {
        goto(`/photos/${id}?similar=1`, { keepfocus: true });
        return;
      }
    }

    // fallback to timeline / parent behavior
    parentPrev();
  }
</script>

<svelte:document
  use:shortcuts={[
    { shortcut: { key: 'ArrowLeft' }, onShortcut: onPreviousAsset },
    { shortcut: { key: 'a' }, onShortcut: onPreviousAsset },
  ]}
/>

<NavigationArea onClick={onPreviousAsset} label={$t('view_previous_asset')}>
  <Icon path={mdiChevronLeft} size="36" ariaHidden />
</NavigationArea>
