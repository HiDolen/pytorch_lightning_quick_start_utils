import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import test from 'node:test';
import {createContext, SourceTextModule, SyntheticModule} from 'node:vm';
import * as categorizationUtils from '../../cli/tensorboard_plugins/shared/histogram/categorization_utils.js';

const context = createContext({
  HTMLElement: class {},
  customElements: {define() {}},
});
const dashboardModule = new SourceTextModule(
  await readFile(
    new URL('../../cli/tensorboard_plugins/shared/histogram/tf_histogram_dashboard.js', import.meta.url),
    'utf8'
  ),
  {context}
);
await dashboardModule.link((specifier) => {
  const exports = specifier === './categorization_utils.js'
    ? categorizationUtils
    : specifier === './color_scale.js'
      ? {createRunsColorScale: undefined}
      : {};
  return new SyntheticModule(Object.keys(exports), function () {
    for (const [name, value] of Object.entries(exports)) {
      this.setExport(name, value);
    }
  }, {context});
});
await dashboardModule.evaluate();

function createDashboard(tags) {
  return Object.assign(
    Object.create(dashboardModule.namespace.TfHistogramDashboard.prototype),
    {
      _selectedRuns: null,
      _knownRuns: new Set(),
      _tagsRequestId: 0,
      tagsProvider: async () => tags,
      _renderRunsSelector() {},
      _renderCategories() {},
      _reloadHistograms: async () => {},
    }
  );
}

const initialTags = {initial: {curve: {displayName: 'Initial'}}};
const updatedTags = {...initialTags, new_run: {curve: {displayName: 'New'}}};

for (const [label, staleTags] of [['existing runs', initialTags], ['no runs', {}]]) {
  test(`a stale refresh with ${label} cannot overwrite a newer run list`, async () => {
    const dashboard = createDashboard(initialTags);
    await dashboard.reload();
    const pending = [];
    dashboard.tagsProvider = () => new Promise(resolve => pending.push(resolve));

    const olderRefresh = dashboard.reload();
    const newerRefresh = dashboard.reload();
    pending[1](updatedTags);
    await newerRefresh;
    pending[0](staleTags);
    await olderRefresh;

    assert.deepEqual([...dashboard._allRuns()], ['initial', 'new_run']);
    assert.deepEqual([...dashboard._selectedRuns], ['initial', 'new_run']);
    assert.equal(dashboard._runToTagInfo, updatedTags);
    assert.equal(dashboard._dataNotFound, false);

    dashboard.tagsProvider = async () => updatedTags;
    await dashboard.reload();
    assert.deepEqual([...dashboard._selectedRuns], ['initial', 'new_run']);
  });
}

test('refresh preserves deselected runs and selects newly discovered runs', async () => {
  const dashboard = createDashboard(initialTags);
  await dashboard.reload();
  dashboard._selectedRuns = [];
  dashboard.tagsProvider = async () => updatedTags;
  await dashboard.reload();

  assert.deepEqual([...dashboard._allRuns()], ['initial', 'new_run']);
  assert.deepEqual([...dashboard._selectedRuns], ['new_run']);

  await dashboard.reload();
  assert.deepEqual([...dashboard._selectedRuns], ['new_run']);
});