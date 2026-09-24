import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import test from 'node:test';
import {createContext, SourceTextModule, SyntheticModule} from 'node:vm';
import {
  normalizeScalars,
  selectTags,
  smoothScalars,
} from '../../cli/tensorboard_plugins/scalar_compare/data.js';

test('scalar points are sorted by step and retain the latest duplicate', () => {
  const events = [[30, 2, 0.4], [10, 0, 1], [20, 2, 0.8], [40, 1, 0.6]];
  assert.deepEqual(normalizeScalars(events), [
    {wallTime: 10, step: 0, value: 1},
    {wallTime: 40, step: 1, value: 0.6},
    {wallTime: 30, step: 2, value: 0.4},
  ]);
  assert.deepEqual(events[0], [30, 2, 0.4]);
});

test('non-finite values remain gaps and malformed coordinates are skipped', () => {
  assert.deepEqual(normalizeScalars([
    [10, 0, 0], [11, 1, NaN], [12, 2, Infinity], [13, 3, 'NaN'],
    [14, 4, -1], [null, 5, 2], [15, NaN, 3], [], null,
  ]).map(point => point.value), [0, null, null, null, -1]);
  assert.throws(() => normalizeScalars({}), /Invalid scalar response/);
});

test('zero smoothing preserves scalar values without mutating the source', () => {
  const points = normalizeScalars([[10, 0, -1], [11, 1, 0], [12, 2, 2]]);
  assert.deepEqual(smoothScalars(points, 0).map(point => point.smoothed), [-1, 0, 2]);
  assert.equal(points[0].smoothed, undefined);
});

test('smoothing is debiased and does not turn missing values into zero', () => {
  const points = normalizeScalars([[10, 0, 2], [11, 1, 4], [12, 2, NaN], [13, 3, 6]]);
  const smoothed = smoothScalars(points, 0.5);
  assert.equal(smoothed[0].smoothed, 2);
  assert.ok(Math.abs(smoothed[1].smoothed - 10 / 3) < 1e-12);
  assert.equal(smoothed[2].smoothed, null);
  assert.ok(Math.abs(smoothed[3].smoothed - 34 / 7) < 1e-12);
});

test('refresh retains selected tags and an explicitly empty selection', () => {
  const tags = ['train/loss', 'val/accuracy', 'val/loss'];
  assert.deepEqual(selectTags(tags, null), ['train/loss', 'val/accuracy']);
  assert.deepEqual(selectTags(tags, ['train/loss', 'val/loss', 'missing']),
    ['train/loss', 'val/loss']);
  assert.deepEqual(selectTags(tags, []), []);
});

const context = createContext({
  HTMLElement: class {},
  customElements: {define() {}},
  URL,
  URLSearchParams,
  AbortController,
});
const entryUrl = new URL('../../cli/tensorboard_plugins/scalar_compare/entry.js', import.meta.url);
const viewModule = new SourceTextModule(await readFile(entryUrl, 'utf8'), {
  context,
  initializeImportMeta(meta) { meta.url = entryUrl.href; },
});
await viewModule.link(specifier => {
  const exports = specifier === './data.js'
    ? {normalizeScalars, selectTags, smoothScalars}
    : {default: {}};
  return new SyntheticModule(Object.keys(exports), function () {
    for (const [name, value] of Object.entries(exports)) this.setExport(name, value);
  }, {context});
});
await viewModule.evaluate();

function createDashboard() {
  return Object.assign(Object.create(viewModule.namespace.ScalarCompareDashboard.prototype), {
    _run: 'baseline',
    _selected: ['train/loss', 'val/loss'],
    _requestId: 0,
    _series: new Map(),
    _failed: new Set(),
    _status(message, error = false) { this.status = {message, error}; },
    _renderRuns() {},
    _renderTags() {},
    _draw() {},
  });
}

test('late tag responses cannot remove newly discovered runs or selected metrics', async () => {
  const dashboard = createDashboard();
  const pending = [];
  dashboard._fetch = url => url === 'tags'
    ? new Promise(resolve => pending.push(resolve))
    : Promise.resolve([[20, 1, 0.2]]);
  const older = dashboard.refresh();
  const newer = dashboard.refresh();
  pending[1]({
    baseline: {'train/loss': {}, 'val/loss': {}},
    fresh: {'train/loss': {}},
  });
  await newer;
  pending[0]({baseline: {'train/loss': {}}});
  await older;

  assert.deepEqual(Object.keys(dashboard._tags), ['baseline', 'fresh']);
  assert.deepEqual(dashboard._selected, ['train/loss', 'val/loss']);
  assert.deepEqual([...dashboard._series.keys()], ['train/loss', 'val/loss']);
  assert.equal(dashboard.status.message, '');
});

test('refresh preserves an explicitly empty metric selection without fetching values', async () => {
  const dashboard = createDashboard();
  const requests = [];
  dashboard._selected = [];
  dashboard._fetch = async url => {
    requests.push(url);
    return {baseline: {'train/loss': {}, 'val/loss': {}}};
  };
  await dashboard.refresh();

  assert.deepEqual(dashboard._selected, []);
  assert.deepEqual(requests, ['tags']);
  assert.equal(dashboard._series.size, 0);
});

test('late scalar responses cannot overwrite a different run', async () => {
  const dashboard = createDashboard();
  const pending = [];
  dashboard._fetch = () => new Promise(resolve => pending.push(resolve));
  const older = dashboard._loadSeries();
  const oldSignal = dashboard._controller.signal;
  dashboard._run = 'train_only';
  dashboard._selected = ['train/loss'];
  const newer = dashboard._loadSeries();
  pending[2]([[20, 1, 0.2]]);
  await newer;
  pending[0]([[10, 0, 99]]);
  pending[1]([[10, 0, 98]]);
  await older;

  assert.equal(oldSignal.aborted, true);
  assert.equal(dashboard._run, 'train_only');
  assert.deepEqual([...dashboard._series.keys()], ['train/loss']);
  assert.equal(dashboard._series.get('train/loss')[0].value, 0.2);
  assert.equal(dashboard.status.message, '');
});

test('one failed metric does not hide the others and can recover on refresh', async () => {
  const dashboard = createDashboard();
  dashboard._fetch = async url => {
    if (new URL(url, 'http://localhost/').searchParams.get('tag') === 'val/loss') {
      throw new Error('HTTP 503');
    }
    return [[10, 0, 0.5]];
  };
  await dashboard._loadSeries();
  assert.deepEqual([...dashboard._series.keys()], ['train/loss']);
  assert.deepEqual([...dashboard._failed], ['val/loss']);
  assert.equal(dashboard.status.error, true);

  dashboard._fetch = async () => [[20, 1, 0.25]];
  await dashboard._loadSeries();
  assert.deepEqual([...dashboard._series.keys()], ['train/loss', 'val/loss']);
  assert.equal(dashboard._failed.size, 0);
  assert.equal(dashboard.status.message, '');
});