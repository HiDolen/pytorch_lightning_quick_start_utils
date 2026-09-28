import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import test from 'node:test';
import {createContext, SourceTextModule, SyntheticModule} from 'node:vm';
import * as categorizationUtils from '../../cli/tensorboard_plugins/shared/histogram/categorization_utils.js';
import * as colorScale from '../../cli/tensorboard_plugins/shared/histogram/color_scale.js';

const context = createContext({
  HTMLElement: class {},
  customElements: {define() {}},
  document: {
    createElement(tag) {
      if (tag === 'tf-histogram-card') return createCard();
      return {
        style: {},
        children: [],
        appendChild(child) { this.children.push(child); },
        replaceChildren() { this.children = []; },
        addEventListener(type, listener) { this[type] = listener; },
      };
    },
  },
});
const dashboardModule = new SourceTextModule(
  await readFile(
    new URL('../../cli/tensorboard_plugins/shared/histogram/tf_histogram_dashboard.js', import.meta.url),
    'utf8'
  ),
  {context}
);
function linkModule(specifier) {
  const exports = specifier === './categorization_utils.js'
    ? categorizationUtils
    : specifier === './color_scale.js'
      ? colorScale
      : {};
  return new SyntheticModule(Object.keys(exports), function () {
    for (const [name, value] of Object.entries(exports)) {
      this.setExport(name, value);
    }
  }, {context});
}
await dashboardModule.link(linkModule);
await dashboardModule.evaluate();

const cardModule = new SourceTextModule(
  await readFile(
    new URL('../../cli/tensorboard_plugins/shared/histogram/tf_histogram_card.js', import.meta.url),
    'utf8'
  ),
  {context}
);
await cardModule.link(linkModule);
await cardModule.evaluate();

function createCard() {
  return Object.assign(Object.create(cardModule.namespace.TfHistogramCard.prototype), {
    _colorScaleFunction: colorScale.defaultColorScale,
    _heading: {},
    $: {chart: {
      setColorScale(scale) { this.colorScale = scale; },
      setTimeProperty() {},
      setMode() {},
    }},
    isConnected: true,
  });
}

function createDashboard(tags) {
  return Object.assign(
    Object.create(dashboardModule.namespace.TfHistogramDashboard.prototype),
    {
      _selectedRuns: null,
      _knownRuns: new Set(),
      _tagsRequestId: 0,
      _cards: new Set(),
      _runsColorScale: colorScale.createRunsColorScale([]),
      _multiCheckbox: context.document.createElement('div'),
      tagsProvider: async () => tags,
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
    const card = dashboard._createCard({run: 'initial', tag: 'curve'});
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
    assert.equal(card._colorScaleFunction, dashboard._runsColorScale);
    assert.equal(card._colorScaleFunction('new_run'), '#0077bb');

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

for (const isConnected of [true, false]) {
  test(`refresh synchronizes colors for ${isConnected ? 'visible' : 'cached'} cards`, async () => {
    const dashboard = createDashboard({b: {curve: {}}});
    await dashboard.reload();
    const card = dashboard._createCard({run: 'b', tag: 'curve'});
    card.isConnected = isConnected;
    assert.equal(card._heading.color, '#ff7043');

    dashboard.tagsProvider = async () => ({b: {curve: {}}, a: {curve: {}}});
    await dashboard.reload();

    const row = dashboard._multiCheckbox.children.find(row => row.children[1].textContent === 'b');
    assert.equal(row.style.color, '#0077bb');
    assert.equal(card._heading.color, row.style.color);
    assert.equal(card.$.chart.colorScale('b'), row.style.color);
    assert.equal(card._colorScaleFunction, dashboard._runsColorScale);
    const newCard = dashboard._createCard({run: 'a', tag: 'curve'});
    assert.equal(newCard._colorScaleFunction, card._colorScaleFunction);
    assert.equal(newCard._heading.color, dashboard._multiCheckbox.children[0].style.color);
  });
}

test('removed runs do not prevent remaining or returning cards from updating colors', async () => {
  const dashboard = createDashboard({a: {curve: {}}, b: {curve: {}}});
  await dashboard.reload();
  const removed = dashboard._createCard({run: 'a', tag: 'curve'});
  const remaining = dashboard._createCard({run: 'b', tag: 'curve'});
  dashboard.tagsProvider = async () => ({b: {curve: {}}});
  await dashboard.reload();

  assert.equal(remaining._heading.color, dashboard._multiCheckbox.children[0].style.color);
  assert.equal(remaining._heading.color, '#ff7043');

  dashboard.tagsProvider = async () => ({});
  await dashboard.reload();
  assert.equal(dashboard._multiCheckbox.children.length, 0);

  dashboard.tagsProvider = async () => ({a: {curve: {}}, b: {curve: {}}});
  await dashboard.reload();
  assert.equal(removed._heading.color, dashboard._multiCheckbox.children[0].style.color);
  assert.equal(remaining._heading.color, dashboard._multiCheckbox.children[1].style.color);
  assert.equal(remaining._heading.color, '#0077bb');
});

test('changing run selection preserves the shared run colors', async () => {
  const dashboard = createDashboard({a: {curve: {}}, b: {curve: {}}});
  await dashboard.reload();
  const card = dashboard._createCard({run: 'b', tag: 'curve'});
  const scale = card._colorScaleFunction;

  dashboard._multiCheckbox.children[0].click();
  assert.deepEqual([...dashboard._selectedRuns], ['b']);
  assert.equal(dashboard._multiCheckbox.children[1].style.color, '#0077bb');
  dashboard._toggleAllRuns();
  dashboard._toggleAllRuns();
  assert.deepEqual([...dashboard._selectedRuns], []);
  dashboard._toggleAllRuns();
  assert.deepEqual([...dashboard._selectedRuns], ['a', 'b']);
  assert.equal(card._colorScaleFunction, scale);
  assert.equal(card._heading.color, dashboard._multiCheckbox.children[1].style.color);
});