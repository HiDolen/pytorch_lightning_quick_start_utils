// Source: tensorboard/plugins/histogram/tf_histogram_dashboard/tf-histogram-dashboard.ts,
//         tensorboard/components/tf_dashboard_common/tf-dashboard-layout.ts,
//         tensorboard/components/tf_dashboard_common/dashboard-style.ts,
//         tensorboard/components/tf_dashboard_common/tf-option-selector.ts,
//         tensorboard/components/tf_runs_selector/tf-runs-selector.ts,
//         tensorboard/components/tf_dashboard_common/tensorboard-color.ts
// Source commit: 012299c95682b6cf89e0eaccb8dc034cf1d61cd2
// Adaptation: Polymer elements ported to a single native custom element; DOM
// structure, styles and interaction flow copied verbatim. Backend requests are
// injected (tagsProvider/dataProvider) so the dashboard is plugin-agnostic;
// paper-* controls are native equivalents. Changing mode/time property updates
// existing cards in place (no DOM rebuild) to preserve the 500ms transition.
import './tf_category_paginated_view.js';
import './tf_histogram_card.js';
import {categorizeRunTagCombinations, getTags} from './categorization_utils.js';
import {createRunsColorScale} from './color_scale.js';

const TEMPLATE = `
    <div id="sidebar">
      <div class="sidebar-slot">
        <div class="settings">
          <div class="sidebar-section">
            <div class="option-selector" id="histogramModeSelector">
              <h3>Histogram mode</h3>
              <div class="content-wrapper">
                <button class="paper-button" id="overlay">overlay</button>
                <button class="paper-button" id="offset">offset</button>
              </div>
            </div>
          </div>
        </div>
        <div class="sidebar-section runs-selector">
          <div id="runs-selector">
            <div id="top-text">
              <h3 id="tooltip-help">Runs</h3>
            </div>
            <div id="multi-checkbox"></div>
            <button class="paper-button x-button" id="toggle-all">
              Toggle All Runs
            </button>
          </div>
        </div>
      </div>
    </div>

    <div id="center">
      <div class="center-slot">
        <div class="no-data-warning" hidden>
          <h3>No curve data was found.</h3>
          <p>Probable causes:</p>
          <ul>
            <li>
              You haven’t written any curve data to your event files.
            </li>
            <li>TensorBoard can’t find your event files.</li>
          </ul>
        </div>
        <div class="tag-filterer" hidden>
          <div class="search-input">
            <input
              id="tag-filter-input"
              type="text"
              placeholder="Filter tags (regex)"
            />
          </div>
          <div class="regex-hint" id="regex-hint"></div>
        </div>
        <div id="categories"></div>
      </div>
    </div>
`;

const STYLE = `
      :host {
        --tb-ui-dark-accent: #757575;
        --tb-ui-light-accent: #e0e0e0;
        --tb-ui-border: #e0e0e0;
        --tb-secondary-text-color: #424242;
        --tb-raised-button-shadow-color: rgba(0, 0, 0, 0.2);
        --primary-background-color: #fff;
        --paper-grey-500: #9e9e9e;
        --sidebar-vertical-padding: 15px;
        --sidebar-left-padding: 30px;
        background-color: #f5f5f5;
        display: flex;
        flex-direction: row;
        height: 100%;
      }

      #sidebar {
        flex: 0 0 var(--tf-dashboard-layout-sidebar-basis, 25%);
        height: 100%;
        max-width: var(--tf-dashboard-layout-sidebar-max-width, 350px);
        min-width: var(--tf-dashboard-layout-sidebar-min-width, 270px);
        overflow-y: auto;
        text-overflow: ellipsis;
      }

      #center {
        flex-grow: 1;
        flex-shrink: 1;
        height: 100%;
        overflow: hidden;
      }

      .center-slot {
        contain: strict;
        height: 100%;
        overflow-x: hidden;
        overflow-y: auto;
        width: 100%;
        will-change: transform;
      }

      .sidebar-slot {
        box-sizing: border-box;
        display: flex;
        flex-direction: column;
        height: 100%;
        margin-right: 10px;
        overflow-x: hidden;
        padding: 5px 0;
        text-overflow: ellipsis;
      }

      .settings {
        min-height: 50px;
        overflow-x: hidden;
        overflow-y: auto;
        will-change: transform;
      }

      .runs-selector {
        display: flex;
        flex-grow: 1;
        min-height: 200px;
      }

      .search-input {
        margin: 10px 5px 0 10px;
      }

      .search-input input {
        font: inherit;
        border: none;
        border-bottom: 1px solid var(--paper-grey-500);
        padding: 4px 0;
        width: 300px;
        outline: none;
      }

      .regex-hint {
        margin: 0 5px 0 10px;
        font-size: 12px;
        color: #f57c00;
      }

      .sidebar-section {
        border-top: solid 1px var(--tb-ui-border);
        margin-right: 10px;
        padding: var(--sidebar-vertical-padding) 0
          var(--sidebar-vertical-padding) var(--sidebar-left-padding);
        position: relative;
        overflow: hidden;
      }

      .sidebar-section:first-of-type {
        border: none;
      }

      .sidebar-section paper-button {
        margin: 5px;
      }

      .sidebar-section > :first-child {
        margin-top: 0;
        padding-top: 0;
      }

      .sidebar-section > :last-child {
        margin-bottom: 0;
        padding-bottom: 0;
      }

      .sidebar-section h3 {
        color: var(--tb-secondary-text-color);
        display: block;
        font-size: 14px;
        font-weight: normal;
        margin: 10px 0 5px;
        pointer-events: none;
      }

      .paper-button {
        display: inline-block;
        position: relative;
        box-sizing: border-box;
        min-width: 5.14em;
        margin: 0 0.29em;
        background: transparent;
        -webkit-tap-highlight-color: rgba(0, 0, 0, 0);
        text-align: center;
        font: inherit;
        text-transform: uppercase;
        outline-width: 0;
        border-radius: 3px;
        user-select: none;
        cursor: pointer;
        z-index: 0;
        padding: 0.7em 0.57em;
        border: none;
      }

      .option-selector .content-wrapper .paper-button {
        background: none;
        color: var(--tb-ui-dark-accent);
        font-size: 13px;
        margin-top: 10px;
      }

      .option-selector .content-wrapper .paper-button.selected {
        background-color: var(--tb-ui-dark-accent);
        color: white !important;
      }

      .option-selector h3 {
        color: var(--tb-secondary-text-color);
        display: block;
        font-size: 14px;
        font-weight: normal;
        margin: 0 0 5px;
        pointer-events: none;
      }

      #runs-selector {
        box-sizing: border-box;
        display: flex;
        flex-direction: column;
        padding-bottom: 10px;
        width: 100%;
      }

      #top-text {
        color: var(--tb-secondary-text-color);
        width: 100%;
        flex-grow: 0;
        flex-shrink: 0;
        padding-right: 16px;
        box-sizing: border-box;
      }

      #tooltip-help {
        color: var(--tb-secondary-text-color);
        margin: 0;
        font-weight: normal;
        font-size: 14px;
        margin-bottom: 5px;
      }

      #multi-checkbox {
        display: flex;
        flex-grow: 1;
        flex-shrink: 1;
        flex-direction: column;
        overflow-y: auto;
      }

      .run-row {
        display: flex;
        align-items: center;
        font-size: 15px;
        margin-top: 5px;
        cursor: pointer;
      }

      .run-row .checkbox {
        width: 16px;
        height: 16px;
        border: 2px solid currentColor;
        border-radius: 2px;
        margin-right: 8px;
        flex: none;
        position: relative;
      }

      .run-row .checkbox svg {
        display: none;
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
      }

      .run-row.checked .checkbox {
        background: currentColor;
        border-color: currentColor;
      }

      .run-row.checked .checkbox svg {
        display: block;
      }

      .run-row .checkbox svg path {
        fill: #fff;
      }

      .run-name {
        word-break: break-all;
      }

      .x-button {
        font-size: 13px;
        background-color: var(--tb-ui-light-accent);
        color: var(--tb-ui-dark-accent);
        margin-left: 0;
        align-self: flex-start;
      }

      .no-data-warning {
        max-width: 540px;
        margin: 80px auto 0 auto;
      }
`;

// run 排序：与 TB Time Series 的 runs 表格一致（无数字前缀在前，数字按
// 数值比较，前缀相同再比剩余后缀；见 webapp/runs/views/runs_table/sorting_utils.ts）
function compareRunNames(a, b) {
  const parseNumericPrefix = (value) => {
    if (!isNaN(parseInt(value))) return parseInt(value);
    for (let i = 0; i < value.length; i++) {
      if (isNaN(parseInt(value[i]))) {
        if (i === 0) return undefined;
        return parseInt(value.slice(0, i));
      }
    }
    return undefined;
  };
  if (a === b) return 0;
  const aP = parseNumericPrefix(a);
  const bP = parseNumericPrefix(b);
  if ((aP === undefined || bP === undefined) && aP !== bP) {
    return aP === undefined ? -1 : 1;
  }
  if (aP !== undefined && bP !== undefined) {
    if (aP === bP) {
      const aS = a.slice(String(aP).length) || undefined;
      const bS = b.slice(String(bP).length) || undefined;
      if (aS === bS) return 0;
      if (aS === undefined) return -1;
      if (bS === undefined) return 1;
      return aS < bS ? -1 : 1;
    }
    return aP < bP ? -1 : 1;
  }
  return a < b ? -1 : 1;
}

export class TfHistogramDashboard extends HTMLElement {
  constructor() {
    super();
    const root = this.attachShadow({mode: 'open'});
    root.innerHTML = `<style>${STYLE}</style>${TEMPLATE}`;
    this._root = root;
    this._categoriesEl = root.getElementById('categories');
    this._noDataWarning = root.querySelector('.no-data-warning');
    this._tagFilterer = root.querySelector('.tag-filterer');
    this._tagFilterInput = root.getElementById('tag-filter-input');
    this._regexHint = root.getElementById('regex-hint');
    this._multiCheckbox = root.getElementById('multi-checkbox');
    this._toggleAll = root.getElementById('toggle-all');

    this._histogramMode = 'offset';
    this._timeProperty = 'step';
    this._selectedRuns = null;
    this._knownRuns = new Set(); // 曾出现过的 run 集合,用于识别刷新时新出现的 run
    this._runToTag = null;
    this._runToTagInfo = null;
    this._dataNotFound = false;
    this._tagFilter = '';
    this._categoryViews = new Map(); // category name -> view element
    this._cards = new Set();

    // Injected by the host plugin:
    //   tagsProvider(): Promise<runToTagInfo>
    //   dataProvider(run, tag): Promise<VzHistogram-shaped data>
    this.tagsProvider = null;
    this.dataProvider = null;
    this.toVz = (data) => data; // backend data -> VzHistogram adapter

    root
      .querySelectorAll('#histogramModeSelector .paper-button')
      .forEach((button) =>
        button.addEventListener('click', () => {
          this._setHistogramMode(button.id);
        })
      );
    this._tagFilterInput.addEventListener('input', () => {
      this._tagFilter = this._tagFilterInput.value;
      this._updateRegexHint();
      this._renderCategories();
    });
    this._toggleAll.addEventListener('click', () => this._toggleAllRuns());
    // 记录 runs 列表内相邻两次点击的时间戳，供双击独选的严格判定
    // 原生 dblclick 阈值约 500ms，太宽松容易误触
    this._prevRunClickAt = 0;
    this._lastRunClickAt = 0;
    this._multiCheckbox.addEventListener(
      'click',
      (e) => {
        this._prevRunClickAt = this._lastRunClickAt;
        this._lastRunClickAt = e.timeStamp;
      },
      true
    );
    this._setHistogramMode('offset');
  }

  connectedCallback() {
    this.reload();
  }

  reload() {
    this._fetchTags().then(() => {
      this._reloadHistograms();
    });
  }

  _fetchTags() {
    return this.tagsProvider().then((runToTagInfo) => {
      const runToTag = {};
      Object.keys(runToTagInfo)
        .sort(compareRunNames)
        .forEach((run) => {
          runToTag[run] = Object.keys(runToTagInfo[run]);
        });
      const tags = getTags(runToTag);
      this._dataNotFound = tags.length === 0;
      this._runToTag = runToTag;
      this._runToTagInfo = runToTagInfo;
      const runs = Object.keys(runToTag);
      if (this._selectedRuns === null) {
        this._selectedRuns = runs;
      } else {
        // 刷新数据时，新出现的 run 自动勾选，且已有 run 保留勾选状态
        this._selectedRuns = this._selectedRuns.filter((run) => runToTag[run]);
        this._selectedRuns = this._selectedRuns.concat(
          runs.filter((run) => !this._knownRuns.has(run))
        );
      }
      runs.forEach((run) => this._knownRuns.add(run));
      this._renderRunsSelector();
      this._renderCategories();
    });
  }

  _reloadHistograms() {
    this._cards.forEach((card) => card.redraw());
  }

  _setHistogramMode(mode) {
    this._histogramMode = mode;
    this._root
      .querySelectorAll('#histogramModeSelector .paper-button')
      .forEach((button) =>
        button.classList.toggle('selected', button.id === mode)
      );
    // Polymer property binding: every existing chart updates in place,
    // triggering the 500ms mode transition instead of rebuilding the DOM.
    this._cards.forEach((card) => card.setHistogramMode(mode));
  }

  _toggleAllRuns() {
    var allSelected = this._selectedRuns.length === this._allRuns().length;
    this._selectedRuns = allSelected ? [] : this._allRuns();
    this._renderRunsSelector();
    this._renderCategories();
  }

  _allRuns() {
    return this._runToTag ? Object.keys(this._runToTag) : [];
  }

  _renderRunsSelector() {
    var runs = this._allRuns();
    var selected = new Set(this._selectedRuns || []);
    var runsColorScale = createRunsColorScale(runs);
    this._multiCheckbox.replaceChildren();
    runs.forEach((run) => {
      var color = runsColorScale(run);
      var row = document.createElement('label');
      row.className = 'run-row' + (selected.has(run) ? ' checked' : '');
      row.style.color = color;
      var checkbox = document.createElement('span');
      checkbox.className = 'checkbox';
      checkbox.innerHTML =
        '<svg width="18" height="18" viewBox="0 0 24 24"><path d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z"/></svg>';
      var name = document.createElement('span');
      name.className = 'run-name';
      name.textContent = run;
      row.appendChild(checkbox);
      row.appendChild(name);
      row.addEventListener('click', () => {
        if (selected.has(run)) {
          selected.delete(run);
        } else {
          selected.add(run);
        }
        this._selectedRuns = runs.filter((r) => selected.has(r));
        this._renderRunsSelector();
        this._renderCategories();
      });
      // 双击复选框进入独立勾选（只保留该 run）
      // 双击间隔 250ms 内才触发
      checkbox.addEventListener('dblclick', (e) => {
        e.preventDefault();
        if (this._lastRunClickAt - this._prevRunClickAt > 250) return;
        this._selectedRuns = [run];
        this._renderRunsSelector();
        this._renderCategories();
      });
      this._multiCheckbox.appendChild(row);
    });
  }

  _shouldOpen(index) {
    return index <= 2;
  }

  get _categories() {
    return categorizeRunTagCombinations(
      this._runToTag,
      this._selectedRuns,
      this._tagFilter
    );
  }

  _renderCategories() {
    if (!this._runToTag) return;
    this._noDataWarning.hidden = !this._dataNotFound;
    this._tagFilterer.hidden = this._dataNotFound;
    var categories = this._dataNotFound ? [] : this._categories;
    var seen = new Set();
    categories.forEach((category, index) => {
      // Polymer dom-if: skip the search-results category when the query is
      // empty.
      if (
        category.metadata.type === 0 /* SEARCH_RESULTS */ &&
        category.name === ''
      ) {
        return;
      }
      seen.add(category.name);
      var view = this._categoryViews.get(category.name);
      if (!view) {
        view = document.createElement('tf-category-paginated-view');
        view.itemFactory = (item) => this._createCard(item);
        view.getCategoryItemKey = (item) => JSON.stringify(item);
        view.initialOpened = this._shouldOpen(index);
        this._categoryViews.set(category.name, view);
        this._categoriesEl.appendChild(view);
      }
      view.setCategory(category);
    });
    this._categoryViews.forEach((view, name) => {
      if (!seen.has(name)) {
        view.remove();
        this._categoryViews.delete(name);
      }
    });
  }

  _createCard(item) {
    var card = document.createElement('tf-histogram-card');
    var runs = this._allRuns();
    card.setRun(item.run);
    card.setTag(item.tag);
    var tagInfo =
      (this._runToTagInfo &&
        this._runToTagInfo[item.run] &&
        this._runToTagInfo[item.run][item.tag]) ||
      {};
    card.setTagMetadata(tagInfo);
    card.setColorScaleFunction(createRunsColorScale(runs));
    card.setTimeProperty(this._timeProperty);
    card.setHistogramMode(this._histogramMode);
    this._cards.add(card);
    this.dataProvider(item.run, item.tag).then((data) => {
      card.setSeriesData(item.run, this.toVz(data));
    });
    return card;
  }

  _updateRegexHint() {
    try {
      new RegExp(this._tagFilter);
      this._regexHint.textContent = '';
    } catch (e) {
      this._regexHint.textContent = 'Invalid regular expression';
    }
  }
}

customElements.define('tf-histogram-dashboard', TfHistogramDashboard);
