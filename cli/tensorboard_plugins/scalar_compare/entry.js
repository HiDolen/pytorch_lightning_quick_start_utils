import d3 from '../shared/histogram/vendor/d3-esm.js';
import {normalizeScalars, selectTags, smoothScalars} from './data.js';

const TEMPLATE = `
  <link rel="stylesheet" href="${new URL('./style.css', import.meta.url)}">
  <aside>
    <label class="section-label" for="run">Run</label>
    <input id="run-filter" type="search" placeholder="Filter runs" aria-label="Filter runs">
    <select id="run" aria-label="Run"></select>
    <div class="section-heading"><h2>Metrics</h2><span id="selection-count"></span></div>
    <input id="tag-filter" type="search" placeholder="Filter metrics" aria-label="Filter metrics">
    <div class="selection-actions">
      <button id="select-all" type="button">Select all</button>
      <button id="clear" type="button">Clear</button>
    </div>
    <div id="tag-list" class="tag-list" aria-label="Metrics"></div>
  </aside>
  <main>
    <header>
      <div class="heading"><h1>Scalar Compare</h1><div id="run-name"></div></div>
      <button id="refresh" type="button" title="Refresh scalar data">Refresh</button>
    </header>
    <div class="toolbar">
      <label>X axis<select id="x-axis"><option value="step">Step</option><option value="wallTime">Wall time</option></select></label>
      <label>Y axis<select id="y-axis"><option value="linear">Linear</option><option value="log">Log</option></select></label>
      <label class="smoothing" for="smoothing">Smoothing <output id="smoothing-value" aria-hidden="true">0</output><input id="smoothing" type="range" min="0" max="0.99" step="0.01" value="0"></label>
      <button id="reset" type="button" title="Reset chart zoom">Reset zoom</button>
    </div>
    <div id="status" role="status" aria-live="polite"></div>
    <div id="chart" class="chart">
      <svg role="img" aria-label="Scalar metrics comparison"></svg>
      <div id="empty" class="empty"></div>
    </div>
    <div class="table-scroll">
      <table aria-label="Metric values"><thead><tr><th>Metric</th><th>Step</th><th>Value</th><th>Smoothed</th><th>Wall time</th></tr></thead><tbody id="values"></tbody></table>
    </div>
  </main>
`;

const compareNames = (left, right) => left.localeCompare(right, undefined, {numeric: true});
const formatValue = value => Number.isFinite(value) ? d3.format('.6~g')(value) : '--';

export class ScalarCompareDashboard extends HTMLElement {
  constructor() {
    super();
    this._root = this.attachShadow({mode: 'open'});
    this._root.innerHTML = TEMPLATE;
    this._tags = {};
    this._run = '';
    this._selected = null;
    this._series = new Map();
    this._failed = new Set();
    this._requestId = 0;
    this._colors = d3.scaleOrdinal(d3.schemeCategory10);
    this._transform = d3.zoomIdentity;
    this._rows = [];
    this._get('refresh').addEventListener('click', () => this.refresh());
    this._get('run-filter').addEventListener('input', () => this._renderRuns());
    this._get('tag-filter').addEventListener('input', () => this._renderTags());
    this._get('run').addEventListener('change', () => {
      this._run = this._get('run').value;
      this._series.clear();
      this._syncSelection(true);
      this._renderTags();
      this._resetZoom();
      this._loadSeries();
    });
    this._get('select-all').addEventListener('click', () => {
      this._selected = [...new Set([...(this._selected || []), ...this._filteredTags()])];
      this._renderTags();
      this._loadSeries();
    });
    this._get('clear').addEventListener('click', () => {
      this._selected = [];
      this._renderTags();
      this._loadSeries();
    });
    for (const axis of ['x-axis', 'y-axis']) {
      this._get(axis).addEventListener('change', () => this._resetZoom());
    }
    this._get('smoothing').addEventListener('input', () => {
      this._get('smoothing-value').value = this._get('smoothing').value;
      this._draw();
    });
    this._get('reset').addEventListener('click', () => this._resetZoom());
    this._svg = d3.select(this._get('chart').querySelector('svg'));
    this._clip = this._svg.append('defs').append('clipPath').attr('id', 'comparison-clip').append('rect');
    this._plot = this._svg.append('g');
    this._xAxis = this._plot.append('g').attr('class', 'axis');
    this._yAxis = this._plot.append('g').attr('class', 'axis');
    this._lines = this._plot.append('g').attr('clip-path', 'url(#comparison-clip)');
    this._hit = this._plot.append('rect').attr('fill', 'transparent').attr('class', 'interaction');
    this._cursor = this._plot.append('line').attr('class', 'cursor').attr('display', 'none');
    this._xLabel = this._svg.append('text').attr('class', 'axis-label');
    this._zoom = d3.zoom().scaleExtent([1, 200]).on('zoom', () => {
      this._transform = d3.event.transform;
      this._draw();
    });
    this._hit.call(this._zoom)
      .on('mousemove.readout', () => this._readout(d3.mouse(this._hit.node())[0]))
      .on('mouseleave.readout', () => {
        this._cursor.attr('display', 'none');
        this._updateValues();
      });
    this._resize = new ResizeObserver(() => this._draw());
  }

  connectedCallback() {
    this._resize.observe(this._get('chart'));
    this.refresh();
  }

  disconnectedCallback() {
    this._controller?.abort();
    this._resize.disconnect();
  }

  _get(id) {
    return this._root.getElementById(id);
  }

  _beginRequest() {
    this._controller?.abort();
    this._controller = new AbortController();
    return {id: ++this._requestId, signal: this._controller.signal};
  }

  async _fetch(url, signal) {
    const response = await fetch(url, {signal, cache: 'no-store'});
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return response.json();
  }

  _status(message, error = false) {
    this._get('status').textContent = message;
    this._get('status').classList.toggle('error', error);
  }

  async refresh() {
    const request = this._beginRequest();
    this._status('Loading scalar tags...');
    try {
      const tags = await this._fetch('tags', request.signal);
      if (request.id !== this._requestId) return;
      const previousRun = this._run;
      this._tags = tags;
      const runs = Object.keys(tags).sort(compareNames);
      if (!Object.hasOwn(tags, this._run)) this._run = runs[0] || '';
      if (previousRun !== this._run) this._series.clear();
      this._syncSelection(previousRun !== this._run);
      this._renderRuns();
      this._renderTags();
      await this._loadSeries();
    } catch (error) {
      if (request.id !== this._requestId || error.name === 'AbortError') return;
      this._status(`Unable to load scalar tags: ${error.message}`, true);
      this._draw();
    }
  }

  _availableTags() {
    return Object.keys(this._tags[this._run] || {}).sort(compareNames);
  }

  _syncSelection(runChanged) {
    const tags = this._availableTags();
    if (!tags.length) {
      this._selected = null;
      return;
    }
    if (runChanged && !selectTags(tags, this._selected || []).length) this._selected = null;
    this._selected = selectTags(tags, this._selected);
  }

  _filteredTags() {
    const filter = this._get('tag-filter').value.toLowerCase();
    return this._availableTags().filter(tag => tag.toLowerCase().includes(filter));
  }

  _renderRuns() {
    const select = this._get('run');
    const filter = this._get('run-filter').value.toLowerCase();
    const runs = Object.keys(this._tags).sort(compareNames)
      .filter(run => run === this._run || run.toLowerCase().includes(filter));
    select.replaceChildren();
    for (const run of runs) {
      const option = document.createElement('option');
      option.value = run;
      option.textContent = run;
      select.appendChild(option);
    }
    select.disabled = !runs.length;
    select.value = this._run;
  }

  _renderTags() {
    const selected = new Set(this._selected || []);
    const list = this._get('tag-list');
    list.replaceChildren();
    for (const tag of this._filteredTags()) {
      const label = document.createElement('label');
      label.className = 'metric-option';
      label.title = tag;
      const input = document.createElement('input');
      input.type = 'checkbox';
      input.checked = selected.has(tag);
      input.addEventListener('change', () => {
        const next = new Set(this._selected || []);
        if (input.checked) next.add(tag);
        else next.delete(tag);
        this._selected = this._availableTags().filter(value => next.has(value));
        this._get('selection-count').textContent = String(this._selected.length);
        this._loadSeries();
      });
      const swatch = document.createElement('span');
      swatch.className = 'swatch';
      swatch.style.backgroundColor = this._colors(tag);
      const name = document.createElement('span');
      name.className = 'metric-name';
      name.textContent = tag;
      label.append(input, swatch, name);
      list.appendChild(label);
    }
    this._get('selection-count').textContent = String(selected.size);
  }

  async _loadSeries() {
    const request = this._beginRequest();
    const run = this._run;
    const tags = [...(this._selected || [])];
    this._series = new Map([...this._series].filter(([tag]) => tags.includes(tag)));
    this._failed.clear();
    this._status(tags.length ? 'Loading scalar values...' : '');
    this._draw();
    const results = await Promise.allSettled(tags.map(async tag => {
      const query = new URLSearchParams({run, tag, format: 'json'});
      const events = await this._fetch(`scalars?${query}`, request.signal);
      return normalizeScalars(events);
    }));
    if (request.id !== this._requestId) return;
    this._series = new Map();
    results.forEach((result, index) => {
      if (result.status === 'fulfilled') this._series.set(tags[index], result.value);
      else this._failed.add(tags[index]);
    });
    this._status(this._failed.size ? `Unable to load: ${[...this._failed].join(', ')}` : '', this._failed.size > 0);
    this._draw();
  }

  _resetZoom() {
    this._hit.call(this._zoom.transform, d3.zoomIdentity);
  }

  _draw() {
    const box = this._get('chart').getBoundingClientRect();
    const outerWidth = Math.max(260, box.width);
    const outerHeight = Math.max(260, box.height);
    const margin = {top: 16, right: 24, bottom: 42, left: 66};
    const width = outerWidth - margin.left - margin.right;
    const height = outerHeight - margin.top - margin.bottom;
    const wallTime = this._get('x-axis').value === 'wallTime';
    const logarithmic = this._get('y-axis').value === 'log';
    const weight = Number(this._get('smoothing').value);
    const valid = value => Number.isFinite(value) && (!logarithmic || value > 0);
    this._viewSeries = [...this._series].map(([tag, points]) => ({
      tag,
      points: smoothScalars(points, weight).map(point => ({
        ...point, x: wallTime ? point.wallTime * 1000 : point.step,
      })).sort((left, right) => left.x - right.x),
    }));
    const allPoints = this._viewSeries.flatMap(series => series.points);
    const values = allPoints.flatMap(point => [point.value, point.smoothed]).filter(valid);
    const xDomain = d3.extent(allPoints, point => point.x);
    if (xDomain[0] === undefined) xDomain.splice(0, 2, 0, 1);
    if (xDomain[0] === xDomain[1]) xDomain[1] += wallTime ? 1000 : 1;
    let yDomain = d3.extent(values);
    if (!values.length) yDomain = logarithmic ? [1, 10] : [0, 1];
    if (yDomain[0] === yDomain[1]) {
      if (logarithmic) yDomain = [yDomain[0] / 2, yDomain[1] * 2];
      else {
        const padding = Math.max(1, Math.abs(yDomain[0]) * 0.05);
        yDomain = [yDomain[0] - padding, yDomain[1] + padding];
      }
    }
    const baseX = (wallTime ? d3.scaleTime() : d3.scaleLinear()).domain(xDomain).range([0, width]);
    this._xScale = this._transform.rescaleX(baseX);
    const yScale = (logarithmic ? d3.scaleLog() : d3.scaleLinear()).domain(yDomain).nice().range([height, 0]);
    this._svg.attr('viewBox', `0 0 ${outerWidth} ${outerHeight}`);
    this._plot.attr('transform', `translate(${margin.left},${margin.top})`);
    this._clip.attr('width', width).attr('height', height);
    this._xAxis.attr('transform', `translate(0,${height})`).call(d3.axisBottom(this._xScale).ticks(Math.max(2, Math.floor(width / 110))));
    this._yAxis.call(d3.axisLeft(yScale).ticks(6, '~g'));
    this._xLabel.attr('x', margin.left + width / 2).attr('y', outerHeight - 5).text(wallTime ? 'Wall time' : 'Step');
    this._hit.attr('width', width).attr('height', height);
    this._zoom.extent([[0, 0], [width, height]]).translateExtent([[0, 0], [width, height]]);
    this._cursor.attr('y1', 0).attr('y2', height).attr('display', 'none');
    for (const kind of ['raw', 'smoothed']) {
      const valueKey = kind === 'raw' ? 'value' : 'smoothed';
      const line = d3.line().defined(point => valid(point[valueKey]))
        .x(point => this._xScale(point.x)).y(point => yScale(point[valueKey]));
      const paths = this._lines.selectAll(`path.${kind}`).data(this._viewSeries, series => series.tag);
      paths.exit().remove();
      paths.enter().append('path').attr('class', kind).merge(paths)
        .attr('d', series => line(series.points)).attr('stroke', series => this._colors(series.tag))
        .attr('opacity', kind === 'raw' ? (weight ? 0.2 : 0) : 1);
    }
    const singlePoints = this._viewSeries.filter(series => series.points.filter(point => valid(point.smoothed)).length === 1)
      .map(series => ({tag: series.tag, point: series.points.find(point => valid(point.smoothed))}));
    const dots = this._lines.selectAll('circle').data(singlePoints, series => series.tag);
    dots.exit().remove();
    dots.enter().append('circle').attr('r', 3).merge(dots)
      .attr('cx', series => this._xScale(series.point.x)).attr('cy', series => yScale(series.point.smoothed))
      .attr('fill', series => this._colors(series.tag));
    this._get('run-name').textContent = this._run;
    this._get('run-name').title = this._run;
    this._get('empty').hidden = values.length > 0;
    this._get('empty').textContent = !this._run ? 'No scalar runs' : !(this._selected || []).length
      ? 'No metrics selected' : logarithmic && allPoints.length ? 'No positive values for log scale' : 'No scalar values';
    this._buildTable();
  }

  _buildTable() {
    this._rows = [];
    const body = this._get('values');
    body.replaceChildren();
    for (const tag of this._selected || []) {
      const row = document.createElement('tr');
      const metric = document.createElement('td');
      const swatch = document.createElement('span');
      swatch.className = 'swatch';
      swatch.style.backgroundColor = this._colors(tag);
      const name = document.createElement('span');
      name.textContent = tag;
      name.title = tag;
      metric.append(swatch, name);
      row.appendChild(metric);
      const cells = Array.from({length: 4}, () => document.createElement('td'));
      row.append(...cells);
      body.appendChild(row);
      this._rows.push({tag, cells});
    }
    this._updateValues();
  }

  _readout(pixel) {
    if (!this._xScale) return;
    this._cursor.attr('x1', pixel).attr('x2', pixel).attr('display', null);
    this._updateValues(Number(this._xScale.invert(pixel)));
  }

  _updateValues(target = null) {
    const bisect = d3.bisector(point => point.x).left;
    const byTag = new Map((this._viewSeries || []).map(series => [series.tag, series.points]));
    for (const {tag, cells} of this._rows) {
      const points = byTag.get(tag) || [];
      let point = points[points.length - 1];
      if (target !== null && points.length) {
        const index = bisect(points, target);
        const before = points[Math.max(0, index - 1)];
        const after = points[Math.min(points.length - 1, index)];
        point = target - before.x <= after.x - target ? before : after;
      }
      const values = point ? [formatValue(point.step), formatValue(point.value), formatValue(point.smoothed), new Date(point.wallTime * 1000).toLocaleString()]
        : ['--', this._failed.has(tag) ? 'Unavailable' : '--', '--', '--'];
      cells.forEach((cell, index) => {cell.textContent = values[index];});
    }
  }
}

customElements.define('scalar-compare-dashboard', ScalarCompareDashboard);

export function render() {
  document.documentElement.style.height = '100%';
  document.body.style.cssText = 'height:100%;margin:0';
  const dashboard = document.createElement('scalar-compare-dashboard');
  document.body.appendChild(dashboard);
  const channel = new MessageChannel();
  channel.port1.onmessage = async event => {
    let message;
    try {message = JSON.parse(event.data);} catch {return;}
    if (!message || message.isReply) return;
    if (message.type === 'experimental.DataReloaded') await dashboard.refresh();
    channel.port1.postMessage(JSON.stringify({type: message.type, id: message.id, payload: null, error: null, isReply: true}));
  };
  window.parent.postMessage('experimental.bootstrap', '*', [channel.port2]);
  window.addEventListener('pagehide', () => channel.port1.close(), {once: true});
}