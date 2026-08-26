// Source: tensorboard/plugins/histogram/tf_histogram_dashboard/tf-histogram-loader.ts
// Source commit: 012299c95682b6cf89e0eaccb8dc034cf1d61cd2
// Adaptation: Polymer element ported to a native custom element; template DOM
// and styles copied verbatim. Data loading is injected by the host (instead of
// the DataLoaderBehavior HTTP path), paper-icon-button by a native equivalent.
import './renderer/vz_histogram_timeseries.js';
import './tf_card_heading.js';
import {defaultColorScale} from './color_scale.js';

const TEMPLATE = `
    <tf-card-heading></tf-card-heading>
    <vz-histogram-timeseries id="chart"></vz-histogram-timeseries>
    <div style="display: flex; flex-direction: row;">
      <button
        class="paper-icon-button"
        title="Expand or collapse card"
      >
        <svg viewBox="0 0 24 24" width="24" height="24"><path fill="currentColor" d="M7 14H5v5h5v-2H7v-3zm-2-4h2V7h3V5H5v5zm12 7h-3v2h5v-5h-2v3zM14 5v2h3v3h2V5h-5z"/></svg>
      </button>
    </div>
`;

const STYLE = `
      :host {
        display: flex;
        flex-direction: column;
        width: 330px;
        height: 235px;
        margin-right: 10px;
        margin-bottom: 15px;
      }
      :host([_expanded]) {
        width: 700px;
        height: 500px;
      }

      vz-histogram-timeseries {
        -moz-user-select: none;
        -webkit-user-select: none;
        will-change: transform;
      }

      .paper-icon-button {
        color: #2196f3;
        border-radius: 100%;
        width: 32px;
        height: 32px;
        padding: 4px;
        background: none;
        border: none;
        cursor: pointer;
      }

      .paper-icon-button[selected] {
        background: var(--tb-ui-light-accent);
      }

      tf-card-heading {
        margin-bottom: 10px;
        width: 90%;
      }
`;

export class TfHistogramCard extends HTMLElement {
  constructor() {
    super();
    const root = this.attachShadow({mode: 'open'});
    root.innerHTML = `<style>${STYLE}</style>${TEMPLATE}`;
    this.$ = {
      chart: root.getElementById('chart'),
    };
    this._heading = root.querySelector('tf-card-heading');
    this._button = root.querySelector('.paper-icon-button');
    this._colorScaleFunction = defaultColorScale;
    this._expanded = false;
  }

  connectedCallback() {
    // 宿主会在过滤/分页时反复重新插入节点，重复连接不得叠加监听
    if (this._listenerBound) return;
    this._listenerBound = true;
    this._button.addEventListener('click', () => this._toggleExpanded());
  }

  setRun(run) {
    this._run = run;
    this._heading.run = run;
    this._updateRunColor();
  }

  setTag(tag) {
    this._tag = tag;
    this._heading.tag = tag;
  }

  setTagMetadata(tagMetadata) {
    this._tagMetadata = tagMetadata;
    this._heading.displayName = tagMetadata && tagMetadata.displayName;
    this._heading.description = tagMetadata && tagMetadata.description;
  }

  _updateRunColor() {
    this._heading.color = this._runColor();
  }

  _runColor() {
    var run = this._run;
    return this._colorScaleFunction(run);
  }

  setColorScaleFunction(fn) {
    this._colorScaleFunction = fn;
    this.$.chart.setColorScale(fn);
    this._updateRunColor();
  }

  setTimeProperty(timeProperty) {
    this.$.chart.setTimeProperty(timeProperty);
  }

  setHistogramMode(histogramMode) {
    this.$.chart.setMode(histogramMode);
  }

  setSeriesData(name, data) {
    this.$.chart.setSeriesData(name, data);
  }

  redraw() {
    this.$.chart.redraw();
  }

  _toggleExpanded(e) {
    this._expanded = !this._expanded;
    if (this._expanded) {
      this.setAttribute('_expanded', '');
    } else {
      this.removeAttribute('_expanded');
    }
    if (this._expanded) {
      this._button.setAttribute('selected', '');
    } else {
      this._button.removeAttribute('selected');
    }
    this.redraw();
  }
}

customElements.define('tf-histogram-card', TfHistogramCard);
