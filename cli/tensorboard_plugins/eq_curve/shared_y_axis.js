// Adaptation (new file, no TensorBoard counterpart): per-tag shared numeric
// axis across runs (default on), toggleable from a sidebar "Y-axis" selector
// injected into the dashboard's settings area. The shared histogram components
// stay pristine — the only hook is the renderer's single-line "_sharedYDomain
// overrides the auto domain" read; the setter is installed on the prototype
// here, effective only in this plugin's module realm (same technique as the
// d3.axisBottom wrapping in frequency_adapter.js).
import {VzHistogramTimeseries} from '../shared/histogram/renderer/vz_histogram_timeseries.js';

VzHistogramTimeseries.prototype.setSharedYDomain = function (domain) {
  this._sharedYDomain = domain;
  this._draw(500);
};

export function enableSharedYAxis(dashboard) {
  var enabled = true;
  var tagYMax = new Map(); // 聚合单调递增,迟到的更大 run 不会让轴回缩

  function domainOf(tag) {
    return tagYMax.has(tag) ? [0, tagYMax.get(tag)] : null;
  }
  function applyToTag(tag) {
    var domain = enabled ? domainOf(tag) : null;
    dashboard.getActiveCards().forEach(function (card) {
      if (card._tag === tag) card.$.chart.setSharedYDomain(domain);
    });
  }
  function refreshAllCards() {
    dashboard.getActiveCards().forEach(function (card) {
      card.$.chart.setSharedYDomain(enabled ? domainOf(card._tag) : null);
    });
  }

  // Sidebar 开关:shared/auto 分段文字,点击即设为该项
  var settings = dashboard.shadowRoot.querySelector('.settings');
  var section = document.createElement('div');
  section.className = 'sidebar-section';
  section.innerHTML =
    '<style>' +
    '.yaxis-seg { font-size: 13px; }' +
    '.yaxis-seg span { cursor: pointer; color: var(--tb-ui-dark-accent); padding: 2px 4px; }' +
    '.yaxis-seg span.on { color: #2196f3; font-weight: bold; }' +
    '.yaxis-seg .sep { color: var(--tb-ui-border); cursor: default; }' +
    '</style>' +
    '<div class="option-selector" id="sharedYAxisSelector">' +
    '<h3>Y-axis</h3>' +
    '<div class="content-wrapper"><span class="yaxis-seg">' +
    '<span data-v="shared">shared</span><span class="sep">|</span><span data-v="auto">auto</span>' +
    '</span></div></div>';
  settings.appendChild(section);
  var segEl = section.querySelector('.yaxis-seg');
  function paint() {
    segEl.querySelectorAll('[data-v]').forEach(function (s) {
      s.classList.toggle('on', (s.dataset.v === 'shared') === enabled);
    });
  }
  segEl.addEventListener('click', function (e) {
    var isShared = e.target.dataset && e.target.dataset.v === 'shared';
    var isAuto = e.target.dataset && e.target.dataset.v === 'auto';
    if ((isShared && enabled) || (isAuto && !enabled)) return;
    enabled = isShared;
    paint();
    refreshAllCards();
  });
  paint();

  // 无论开关状态都聚合，切回 shared 时即时生效；
  // 注入与否由 applyToTag 按当前开关决定。
  dashboard.addEventListener('card-data-updated', function (e) {
    var card = e.detail.card;
    var vz = card.$.chart._data || [];
    var yMax = -Infinity;
    for (var i = 0; i < vz.length; i++) {
      var bins = vz[i].bins;
      for (var j = 0; j < bins.length; j++) {
        if (bins[j].y > yMax) yMax = bins[j].y;
      }
    }
    var prev = tagYMax.has(card._tag) ? tagYMax.get(card._tag) : -Infinity;
    if (yMax > prev) tagYMax.set(card._tag, yMax);
    applyToTag(card._tag);
  });
  dashboard.addEventListener('active-cards-changed', refreshAllCards);
}
