// 全局 step 区间滑条，挂在 dashboard 中央栏顶部
import {VzHistogramTimeseries} from './histogram/renderer/vz_histogram_timeseries.js';

const SLIDER_TEMPLATE = `
  <h3>Step Range</h3>
  <div class="content-wrapper">
    <div class="eq-srs-track-wrap" title="Drag endpoints to select the step range; double-click to reset">
      <div class="eq-srs-track"></div>
      <div class="eq-srs-range"></div>
      <div class="eq-srs-thumb" data-which="lo"></div>
      <div class="eq-srs-thumb" data-which="hi"></div>
    </div>
    <div class="eq-srs-ticks"></div>
  </div>
`;

const SLIDER_STYLE = `
  .eq-srs-track-wrap {
    position: relative; height: 26px; touch-action: none;
    margin: 6px 2px 0;
  }
  .eq-srs-track {
    position: absolute; left: 7px; right: 7px; top: 50%; height: 2px;
    transform: translateY(-50%); background: var(--tb-ui-border); border-radius: 1px;
  }
  .eq-srs-range {
    position: absolute; top: 50%; height: 2px; transform: translateY(-50%);
    background: #2196f3; border-radius: 1px;
    transition: left .15s ease-out, width .15s ease-out;
  }
  .eq-srs-thumb {
    position: absolute; top: 50%; width: 12px; height: 12px;
    transform: translate(-50%, -50%); border-radius: 50%;
    background: #2196f3; border: 2px solid #fff;
    box-shadow: 0 1px 3px rgba(0,0,0,.35); cursor: grab; z-index: 2;
    transition: left .12s ease-out, transform .1s;
  }
  .eq-srs-thumb:hover, .eq-srs-thumb.active {
    transform: translate(-50%, -50%) scale(1.25);
  }
  .eq-srs-thumb.active { cursor: grabbing; }
  .eq-srs-ticks {
    position: relative; height: 15px; margin: 3px 2px 0;
  }
  .eq-srs-tick {
    position: absolute; top: 0; width: 0; pointer-events: none;
  }
  .eq-srs-tick-line {
    position: absolute; top: 0; left: 0; transform: translateX(-50%);
    width: 1px; height: 3px; background: #bdbdbd;
  }
  .eq-srs-tick-label {
    position: absolute; top: 4px; left: 0; transform: translateX(-50%);
    font-size: 10px; line-height: 1.3; color: #9e9e9e;
    white-space: nowrap;
  }
  /* 首尾标签边缘对齐，避免溢出被 sidebar 的 overflow:hidden 裁剪 */
  .eq-srs-tick.first .eq-srs-tick-label { transform: none; }
  .eq-srs-tick.last .eq-srs-tick-label { transform: translateX(-100%); }
`;

const clamp = (v, a, b) => Math.max(a, Math.min(b, v));

// 大 step 统一 SI 缩写（37.2k / 75k / 1.5M），与 TB 坐标轴风格一致
function fmtStep(v) {
  if (v < 1000) return '' + v;
  for (const [scale, suffix] of [[1e9, 'G'], [1e6, 'M'], [1e3, 'k']]) {
    if (v >= scale) {
      const s = (Math.round(v / (scale / 10)) / 10).toFixed(1).replace(/\.0$/, '');
      return s + suffix;
    }
  }
}

// d3 风格标准步长：1/2/5 × 10^k
function niceStep(raw) {
  const p = Math.pow(10, Math.floor(Math.log10(raw)));
  const e = raw / p;
  return p * (e >= 7.07 ? 10 : e >= 3.16 ? 5 : e >= 1.41 ? 2 : 1);
}

// paper-slider 观感的双端水平滑条；steps 为可选 step 值列表（按下标均匀分布）
class StepRangeSlider {
  constructor(row, onChange) {
    this.row = row;
    this.onChange = onChange;
    this.steps = [];
    this.lo = 0;
    this.hi = 0;
    this.drag = null;
    row.innerHTML = SLIDER_TEMPLATE;
    this.wrap = row.querySelector('.eq-srs-track-wrap');
    this.rangeEl = row.querySelector('.eq-srs-range');
    this.ticksEl = row.querySelector('.eq-srs-ticks');
    this.thumbs = {
      lo: row.querySelector('.eq-srs-thumb[data-which="lo"]'),
      hi: row.querySelector('.eq-srs-thumb[data-which="hi"]'),
    };
    for (const which of ['lo', 'hi']) this._attach(this.thumbs[which], which);
    this.wrap.addEventListener('dblclick', () => this.reset());
  }

  // 数据重载后更新可选域，尽量保持已选值
  setSteps(steps, loVal, hiVal) {
    this.steps = steps;
    this.lo = Math.max(0, steps.indexOf(loVal));
    this.hi = steps.indexOf(hiVal);
    if (this.hi < 0) this.hi = steps.length - 1;
    this._renderTicks();
    this.render();
  }

  // 自适应刻度：中间刻度取 1/2/5 标准步长的整洁值（20k/40k…），首尾
  // 忠实于真实 step 边界；渲染后按标签实际位置清除重叠（首尾优先保留）
  _renderTicks() {
    this.ticksEl.replaceChildren();
    const n = this.steps.length;
    if (n === 0) return;
    const width = this.wrap.clientWidth;
    const min = this.steps[0];
    const max = this.steps[n - 1];
    const vals = [min];
    if (max > min) {
      const target = Math.max(2, Math.floor(width / 44));
      const step = niceStep((max - min) / target);
      for (let v = Math.ceil(min / step) * step; v < max; v += step) {
        if (v > min && max - v >= step / 2) vals.push(v);
      }
      vals.push(max);
    }
    for (let i = 0; i < vals.length; i++) {
      const tick = document.createElement('div');
      tick.className = 'eq-srs-tick';
      if (i === 0) tick.classList.add('first');
      if (i === vals.length - 1) tick.classList.add('last');
      tick.style.left = this.valToX(vals[i]) + 'px';
      const line = document.createElement('span');
      line.className = 'eq-srs-tick-line';
      const label = document.createElement('span');
      label.className = 'eq-srs-tick-label';
      label.textContent = fmtStep(vals[i]);
      tick.appendChild(line);
      tick.appendChild(label);
      this.ticksEl.appendChild(tick);
    }
    // 过渡态（数据渐进到达）下刻度可能被压到一起，按渲染后的真实位置去重
    const kids = [...this.ticksEl.children];
    const rects = kids.map((k) =>
      k.querySelector('.eq-srs-tick-label').getBoundingClientRect()
    );
    const dropped = new Set();
    let prev = 0;
    for (let i = 1; i < kids.length; i++) {
      if (rects[i].left < rects[prev].right + 3) {
        if (i === kids.length - 1) dropped.add(prev); // 尾刻度忠实边界，优先保留
        else {
          dropped.add(i);
          continue;
        }
      }
      prev = i;
    }
    dropped.forEach((i) => kids[i].remove());
  }

  idxToX(i) {
    const w = this.wrap.clientWidth;
    const pad = 8; // thumb 半径留白
    if (this.steps.length <= 1) return w / 2;
    return pad + (i / (this.steps.length - 1)) * (w - 2 * pad);
  }

  // 任意值 → x：换算到最近 step 索引再映射，保证与锚点吸附位置一致
  valToX(v) {
    const n = this.steps.length;
    if (n <= 1) return this.wrap.clientWidth / 2;
    let lo = 0, hi = n - 1; // 二分：第一个 >= v 的索引
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (this.steps[mid] < v) lo = mid + 1;
      else hi = mid;
    }
    return this.idxToX(lo);
  }

  clientXToIdx(x) {
    const r = this.wrap.getBoundingClientRect();
    const pad = 8;
    if (this.steps.length <= 1) return 0;
    const t = clamp((x - r.left - pad) / (r.width - 2 * pad), 0, 1);
    return Math.round(t * (this.steps.length - 1));
  }

  _setDrag(which) {
    this.drag = which;
    this.thumbs.lo.classList.toggle('active', which === 'lo');
    this.thumbs.hi.classList.toggle('active', which === 'hi');
  }

  _endDrag() {
    this.drag = null;
    this._startX = null;
    this.thumbs.lo.classList.remove('active');
    this.thumbs.hi.classList.remove('active');
  }

  _attach(thumb, which) {
    thumb.addEventListener('pointerdown', (e) => {
      if (e.button !== 0) return;
      // 两端重叠时（如都拖到最右）命中的必是上层 thumb，无法区分是哪端，
      // 先挂起，由首次拖动方向决定要移动的那一端
      if (this.lo === this.hi) {
        this.drag = 'pending';
        this._startX = e.clientX;
      } else {
        this._setDrag(which);
      }
      thumb.setPointerCapture(e.pointerId);
      e.preventDefault();
      e.stopPropagation();
    });
    thumb.addEventListener('pointermove', (e) => {
      // 跨 iframe 边界松开鼠标时 pointerup 可能丢失，drag 状态残留导致
      // 锚点"粘"在鼠标上；发现左键已松开就立即结束拖动
      if ((e.buttons & 1) === 0) {
        if (this.drag) this._endDrag();
        return;
      }
      if (this.drag === 'pending') {
        const dx = e.clientX - this._startX;
        if (dx > 2) this._setDrag('hi');
        else if (dx < -2) this._setDrag('lo');
        else return;
      } else if (this.drag === null) {
        return;
      }
      // 吸附到最近 step；位置变化由 CSS transition 平滑过渡
      const w = this.drag;
      const i = this.clientXToIdx(e.clientX);
      const ni =
        w === 'lo'
          ? clamp(i, 0, this.hi)
          : clamp(i, this.lo, this.steps.length - 1);
      if (ni !== this[w]) {
        this[w] = ni;
        this.render();
        this.onChange(this.steps[this.lo], this.steps[this.hi]);
      }
    });
    // up 与 lostpointercapture 均做幂等清理，防 capture 异常释放时状态残留
    thumb.addEventListener('pointerup', () => this._endDrag());
    thumb.addEventListener('pointercancel', () => this._endDrag());
    thumb.addEventListener('lostpointercapture', () => this._endDrag());
  }

  reset() {
    if (this.steps.length === 0) return;
    this.lo = 0;
    this.hi = this.steps.length - 1;
    this.render();
    this.onChange(this.steps[this.lo], this.steps[this.hi]);
  }

  render() {
    const xLo = this.idxToX(this.lo);
    const xHi = this.idxToX(this.hi);
    this.thumbs.lo.style.left = xLo + 'px';
    this.thumbs.hi.style.left = xHi + 'px';
    this.rangeEl.style.left = xLo + 'px';
    this.rangeEl.style.width = Math.max(0, xHi - xLo) + 'px';
  }
}

let _dashboard = null;
let _slider = null;
let _row = null;
let _steps = [];
let _lo = null;
let _hi = null;

// 单卡过滤：区间外的 g.histogram 整组淡出（opacity 为组级，不干扰内部
// 样式）；同时设置渲染器 _stepFilter，使 hover 忽略被过滤的切片
function applyFilterToChart(chart) {
  if (!chart) return;
  if (_lo === null || _hi === null) {
    chart._stepFilter = null;
    return;
  }
  chart._stepFilter = (step) => step >= _lo && step <= _hi;
  const root = chart.shadowRoot;
  if (!root) return;
  root.querySelectorAll('g.histogram').forEach((g) => {
    const d = g.__data__;
    if (!d) return;
    g.style.opacity = d.step >= _lo && d.step <= _hi ? '' : '0.05';
  });
}

function applyFilter() {
  if (!_dashboard) return;
  _dashboard._cards.forEach((card) => applyFilterToChart(card.$.chart));
}

// 重算全局 step 并集；域变化时保持用户主动选择的端点值（旧值若在旧域
// 边界上则视为"未主动选择"，跟随新边界，保证默认始终是全范围）
function refreshSteps() {
  const set = new Set();
  _dashboard._cards.forEach((card) => {
    const data = (card.$.chart && card.$.chart._data) || [];
    data.forEach((d) => set.add(d.step));
  });
  const steps = Array.from(set).sort((a, b) => a - b);
  if (steps.length === 0) return;
  const same =
    steps.length === _steps.length && steps.every((s, i) => s === _steps[i]);
  if (same) return;
  const keepLo = _lo !== null && _steps.length && _lo !== _steps[0];
  const keepHi =
    _hi !== null && _steps.length && _hi !== _steps[_steps.length - 1];
  const lo = keepLo && steps.includes(_lo) ? _lo : steps[0];
  const hi = keepHi && steps.includes(_hi) ? _hi : steps[steps.length - 1];
  _steps = steps;
  _lo = lo;
  _hi = hi;
  _row.hidden = false;
  _slider.setSteps(steps, lo, hi);
  applyFilter();
}

export function installStepRangeSlider(dashboard) {
  if (_dashboard) return;
  _dashboard = dashboard;
  const root = dashboard._root;
  const style = document.createElement('style');
  style.textContent = SLIDER_STYLE;
  root.appendChild(style);
  // 挂到左侧 settings 区，作为与 Histogram mode / Y-axis 同款的 sidebar-section
  _row = document.createElement('div');
  _row.className = 'sidebar-section option-selector';
  _row.hidden = true;
  const settings = root.querySelector('.settings');
  settings.appendChild(_row);
  _slider = new StepRangeSlider(_row, (lo, hi) => {
    _lo = lo;
    _hi = hi;
    applyFilter();
  });
  // 数据每次到达后（新卡片 / 刷新重取）重算可选域；
  // setTimeout 保证在 setSeriesData 写入卡片之后再收集
  const origProvider = dashboard.dataProvider;
  dashboard.dataProvider = (run, tag) =>
    origProvider(run, tag).then((d) => {
      setTimeout(refreshSteps, 0);
      return d;
    });
  // redraw 后自动套用过滤，覆盖新数据、展开、模式切换等一切重画路径
  const origRedraw = VzHistogramTimeseries.prototype.redraw;
  VzHistogramTimeseries.prototype.redraw = function (...args) {
    const result = origRedraw.apply(this, args);
    applyFilterToChart(this);
    return result;
  };
}
