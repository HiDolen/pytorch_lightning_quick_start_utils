// 在 offset 模式下曲线下方填充淡出
import {VzHistogramTimeseries} from '../histogram/renderer/vz_histogram_timeseries.js';

const NS = 'http://www.w3.org/2000/svg';
const SOLID_PX = 1
; // 曲线下方实心距离(px)
const FADE_PX = 20; // 渐隐跨度(px)，高斯约 6σ
const BLUR_X_PX = 3.0; // 横向模糊(px)，抑制渐隐带端点的横向渗色
const HIDE_CSS = '.off-fade-src{fill-opacity:0 !important;}';

let uid = 0; // 滤镜 id 计数
let installed = false;

export function enableOffsetFillFade() {
  if (installed) return; // 防重复包装 _draw
  installed = true;
  const origDraw = VzHistogramTimeseries.prototype._draw;
  VzHistogramTimeseries.prototype._draw = function (duration) {
    const chart = this;
    const result = origDraw.call(this, duration);
    // d3 经 transition 提交 d/transform/fill(duration=0 也在下一帧提交),
    // 淡出层必须等首帧提交后再构建
    requestAnimationFrame(() => {
      const pairs = applyFade(chart, duration);
      if (pairs && duration > 0) startSyncLoop(chart, pairs, duration);
    });
    return result;
  };
}

// 过渡期间逐帧同步；token 防过期：新的 _draw 会换 token，旧循环自行退出
function startSyncLoop(chart, pairs, duration, onEnd) {
  const token = chart._offFadeToken;
  const t0 = performance.now();
  const tick = () => {
    if (chart._offFadeToken !== token) return;
    syncFrame(pairs);
    if (performance.now() - t0 < duration + 80) requestAnimationFrame(tick);
    else if (onEnd) onEnd();
  };
  requestAnimationFrame(tick);
}

// 拆除淡出层并恢复源填充
function teardown(chart) {
  const svg = chart.$.svg;
  svg.querySelectorAll('.off-fade').forEach((el) => el.remove());
  svg.querySelectorAll('.outline').forEach((p) =>
    p.classList.remove('off-fade-src')
  );
  chart._offFadePairs = null;
}

// 全量重建淡出层；返回逐帧同步所需的 {items, off, blur}，非 offset 模式返回 null。
// 切到非 offset 时若正处过渡，让现有淡出层随 fill-opacity 动画一起淡完再拆，
// 避免源填充以中途不透明度显形造成全涂色突变
function applyFade(chart, duration) {
  const svg = chart.$.svg;
  chart._offFadeToken = {};
  if (chart.mode !== 'offset') {
    const pairs = chart._offFadePairs;
    if (pairs && duration > 0) {
      startSyncLoop(chart, pairs, duration, () => teardown(chart));
    } else {
      teardown(chart);
    }
    return null;
  }
  teardown(chart);
  // 注入隐藏源填充的样式（每个 shadow root 一次；!important 压过 d3 内联动画）
  const root = chart.shadowRoot;
  if (!root.querySelector('.off-fade-css')) {
    const st = document.createElement('style');
    st.className = 'off-fade-css';
    st.textContent = HIDE_CSS;
    root.appendChild(st);
  }
  const paths = svg.querySelectorAll('.outline');
  if (!paths.length) return null;
  // 共享滤镜 def（每图唯一 id）
  const fid = 'off-fade-' + ++uid;
  const defs = document.createElementNS(NS, 'defs');
  defs.setAttribute('class', 'off-fade');
  const filter = document.createElementNS(NS, 'filter');
  filter.setAttribute('id', fid);
  // 固定大区域：百分比区域随扁平切片 bbox 变化，会裁掉下移后的形状
  filter.setAttribute('filterUnits', 'userSpaceOnUse');
  filter.setAttribute('x', '-500');
  filter.setAttribute('y', '-1000');
  filter.setAttribute('width', '2000');
  filter.setAttribute('height', '4000');
  const off = document.createElementNS(NS, 'feOffset');
  off.setAttribute('in', 'SourceGraphic');
  off.setAttribute('result', 'o');
  const blur = document.createElementNS(NS, 'feGaussianBlur');
  blur.setAttribute('in', 'o');
  blur.setAttribute('result', 'b');
  const comp = document.createElementNS(NS, 'feComposite');
  comp.setAttribute('in', 'SourceGraphic');
  comp.setAttribute('in2', 'b');
  comp.setAttribute('operator', 'out');
  filter.appendChild(off);
  filter.appendChild(blur);
  filter.appendChild(comp);
  defs.appendChild(filter);
  svg.appendChild(defs);
  // 每个切片一条克隆填充 path，挂滤镜
  const items = [];
  paths.forEach((p) => {
    if (!scaleOf(p).sy) return;
    p.classList.add('off-fade-src');
    const el = document.createElementNS(NS, 'path');
    el.setAttribute('class', 'off-fade');
    el.setAttribute('stroke', 'none');
    el.setAttribute('fill', p.style.fill || getComputedStyle(p).fill);
    el.setAttribute('filter', 'url(#' + fid + ')');
    p.parentNode.insertBefore(el, p.nextSibling);
    items.push({src: p, el});
  });
  if (!items.length) {
    defs.remove();
    return null;
  }
  const pairs = {items, off, blur};
  syncFrame(pairs); // 初值：对齐当前几何与透明度
  chart._offFadePairs = pairs;
  return pairs;
}

// path transform 的 scale（缺失返回 {0,0}）
function scaleOf(p) {
  const m = (p.getAttribute('transform') || '').match(
    /scale\(\s*([-\d.]+)\s*,\s*([-\d.]+)/
  );
  return m ? {sx: parseFloat(m[1]), sy: parseFloat(m[2])} : {sx: 0, sy: 0};
}

// 同步一帧：把 src 的 d/transform/fill-opacity 复制到克隆 path；dy/σ 换算成
// path 单位，使 50% 点落在曲线下 SOLID+FADE/2、渐隐全跨 FADE（屏幕像素恒定）
function syncFrame(pairs) {
  let sx = 0;
  let sy = 0;
  pairs.items.forEach((pair) => {
    const src = pair.src;
    const d = src.getAttribute('d');
    const tr = src.getAttribute('transform') || '';
    const s = scaleOf(src);
    let fo = parseFloat(src.style.fillOpacity);
    if (isNaN(fo)) fo = 1;
    if (!d || !s.sy) return;
    if (!sx) {
      sx = s.sx;
      sy = s.sy;
    }
    pair.el.setAttribute('d', d);
    if (tr !== pair.lastTr) {
      pair.lastTr = tr;
      pair.el.setAttribute('transform', tr);
    }
    if (fo !== pair.lastFo) {
      pair.lastFo = fo;
      pair.el.style.fillOpacity = String(fo);
    }
  });
  if (sx) {
    pairs.off.setAttribute('dy', (SOLID_PX + FADE_PX / 2) / sy);
    pairs.blur.setAttribute(
      'stdDeviation',
      BLUR_X_PX / sx + ' ' + FADE_PX / 6 / sy
    );
  }
}
