// offset 模式下的单卡变化曲线面板：悬停卡片时，展示悬停 x 位置的数值
// 随 step 变化的曲线（y 轴域与源卡片数值轴一致），停靠在左侧控制面板区域。
// 单 step 模式下由聚合读数面板（hover_readout）接管，此处不显示。
import d3 from '../histogram/vendor/d3-esm.js';
import {pickTextColor} from '../histogram/tf_card_heading.js';
import {isSingleStepMode, getStepRange} from './step_range_slider.js';

const PANEL_STYLE = `
  .eq-scr-panel {
    position: fixed; z-index: 10; box-sizing: border-box;
    background: #fff; border: 1px solid #e0e0e0; border-radius: 4px;
    box-shadow: 0 2px 8px rgba(0,0,0,.18);
    padding: 6px 10px; font-size: 12px; line-height: 1.6;
    pointer-events: none; display: none;
    font-family: Roboto, "Helvetica Neue", sans-serif;
  }
  .eq-scr-panel .head {
    display: flex; align-items: center; gap: 6px;
    color: #212121; max-width: 280px;
  }
  .eq-scr-panel .run-badge {
    flex: none; display: inline-block; font-size: 11px; font-weight: bold;
    border-radius: 3px; padding: 1px 4px 2px; white-space: nowrap;
    max-width: 120px; box-sizing: border-box;
    overflow: hidden; text-overflow: ellipsis;
  }
  .eq-scr-panel .name {
    color: #616161; white-space: nowrap;
    overflow: hidden; text-overflow: ellipsis;
  }
  .eq-scr-panel .sub {
    color: #757575; margin-bottom: 2px;
    font-variant-numeric: tabular-nums;
  }
  .eq-scr-panel .sub b { color: #212121; font-weight: 500; }
  .eq-scr-panel text { font-family: Roboto, "Helvetica Neue", sans-serif; }
`;

const fmt = d3.format('.4g');
const SVG_NS = 'http://www.w3.org/2000/svg';
const SVG_H = 112;
const MARGIN = {top: 8, right: 10, bottom: 16, left: 38};

export function enableStepCurveReadout(dashboard, formatX) {
  const fmtX = formatX || ((x) => 'x ' + fmt(x));
  const root = dashboard._root;
  const style = document.createElement('style');
  style.textContent = PANEL_STYLE;
  root.appendChild(style);
  const panel = document.createElement('div');
  panel.className = 'eq-scr-panel';
  root.appendChild(panel);

  // 停靠在左侧 sidebar 区域
  function position() {
    const slot = root.querySelector('.sidebar-slot');
    const r = slot && slot.getBoundingClientRect();
    panel.style.left = (r ? r.left : 8) + 'px';
    panel.style.top = (r ? r.top : 8) + 'px';
  }

  function buildSvg(pts, yDomain, curPt, runColor, svgW) {
    const svg = document.createElementNS(SVG_NS, 'svg');
    svg.setAttribute('width', svgW);
    svg.setAttribute('height', SVG_H);
    const xDomain = d3.extent(pts, (p) => p.step);
    if (xDomain[0] === xDomain[1]) xDomain[1] = xDomain[0] + 1;
    const xScale = d3
      .scaleLinear()
      .domain(xDomain)
      .range([MARGIN.left, svgW - MARGIN.right]);
    const yScale = d3
      .scaleLinear()
      .domain(yDomain)
      .range([SVG_H - MARGIN.bottom, MARGIN.top]);
    const g = d3.select(svg);
    // 数值 0 基准线
    if (yDomain[0] < 0 && yDomain[1] > 0) {
      g.append('line')
        .attr('x1', MARGIN.left)
        .attr('x2', svgW - MARGIN.right)
        .attr('y1', yScale(0))
        .attr('y2', yScale(0))
        .attr('stroke', '#e0e0e0')
        .attr('stroke-dasharray', '3,3');
    }
    const line = d3
      .line()
      .x((p) => xScale(p.step))
      .y((p) => yScale(p.value));
    g.append('path')
      .attr('d', line(pts))
      .attr('fill', 'none')
      .attr('stroke', runColor)
      .attr('stroke-width', 1.5);
    // 当前 step 标记：竖参考线 + 圆点
    if (curPt) {
      g.append('line')
        .attr('x1', xScale(curPt.step))
        .attr('x2', xScale(curPt.step))
        .attr('y1', MARGIN.top)
        .attr('y2', SVG_H - MARGIN.bottom)
        .attr('stroke', '#bdbdbd');
      g.append('circle')
        .attr('cx', xScale(curPt.step))
        .attr('cy', yScale(curPt.value))
        .attr('r', 3)
        .attr('fill', runColor)
        .attr('stroke', '#fff')
        .attr('stroke-width', 1.5);
    }
    // y 轴域标签（与源卡片数值轴一致的上下界）
    g.append('text')
      .attr('x', MARGIN.left - 4)
      .attr('y', MARGIN.top + 4)
      .attr('text-anchor', 'end')
      .attr('font-size', 10)
      .attr('fill', '#9e9e9e')
      .text(fmt(yDomain[1]));
    g.append('text')
      .attr('x', MARGIN.left - 4)
      .attr('y', SVG_H - MARGIN.bottom + 4)
      .attr('text-anchor', 'end')
      .attr('font-size', 10)
      .attr('fill', '#9e9e9e')
      .text(fmt(yDomain[0]));
    // step 轴两端标签
    g.append('text')
      .attr('x', MARGIN.left)
      .attr('y', SVG_H - 3)
      .attr('font-size', 10)
      .attr('fill', '#9e9e9e')
      .text(xDomain[0]);
    g.append('text')
      .attr('x', svgW - MARGIN.right)
      .attr('y', SVG_H - 3)
      .attr('text-anchor', 'end')
      .attr('font-size', 10)
      .attr('fill', '#9e9e9e')
      .text(xDomain[1]);
    return svg;
  }

  function onHover(card, detail) {
    const chart = card.$.chart;
    if (!detail || chart.mode !== 'offset' || isSingleStepMode()) {
      panel.style.display = 'none';
      return;
    }
    const data = chart._data || [];
    const tp = chart.timeProperty;
    const bisect = d3.bisector((b) => b[chart.x] + b[chart.dx]).left;
    // step 范围跟随滑条选择，区间外的不进入曲线
    const range = getStepRange();
    // 与渲染器 hover 相同的 bin 定位：每个 step 取悬停 x 所在 bin 的数值
    const pts = data
      .filter((d) => !range || (d[tp] >= range.lo && d[tp] <= range.hi))
      .map((d) => {
        const bins = d[chart.bins];
        const i = Math.min(bins.length - 1, bisect(bins, detail.value));
        return {step: d[tp], value: bins[i][chart.y]};
      })
      .sort((a, b) => a.step - b.step);
    if (!pts.length) {
      panel.style.display = 'none';
      return;
    }
    // y 轴域与源卡片一致：共享域优先，缺省按数据自动（同渲染器逻辑）
    let yDomain = chart._sharedYDomain || [
      0,
      d3.max(data, (d) => d3.max(d[chart.bins], (b) => b[chart.y])),
    ];
    if (yDomain[0] === yDomain[1]) yDomain = [yDomain[0], yDomain[1] + 1];

    const curPt = pts.find((p) => p.step === detail.step) || null;
    const runColor = card._colorScaleFunction(card._run);
    const tagLabel = (card._heading && card._heading.displayName) || card._tag || '';

    const head = document.createElement('div');
    head.className = 'head';
    if (card._run) {
      const badge = document.createElement('span');
      badge.className = 'run-badge';
      badge.textContent = card._run;
      badge.title = card._run;
      badge.style.background = runColor;
      badge.style.color = pickTextColor(runColor);
      head.appendChild(badge);
    }
    const name = document.createElement('span');
    name.className = 'name';
    name.textContent = tagLabel;
    name.title = card._run ? card._run + ' / ' + tagLabel : tagLabel;
    head.appendChild(name);

    const sub = document.createElement('div');
    sub.className = 'sub';
    sub.innerHTML =
      'x = <b>' + fmtX(detail.value) + '</b>' +
      (curPt ? ' · step <b>' + fmt(curPt.step) + '</b> · <b>' + fmt(curPt.value) + '</b>' : '');

    // 面板宽度贴合左侧控制栏，SVG 随之自适应
    const slot = root.querySelector('.sidebar-slot');
    const slotW = slot ? slot.getBoundingClientRect().width : 292;
    const svgW = Math.max(160, Math.round(slotW) - 20);
    panel.style.width = svgW + 20 + 'px';

    panel.replaceChildren(head, sub, buildSvg(pts, yDomain, curPt, runColor, svgW));
    panel.style.display = 'block';
    position();
  }

  // 卡片随数据渐进创建，数据到达后再绑定 hover 监听
  function attach(card) {
    const chart = card && card.$.chart;
    if (!chart || chart.__scrBound) return;
    chart.__scrBound = true;
    chart.addEventListener('histogram-hover', (e) => onHover(card, e.detail));
  }

  dashboard.addEventListener('card-data-updated', (e) => attach(e.detail.card));
  dashboard.getActiveCards().forEach(attach);
}
