// 单 step 模式下的聚合读数面板：悬停任意卡片时，把所有卡片在当前
// 悬停 x 值、单 step 处的数值汇总到一个跟随鼠标的浮层。
// 仅单 step 模式激活。
import d3 from './histogram/vendor/d3-esm.js';
import {pickTextColor} from './histogram/tf_card_heading.js';
import {isSingleStepMode} from './step_range_slider.js';

const PANEL_STYLE = `
  .eq-hro-panel {
    position: fixed; z-index: 10;
    background: #fff; border: 1px solid #e0e0e0; border-radius: 4px;
    box-shadow: 0 2px 8px rgba(0,0,0,.18);
    padding: 6px 12px; font-size: 12px; line-height: 1.7;
    pointer-events: none; display: none; min-width: 240px;
    font-family: Roboto, "Helvetica Neue", sans-serif;
  }
  .eq-hro-panel .head {
    font-weight: 500; color: #212121; margin-bottom: 2px;
    border-bottom: 1px solid #eee; padding-bottom: 2px;
    font-variant-numeric: tabular-nums;
  }
  .eq-hro-panel .row { display: flex; align-items: center; gap: 6px; }
  .eq-hro-panel .row.cur { background: #e3f2fd; border-radius: 3px; }
  .eq-hro-panel .name {
    color: #616161; white-space: nowrap;
    overflow: hidden; text-overflow: ellipsis; max-width: 200px;
  }
  /* run 徽章与卡片标题（tf-card-heading）一样的配色机制。
     外层 .run-slot 固定等宽保证各行列对齐，徽章染色自适应文字、长度过长截断 */
  .eq-hro-panel .run-slot {
    flex: none; overflow: hidden; box-sizing: border-box;
  }
  .eq-hro-panel .run-badge {
    display: inline-block; font-size: 11px; font-weight: bold;
    border-radius: 3px; padding: 1px 4px 2px; white-space: nowrap;
    max-width: 100%; box-sizing: border-box;
    overflow: hidden; text-overflow: ellipsis;
  }
  .eq-hro-panel .spacer { flex: 1; }
  .eq-hro-panel .val {
    min-width: 64px; text-align: left;
    font-variant-numeric: tabular-nums; color: #212121; white-space: nowrap;
  }
  .eq-hro-panel .delta {
    position: relative; min-width: 72px; text-align: left; color: #757575;
    font-variant-numeric: tabular-nums; white-space: nowrap;
  }
  /* 差值幅度条：长度 ∝ |差值|/最大|差值|，颜色按正负 */
  .eq-hro-panel .delta-bar {
    position: absolute; left: 0; top: 3px; bottom: 3px; border-radius: 2px;
  }
  .eq-hro-panel .delta-text { position: relative; }
`;

const fmt = d3.format('.4g');

// 与渲染器 hoverXIndex 相同的 bin 定位：按 x+dx 二分
function binAt(chart, v, step) {
  const data = chart._data || [];
  const tp = chart.timeProperty;
  const d = data.find((d) => d[tp] === step);
  if (!d) return null;
  const bins = d[chart.bins];
  const bisect = d3.bisector((b) => b[chart.x] + b[chart.dx]).left;
  const index = Math.min(bins.length - 1, bisect(bins, v));
  return bins[index];
}

function valueAt(chart, v, step) {
  const bin = binAt(chart, v, step);
  return bin ? bin[chart.y] : null;
}

export function enableHoverReadout(dashboard, formatX) {
  const fmtX = formatX || ((x) => 'x ' + fmt(x));
  const root = dashboard._root;
  const style = document.createElement('style');
  style.textContent = PANEL_STYLE;
  root.appendChild(style);
  const panel = document.createElement('div');
  panel.className = 'eq-hro-panel';
  root.appendChild(panel);

  let mouse = {x: 0, y: 0};
  document.addEventListener(
    'mousemove',
    (e) => {
      mouse.x = e.clientX;
      mouse.y = e.clientY;
      if (panel.style.display === 'block') position();
    },
    {passive: true}
  );

  function position() {
    const w = panel.offsetWidth || 260;
    const h = panel.offsetHeight || 80;
    const gap = 18;
    // 面板右移时，空间不足则贴住右边但不溢出，此后右移不再改变 x
    const left =
      mouse.x + gap + w > window.innerWidth
        ? window.innerWidth - w - 8
        : mouse.x + gap;
    panel.style.left = left + 'px';
    panel.style.top = Math.min(mouse.y + gap, window.innerHeight - h - 8) + 'px';
  }

  function buildRow(tagLabel, run, runColor, valueText, deltaText, isCur, diff, ratio) {
    const row = document.createElement('div');
    row.className = 'row' + (isCur ? ' cur' : '');
    if (run) {
      const slot = document.createElement('span');
      slot.className = 'run-slot';
      const badge = document.createElement('span');
      badge.className = 'run-badge';
      badge.textContent = run;
      badge.title = run;
      badge.style.background = runColor;
      badge.style.color = pickTextColor(runColor);
      slot.appendChild(badge);
      row.appendChild(slot);
    }
    const name = document.createElement('span');
    name.className = 'name';
    name.textContent = tagLabel;
    name.title = run ? run + ' / ' + tagLabel : tagLabel;
    row.appendChild(name);
    const spacer = document.createElement('span');
    spacer.className = 'spacer';
    const val = document.createElement('span');
    val.className = 'val';
    val.textContent = valueText;
    const delta = document.createElement('span');
    delta.className = 'delta';
    if (deltaText) {
      const rgb = diff < 0 ? '198,40,40' : '46,125,50';
      const bar = document.createElement('span');
      bar.className = 'delta-bar';
      bar.style.width = Math.round(ratio * 64) + 'px';
      bar.style.background = 'rgba(' + rgb + ',0.22)';
      const text = document.createElement('span');
      text.className = 'delta-text';
      text.textContent = deltaText;
      // 幅度越大的差值，颜色越鲜明，文字也越粗
      text.style.color = 'rgba(' + rgb + ',' + (0.55 + 0.45 * ratio).toFixed(2) + ')';
      if (ratio > 0.66) text.style.fontWeight = '500';
      delta.appendChild(bar);
      delta.appendChild(text);
    }
    row.appendChild(spacer);
    row.appendChild(val);
    row.appendChild(delta);
    return row;
  }

  function onHover(sourceCard, detail) {
    if (!detail || !isSingleStepMode()) {
      panel.style.display = 'none';
      return;
    }
    
    function tagCategory(tag) {
      const i = tag.indexOf('/');
      return i >= 0 ? tag.slice(0, i) : tag;
    }
    // 只对比与源卡片同分类（同 tag 前缀）下的卡片
    const curCategory = tagCategory(sourceCard._tag || '');
    const cards = [...dashboard._cards].filter(
      (card) => tagCategory(card._tag || '') === curCategory
    );

    const srcChart = sourceCard.$.chart;
    const curVal = valueAt(srcChart, detail.value, detail.step);
    // 头部显示吸附到源卡实际 bin 中心的 x 值（由插件入口决定展示格式）
    const curBin = binAt(srcChart, detail.value, detail.step);
    const head = document.createElement('div');
    head.className = 'head';
    head.textContent = curBin
      ? fmtX(curBin[srcChart.x] + curBin[srcChart.dx] / 2)
      : fmtX(detail.value);
    panel.replaceChildren(head);
    // 先收集各行差值，求最大幅度后再按比例渲染
    const rowsData = cards.map((card) => {
      const value = valueAt(card.$.chart, detail.value, detail.step);
      const isCur = card === sourceCard;
      const diff =
        !isCur && value != null && curVal != null ? value - curVal : null;
      return {card, value, isCur, diff};
    });
    const maxAbs = Math.max(
      0,
      ...rowsData.map((r) => (r.diff == null ? 0 : Math.abs(r.diff)))
    );
    rowsData.forEach(({card, value, isCur, diff}) => {
      const tagLabel =
        (card._heading && card._heading.displayName) || card._tag || '';
      const runColor = card._colorScaleFunction(card._run);
      const delta = diff == null ? '' : (diff > 0 ? '+' : '') + fmt(diff);
      const ratio =
        diff == null || maxAbs === 0 ? 0 : Math.abs(diff) / maxAbs;
      panel.appendChild(
        buildRow(
          tagLabel,
          card._run,
          runColor,
          value == null ? '—' : fmt(value),
          delta,
          isCur,
          diff,
          ratio
        )
      );
    });
    panel.style.display = 'block';
    // 统一第一列宽度，保证第二列对齐
    const badges = [...panel.querySelectorAll('.run-badge')];
    if (badges.length) {
      const slotW = Math.min(220, Math.max(...badges.map((b) => b.offsetWidth)));
      panel
        .querySelectorAll('.run-slot')
        .forEach((s) => (s.style.width = slotW + 'px'));
    }
    position();
  }

  // 卡片随数据渐进创建，数据到达后再绑定 hover 监听
  const origProvider = dashboard.dataProvider;
  dashboard.dataProvider = (run, tag) =>
    origProvider(run, tag).then((d) => {
      setTimeout(attach, 0);
      return d;
    });

  function attach() {
    dashboard._cards.forEach((card) => {
      const chart = card.$.chart;
      if (!chart || chart.__hroBound) return;
      chart.__hroBound = true;
      chart.addEventListener('histogram-hover', (e) => onHover(card, e.detail));
    });
  }
}
