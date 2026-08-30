// 跨卡 hover 联动：悬停一张卡的某个 step 后，同一折叠分组内的其他卡在
// 相同 x 值、相同 step 处显示各自的悬停标记与 tooltip，便于对比数值。
import {VzHistogramTimeseries} from '../histogram/renderer/vz_histogram_timeseries.js';

export function enableHoverLink(dashboard) {
  // 不能包装 connectedCallback：custom element 生命周期回调在 define() 时
  // 已被捕获，事后改原型无效；改在首帧 _draw 挂监听
  const origDraw = VzHistogramTimeseries.prototype._draw;
  VzHistogramTimeseries.prototype._draw = function (...args) {
    if (!this._hoverLinkBound) {
      this._hoverLinkBound = true;
      this.addEventListener('histogram-hover', (e) => onHover(this, e.detail));
    }
    return origDraw.apply(this, args);
  };

  // 源图表 -> 所在分组的当前渲染卡片列表；找不到（未挂载）返回 null
  function groupCardsOf(chart) {
    for (const view of dashboard._categoryViews.values()) {
      const cards = [...view._items.children].filter(
        (el) => el.tagName === 'TF-HISTOGRAM-CARD'
      );
      if (cards.some((c) => c.$.chart === chart)) return cards;
    }
    return null;
  }

  function onHover(sourceChart, detail) {
    const cards = groupCardsOf(sourceChart);
    if (!cards) return;
    cards.forEach((card) => {
      if (card.$.chart === sourceChart) return;
      if (detail) card.$.chart.setLinkedHover(detail.value, detail.step);
      else card.$.chart.clearLinkedHover();
    });
  }
}
