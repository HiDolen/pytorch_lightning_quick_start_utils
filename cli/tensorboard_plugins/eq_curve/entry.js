// Adaptation (new file, no TensorBoard counterpart): entry point for the
// EQ Curves plugin. Identical to the XY Curves entry except that data is
// adapted onto a log10 frequency axis (see frequency_adapter.js).
import '../shared/histogram/tf_histogram_dashboard.js';
import {eqCurvesToVz, logToHz} from './frequency_adapter.js';
import {enableSharedYAxis} from './shared_y_axis.js';
import {installStepRangeSlider} from '../shared/step_range_slider.js';
import {enableHoverReadout} from '../shared/hover_readout.js';
import {enableOffsetFillFade} from '../shared/offset_fill_fade.js';
import {enableHoverLink} from '../shared/hover_link.js';

// 接入宿主的刷新广播（experimental IPC），保证 TensorBoard 刷新日志能立即更新
function listenForReload(onReload) {
  const channel = new MessageChannel();
  channel.port1.onmessage = (event) => {
    let msg = null;
    try {
      msg = JSON.parse(event.data);
    } catch (e) {
      return;
    }
    if (!msg || msg.isReply) return;
    // 宿主按 sendMessage 语义等待回复，必须回一条 reply
    channel.port1.postMessage(
      JSON.stringify({type: msg.type, id: msg.id, payload: null, error: null, isReply: true})
    );
    if (msg.type === 'experimental.DataReloaded') onReload();
  };
  window.parent.postMessage('experimental.bootstrap', '*', [channel.port2]);
}

export function render() {
  enableOffsetFillFade();
  var dashboard = document.createElement('tf-histogram-dashboard');
  dashboard.tagsProvider = async () => {
    const response = await fetch('tags');
    return await response.json();
  };
  dashboard.dataProvider = async (run, tag) => {
    const response = await fetch(
      'data?run=' + encodeURIComponent(run) + '&tag=' + encodeURIComponent(tag)
    );
    return await response.json();
  };
  dashboard.toVz = eqCurvesToVz;
  enableSharedYAxis(dashboard);
  enableHoverLink(dashboard);
  installStepRangeSlider(dashboard);
  // 读数面板头部显示 bin 的真实频率
  enableHoverReadout(dashboard, (logHz) => {
    const hz = logToHz(logHz);
    return (hz >= 1000 ? Math.round(hz) : Math.round(hz * 10) / 10) + ' Hz';
  });
  listenForReload(() => dashboard.reload());
    // 重新拉取数据，在卡片上更新数据
  dashboard._reloadHistograms = () => {
    dashboard._cards.forEach((card) => {
      dashboard.dataProvider(card._run, card._tag).then((data) => {
        card.setSeriesData(card._run, dashboard.toVz(data));
      });
    });
  };
  dashboard.style.display = 'flex';
  dashboard.style.height = '100%';
  document.body.style.margin = '0';
  document.documentElement.style.height = '100%';
  document.body.style.height = '100%';
  document.body.appendChild(dashboard);
}
