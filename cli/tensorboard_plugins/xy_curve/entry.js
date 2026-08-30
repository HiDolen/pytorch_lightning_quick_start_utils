// Adaptation (new file, no TensorBoard counterpart): entry point for the
// XY Curves plugin. It mounts the unmodified histogram dashboard components and
// wires them to the plugin backend; the only data transform is
// curvesToVz (exact curve points -> VzHistogram bins shape).
import '../shared/histogram/tf_histogram_dashboard.js';
import {curvesToVz} from '../shared/histogram/data/exact_curve_adapter.js';
import {installStepRangeSlider} from '../shared/curve_dashboard/step_range_slider.js';
import {enableHoverReadout} from '../shared/curve_dashboard/hover_readout.js';
import {enableStepCurveReadout} from '../shared/curve_dashboard/step_curve_readout.js';
import {enableOffsetFillFade} from '../shared/curve_dashboard/offset_fill_fade.js';
import {enableHoverLink} from '../shared/curve_dashboard/hover_link.js';

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
  dashboard.toVz = curvesToVz;
  enableHoverLink(dashboard);
  installStepRangeSlider(dashboard);
  enableHoverReadout(dashboard);
  enableStepCurveReadout(dashboard);
  listenForReload(() => dashboard.reload());
  dashboard.style.display = 'flex';
  dashboard.style.height = '100%';
  document.body.style.margin = '0';
  document.documentElement.style.height = '100%';
  document.body.style.height = '100%';
  document.body.appendChild(dashboard);
}
