// Adaptation (new file, no TensorBoard counterpart): entry point for the
// XY Curves plugin. It mounts the unmodified histogram dashboard components and
// wires them to the plugin backend; the only data transform is
// curvesToVz (exact curve points -> VzHistogram bins shape).
import '../shared/histogram/tf_histogram_dashboard.js';
import {curvesToVz} from '../shared/histogram/data/exact_curve_adapter.js';

export function render() {
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
  dashboard.style.display = 'flex';
  dashboard.style.height = '100%';
  document.body.style.margin = '0';
  document.documentElement.style.height = '100%';
  document.body.style.height = '100%';
  document.body.appendChild(dashboard);
}
