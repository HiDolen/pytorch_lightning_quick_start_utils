// Adaptation (new file, no TensorBoard counterpart): entry point for the
// EQ Curves plugin. Identical to the XY Curves entry except that data is
// adapted onto a log10 frequency axis (see frequency_adapter.js).
import '../shared/histogram/tf_histogram_dashboard.js';
import {eqCurvesToVz} from './frequency_adapter.js';
import {enableSharedYAxis} from './shared_y_axis.js';

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
  dashboard.toVz = eqCurvesToVz;
  enableSharedYAxis(dashboard);
  dashboard.style.display = 'flex';
  dashboard.style.height = '100%';
  document.body.style.margin = '0';
  document.documentElement.style.height = '100%';
  document.body.style.height = '100%';
  document.body.appendChild(dashboard);
}
