// Adaptation (new file, no TensorBoard counterpart): logarithmic frequency
// axis helpers for the EQ Curves plugin. Frequencies in Hz are mapped onto a
// log10 axis (the audio-engineering standard: every decade has equal width,
// so 20Hz..1kHz spans roughly half of a 20Hz..20kHz axis) before being handed
// to the shared (unmodified) histogram components.
//
// The x axis of the renderer is the frequency axis in both overlay and offset
// mode. Since this plugin runs in its own iframe with its own module realm,
// we relabel it by wrapping d3.axisBottom here — the shared lib stays pristine.
import d3 from '../shared/histogram/vendor/d3-esm.js';
import {VzHistogramTimeseries} from '../shared/histogram/renderer/vz_histogram_timeseries.js';
import {curvesToVz} from '../shared/histogram/data/exact_curve_adapter.js';

export function hzToLog(hz) {
  return Math.log10(hz);
}

export function logToHz(logHz) {
  return Math.pow(10, logHz);
}

// Round to a friendly audio-style label: "63" / "250" / "1k" / "2.5k" / "16k".
export function formatHz(hz) {
  if (hz < 1) return '0';
  if (hz < 1000) return String(Math.round(hz / 10) * 10);
  var kHz = hz / 1000;
  return (kHz >= 10 ? Math.round(kHz) : Math.round(kHz * 10) / 10) + 'k';
}

// Musical tick positions: the 1-2-5 series per decade (20, 50, 100, 200, 500,
// 1k, 2k, 5k, 10k, 20k...), equally spaced on a log axis. Falls back to
// sparser 1-5 / 1-only series when the axis is too narrow to fit them.
export function musicTickValues(logLo, logHi) {
  var series = [
    [1, 2, 5],
    [1, 5],
    [1],
  ];
  var values = [];
  for (var s = 0; s < series.length; s++) {
    values = [];
    for (var d = Math.floor(logLo); d <= Math.ceil(logHi); d++) {
      for (var i = 0; i < series[s].length; i++) {
        var v = d + Math.log10(series[s][i]);
        if (v >= logLo - 1e-9 && v <= logHi + 1e-9) values.push(v);
      }
    }
    if (values.length <= 14) break;
  }
  return values;
}

const origAxisBottom = d3.axisBottom;
d3.axisBottom = function (scale) {
  var axis = origAxisBottom(scale);
  var domain = scale.domain();
  if (domain[0] > 0) {
    // Positive domain == log-frequency axis (the only axisBottom in this
    // plugin's realm); relabel ticks with musical 1-2-5 frequencies.
    axis
      .tickValues(musicTickValues(domain[0], domain[1]))
      .tickFormat(function (logHz) {
        return formatHz(logToHz(logHz));
      });
  }
  return axis;
};

// shared 渲染器会对 x 轴 domain 调 `.nice()`，
// 而它作用在裸 log10 数值上会被取整，对音乐频率轴毫无意义。
// 正 domain（即 log 频率轴，同 axisBottom 补丁的
// 判定）保留数据的精确范围。
const origScaleLinear = d3.scaleLinear;
d3.scaleLinear = function () {
  const scale = origScaleLinear();
  const origNice = scale.nice;
  scale.nice = function () {
    if (scale.domain()[0] > 0) return scale;
    return origNice.apply(scale, arguments);
  };
  return scale;
};

// domain 改为数据精确范围后，首尾刻度即实测频率范围，但 shared CSS 会
// 隔一个隐藏刻度标签，可能吞掉最左边的 "20"。参照渲染器自己的 `.small`
// 规则让首尾标签始终显示，且只作用于频率轴（.axis.x）。
const origRedraw = VzHistogramTimeseries.prototype.redraw;
VzHistogramTimeseries.prototype.redraw = function () {
  const result = origRedraw.apply(this, arguments);
  const root = this.shadowRoot;
  if (root && !root.getElementById('eq-endpoint-ticks')) {
    const style = document.createElement('style');
    style.id = 'eq-endpoint-ticks';
    style.textContent =
      '.medium .axis.x .tick:first-of-type text,' +
      '.medium .axis.x .tick:last-of-type text,' +
      '.large .axis.x .tick:first-of-type text,' +
      '.large .axis.x .tick:last-of-type text{display: block;}';
    root.appendChild(style);
  }
  return result;
};

// The renderer writes x-axis hover labels with its own stock numeric format,
// which would show the raw log10 value ("1.30" instead of "20"). Its mousemove
// listener is bound earlier, so this document-level listener runs after it and
// rewrites the label in Hz.
document.addEventListener('mousemove', function (event) {
  var path = event.composedPath();
  for (var i = 0; i < path.length; i++) {
    var node = path[i];
    if (node && node.tagName === 'VZ-HISTOGRAM-TIMESERIES') {
      var label = node.shadowRoot.querySelector('.x-axis-hover text');
      if (label) {
        var logHz = parseFloat(label.textContent);
        if (!isNaN(logHz)) {
          label.textContent = formatHz(logToHz(logHz));
        }
      }
      return;
    }
  }
});

// Backend data ([[hz, gain], ...] per step) -> VzHistogram shape, with the
// x coordinates transformed into log10(Hz) space (non-positive Hz dropped).
export function eqCurvesToVz(curves) {
  if (!curves || !curves.length) return [];
  var logCurves = curves.map(function (datum) {
    return {
      wall_time: datum.wall_time,
      step: datum.step,
      points: datum.points
        .filter(function (point) {
          return point[0] > 0;
        })
        .map(function (point) {
          return [hzToLog(point[0]), point[1]];
        }),
    };
  });
  return curvesToVz(logCurves);
}
