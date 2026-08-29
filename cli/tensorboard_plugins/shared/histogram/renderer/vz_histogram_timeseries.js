// Source: tensorboard/plugins/histogram/vz_histogram_timeseries/vz-histogram-timeseries.ts
// Source commit: 012299c95682b6cf89e0eaccb8dc034cf1d61cd2
// Adaptation: Polymer element ported to a native custom element; the template
// DOM, CSS, and the entire _draw body are copied verbatim (types stripped).
// Polymer property observers are replaced by explicit setters that invoke the
// exact same redraw paths (mode change => 500ms transition, everything else 0).
// d3 is the unmodified UMD bundle re-exported as an ES module.
import d3 from '../vendor/d3-esm.js';

const TEMPLATE = `
    <div id="tooltip"><span></span></div>
    <svg id="svg">
      <g>
        <g class="axis x"></g>
        <g class="axis y"></g>
        <g class="axis y slice"></g>
        <g class="stage">
          <rect class="background"></rect>
        </g>
        <g class="x-axis-hover"></g>
        <g class="y-axis-hover"></g>
        <g class="y-slice-axis-hover"></g>
      </g>
    </svg>
`;

const STYLE = `
      :host {
        color: #aaa;
        display: flex;
        flex-direction: column;
        flex-grow: 1;
        flex-shrink: 1;
        position: relative;
        --vz-histogram-timeseries-hover-bg-color: #fff;
        --vz-histogram-timeseries-outline-color: #fff;
        --vz-histogram-timeseries-hover-outline-color: #000;
      }

      svg {
        font-family: roboto, sans-serif;
        overflow: visible;
        display: block;
        width: 100%;
        flex-grow: 1;
        flex-shrink: 1;
      }

      text {
        fill: currentColor;
      }

      #tooltip {
        position: absolute;
        display: block;
        opacity: 0;
        font-weight: bold;
        font-size: 11px;
      }

      .background {
        fill-opacity: 0;
        fill: red;
      }

      .histogram {
        pointer-events: none;
      }

      .hover {
        font-size: 9px;
        dominant-baseline: middle;
        opacity: 0;
      }

      .hover circle {
        stroke: white;
        stroke-opacity: 0.5;
        stroke-width: 1px;
      }

      .hover text {
        fill: black;
        opacity: 0;
      }

      .hover.hover-closest circle {
        fill: var(--vz-histogram-timeseries-hover-outline-color) !important;
      }

      .hover.hover-closest text {
        opacity: 1;
      }

      .baseline {
        stroke: black;
        stroke-opacity: 0.1;
      }

      .outline {
        fill: none;
        stroke: var(--vz-histogram-timeseries-outline-color);
        stroke-opacity: 0.5;
      }

      .outline.outline-hover {
        stroke: var(--vz-histogram-timeseries-hover-outline-color) !important;
        stroke-opacity: 1;
      }

      .x-axis-hover,
      .y-axis-hover,
      .y-slice-axis-hover {
        pointer-events: none;
      }

      .x-axis-hover .label,
      .y-axis-hover .label,
      .y-slice-axis-hover .label {
        opacity: 0;
        font-weight: bold;
        font-size: 11px;
        text-anchor: end;
      }

      .x-axis-hover text {
        text-anchor: middle;
      }

      .y-axis-hover text,
      .y-slice-axis-hover text {
        text-anchor: start;
      }

      .x-axis-hover line,
      .y-axis-hover line,
      .y-slice-axis-hover line {
        stroke: currentColor;
      }

      .x-axis-hover rect,
      .y-axis-hover rect,
      .y-slice-axis-hover rect {
        fill: var(--vz-histogram-timeseries-hover-bg-color);
      }

      #tooltip,
      .x-axis-hover text,
      .y-axis-hover text,
      .y-slice-axis-hover text {
        color: var(--vz-histogram-timeseries-hover-outline-color);
      }

      .axis {
        font-size: 11px;
      }

      .axis path.domain {
        fill: none;
      }

      .axis .tick line {
        stroke: #ddd;
      }

      .axis.slice {
        opacity: 0;
      }

      .axis.slice .tick line {
        stroke-dasharray: 2;
      }

      .small .axis text {
        display: none;
      }
      .small .axis .tick:first-of-type text {
        display: block;
      }
      .small .axis .tick:last-of-type text {
        display: block;
      }
      .medium .axis text {
        display: none;
      }
      .medium .axis .tick:nth-child(2n + 1) text {
        display: block;
      }
      .large .axis text {
        display: none;
      }
      .large .axis .tick:nth-child(2n + 1) text {
        display: block;
      }
`;

export class VzHistogramTimeseries extends HTMLElement {
  constructor() {
    super();
    this.mode = 'offset';
    this.timeProperty = 'step';
    this.bins = 'bins';
    this.x = 'x';
    this.dx = 'dx';
    this.y = 'y';
    this.colorScale = d3.scaleOrdinal(d3.schemeCategory10);
    this.modeTransitionDuration = 500;
    this._attached = false;
    this._name = null;
    this._data = null;
    const root = this.attachShadow({mode: 'open'});
    root.innerHTML = `<style>${STYLE}</style>${TEMPLATE}`;
    this.$ = {
      tooltip: root.getElementById('tooltip'),
      svg: root.getElementById('svg'),
    };
  }

  connectedCallback() {
    this._attached = true;
    this._redrawOnChange();
  }

  disconnectedCallback() {
    this._attached = false;
  }

  setMode(mode) {
    if (this.mode === mode) return;
    this.mode = mode;
    this._modeRedraw();
  }

  setTimeProperty(timeProperty) {
    if (this.timeProperty === timeProperty) return;
    this.timeProperty = timeProperty;
    this._redrawOnChange();
  }

  setColorScale(colorScale) {
    this.colorScale = colorScale;
    this._redrawOnChange();
  }

  setSeriesData(name, data) {
    this._name = name;
    this._data = data;
    this.redraw();
  }

  _redrawOnChange() {
    this.redraw();
  }

  redraw() {
    this._draw(0);
  }

  _modeRedraw() {
    this._draw(this.modeTransitionDuration);
  }

  _draw(duration) {
    if (!this._attached || !this._data) {
      return;
    }
    //
    // Data verification
    //
    if (duration === undefined)
      throw new Error('vz-histogram-timeseries _draw needs duration');
    if (this._data.length <= 0) throw new Error('Not enough steps in the data');
    if (!this._data[0].hasOwnProperty(this.bins))
      throw new Error("No bins property of '" + this.bins + "' in data");
    if (this._data[0][this.bins].length <= 0)
      throw new Error('Must have at least one bin in bins in data');
    if (!this._data[0][this.bins][0].hasOwnProperty(this.x))
      throw new Error("No x property '" + this.x + "' on bins data");
    if (!this._data[0][this.bins][0].hasOwnProperty(this.dx))
      throw new Error("No dx property '" + this.dx + "' on bins data");
    if (!this._data[0][this.bins][0].hasOwnProperty(this.y))
      throw new Error("No y property '" + this.y + "' on bins data");
    //
    // Initialization
    //
    var timeProp = this.timeProperty;
    var xProp = this.x;
    var binsProp = this.bins;
    var dxProp = this.dx;
    var yProp = this.y;
    var data = this._data;
    var name = this._name;
    var mode = this.mode;
    var color = d3.hcl(this.colorScale(name));
    var tooltip = d3.select(this.$.tooltip);
    var xAccessor = function (d) {
      return d[xProp];
    };
    var yAccessor = function (d) {
      return d[yProp];
    };
    var dxAccessor = function (d) {
      return d[dxProp];
    };
    var xRightAccessor = function (d) {
      return d[xProp] + d[dxProp];
    };
    var timeAccessor = function (d) {
      return d[timeProp];
    };
    if (timeProp === 'relative') {
      timeAccessor = function (d) {
        return d.wall_time - data[0].wall_time;
      };
    }
    var brect = this.$.svg.getBoundingClientRect();
    var outerWidth = brect.width,
      outerHeight = brect.height;
    var sliceHeight,
      margin = {top: 5, right: 60, bottom: 20, left: 24};
    if (mode === 'offset') {
      sliceHeight = outerHeight / 2.5;
      margin.top = sliceHeight + 5;
    } else {
      sliceHeight = outerHeight - margin.top - margin.bottom;
    }
    var width = outerWidth - margin.left - margin.right,
      height = outerHeight - margin.top - margin.bottom;
    var leftMin = d3.min(data, xAccessor),
      rightMax = d3.max(data, xRightAccessor);
    //
    // Text formatters
    //
    var format = d3.format('.3n');
    var yAxisFormat = d3.format('.0f');
    if (timeProp === 'wall_time') {
      yAxisFormat = d3.timeFormat('%m/%d %X');
    } else if (timeProp === 'relative') {
      yAxisFormat = function (d) {
        return d3.format('.1r')(d / 3.6e6) + 'h'; // Convert to hours.
      };
    }
    //
    // Calculate the extents
    //
    var xExtents = data.map(function (d, i) {
      return [
        d3.min(d[binsProp], xAccessor),
        d3.max(d[binsProp], xRightAccessor),
      ];
    });
    var yExtents = data.map(function (d) {
      return d3.extent(d[binsProp], yAccessor);
    });
    //
    // Scales and axis
    //
    var outlineCanvasSize = 500;
    var extent = d3.extent(data, timeAccessor);

    var yScale = (timeProp === 'wall_time' ? d3.scaleTime() : d3.scaleLinear())
      .domain(extent)
      .range([0, mode === 'offset' ? height : 0]);
    // 宿主可经 _sharedYDomain 锁定数值域；缺省则按本卡数据自动。
    var ySliceScale = d3
      .scaleLinear()
      .domain(this._sharedYDomain || [
        0,
        d3.max(data, function (d, i) {
          return yExtents[i][1];
        }),
      ])
      .range([sliceHeight, 0]);
    var yLineScale = d3
      .scaleLinear()
      .domain(ySliceScale.domain())
      .range([outlineCanvasSize, 0]);
    var xScale = d3
      .scaleLinear()
      .domain([
        d3.min(data, function (d, i) {
          return xExtents[i][0];
        }),
        d3.max(data, function (d, i) {
          return xExtents[i][1];
        }),
      ])
      .nice()
      .range([0, width]);
    var xLineScale = d3
      .scaleLinear()
      .domain(xScale.domain())
      .range([0, outlineCanvasSize]);
    const fillColor = d3
      .scaleLinear()
      .domain(d3.extent(data, timeAccessor))
      .range([color.brighter(), color.darker()])
      .interpolate(d3.interpolateHcl);
    var xAxis = d3.axisBottom(xScale).ticks(Math.max(2, width / 20));
    var yAxis = d3
      .axisRight(yScale)
      .ticks(Math.max(2, height / 15))
      .tickFormat(yAxisFormat);
    var ySliceAxis = d3
      .axisRight(ySliceScale)
      .ticks(Math.max(2, height / 15))
      .tickSize(width + 5)
      .tickFormat(format);
    var xBinCentroid = function (d) {
      return d[xProp] + d[dxProp] / 2;
    };
    var linePath = d3
      .line()
      .x(function (d) {
        return xLineScale(xBinCentroid(d));
      })
      .y(function (d) {
        return yLineScale(d[yProp]);
      });
    var path = function (d) {
      // Draw a line from 0 to the first point and from the last point to 0.
      return (
        'M' +
        xLineScale(xBinCentroid(d[0])) +
        ',' +
        yLineScale(0) +
        'L' +
        linePath(d).slice(1) +
        'L' +
        xLineScale(xBinCentroid(d[d.length - 1])) +
        ',' +
        yLineScale(0)
      );
    };
    //
    // Render
    //
    var svgNode = this.$.svg;
    var svg = d3.select(svgNode);
    var svgTransition = svg.transition().duration(duration);
    var g = svg
      .select('g')
      .classed('small', function () {
        return width > 0 && width <= 150;
      })
      .classed('medium', function () {
        return width > 150 && width <= 300;
      })
      .classed('large', function () {
        return width > 300;
      });
    var gTransition = svgTransition
      .select('g')
      .attr('transform', 'translate(' + margin.left + ',' + margin.top + ')');
    var bisect = d3.bisector(xRightAccessor).left;
    var self = this;
    var stage = g
      .select('.stage')
      .on('mouseover', function () {
        showHover();
      })
      .on('mouseout', function () {
        clearHover();
        emitHover(null);
      })
      .on('mousemove', onMouseMove);
    var background = stage
      .select('.background')
      .attr('transform', 'translate(' + -margin.left + ',' + -margin.top + ')')
      .attr('width', outerWidth)
      .attr('height', outerHeight);
    var histogram = stage.selectAll('.histogram').data(data),
      histogramExit = histogram.exit().remove(),
      histogramEnter = histogram.enter().append('g').attr('class', 'histogram'),
      histogramUpdate = histogramEnter.merge(histogram).sort(function (a, b) {
        return timeAccessor(a) - timeAccessor(b);
      }),
      histogramTransition = gTransition
        .selectAll('.histogram')
        .attr('transform', function (d) {
          return (
            'translate(0, ' +
            (mode === 'offset' ? yScale(timeAccessor(d)) - sliceHeight : 0) +
            ')'
          );
        });
    var baselineEnter = histogramEnter.append('line').attr('class', 'baseline'),
      baselineUpdate = histogramTransition
        .select('.baseline')
        .style('stroke-opacity', function (d) {
          return mode === 'offset' ? 0.1 : 0;
        })
        .attr('y1', sliceHeight)
        .attr('y2', sliceHeight)
        .attr('x2', width);
    var outlineEnter = histogramEnter.append('path').attr('class', 'outline'),
      outlineUpdate = histogramUpdate
        .select('.outline')
        .attr('vector-effect', 'non-scaling-stroke')
        .style('stroke-width', 1),
      // d 放 transition 链:数值域变化时曲线随轴一起 morph
      outlineTransition = histogramTransition
        .select('.outline')
        .attr('d', function (d) {
          return path(d[binsProp]);
        })
        .attr(
          'transform',
          'scale(' +
            width / outlineCanvasSize +
            ', ' +
            sliceHeight / outlineCanvasSize +
            ')'
        )
        .style('stroke', function (d) {
          return mode === 'offset' ? '' : fillColor(timeAccessor(d));
        })
        .style('fill-opacity', function (d) {
          return mode === 'offset' ? 1 : 0;
        })
        .style('fill', function (d) {
          return fillColor(timeAccessor(d));
        });
    var hoverEnter = histogramEnter.append('g').attr('class', 'hover');
    var hoverUpdate = histogramUpdate
      .select('.hover')
      .style('fill', function (d) {
        return fillColor(timeAccessor(d));
      });
    hoverEnter.append('circle').attr('r', 2);
    hoverEnter.append('text').style('display', 'none').attr('dx', 4);
    var xAxisHover = g.select('.x-axis-hover').selectAll('.label').data(['x']),
      xAxisHoverEnter = xAxisHover.enter().append('g').attr('class', 'label'),
      xAxisHoverUpdate = xAxisHover.merge(xAxisHoverEnter);
    xAxisHoverEnter
      .append('rect')
      .attr('x', -20)
      .attr('y', 6)
      .attr('width', 40)
      .attr('height', 14);
    xAxisHoverEnter
      .append('line')
      .attr('x1', 0)
      .attr('x2', 0)
      .attr('y1', 0)
      .attr('y2', 6);
    xAxisHoverEnter.append('text').attr('dy', 18);
    var yAxisHover = g.select('.y-axis-hover').selectAll('.label').data(['y']),
      yAxisHoverEnter = yAxisHover.enter().append('g').attr('class', 'label'),
      yAxisHoverUpdate = yAxisHover.merge(yAxisHoverEnter);
    yAxisHoverEnter
      .append('rect')
      .attr('x', 8)
      .attr('y', -6)
      .attr('width', 40)
      .attr('height', 14);
    yAxisHoverEnter
      .append('line')
      .attr('x1', 0)
      .attr('x2', 6)
      .attr('y1', 0)
      .attr('y2', 0);
    yAxisHoverEnter.append('text').attr('dx', 8).attr('dy', 4);
    var ySliceAxisHover = g
        .select('.y-slice-axis-hover')
        .selectAll('.label')
        .data(['y']),
      ySliceAxisHoverEnter = ySliceAxisHover
        .enter()
        .append('g')
        .attr('class', 'label'),
      ySliceAxisHoverUpdate = ySliceAxisHover.merge(ySliceAxisHoverEnter);
    ySliceAxisHoverEnter
      .append('rect')
      .attr('x', 8)
      .attr('y', -6)
      .attr('width', 40)
      .attr('height', 14);
    ySliceAxisHoverEnter
      .append('line')
      .attr('x1', 0)
      .attr('x2', 6)
      .attr('y1', 0)
      .attr('y2', 0);
    ySliceAxisHoverEnter.append('text').attr('dx', 8).attr('dy', 4);
    gTransition
      .select('.y.axis.slice')
      .style('opacity', mode === 'offset' ? 0 : 1)
      .attr(
        'transform',
        'translate(0, ' + (mode === 'offset' ? -sliceHeight : 0) + ')'
      )
      .call(ySliceAxis);
    gTransition
      .select('.x.axis')
      .attr('transform', 'translate(0, ' + height + ')')
      .call(xAxis);
    gTransition
      .select('.y.axis')
      .style('opacity', mode === 'offset' ? 1 : 0)
      .attr(
        'transform',
        'translate(' + width + ', ' + (mode === 'offset' ? 0 : height) + ')'
      )
      .call(yAxis);
    gTransition.selectAll('.tick text').attr('fill', '#aaa');
    gTransition.selectAll('.axis path.domain').attr('stroke', 'none');
    function showHover() {
      hoverUpdate.style('opacity', 1);
      xAxisHoverUpdate.style('opacity', 1);
      yAxisHoverUpdate.style('opacity', 1);
      ySliceAxisHoverUpdate.style('opacity', 1);
      tooltip.style('opacity', 1);
    }
    function clearHover() {
      hoverUpdate.style('opacity', 0);
      xAxisHoverUpdate.style('opacity', 0);
      yAxisHoverUpdate.style('opacity', 0);
      ySliceAxisHoverUpdate.style('opacity', 0);
      hoverUpdate.classed('hover-closest', false);
      outlineUpdate.classed('outline-hover', false);
      tooltip.style('opacity', 0);
    }
    // 联动 hook：宿主监听此事件同步其他图表（detail: {value, step} | null）
    function emitHover(detail) {
      self.dispatchEvent(new CustomEvent('histogram-hover', {detail: detail}));
    }
    function onMouseMove() {
      var m = d3.mouse(this),
        v = xScale.invert(m[0]);
      var step = updateHover(v, null, m);
      emitHover(step == null ? null : {value: v, step: step});
    }
    // 悬停标记全量更新。m 为 stage 局部鼠标坐标：有鼠标时按鼠标 y 判定最近
    // 切片、tooltip 跟随鼠标；联动模式（m 为 null）由 forcedStep 指定目标
    // step、tooltip 落在最近切片的标记点上。返回最近切片的 step。
    function updateHover(v, forcedStep, m) {
      showHover();
      function hoverXIndex(d) {
        return Math.min(d[binsProp].length - 1, bisect(d[binsProp], v));
      }
      var closestSliceData;
      var closestSliceDistance = Infinity;
      var lastSliceData;
      hoverUpdate.attr('transform', function (d, i) {
        var index = hoverXIndex(d);
        lastSliceData = d;
        var x = xScale(
          d[binsProp][index][xProp] + d[binsProp][index][dxProp] / 2
        );
        var y = ySliceScale(d[binsProp][index][yProp]);
        var globalY =
          mode === 'offset' ? yScale(timeAccessor(d)) - (sliceHeight - y) : y;
        var dist = m
          ? Math.abs(m[1] - globalY)
          : Math.abs(timeAccessor(d) - forcedStep);
        if (dist < closestSliceDistance) {
          closestSliceDistance = dist;
          closestSliceData = d;
        }
        return 'translate(' + x + ',' + y + ')';
      });
      hoverUpdate.select('text').text(function (d) {
        var index = hoverXIndex(d);
        return d[binsProp][index][yProp];
      });
      hoverUpdate.classed('hover-closest', function (d) {
        return d === closestSliceData;
      });
      outlineUpdate.classed('outline-hover', function (d) {
        return d === closestSliceData;
      });
      var index = hoverXIndex(lastSliceData);
      xAxisHoverUpdate
        .attr('transform', function (d) {
          return (
            'translate(' +
            xScale(
              lastSliceData[binsProp][index][xProp] +
                lastSliceData[binsProp][index][dxProp] / 2
            ) +
            ', ' +
            height +
            ')'
          );
        })
        .select('text')
        .text(function (d) {
          return format(
            lastSliceData[binsProp][index][xProp] +
              lastSliceData[binsProp][index][dxProp] / 2
          );
        });
      var fy = yAxis.tickFormat();
      yAxisHoverUpdate
        .attr('transform', function (d) {
          return (
            'translate(' +
            width +
            ', ' +
            (mode === 'offset' ? yScale(timeAccessor(closestSliceData)) : 0) +
            ')'
          );
        })
        .style('display', mode === 'offset' ? '' : 'none')
        .select('text')
        .text(function (d) {
          return fy(timeAccessor(closestSliceData));
        });
      var fsy = ySliceAxis.tickFormat();
      ySliceAxisHoverUpdate
        .attr('transform', function (d) {
          return (
            'translate(' +
            width +
            ', ' +
            (mode === 'offset'
              ? 0
              : ySliceScale(closestSliceData[binsProp][index][yProp])) +
            ')'
          );
        })
        .style('display', mode === 'offset' ? 'none' : '')
        .select('text')
        .text(function (d) {
          return fsy(closestSliceData[binsProp][index][yProp]);
        });
      var svgMouse = m
        ? d3.mouse(svgNode)
        : (function () {
            // 联动模式：tooltip 落在最近切片标记点（stage 坐标 -> svg 坐标）
            var idx = hoverXIndex(closestSliceData);
            var cx = xScale(
              closestSliceData[binsProp][idx][xProp] +
                closestSliceData[binsProp][idx][dxProp] / 2
            );
            var cy =
              mode === 'offset'
                ? yScale(timeAccessor(closestSliceData)) -
                  (sliceHeight -
                    ySliceScale(closestSliceData[binsProp][idx][yProp]))
                : ySliceScale(closestSliceData[binsProp][idx][yProp]);
            return [cx + margin.left, cy + margin.top];
          })();
      tooltip
        .style(
          'transform',
          'translate(' + (svgMouse[0] + 15) + 'px,' + (svgMouse[1] - 15) + 'px)'
        )
        .select('span')
        .text(
          mode === 'offset'
            ? fsy(closestSliceData[binsProp][index][yProp])
            : (timeProp === 'step' ? 'step ' : '') +
                fy(timeAccessor(closestSliceData))
        );
      return closestSliceData ? timeAccessor(closestSliceData) : null;
    }
    // 供 setLinkedHover/clearLinkedHover 复用（_draw 重建后自动更新）
    this._hoverUpdate = updateHover;
    this._hoverClear = clearHover;
  }

  // 联动驱动：在其他图表 hover 时，在本图相同 x 值、相同 step 处显示标记
  setLinkedHover(value, step) {
    if (this._hoverUpdate) this._hoverUpdate(value, step, null);
  }

  clearLinkedHover() {
    if (this._hoverClear) this._hoverClear();
  }
}

customElements.define('vz-histogram-timeseries', VzHistogramTimeseries);
