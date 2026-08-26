// Adaptation (new file, no TensorBoard counterpart): converts exact curve
// points from the xy-curve plugin backend into the VzHistogram shape consumed
// by the (unmodified) vz-histogram-timeseries renderer. Points are preserved
// exactly: each bin's centroid equals the original x, with a uniform width
// equal to the smallest positive x-gap across the series (mirroring the
// uniform-width invariant of histogramCore.intermediateToD3).

export function curvesToVz(curves) {
  if (!curves || !curves.length) return [];
  var minGap = Infinity;
  curves.forEach((datum) => {
    var points = datum.points;
    for (var i = 1; i < points.length; i++) {
      var gap = points[i][0] - points[i - 1][0];
      if (gap > 0 && gap < minGap) {
        minGap = gap;
      }
    }
  });
  if (!isFinite(minGap)) {
    // Degenerate single-point curves: use a nominal width, like the original
    // zero-width-range fallback in histogramCore.intermediateToD3.
    minGap = 1;
  }
  return curves.map((datum) => ({
    wall_time: datum.wall_time,
    step: datum.step,
    bins: datum.points.map((point) => ({
      x: point[0] - minGap / 2,
      dx: minGap,
      y: point[1],
    })),
  }));
}
