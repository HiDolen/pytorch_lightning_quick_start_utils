// Source: tensorboard/plugins/histogram/tf_histogram_dashboard/histogramCore.ts
// Source commit: 012299c95682b6cf89e0eaccb8dc034cf1d61cd2
// Adaptation: TypeScript types stripped; logic copied verbatim. Reusable by
// any plugin that needs the original histogram data pipeline.
import d3 from '../vendor/d3-esm.js';

export function backendToIntermediate(histogram) {
  const [wall_time, step, buckets] = histogram;
  return {
    wall_time,
    step,
    min: d3.min(buckets.map(([left, ,]) => left)),
    max: d3.max(buckets.map(([, right]) => right)),
    buckets: buckets.map(([left, right, count]) => ({left, right, count})),
  };
}

/**
 * Convert histogram data to the standard D3 format to make it more
 * compatible and easier to visualize. When rendering histograms, having
 * access to the left edge and width of each bin makes things quite a
 * bit easier, so we include these in the result. We also convert the
 * bins to have a uniform width, which makes the visualization easier
 * to understand.
 *
 * @param histogram
 * @param min The leftmost edge. The binning will start on it.
 * @param max The rightmost edge. The binning will end on it.
 * @param numBins The number of bins of the converted data. The default
 * of 30 is sensible: if you use more, you start to get artifacts
 * because the event data is stored in buckets, and you start being able
 * to see the aliased borders between each bucket.
 *
 * @return A list of histogram bins. Each bin has an `x` (left
 *     edge), a `dx` (width), and a `y` (count). If the given
 *     right edges are inclusive, then these left edges (`x`) are
 *     exclusive.
 */
export function intermediateToD3(histogram, min, max, numBins = 30) {
  if (min === undefined || max == undefined) {
    min = 0;
    max = 0;
  }
  if (max === min) {
    // If the output range is 0 width, use a default non 0 range for
    // visualization purpose.
    max = min * 1.1 + 1;
    min = min / 1.1 - 1;
  }
  // Terminology note: _buckets_ are the input to this function,
  // while _bins_ are our output.
  const binWidth = (max - min) / numBins;
  let bucketIndex = 0;
  const d3HistogramBins = [];
  for (let i = 0; i < numBins; i++) {
    const binLeft = min + i * binWidth;
    const binRight = binLeft + binWidth;
    // Take the count of each existing bucket, multiply it by the
    // proportion of overlap with the new bin, then sum and store as the
    // count for the new bin. If no overlap, will add to zero; if 100%
    // overlap, will include the full count into new bin.
    let binY = 0;
    while (bucketIndex < histogram.buckets.length) {
      // Clip the right edge because right-most edge can be
      // infinite-sized.
      const bucketRight = Math.min(max, histogram.buckets[bucketIndex].right);
      const bucketLeft = Math.max(min, histogram.buckets[bucketIndex].left);
      const bucketWidth = bucketRight - bucketLeft;
      if (bucketWidth > 0) {
        const intersect =
          Math.min(bucketRight, binRight) - Math.max(bucketLeft, binLeft);
        const count =
          (intersect / (bucketRight - bucketLeft)) *
          histogram.buckets[bucketIndex].count;
        binY += intersect > 0 ? count : 0;
      } else {
        const isFinalBin = binRight >= max;
        const singleValueOverlap =
          binLeft <= bucketLeft &&
          (isFinalBin ? bucketRight <= binRight : bucketRight < binRight);
        binY += singleValueOverlap ? histogram.buckets[bucketIndex].count : 0;
      }
      // If `bucketRight` is bigger than `binRight`, then this bin is
      // finished and there is data for the next bin, so don't increment
      // `bucketIndex`.
      if (bucketRight > binRight) {
        break;
      }
      bucketIndex++;
    }
    d3HistogramBins.push({x: binLeft, dx: binWidth, y: binY});
  }
  return d3HistogramBins;
}

export function backendToVz(histograms) {
  const intermediateHistograms = histograms.map(backendToIntermediate);
  const minmin = d3.min(intermediateHistograms, (h) => h.min);
  const maxmax = d3.max(intermediateHistograms, (h) => h.max);
  return intermediateHistograms.map((h) => ({
    wall_time: h.wall_time,
    step: h.step,
    bins: intermediateToD3(h, minmin, maxmax),
  }));
}
