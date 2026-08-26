// Source: tensorboard/components/tf_color_scale/colorScale.ts,
//         tensorboard/components/tf_color_scale/palettes.ts
// Source commit: 012299c95682b6cf89e0eaccb8dc034cf1d61cd2
// Adaptation: The store-backed runsColorScale is exposed as a factory over an
// explicit run list; palette and mapping logic copied verbatim.
import d3 from './vendor/d3-esm.js';

export function createRunsColorScale(runs) {
  const colorScale = new ColorScale();
  colorScale.setDomain(runs);
  return (runName) => colorScale.getColor(runName);
}

// Default color function (same default as the original renderer property).
export function defaultColorScale(name) {
  return d3.scaleOrdinal(d3.schemeCategory10)(name);
}

const tensorboardColorBlindAssist = [
  '#ff7043', // orange
  '#0077bb', // blue
  '#cc3311', // red
  '#33bbee', // cyan
  '#ee3377', // magenta
  '#009988', // teal
  '#bbbbbb', // grey
];

const standard = tensorboardColorBlindAssist;

export class ColorScale {
  constructor(palette = standard) {
    this.palette = palette;
    this.identifiers = d3.map();
  }
  setDomain(strings) {
    this.identifiers = d3.map();
    strings.forEach((s, i) => {
      this.identifiers.set(s, this.palette[i % this.palette.length]);
    });
    return this;
  }
  getColor(s) {
    if (!this.identifiers.has(s)) {
      throw new Error(`String ${s} was not in the domain.`);
    }
    return this.identifiers.get(s);
  }
}
