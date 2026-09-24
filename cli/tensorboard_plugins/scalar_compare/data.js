export function normalizeScalars(events) {
  if (!Array.isArray(events)) throw new Error('Invalid scalar response');
  const byStep = new Map();
  for (const event of events) {
    if (!Array.isArray(event) || event.length < 3) continue;
    const [wallTime, step, rawValue] = event;
    if (!Number.isFinite(wallTime) || !Number.isFinite(step)) continue;
    const previous = byStep.get(step);
    if (previous && previous.wallTime > wallTime) continue;
    byStep.set(step, {
      wallTime,
      step,
      value: Number.isFinite(rawValue) ? rawValue : null,
    });
  }
  return [...byStep.values()].sort((left, right) => left.step - right.step);
}

export function smoothScalars(points, weight) {
  const smoothing = Math.min(0.99, Math.max(0, Number(weight) || 0));
  let accumulated = 0;
  let mass = 0;
  return points.map(point => {
    if (point.value === null) return {...point, smoothed: null};
    accumulated = smoothing * accumulated + (1 - smoothing) * point.value;
    mass = smoothing * mass + 1 - smoothing;
    return {...point, smoothed: accumulated / mass};
  });
}

export function selectTags(available, selected) {
  if (selected === null) return available.slice(0, 2);
  const availableSet = new Set(available);
  return selected.filter(tag => availableSet.has(tag));
}