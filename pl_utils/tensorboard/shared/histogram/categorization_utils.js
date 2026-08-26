// Source: tensorboard/components/vz_sorting/sorting.ts,
//         tensorboard/components/tf_backend/backend.ts,
//         tensorboard/components/tf_categorization_utils/categorizationUtils.ts
// Source commit: 012299c95682b6cf89e0eaccb8dc034cf1d61cd2
// Adaptation: TypeScript types stripped; logic copied verbatim. lodash union /
// flatten / pick are inlined so the module stays dependency-free.

export function compareTagNames(a, b) {
  let ai = 0;
  let bi = 0;
  while (true) {
    if (ai === a.length) {
      return bi === b.length ? 0 : -1;
    }
    if (bi === b.length) {
      return 1;
    }
    if (isDigit(a[ai]) && isDigit(b[bi])) {
      const ais = ai;
      const bis = bi;
      ai = consumeNumber(a, ai + 1);
      bi = consumeNumber(b, bi + 1);
      const an = parseFloat(a.slice(ais, ai));
      const bn = parseFloat(b.slice(bis, bi));
      if (an < bn) {
        return -1;
      }
      if (an > bn) {
        return 1;
      }
      continue;
    }
    if (isBreak(a[ai])) {
      if (!isBreak(b[bi])) {
        return -1;
      }
    } else if (isBreak(b[bi])) {
      return 1;
    } else if (a[ai] < b[bi]) {
      return -1;
    } else if (a[ai] > b[bi]) {
      return 1;
    }
    ai++;
    bi++;
  }
}

function consumeNumber(s, i) {
  const State = {
    NATURAL: 0,
    REAL: 1,
    EXPONENT_SIGN: 2,
    EXPONENT: 3,
  };
  let state = State.NATURAL;
  for (; i < s.length; i++) {
    if (state === State.NATURAL) {
      if (s[i] === '.') {
        state = State.REAL;
      } else if (s[i] === 'e' || s[i] === 'E') {
        state = State.EXPONENT_SIGN;
      } else if (!isDigit(s[i])) {
        break;
      }
    } else if (state === State.REAL) {
      if (s[i] === 'e' || s[i] === 'E') {
        state = State.EXPONENT_SIGN;
      } else if (!isDigit(s[i])) {
        break;
      }
    } else if (state === State.EXPONENT_SIGN) {
      if (isDigit(s[i]) || s[i] === '+' || s[i] === '-') {
        state = State.EXPONENT;
      } else {
        break;
      }
    } else if (state === State.EXPONENT) {
      if (!isDigit(s[i])) {
        break;
      }
    }
  }
  return i;
}

function isDigit(c) {
  return '0' <= c && c <= '9';
}

function isBreak(c) {
  // TODO(@jart): Remove underscore when people stop using it like a slash.
  return c === '/' || c === '_' || isDigit(c);
}

export function getTags(r) {
  const tags = new Set();
  Object.values(r).forEach((tagList) => tagList.forEach((t) => tags.add(t)));
  return Array.from(tags).sort(compareTagNames);
}

export const CategoryType = {
  SEARCH_RESULTS: 0,
  PREFIX_GROUP: 1,
};

/**
 * Compute a category containing the search results for the given query.
 */
export function categorizeBySearchQuery(xs, query) {
  const re = (() => {
    try {
      return new RegExp(query);
    } catch (e) {
      return null;
    }
  })();
  return {
    name: query,
    metadata: {
      type: CategoryType.SEARCH_RESULTS,
      validRegex: !!re,
      universalRegex: query === '.*',
    },
    items: re ? xs.filter((x) => x.match(re)) : [],
  };
}

/**
 * Compute the quotient set $X/{\sim}$, where $a \sim b$ if $a$ and $b$
 * share a common `separator`-prefix. Order is preserved.
 */
export function categorizeByPrefix(xs, separator = '/') {
  const categories = [];
  const categoriesByName = {};
  xs.forEach((x) => {
    const index = x.indexOf(separator);
    const name = index >= 0 ? x.slice(0, index) : x;
    if (!categoriesByName[name]) {
      const category = {
        name,
        metadata: {type: CategoryType.PREFIX_GROUP},
        items: [],
      };
      categoriesByName[name] = category;
      categories.push(category);
    }
    categoriesByName[name].items.push(x);
  });
  return categories;
}

/*
 * Compute the standard categorization of the given input, including
 * both search categories and prefix categories.
 */
export function categorize(xs, query = '') {
  const byFilter = [categorizeBySearchQuery(xs, query)];
  const byPrefix = categorizeByPrefix(xs);
  return Array().concat(byFilter, byPrefix);
}

function createTagToRuns(runToTag) {
  const tagToRun = new Map();
  Object.keys(runToTag).forEach((run) => {
    runToTag[run].forEach((tag) => {
      const runs = tagToRun.get(tag) || [];
      runs.push(run);
      tagToRun.set(tag, runs);
    });
  });
  return tagToRun;
}

function compareTagRun(a, b) {
  const c = compareTagNames(a.tag, b.tag);
  if (c != 0) {
    return c;
  }
  return compareTagNames(a.run, b.run);
}

export function categorizeRunTagCombinations(runToTag, selectedRuns, query) {
  // Inline of categorizeTags: tagToRuns honors selectedRuns only.
  const picked = {};
  (selectedRuns || []).forEach((run) => {
    if (runToTag[run]) picked[run] = runToTag[run];
  });
  const tags = getTags(runToTag);
  const categories = categorize(tags, query);
  const tagToRuns = createTagToRuns(picked);
  const tagCategories = categories.map(({name, metadata, items}) => ({
    name,
    metadata,
    items: items.map((tag) => ({
      tag,
      runs: (tagToRuns.get(tag) || []).slice(),
    })),
  }));
  function explodeCategory(tagCategory) {
    const items = [].concat(
      ...tagCategory.items.map(({tag, runs}) =>
        runs.map((run) => ({tag, run}))
      )
    );
    items.sort(compareTagRun);
    return {
      name: tagCategory.name,
      metadata: tagCategory.metadata,
      items,
    };
  }
  return tagCategories.map(explodeCategory);
}
