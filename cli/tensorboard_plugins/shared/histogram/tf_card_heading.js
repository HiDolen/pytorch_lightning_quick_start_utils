// Source: tensorboard/components/tf_card_heading/tf-card-heading.ts,
//         tensorboard/components/tf_card_heading/tf-card-heading-style.ts,
//         tensorboard/components/tf_card_heading/util.ts
// Source commit: 012299c95682b6cf89e0eaccb8dc034cf1d61cd2
// Adaptation: Polymer element ported to a native custom element; template DOM
// and styles copied verbatim. paper-icon-button/paper-dialog are replaced by
// minimal native equivalents (only used when a description exists).

export function pickTextColor(background) {
  const rgb = convertHexToRgb(background);
  if (!rgb) {
    return 'inherit';
  }
  // See: http://www.w3.org/TR/AERT#color-contrast
  const brightness = Math.round(
    (rgb[0] * 299 + rgb[1] * 587 + rgb[2] * 114) / 1000
  );
  return brightness > 125 ? 'inherit' : '#eee';
}

function convertHexToRgb(color) {
  if (!color) {
    return null;
  }
  let m = color.match(/^#([0-9a-f]{1,2})([0-9a-f]{1,2})([0-9a-f]{1,2})$/);
  if (!m) {
    return null;
  }
  if (color.length == 4) {
    for (var i = 1; i <= 3; i++) {
      m[i] = m[i] + m[i];
    }
  }
  return [parseInt(m[1], 16), parseInt(m[2], 16), parseInt(m[3], 16)];
}

const TEMPLATE = `
    <div class="container">
      <figcaption class="content">
        <div class="heading-row">
          <div itemprop="name" class="heading-label name"></div>
        </div>
        <div class="heading-row">
          <span class="run-wrap">
            <span
              itemprop="run"
              id="heading-run"
              class="heading-label run"
            ></span>
          </span>
        </div>
        <slot></slot>
      </figcaption>
      <button class="paper-icon-button" title="Show summary description" hidden>
        <svg viewBox="0 0 24 24" width="20" height="20"><path fill="currentColor" d="M11 7h2v2h-2zm0 4h2v6h-2zm1-9C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 18c-4.41 0-8-3.59-8-8s3.59-8 8-8 8 3.59 8 8-3.59 8-8 8z"/></svg>
      </button>
      <div class="paper-dialog" role="dialog">
        <div class="paper-dialog-scrollable"></div>
      </div>
    </div>
`;

const STYLE = `
      :host {
        display: block;
      }

      .container {
        display: flex;
      }

      figcaption {
        width: 100%;
      }

      /** Horizontal line of labels. */
      .heading-row {
        margin-top: -4px;
        display: flex;
        flex-direction: row;
        flex-wrap: wrap;
      }

      /** Piece of text in the figure caption. */
      .heading-label {
        flex-grow: 1;
        margin-top: 4px;
        max-width: 100%;
        word-wrap: break-word;
      }

      /** Makes label show on the right. */
      .heading-right {
        flex-grow: 0;
      }

      .content {
        font-size: 12px;
        flex-grow: 1;
      }

      .name {
        font-size: 14px;
        /* 单行截断,保证各卡片头部等高 */
        display: -webkit-box;
        -webkit-box-orient: vertical;
        -webkit-line-clamp: 1;
        overflow: hidden;
      }

      .run {
        font-size: 11px;
        width: auto;
        border-radius: 3px;
        font-weight: bold;
        padding: 1px 4px 2px;
        display: block;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
      }

      /* run 徽章独占一行,宽度贴合内容,超长截断 */
      .run-wrap {
        flex-shrink: 1;
        min-width: 0;
        max-width: 100%;
      }

      .paper-icon-button {
        flex-grow: 0;
        background: none;
        border: none;
        padding: 4px;
        width: 32px;
        height: 32px;
        border-radius: 100%;
        cursor: pointer;
        color: inherit;
      }

      .paper-dialog-scrollable {
        max-width: 640px;
      }

      #heading-run {
        background: var(--tf-card-heading-background-color);
        color: var(--tf-card-heading-color);
      }

      .paper-dialog {
        display: none;
        position: fixed;
        z-index: 100;
        background: white;
        color: rgba(0, 0, 0, 0.87);
        border-radius: 2px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
        padding: 16px;
        max-width: 640px;
        max-height: 60vh;
        overflow-y: auto;
        white-space: pre-wrap;
      }
`;

export class TfCardHeading extends HTMLElement {
  constructor() {
    super();
    const root = this.attachShadow({mode: 'open'});
    root.innerHTML = `<style>${STYLE}</style>${TEMPLATE}`;
    this._root = root;
    this._nameEl = root.querySelector('.name');
    this._runEl = root.querySelector('#heading-run');
    this._infoButton = root.querySelector('.paper-icon-button');
    this._dialog = root.querySelector('.paper-dialog');
    this._dialogScrollable = root.querySelector('.paper-dialog-scrollable');
  }

  connectedCallback() {
    // 宿主会在过滤/分页时反复重新插入节点，重复连接不得叠加监听
    if (!this._listenerBound) {
      this._listenerBound = true;
      this._infoButton.addEventListener('click', () =>
        this._toggleDescriptionDialog()
      );
    }
    this._update();
  }

  set displayName(v) {
    this._displayName = v;
    this._update();
  }
  get displayName() {
    return this._displayName;
  }

  set tag(v) {
    this._tag = v;
    this._update();
  }
  get tag() {
    return this._tag;
  }

  set run(v) {
    this._run = v;
    this._update();
  }
  get run() {
    return this._run;
  }

  set description(v) {
    this._description = v;
    this._update();
  }
  get description() {
    return this._description;
  }

  set color(v) {
    this._color = v;
    this._updateHeadingStyle();
  }
  get color() {
    return this._color;
  }

  _toggleDescriptionDialog() {
    this._dialog.style.display =
      this._dialog.style.display === 'block' ? 'none' : 'block';
  }

  _updateHeadingStyle() {
    var runBackground = this._computeRunBackground(this._color);
    var runColor = this._computeRunColor(this._color);
    this.style.setProperty('--tf-card-heading-background-color', runBackground);
    this.style.setProperty('--tf-card-heading-color', runColor);
  }

  _computeRunBackground(color) {
    return color || 'none';
  }

  _computeRunColor(color) {
    return pickTextColor(color);
  }

  _update() {
    var displayName = this._displayName || null;
    var tag = this._tag || null;
    var run = this._run || null;
    var description = this._description || null;
    var nameLabel = displayName || tag || '';
    if (nameLabel) {
      this._nameEl.textContent = nameLabel;
      this._nameEl.style.display = '';
    } else {
      this._nameEl.style.display = 'none';
    }
    if (run) {
      this._runEl.textContent = run;
      this._runEl.parentElement.style.display = '';
    } else {
      this._runEl.parentElement.style.display = 'none';
    }
    if (description) {
      this._infoButton.hidden = false;
      this._dialogScrollable.textContent = description;
    } else {
      this._infoButton.hidden = true;
    }
  }
}

customElements.define('tf-card-heading', TfCardHeading);
