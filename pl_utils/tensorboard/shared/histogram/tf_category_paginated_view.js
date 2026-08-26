// Source: tensorboard/components/tf_paginated_view/tf-category-paginated-view.ts
// Source commit: 012299c95682b6cf89e0eaccb8dc034cf1d61cd2
// Adaptation: Polymer element ported to a native custom element; template DOM,
// styles, pagination logic and item caching copied verbatim. The Polymer
// slot/dom-repeat item rendering is replaced by an injected itemFactory, and
// iron-collapse/iron-icon/paper-* controls by native equivalents.
const EXPAND_MORE_ICON =
  '<svg class="expand-arrow" viewBox="0 0 24 24" width="24" height="24"><path fill="currentColor" d="M16.59 8.59L12 13.17 7.41 8.59 6 10l6 6 6-6z"/></svg>';

const TEMPLATE = `
      <button class="heading">
        <span class="name">
          <span class="category-name"></span>
        </span>
        <span class="count">
          <span class="count-value"></span>
          ${EXPAND_MORE_ICON}
        </span>
      </button>
      <div class="collapse">
        <div class="content">
          <span id="top-of-container"></span>
          <div class="big-page-buttons top-page-buttons" style="margin-bottom: 10px;" hidden>
            <button class="paper-button previous-page">Previous page</button>
            <button class="paper-button next-page">Next page</button>
          </div>
          <div id="items"></div>
          <div id="controls-container" hidden>
            <div style="display: inline-block; padding: 0 5px">
              Page
              <input
                id="page-input"
                type="number"
                min="1"
              />
              of <span class="page-count"></span>
            </div>
          </div>
          <div class="big-page-buttons bottom-page-buttons" style="margin-top: 10px;" hidden>
            <button class="paper-button previous-page">Previous page</button>
            <button class="paper-button next-page">Next page</button>
          </div>
        </div>
      </div>
`;

const STYLE = `
      /* Port note: Polymer dom-if was replaced by the hidden attribute; author
         display rules below would otherwise override the UA [hidden] style. */
      [hidden] {
        display: none !important;
      }

      :host {
        display: block;
        margin: 0 5px 1px 10px;
        --paper-grey-500: #9e9e9e;
      }

      :host(:first-of-type) {
        margin-top: 10px;
      }

      :host(:last-of-type) {
        margin-bottom: 20px;
      }

      .heading {
        background-color: var(--primary-background-color, #fff);
        border: none;
        color: inherit;
        cursor: pointer;
        width: 100%;
        font-size: 15px;
        line-height: 1;
        box-shadow: 0 1px 5px rgba(0, 0, 0, 0.2);
        padding: 10px 15px;
        display: flex;
        align-items: center;
        justify-content: space-between;
      }

      [open-button] {
        border-bottom-left-radius: 0 !important;
        border-bottom-right-radius: 0 !important;
      }

      [open-button] .expand-arrow {
        transform: rotateZ(180deg);
      }

      .name {
        display: inline-flex;
        overflow: hidden;
      }

      .light {
        color: var(--paper-grey-500);
      }

      .category-name {
        white-space: pre;
        overflow: hidden;
        text-overflow: ellipsis;
        padding: 2px 0;
      }

      .count {
        margin: 0 5px;
        font-size: 12px;
        color: var(--paper-grey-500);
        display: flex;
        align-items: center;
        flex: none;
      }

      .count .expand-arrow {
        transition: transform 0.25s;
      }

      .collapse {
        display: none;
      }

      .collapse[opened] {
        display: block;
      }

      .content {
        display: flex;
        flex-direction: column;
        background-color: var(--primary-background-color, #fff);
        border-bottom-left-radius: 2px;
        border-bottom-right-radius: 2px;
        border-top: none;
        border: 1px solid #dedede;
        padding: 15px;
      }

      #controls-container {
        justify-content: center;
        display: flex;
        flex-direction: row;
        flex-grow: 0;
        flex-shrink: 0;
        width: 100%;
      }

      .big-page-buttons {
        display: flex;
      }

      .paper-button {
        display: inline-block;
        position: relative;
        box-sizing: border-box;
        min-width: 5.14em;
        margin: 0 0.29em;
        background: transparent;
        text-align: center;
        font: inherit;
        text-transform: uppercase;
        outline-width: 0;
        border-radius: 3px;
        user-select: none;
        cursor: pointer;
        z-index: 0;
        padding: 0.7em 0.57em;
        border: none;
      }

      .big-page-buttons .paper-button {
        background-color: var(--tb-ui-light-accent, #e0e0e0);
        color: var(--tb-ui-dark-accent, #757575);
        display: inline-block;
        flex-basis: 0;
        flex-grow: 1;
        flex-shrink: 1;
        font-size: 13px;
      }

      .big-page-buttons .paper-button[disabled] {
        background: none;
        color: #a8a8a8;
        pointer-events: none;
        cursor: auto;
      }

      #items {
        display: flex;
        flex-direction: row;
        flex-wrap: wrap;
      }

      #page-input {
        display: inline-block;
        width: 100%;
        font: inherit;
        border: none;
        border-bottom: 1px solid var(--paper-grey-500);
        text-align: center;
        background: transparent;
      }
`;

export class TfCategoryPaginatedView extends HTMLElement {
  constructor() {
    super();
    const root = this.attachShadow({mode: 'open'});
    root.innerHTML = `<style>${STYLE}</style>${TEMPLATE}`;
    this._root = root;
    this._headingButton = root.querySelector('.heading');
    this._categoryName = root.querySelector('.category-name');
    this._countValue = root.querySelector('.count-value');
    this._collapse = root.querySelector('.collapse');
    this._items = root.getElementById('items');
    this._controlsContainer = root.getElementById('controls-container');
    this._pageCount = root.querySelector('.page-count');
    this._pageInput = root.getElementById('page-input');
    this._topButtons = root.querySelector('.top-page-buttons');
    this._bottomButtons = root.querySelector('.bottom-page-buttons');
    this.category = null;
    this.initialOpened = null;
    this.opened = false;
    this.disablePagination = false;
    this.getCategoryItemKey = (item) => JSON.stringify(item);
    this.itemFactory = null;
    this._cache = [];
    this._cacheSize = 24;
    this._limit = 12;
    this._activeIndex = 0;
    this._pageInputRawValue = '';
    this._pageInputFocused = false;
    this._rendered = false;
  }

  connectedCallback() {
    // ready(): Polymer initialization.
    this.opened = this.initialOpened == null ? true : this.initialOpened;
    this._headingButton.addEventListener('click', () => this._togglePane());
    this._pageInput.addEventListener('input', (e) =>
      this._handlePageInputEvent(e)
    );
    this._pageInput.addEventListener('change', () => this._handlePageChangeEvent());
    this._pageInput.addEventListener('focus', () => this._handlePageFocusEvent());
    this._pageInput.addEventListener('blur', () => this._handlePageBlurEvent());
    this._root
      .querySelectorAll('.previous-page')
      .forEach((b) => b.addEventListener('click', () => this._performPreviousPage()));
    this._root
      .querySelectorAll('.next-page')
      .forEach((b) => b.addEventListener('click', () => this._performNextPage()));
    this._rendered = true;
    this._syncOpened();
    this._updateHeading();
    this._updatePagination();
    this._updateRenderedItems();
  }

  setCategory(category) {
    this.category = category;
    if (!this._rendered) return;
    this._updateHeading();
    this._clampActiveIndex();
    this._updatePagination();
    this._updateRenderedItems();
  }

  _togglePane() {
    this.opened = !this.opened;
    this._syncOpened();
    this._updateRenderedItems();
  }

  _syncOpened() {
    if (this.opened) {
      this._collapse.setAttribute('opened', '');
      this._headingButton.setAttribute('open-button', '');
    } else {
      this._collapse.removeAttribute('opened');
      this._headingButton.removeAttribute('open-button');
    }
  }

  _updateHeading() {
    if (!this.category) return;
    // PREFIX_GROUP heading; SEARCH_RESULTS headings are not used by this port.
    this._categoryName.textContent = this.category.name;
    this._categoryName.title = this.category.name;
    var count = this._count;
    this._countValue.textContent = this._hasMultiple ? String(count) : '';
  }

  get _count() {
    return this.category.items.length;
  }

  get _hasMultiple() {
    return this._count > 1;
  }

  get _itemsRendered() {
    return this._rendered && this.opened;
  }

  _updatePagination() {
    var multiplePagesExist = this._multiplePagesExist;
    this._topButtons.hidden = !multiplePagesExist;
    this._bottomButtons.hidden = !multiplePagesExist;
    this._controlsContainer.hidden = !multiplePagesExist;
    if (!multiplePagesExist) return;
    var pageCount = this._pageCountComputed;
    this._pageCountValue = pageCount;
    this._pageCount.textContent = String(pageCount);
    this._pageInput.max = pageCount;
    this._updateInputWidth();
    this._updatePageInputValue();
    this._root
      .querySelectorAll('.previous-page')
      .forEach((b) => (b.disabled = !this._hasPreviousPage));
    this._root
      .querySelectorAll('.next-page')
      .forEach((b) => (b.disabled = !this._hasNextPage));
  }

  _updateRenderedItems() {
    var itemsRendered = this._itemsRendered;
    var limit = this._limit;
    var activeIndex = this._activeIndex;
    var disablePagination = this.disablePagination;
    if (!itemsRendered) return;
    if (!this.category) return;
    const activePageIndex = Math.floor(activeIndex / limit);
    const items = this.category.items || [];
    const domItems = disablePagination
      ? items
      : items.slice(activePageIndex * limit, (activePageIndex + 1) * limit);
    this.updateDom(domItems);
  }

  updateDom(domItems) {
    var cache = this._cache;
    var keys = domItems.map((item) => this.getCategoryItemKey(item));
    var keySet = new Set(keys);
    // Evict cached elements beyond the cache size (least recently used first).
    for (var i = 0; cache.length > this._cacheSize && i < cache.length; ) {
      if (!keySet.has(cache[i].key)) {
        cache.splice(i, 1);
      } else {
        i++;
      }
    }
    var newChildren = domItems.map((item, i) => {
      var key = keys[i];
      var entry = null;
      var index = cache.findIndex((e) => e.key === key);
      if (index >= 0) {
        entry = cache.splice(index, 1)[0];
      } else {
        entry = {key: key, element: this.itemFactory(item)};
      }
      cache.push(entry);
      return entry.element;
    });
    this._items.replaceChildren(...newChildren);
  }

  setCacheSize(size) {
    this._cacheSize = size;
  }

  get _currentPage() {
    var limit = this._limit;
    var activeIndex = this._activeIndex;
    return Math.floor(activeIndex / limit) + 1;
  }

  get _pageCountComputed() {
    return this.category ? Math.ceil(this.category.items.length / this._limit) : 0;
  }

  get _multiplePagesExist() {
    var pageCount = this._pageCountComputed;
    var disablePagination = this.disablePagination;
    return !disablePagination && pageCount > 1;
  }

  get _hasPreviousPage() {
    var currentPage = this._currentPage;
    return currentPage > 1;
  }

  get _hasNextPage() {
    var currentPage = this._currentPage;
    var pageCount = this._pageCountComputed;
    return currentPage < pageCount;
  }

  _computeInputWidth(pageCount) {
    // Add 20px for the +/- arrows added by browsers.
    return `calc(${pageCount.toString().length}em + 20px)`;
  }

  _updateInputWidth() {
    this._pageInput.style.width = this._computeInputWidth(this._pageCountValue);
  }

  /**
   * Update _activeIndex, maintaining its range invariant.
   */
  _setActiveIndex(index) {
    const maxIndex = (this.category.items || []).length - 1;
    if (index > maxIndex) {
      index = maxIndex;
    }
    if (index < 0) {
      index = 0;
    }
    this._activeIndex = index;
    this._limitChanged();
    this._updatePagination();
    this._updateRenderedItems();
  }

  _clampActiveIndex() {
    if (this.category) this._setActiveIndex(this._activeIndex);
  }

  _limitChanged() {
    this.setCacheSize(this._limit * 2);
  }

  _performPreviousPage() {
    this._setActiveIndex(this._activeIndex - this._limit);
  }

  _performNextPage() {
    this._setActiveIndex(this._activeIndex + this._limit);
  }

  get _pageInputValue() {
    return this._pageInputFocused
      ? this._pageInputRawValue
      : this._currentPage.toString();
  }

  _handlePageInputEvent(e) {
    this._pageInputRawValue = e.target.value;
    const oneIndexedPage = Number(e.target.value || NaN);
    if (isNaN(oneIndexedPage)) return;
    const page =
      Math.max(1, Math.min(oneIndexedPage, this._pageCountComputed)) - 1;
    this._setActiveIndex(this._limit * page);
  }

  _handlePageChangeEvent() {
    // Occurs on Enter, etc. Commit the true state.
    this._pageInputRawValue = this._currentPage.toString();
    this._updatePageInputValue();
  }

  _handlePageFocusEvent() {
    // Discard any old (or uninitialized) state before we grant focus.
    this._pageInputRawValue = this._pageInputValue;
    this._pageInputFocused = true;
  }

  _handlePageBlurEvent() {
    this._pageInputFocused = false;
    this._updatePageInputValue();
  }

  _updatePageInputValue() {
    this._pageInput.value = this._pageInputValue;
  }
}

customElements.define('tf-category-paginated-view', TfCategoryPaginatedView);
