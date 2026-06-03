(function () {
  'use strict';

  function getRoot() {
    return document.querySelector('.article-content.markdown-body')
      || document.querySelector('.article-content');
  }

  function hasRenderedMath(root) {
    return !!(
      root.querySelector('.mathjax-svg')
      || root.querySelector('mjx-container')
      || root.querySelector('.katex')
    );
  }

  function hasRawMath(text) {
    return text.indexOf('$$') !== -1
      || text.indexOf('\\begin{') !== -1
      || text.indexOf('\\mathbf{') !== -1;
  }

  function needsMathFallback() {
    var root = getRoot();
    if (!root) {
      return false;
    }

    if (hasRenderedMath(root)) {
      return false;
    }

    return hasRawMath(root.textContent || '');
  }

  function bindSwupFallback() {
    var swup = window.swup;
    if (!swup || !swup.hooks || swup.__mathFallbackBound) {
      return;
    }

    swup.__mathFallbackBound = true;
    swup.hooks.on('page:view', function () {
      if (needsMathFallback()) {
        window.location.reload();
      }
    });
  }

  function init() {
    if (needsMathFallback()) {
      window.location.reload();
      return;
    }
    bindSwupFallback();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

  window.addEventListener('redefine:swup:ready', bindSwupFallback);
})();
