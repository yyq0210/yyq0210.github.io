(function () {
  'use strict';

  function needsMathFallback() {
    var root = document.querySelector('.article-content.markdown-body')
      || document.querySelector('.article-content');
    if (!root) {
      return false;
    }

    if (root.querySelector('mjx-container')) {
      return false;
    }

    var text = root.textContent || '';
    return text.indexOf('$$') !== -1 || text.indexOf('\\begin{') !== -1;
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
