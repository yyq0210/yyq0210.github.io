(function () {
  'use strict';

  function getMathRoot() {
    return document.querySelector('.article-content.markdown-body')
      || document.querySelector('.article-content');
  }

  function typesetMath() {
    var root = getMathRoot();
    if (!root || !window.MathJax || typeof MathJax.typesetPromise !== 'function') {
      return Promise.resolve();
    }

    return MathJax.typesetClear([root]).then(function () {
      return MathJax.typesetPromise([root]);
    }).catch(function (err) {
      console.error('MathJax typeset failed:', err);
    });
  }

  function bindSwup() {
    var swup = window.swup;
    if (!swup || !swup.hooks || swup.__mathjaxHookBound) {
      return;
    }

    swup.__mathjaxHookBound = true;
    swup.hooks.on('page:view', function () {
      typesetMath();
    });
  }

  function init() {
    typesetMath();
    bindSwup();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

  window.addEventListener('redefine:swup:ready', function () {
    bindSwup();
    typesetMath();
  });
})();
