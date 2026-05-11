// ===========================
// RESULT PAGE SCRIPTS — VERITAS
// ===========================

document.addEventListener('DOMContentLoaded', function () {

  // ---- Animate Score Bar on Load ----
  const fill = document.querySelector('.score-bar-fill');
  const scoreTarget = 72; // percent

  if (fill) {
    // Small delay so transition is visible
    setTimeout(() => {
      fill.style.width = scoreTarget + '%';
    }, 200);
  }

  // ---- Save as Image (placeholder) ----
  const btnSaveImage = document.querySelector('[data-action="save-image"]');
  if (btnSaveImage) {
    btnSaveImage.addEventListener('click', function () {
      alert('Save as Image — feature coming soon.');
    });
  }

  // ---- Export PDF (placeholder) ----
  const btnExportPDF = document.querySelector('[data-action="export-pdf"]');
  if (btnExportPDF) {
    btnExportPDF.addEventListener('click', function () {
      window.print();
    });
  }

});