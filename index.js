// ===========================
// INDEX PAGE SCRIPTS — VERITAS
// ===========================

document.addEventListener('DOMContentLoaded', function () {

  // ---- Analyze Link Button ----
  const btnAnalyze = document.querySelector('.btn-analyze');
  const linkInput = document.getElementById('link-input');

  btnAnalyze.addEventListener('click', function (e) {
    const url = linkInput.value.trim();
    if (!url) {
      e.preventDefault();
      linkInput.style.borderBottomColor = '#c0392b';
      linkInput.placeholder = 'Please enter a URL first';
      linkInput.focus();
      setTimeout(() => {
        linkInput.style.borderBottomColor = '#aaa';
        linkInput.placeholder = 'https://example.com/article';
      }, 2000);
      return;
    }
    // Navigate to result page
    window.location.href = 'result.html';
  });

  // Reset border on input
  linkInput.addEventListener('input', function () {
    linkInput.style.borderBottomColor = '#aaa';
  });

  // ---- Upload Zone ----
  const uploadZone = document.querySelector('.upload-zone');
  const fileInput = document.getElementById('file-input');

  uploadZone.addEventListener('click', function () {
    fileInput.click();
  });

  fileInput.addEventListener('change', function () {
    const file = fileInput.files[0];
    if (file) {
      const label = uploadZone.querySelector('.select-label');
      const hint = uploadZone.querySelector('.select-hint');
      label.textContent = file.name;
      hint.textContent = (file.size / 1024 / 1024).toFixed(2) + ' MB';
      uploadZone.style.borderColor = '#555';
    }
  });

  // ---- Drag & Drop Support ----
  uploadZone.addEventListener('dragover', function (e) {
    e.preventDefault();
    uploadZone.style.borderColor = '#555';
  });

  uploadZone.addEventListener('dragleave', function () {
    uploadZone.style.borderColor = 'var(--dashed)';
  });

  uploadZone.addEventListener('drop', function (e) {
    e.preventDefault();
    uploadZone.style.borderColor = '#555';
    const file = e.dataTransfer.files[0];
    if (file) {
      const label = uploadZone.querySelector('.select-label');
      const hint = uploadZone.querySelector('.select-hint');
      label.textContent = file.name;
      hint.textContent = (file.size / 1024 / 1024).toFixed(2) + ' MB';
    }
  });

});