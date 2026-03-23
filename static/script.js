/**
 * SkillLens AI — Client-side interactions
 */
document.addEventListener("DOMContentLoaded", () => {
  const fileInput = document.getElementById("resume");
  const fileNameEl = document.getElementById("file-name");
  const uploadSuccessEl = document.getElementById("upload-success");
  const dropzone = document.getElementById("file-dropzone");
  const form = document.getElementById("resume-form");
  const submitBtn = document.getElementById("submit-btn");
  const btnText = submitBtn?.querySelector(".btn-text");
  const btnLoader = submitBtn?.querySelector(".btn-loader");

  const MAX_FILE_SIZE = 5 * 1024 * 1024; // 5MB

  function showUploadSuccess() {
    if (uploadSuccessEl) uploadSuccessEl.classList.remove("hidden");
  }

  function hideUploadSuccess() {
    if (uploadSuccessEl) uploadSuccessEl.classList.add("hidden");
  }

  function validateFile(file) {
    if (!file) return { valid: false, message: "Please select a file." };
    const ext = file.name.split(".").pop()?.toLowerCase();
    if (!["pdf", "docx"].includes(ext)) {
      return { valid: false, message: "Unsupported file type. Please upload PDF or DOCX." };
    }
    if (file.size > MAX_FILE_SIZE) {
      return { valid: false, message: "File too large. Maximum size is 5 MB." };
    }
    return { valid: true };
  }

  function updateFileDisplay(file) {
    if (!fileNameEl) return;
    if (file) {
      const validation = validateFile(file);
      fileNameEl.textContent = `${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
      if (validation.valid) {
        showUploadSuccess();
        fileNameEl.style.color = "#10B981";
      } else {
        hideUploadSuccess();
        fileNameEl.style.color = "#EF4444";
      }
    } else {
      fileNameEl.textContent = "";
      hideUploadSuccess();
    }
  }

  if (fileInput) {
    fileInput.addEventListener("change", () => {
      updateFileDisplay(fileInput.files?.[0]);
    });
  }

  if (dropzone) {
    dropzone.addEventListener("dragover", (e) => {
      e.preventDefault();
      dropzone.style.borderColor = "var(--primary)";
      dropzone.style.background = "#F0F7FF";
    });

    dropzone.addEventListener("dragleave", () => {
      dropzone.style.borderColor = "";
      dropzone.style.background = "";
    });

    dropzone.addEventListener("drop", (e) => {
      e.preventDefault();
      dropzone.style.borderColor = "";
      dropzone.style.background = "";
      const files = e.dataTransfer?.files;
      if (files?.length) {
        fileInput.files = files;
        updateFileDisplay(files[0]);
      }
    });
  }

  if (form && submitBtn) {
    form.addEventListener("submit", () => {
      if (btnText) btnText.classList.add("hidden");
      if (btnLoader) btnLoader.classList.remove("hidden");
      submitBtn.disabled = true;
    });
  }

  // FAQ Accordion
  const faqItems = document.querySelectorAll(".faq-item");
  faqItems.forEach(item => {
    const question = item.querySelector(".faq-question");
    question.addEventListener("click", () => {
      const isActive = item.classList.contains("active");
      // Close all
      faqItems.forEach(i => i.classList.remove("active"));
      // Toggle current
      if (!isActive) item.classList.add("active");
    });
  });
});
