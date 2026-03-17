/**
 * app.js — PlantGuard AI Frontend Logic
 * ----------------------------------------
 * Handles:
 *  - Drag & drop and file input for image upload
 *  - Image preview
 *  - Sending image to /predict API via fetch
 *  - Rendering prediction results dynamically
 *  - Loading state animation with step progression
 */

// ─────────────────────────────────────────────
// DOM Element References
// ─────────────────────────────────────────────
const dropZone = document.getElementById("dropZone");
const dropZoneContent = document.getElementById("dropZoneContent");
const previewContainer = document.getElementById("previewContainer");
const previewImage = document.getElementById("previewImage");
const fileInput = document.getElementById("fileInput");
const analyzeBtn = document.getElementById("analyzeBtn");
const analyzeBtnText = document.getElementById("analyzeBtnText");
const analyzeBtnSpinner = document.getElementById("analyzeBtnSpinner");
const changeImageBtn = document.getElementById("changeImageBtn");

const uploadCard = document.getElementById("uploadCard");
const loadingCard = document.getElementById("loadingCard");
const errorCard = document.getElementById("errorCard");
const resultCard = document.getElementById("resultCard");

const errorMessage = document.getElementById("errorMessage");
const retryBtn = document.getElementById("retryBtn");
const newAnalysisBtn = document.getElementById("newAnalysisBtn");

// Loading steps
const step1 = document.getElementById("step1");
const step2 = document.getElementById("step2");
const step3 = document.getElementById("step3");

// Result elements
const warningBanner = document.getElementById("warningBanner");
const warningMessage = document.getElementById("warningMessage");
const cropEmoji = document.getElementById("cropEmoji");
const cropName = document.getElementById("cropName");
const cropConfBar = document.getElementById("cropConfBar");
const cropConfPct = document.getElementById("cropConfPct");
const diseaseName = document.getElementById("diseaseName");
const severityBadge = document.getElementById("severityBadge");
const diseaseConfBar = document.getElementById("diseaseConfBar");
const diseaseConfPct = document.getElementById("diseaseConfPct");
const resultImage = document.getElementById("resultImage");
const diseaseDescription = document.getElementById("diseaseDescription");
const symptomsList = document.getElementById("symptomsList");
const treatmentList = document.getElementById("treatmentList");
const preventionList = document.getElementById("preventionList");
const confidenceBreakdown = document.getElementById("confidenceBreakdown");


// ─────────────────────────────────────────────
// State
// ─────────────────────────────────────────────
let selectedFile = null;


// ─────────────────────────────────────────────
// Crop Emoji Map
// ─────────────────────────────────────────────
const CROP_EMOJIS = {
    tomato: "🍅",
    potato: "🥔",
    rice: "🌾",
    corn: "🌽",
};

const SEVERITY_CLASSES = {
    "None": "severity-none",
    "Low": "severity-low",
    "Moderate": "severity-moderate",
    "High": "severity-high",
    "Unknown": "severity-unknown",
};


// ─────────────────────────────────────────────
// File Handling
// ─────────────────────────────────────────────

/**
 * Handle a file being selected (via input or drag-drop).
 * Shows a preview and enables the analyze button.
 */
function handleFileSelect(file) {
    if (!file || !file.type.startsWith("image/")) {
        showError("Please select a valid image file (JPG, PNG, WEBP, BMP).");
        return;
    }

    selectedFile = file;

    // Show image preview
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        dropZoneContent.style.display = "none";
        previewContainer.style.display = "block";
        analyzeBtn.disabled = false;
    };
    reader.readAsDataURL(file);
}

// File input change
fileInput.addEventListener("change", (e) => {
    if (e.target.files.length > 0) {
        handleFileSelect(e.target.files[0]);
    }
});

// Click on drop zone to open file picker
dropZone.addEventListener("click", (e) => {
    // Don't trigger if clicking the change button
    if (e.target === changeImageBtn || changeImageBtn.contains(e.target)) return;
    fileInput.click();
});

// Change image button
changeImageBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    resetUpload();
});

// Reset to upload state
function resetUpload() {
    selectedFile = null;
    fileInput.value = "";
    previewImage.src = "";
    previewContainer.style.display = "none";
    dropZoneContent.style.display = "block";
    analyzeBtn.disabled = true;
}


// ─────────────────────────────────────────────
// Drag & Drop
// ─────────────────────────────────────────────

dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("drag-over");
});

dropZone.addEventListener("dragleave", () => {
    dropZone.classList.remove("drag-over");
});

dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("drag-over");
    const file = e.dataTransfer.files[0];
    if (file) handleFileSelect(file);
});


// ─────────────────────────────────────────────
// UI State Management
// ─────────────────────────────────────────────

function showSection(section) {
    uploadCard.style.display = "none";
    loadingCard.style.display = "none";
    errorCard.style.display = "none";
    resultCard.style.display = "none";

    if (section === "upload") uploadCard.style.display = "block";
    if (section === "loading") loadingCard.style.display = "block";
    if (section === "error") errorCard.style.display = "block";
    if (section === "result") resultCard.style.display = "block";
}

function showError(message) {
    errorMessage.textContent = message;
    showSection("error");
}

// Retry button
retryBtn.addEventListener("click", () => {
    showSection("upload");
});

// New analysis button
newAnalysisBtn.addEventListener("click", () => {
    resetUpload();
    showSection("upload");
});


// ─────────────────────────────────────────────
// Loading Animation Steps
// ─────────────────────────────────────────────

/**
 * Animate the loading steps to simulate progress.
 * Step 1 activates immediately, Step 2 after 1.5s, Step 3 after 3s.
 */
function animateLoadingSteps() {
    // Reset all steps
    [step1, step2, step3].forEach((s) => {
        s.classList.remove("active", "done");
        const dot = s.querySelector(".step-dot");
        dot.classList.remove("step-dot--active", "step-dot--done");
    });

    // Step 1 — active immediately
    activateStep(step1);

    // Step 2 — after 1.5s
    setTimeout(() => {
        completeStep(step1);
        activateStep(step2);
    }, 1500);

    // Step 3 — after 3s
    setTimeout(() => {
        completeStep(step2);
        activateStep(step3);
    }, 3000);
}

function activateStep(stepEl) {
    stepEl.classList.add("active");
    const dot = stepEl.querySelector(".step-dot");
    dot.classList.add("step-dot--active");
}

function completeStep(stepEl) {
    stepEl.classList.remove("active");
    stepEl.classList.add("done");
    const dot = stepEl.querySelector(".step-dot");
    dot.classList.remove("step-dot--active");
    dot.classList.add("step-dot--done");
}


// ─────────────────────────────────────────────
// Analyze Button — Main Prediction Flow
// ─────────────────────────────────────────────

analyzeBtn.addEventListener("click", async () => {
    if (!selectedFile) return;

    // Show loading state
    showSection("loading");
    animateLoadingSteps();

    // Build form data
    const formData = new FormData();
    formData.append("image", selectedFile);

    try {
        const response = await fetch("/predict", {
            method: "POST",
            body: formData,
        });

        const data = await response.json();

        if (!response.ok || !data.success) {
            showError(data.error || "An unexpected error occurred. Please try again.");
            return;
        }

        // Render results
        renderResult(data);
        showSection("result");

    } catch (err) {
        showError(
            "Could not connect to the server. Make sure the Flask app is running."
        );
    }
});


// ─────────────────────────────────────────────
// Result Rendering
// ─────────────────────────────────────────────

/**
 * Populate the result card with prediction data.
 * @param {Object} data - The JSON response from /predict
 */
function renderResult(data) {
    const treatment = data.treatment || {};

    // ── Warning ────────────────────────────────────────────────────────────
    if (data.low_confidence_warning) {
        warningBanner.style.display = "flex";
        warningMessage.textContent = data.warning_message;
    } else {
        warningBanner.style.display = "none";
    }

    // ── Stage 1: Crop ──────────────────────────────────────────────────────
    const crop = (data.crop || "unknown").toLowerCase();
    cropEmoji.textContent = CROP_EMOJIS[crop] || "🌿";
    cropName.textContent = capitalize(crop);

    // Animate confidence bar after a short delay (allows CSS transition)
    setTimeout(() => {
        cropConfBar.style.width = `${data.crop_confidence}%`;
    }, 100);
    cropConfPct.textContent = `${data.crop_confidence}%`;

    // ── Stage 2: Disease ───────────────────────────────────────────────────
    const displayName = treatment.display_name || capitalize(data.disease || "Unknown");
    diseaseName.textContent = displayName;

    // Severity badge
    const severity = treatment.severity || "Unknown";
    severityBadge.textContent = severity;
    severityBadge.className = "severity-badge " + (SEVERITY_CLASSES[severity] || "severity-unknown");

    setTimeout(() => {
        diseaseConfBar.style.width = `${data.disease_confidence}%`;
    }, 200);
    diseaseConfPct.textContent = `${data.disease_confidence}%`;

    // ── Uploaded Image ─────────────────────────────────────────────────────
    if (data.image_url) {
        resultImage.src = data.image_url;
        resultImage.parentElement.style.display = "block";
    } else {
        resultImage.parentElement.style.display = "none";
    }

    // ── Description ────────────────────────────────────────────────────────
    diseaseDescription.textContent = treatment.description || "No description available.";

    // ── Symptoms ───────────────────────────────────────────────────────────
    renderList(symptomsList, treatment.symptoms || []);

    // ── Treatment ──────────────────────────────────────────────────────────
    renderList(treatmentList, treatment.treatment || []);

    // ── Prevention ─────────────────────────────────────────────────────────
    renderList(preventionList, treatment.prevention || []);

    // ── Confidence Breakdown ───────────────────────────────────────────────
    renderConfidenceBreakdown(data);
}


/**
 * Render a list of strings as <li> elements.
 */
function renderList(listEl, items) {
    listEl.innerHTML = "";
    if (items.length === 0) {
        const li = document.createElement("li");
        li.textContent = "No information available.";
        listEl.appendChild(li);
        return;
    }
    items.forEach((item) => {
        const li = document.createElement("li");
        li.textContent = item;
        listEl.appendChild(li);
    });
}


/**
 * Render the confidence breakdown for all crop and disease predictions.
 */
function renderConfidenceBreakdown(data) {
    confidenceBreakdown.innerHTML = "";

    // Crop predictions
    if (data.crop_all_predictions) {
        const heading = document.createElement("p");
        heading.style.cssText = "font-size:11px;text-transform:uppercase;letter-spacing:0.8px;color:var(--text-muted);margin-bottom:8px;font-weight:600;";
        heading.textContent = "Crop Predictions";
        confidenceBreakdown.appendChild(heading);

        const sorted = Object.entries(data.crop_all_predictions).sort((a, b) => b[1] - a[1]);
        sorted.forEach(([label, pct]) => {
            confidenceBreakdown.appendChild(createBreakdownItem(label, pct));
        });
    }

    // Disease predictions
    if (data.disease_all_predictions) {
        const heading = document.createElement("p");
        heading.style.cssText = "font-size:11px;text-transform:uppercase;letter-spacing:0.8px;color:var(--text-muted);margin:16px 0 8px;font-weight:600;";
        heading.textContent = "Disease Predictions";
        confidenceBreakdown.appendChild(heading);

        const sorted = Object.entries(data.disease_all_predictions).sort((a, b) => b[1] - a[1]);
        sorted.forEach(([label, pct]) => {
            confidenceBreakdown.appendChild(createBreakdownItem(label, pct));
        });
    }
}


/**
 * Create a single confidence breakdown bar item.
 */
function createBreakdownItem(label, pct) {
    const item = document.createElement("div");
    item.className = "breakdown-item";

    const labelEl = document.createElement("span");
    labelEl.className = "breakdown-label";
    labelEl.textContent = label.replace(/_/g, " ");

    const track = document.createElement("div");
    track.className = "breakdown-bar-track";

    const fill = document.createElement("div");
    fill.className = "breakdown-bar-fill";
    fill.style.width = "0%";
    setTimeout(() => { fill.style.width = `${pct}%`; }, 300);

    track.appendChild(fill);

    const pctEl = document.createElement("span");
    pctEl.className = "breakdown-pct";
    pctEl.textContent = `${pct}%`;

    item.appendChild(labelEl);
    item.appendChild(track);
    item.appendChild(pctEl);

    return item;
}


// ─────────────────────────────────────────────
// Utility
// ─────────────────────────────────────────────

function capitalize(str) {
    if (!str) return "";
    return str.charAt(0).toUpperCase() + str.slice(1);
}
