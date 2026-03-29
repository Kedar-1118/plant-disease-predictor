/**
 * app.js - PlantGuard AI Frontend Logic
 */

const dropZone = document.getElementById("dropZone");
const dropZoneContent = document.getElementById("dropZoneContent");
const previewContainer = document.getElementById("previewContainer");
const previewImage = document.getElementById("previewImage");
const fileInput = document.getElementById("fileInput");
const analyzeBtn = document.getElementById("analyzeBtn");
const changeImageBtn = document.getElementById("changeImageBtn");

const uploadCard = document.getElementById("uploadCard");
const loadingCard = document.getElementById("loadingCard");
const errorCard = document.getElementById("errorCard");
const resultCard = document.getElementById("resultCard");

const errorMessage = document.getElementById("errorMessage");
const retryBtn = document.getElementById("retryBtn");
const newAnalysisBtn = document.getElementById("newAnalysisBtn");

const step1 = document.getElementById("step1");
const step2 = document.getElementById("step2");
const step3 = document.getElementById("step3");

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

let selectedFile = null;

const CROP_EMOJIS = {
    tomato: "??",
    potato: "??",
    corn: "??",
};

const SEVERITY_CLASSES = {
    None: "severity-none",
    Low: "severity-low",
    Moderate: "severity-moderate",
    High: "severity-high",
    Unknown: "severity-unknown",
};

function handleFileSelect(file) {
    if (!file || !file.type.startsWith("image/")) {
        showError("Please select a valid image file (JPG, PNG, WEBP, BMP).");
        return;
    }

    selectedFile = file;
    const reader = new FileReader();
    reader.onload = (event) => {
        previewImage.src = event.target.result;
        dropZoneContent.style.display = "none";
        previewContainer.style.display = "block";
        analyzeBtn.disabled = false;
    };
    reader.readAsDataURL(file);
}

fileInput.addEventListener("change", (event) => {
    if (event.target.files.length > 0) {
        handleFileSelect(event.target.files[0]);
    }
});

dropZone.addEventListener("click", (event) => {
    if (event.target === changeImageBtn || changeImageBtn.contains(event.target)) return;
    fileInput.click();
});

changeImageBtn.addEventListener("click", (event) => {
    event.stopPropagation();
    resetUpload();
});

function resetUpload() {
    selectedFile = null;
    fileInput.value = "";
    previewImage.src = "";
    previewContainer.style.display = "none";
    dropZoneContent.style.display = "block";
    analyzeBtn.disabled = true;
}

dropZone.addEventListener("dragover", (event) => {
    event.preventDefault();
    dropZone.classList.add("drag-over");
});

dropZone.addEventListener("dragleave", () => {
    dropZone.classList.remove("drag-over");
});

dropZone.addEventListener("drop", (event) => {
    event.preventDefault();
    dropZone.classList.remove("drag-over");
    const file = event.dataTransfer.files[0];
    if (file) handleFileSelect(file);
});

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

function getErrorMessage(error) {
    if (!error) return "An unexpected error occurred. Please try again.";
    if (typeof error === "string") return error;

    const message = error.message || "An unexpected error occurred. Please try again.";
    const supportedCrops = error.details?.supported_crops;
    if (Array.isArray(supportedCrops) && supportedCrops.length > 0) {
        return `${message} Supported crops: ${supportedCrops.join(", ")}.`;
    }
    return message;
}

retryBtn.addEventListener("click", () => {
    showSection("upload");
});

newAnalysisBtn.addEventListener("click", () => {
    resetUpload();
    showSection("upload");
});

function animateLoadingSteps() {
    [step1, step2, step3].forEach((step) => {
        step.classList.remove("active", "done");
        const dot = step.querySelector(".step-dot");
        dot.classList.remove("step-dot--active", "step-dot--done");
    });

    activateStep(step1);
    setTimeout(() => {
        completeStep(step1);
        activateStep(step2);
    }, 1500);
    setTimeout(() => {
        completeStep(step2);
        activateStep(step3);
    }, 3000);
}

function activateStep(stepEl) {
    stepEl.classList.add("active");
    stepEl.querySelector(".step-dot").classList.add("step-dot--active");
}

function completeStep(stepEl) {
    stepEl.classList.remove("active");
    stepEl.classList.add("done");
    const dot = stepEl.querySelector(".step-dot");
    dot.classList.remove("step-dot--active");
    dot.classList.add("step-dot--done");
}

analyzeBtn.addEventListener("click", async () => {
    if (!selectedFile) return;

    showSection("loading");
    animateLoadingSteps();

    const formData = new FormData();
    formData.append("image", selectedFile);

    try {
        const response = await fetch("/predict", {
            method: "POST",
            body: formData,
        });

        const data = await response.json();
        if (!response.ok || !data.success) {
            showError(getErrorMessage(data.error));
            return;
        }

        renderResult(data);
        showSection("result");
    } catch (_error) {
        showError("Could not connect to the server. Make sure the Flask app is running.");
    }
});

function renderResult(data) {
    const treatment = data.treatment || {};

    if (data.low_confidence_warning) {
        warningBanner.style.display = "flex";
        warningMessage.textContent = data.warning_message;
    } else {
        warningBanner.style.display = "none";
    }

    const crop = (data.crop || "unknown").toLowerCase();
    cropEmoji.textContent = CROP_EMOJIS[crop] || "??";
    cropName.textContent = capitalize(crop);

    setTimeout(() => {
        cropConfBar.style.width = `${data.crop_confidence}%`;
    }, 100);
    cropConfPct.textContent = `${data.crop_confidence}%`;

    diseaseName.textContent = treatment.display_name || capitalize(data.disease || "Unknown");
    const severity = treatment.severity || "Unknown";
    severityBadge.textContent = severity;
    severityBadge.className = `severity-badge ${SEVERITY_CLASSES[severity] || "severity-unknown"}`;

    setTimeout(() => {
        diseaseConfBar.style.width = `${data.disease_confidence}%`;
    }, 200);
    diseaseConfPct.textContent = `${data.disease_confidence}%`;

    if (previewImage.src) {
        resultImage.src = previewImage.src;
        resultImage.parentElement.style.display = "block";
    } else {
        resultImage.parentElement.style.display = "none";
    }

    diseaseDescription.textContent = treatment.description || "No description available.";
    renderList(symptomsList, treatment.symptoms || []);
    renderList(treatmentList, treatment.treatment || []);
    renderList(preventionList, treatment.prevention || []);
    renderConfidenceBreakdown(data);

    // ── Grad-CAM Heatmap ──────────────────────────────────────────────
    const gradcamSection = document.getElementById("gradcamSection");
    const gradcamOriginal = document.getElementById("gradcamOriginal");
    const gradcamOverlay = document.getElementById("gradcamOverlay");

    if (data.gradcam_image) {
        gradcamOriginal.src = previewImage.src;
        gradcamOverlay.src = data.gradcam_image;
        gradcamSection.style.display = "block";
    } else {
        gradcamSection.style.display = "none";
    }

    // ── Inference Latency ─────────────────────────────────────────────
    const latencyDisplay = document.getElementById("latencyDisplay");
    const latencyValue = document.getElementById("latencyValue");

    if (data.latency_ms) {
        latencyValue.textContent = data.latency_ms;
        latencyDisplay.style.display = "block";
    } else {
        latencyDisplay.style.display = "none";
    }
}

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

function renderConfidenceBreakdown(data) {
    confidenceBreakdown.innerHTML = "";

    if (data.crop_all_predictions) {
        const heading = document.createElement("p");
        heading.style.cssText = "font-size:11px;text-transform:uppercase;letter-spacing:0.8px;color:var(--text-muted);margin-bottom:8px;font-weight:600;";
        heading.textContent = "Crop Predictions";
        confidenceBreakdown.appendChild(heading);

        Object.entries(data.crop_all_predictions)
            .sort((a, b) => b[1] - a[1])
            .forEach(([label, pct]) => confidenceBreakdown.appendChild(createBreakdownItem(label, pct)));
    }

    if (data.disease_all_predictions) {
        const heading = document.createElement("p");
        heading.style.cssText = "font-size:11px;text-transform:uppercase;letter-spacing:0.8px;color:var(--text-muted);margin:16px 0 8px;font-weight:600;";
        heading.textContent = "Disease Predictions";
        confidenceBreakdown.appendChild(heading);

        Object.entries(data.disease_all_predictions)
            .sort((a, b) => b[1] - a[1])
            .forEach(([label, pct]) => confidenceBreakdown.appendChild(createBreakdownItem(label, pct)));
    }
}

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
    setTimeout(() => {
        fill.style.width = `${pct}%`;
    }, 300);

    track.appendChild(fill);

    const pctEl = document.createElement("span");
    pctEl.className = "breakdown-pct";
    pctEl.textContent = `${pct}%`;

    item.appendChild(labelEl);
    item.appendChild(track);
    item.appendChild(pctEl);

    return item;
}

function capitalize(str) {
    if (!str) return "";
    return str.charAt(0).toUpperCase() + str.slice(1);
}
