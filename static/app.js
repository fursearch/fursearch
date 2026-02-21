/**
 * Fursuit Scanner - Client-side CLIP classification with localStorage persistence
 */

import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2';

// Use local models instead of Hugging Face
env.localModelPath = 'models/';
env.allowRemoteModels = true;

const STORAGE_KEY = 'fursuit_scanner_state';
const LABELS = [
    'a photo of a fursuit',
    'a photo of a person in an animal costume',
    'a photo of a mascot',
    'a regular photo of people',
    'a photo of nature or objects'
];

// State
let classifier = null;
let files = [];
let isRunning = false;
let isPaused = false;
let currentIndex = 0;
let processedFiles = new Set();
let results = [];
let startTime = null;
let threshold = 50;

// Folder selection state
const IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp'];

// DOM elements
const fileInput = document.getElementById('fileInput');
const fileCount = document.getElementById('fileCount');
const startBtn = document.getElementById('startBtn');
const pauseBtn = document.getElementById('pauseBtn');
const progressSection = document.getElementById('progressSection');
const progressBar = document.getElementById('progressBar');
const progressCount = document.getElementById('progressCount');
const progressEta = document.getElementById('progressEta');
const currentFile = document.getElementById('currentFile');
const resultsGrid = document.getElementById('resultsGrid');
const modelStatus = document.getElementById('modelStatus');
const thresholdInput = document.getElementById('threshold');
const thresholdValue = document.getElementById('thresholdValue');
const statTotal = document.getElementById('statTotal');
const statMatches = document.getElementById('statMatches');
const statRate = document.getElementById('statRate');
const resumeBanner = document.getElementById('resumeBanner');
const resumeBtn = document.getElementById('resumeBtn');
const clearResumeBtn = document.getElementById('clearResumeBtn');
const resumeCount = document.getElementById('resumeCount');
const clearResultsBtn = document.getElementById('clearResultsBtn');
const folderInput = document.getElementById('folderInput');
const folderMonitor = document.getElementById('folderMonitor');
const folderPath = document.getElementById('folderPath');
const monitorStatus = document.getElementById('monitorStatus');
const refreshFolderBtn = document.getElementById('refreshFolderBtn');
const clearFolderBtn = document.getElementById('clearFolderBtn');

// Initialize
async function init() {
    loadState();
    setupEventListeners();
    await loadModel();
}

function setupEventListeners() {
    fileInput.addEventListener('change', handleFileSelect);
    startBtn.addEventListener('click', startScan);
    pauseBtn.addEventListener('click', togglePause);
    thresholdInput.addEventListener('input', handleThresholdChange);
    resumeBtn.addEventListener('click', resumeScan);
    clearResumeBtn.addEventListener('click', clearAndStartFresh);
    clearResultsBtn.addEventListener('click', clearAllResults);
    folderInput.addEventListener('change', handleFolderSelect);
    refreshFolderBtn.addEventListener('click', () => folderInput.click());
    clearFolderBtn.addEventListener('click', clearFolder);
}

async function loadModel() {
    modelStatus.classList.add('active', 'loading');
    modelStatus.textContent = 'Loading CLIP model from local server...';

    try {
        classifier = await pipeline(
            'zero-shot-image-classification',
            'clip-vit-base-patch32',
            { progress_callback: (progress) => {
                if (progress.status === 'loading') {
                    modelStatus.textContent = `Loading model: ${progress.name || 'initializing'}...`;
                }
            }}
        );

        modelStatus.classList.remove('loading');
        modelStatus.classList.add('ready');
        modelStatus.textContent = 'CLIP model loaded and ready!';
        updateUI();

        setTimeout(() => {
            modelStatus.classList.remove('active');
        }, 2000);

    } catch (error) {
        modelStatus.classList.remove('loading');
        modelStatus.classList.add('error');
        modelStatus.textContent = `Error loading model: ${error.message}`;
        console.error('Model loading error:', error);
    }
}

function handleFileSelect(e) {
    // Clear folder mode when selecting individual files
    folderMonitor.classList.remove('active');
    folderInput.value = '';

    files = Array.from(e.target.files);
    fileCount.textContent = `${files.length} files selected`;
    startBtn.disabled = files.length === 0 || !classifier;

    // Check for resumable state
    const savedFiles = files.map(f => f.name);
    const canResume = results.some(r => savedFiles.includes(r.name));

    if (canResume && processedFiles.size > 0) {
        resumeBanner.classList.add('active');
        resumeCount.textContent = processedFiles.size;
    } else {
        resumeBanner.classList.remove('active');
    }
}

function handleThresholdChange(e) {
    threshold = parseInt(e.target.value);
    thresholdValue.textContent = `${threshold}%`;
    renderResults();
}

async function startScan() {
    if (!classifier || files.length === 0) return;

    // Reset state for fresh scan
    currentIndex = 0;
    processedFiles = new Set();
    results = [];
    startTime = Date.now();

    isRunning = true;
    isPaused = false;
    updateUI();
    await runScan();
}

async function resumeScan() {
    if (!classifier || files.length === 0) return;

    // Find the first unprocessed file
    currentIndex = 0;
    for (let i = 0; i < files.length; i++) {
        if (!processedFiles.has(files[i].name)) {
            currentIndex = i;
            break;
        }
    }

    startTime = Date.now();
    isRunning = true;
    isPaused = false;
    resumeBanner.classList.remove('active');
    updateUI();
    await runScan();
}

function clearAndStartFresh() {
    processedFiles = new Set();
    results = [];
    currentIndex = 0;
    saveState();
    resumeBanner.classList.remove('active');
    renderResults();
    updateStats();
}

function clearAllResults() {
    processedFiles = new Set();
    results = [];
    currentIndex = 0;
    localStorage.removeItem(STORAGE_KEY);
    renderResults();
    updateStats();
}

function togglePause() {
    isPaused = !isPaused;
    pauseBtn.textContent = isPaused ? 'Resume' : 'Pause';

    if (!isPaused && isRunning) {
        runScan();
    }
}

async function runScan() {
    progressSection.classList.add('active');

    while (currentIndex < files.length && isRunning && !isPaused) {
        const file = files[currentIndex];

        // Skip already processed files
        if (processedFiles.has(file.name)) {
            currentIndex++;
            continue;
        }

        currentFile.textContent = `Processing: ${file.name}`;
        updateProgress();

        try {
            const score = await classifyImage(file);
            const isFursuit = score >= threshold / 100;

            const result = {
                name: file.name,
                score: score,
                isFursuit: isFursuit,
                timestamp: Date.now(),
                dataUrl: isFursuit ? await fileToDataUrl(file) : null
            };

            results.push(result);
            processedFiles.add(file.name);

            if (isFursuit) {
                addResultCard(result);
            }

            updateStats();
            saveState();

        } catch (error) {
            console.error(`Error processing ${file.name}:`, error);
        }

        currentIndex++;
        updateProgress();
    }

    if (currentIndex >= files.length) {
        isRunning = false;
        currentFile.textContent = 'Scan complete!';
    }

    updateUI();
}

async function classifyImage(file) {
    const dataUrl = await fileToDataUrl(file);

    const output = await classifier(dataUrl, LABELS);

    // Calculate fursuit score from top results
    // Sum of fursuit-related labels vs non-fursuit
    let fursuitScore = 0;
    for (const result of output) {
        if (result.label.includes('fursuit') ||
            result.label.includes('animal costume') ||
            result.label.includes('mascot')) {
            fursuitScore += result.score;
        }
    }

    return fursuitScore;
}

function fileToDataUrl(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result);
        reader.onerror = reject;
        reader.readAsDataURL(file);
    });
}

function updateProgress() {
    const total = files.length;
    const processed = currentIndex;
    const pct = Math.round((processed / total) * 100);

    progressBar.style.width = `${pct}%`;
    progressBar.textContent = `${pct}%`;
    progressCount.textContent = `${processed} / ${total}`;

    // Calculate ETA
    if (startTime && processed > 0) {
        const elapsed = (Date.now() - startTime) / 1000;
        const rate = processed / elapsed;
        const remaining = total - processed;
        const eta = remaining / rate;

        if (eta < 60) {
            progressEta.textContent = `~${Math.round(eta)}s remaining`;
        } else {
            progressEta.textContent = `~${Math.round(eta / 60)}m remaining`;
        }

        statRate.textContent = rate.toFixed(1);
    }
}

function updateStats() {
    const matches = results.filter(r => r.score >= threshold / 100);
    statTotal.textContent = processedFiles.size;
    statMatches.textContent = matches.length;
}

function updateUI() {
    startBtn.disabled = isRunning || !classifier || files.length === 0;
    pauseBtn.disabled = !isRunning;
    pauseBtn.textContent = isPaused ? 'Resume' : 'Pause';
}

function renderResults() {
    const matches = results.filter(r => r.score >= threshold / 100);

    if (matches.length === 0) {
        resultsGrid.innerHTML = '<div class="no-results">No fursuit images found yet. Select photos and start scanning.</div>';
        return;
    }

    resultsGrid.innerHTML = '';
    // Sort by score descending
    matches.sort((a, b) => b.score - a.score);

    for (const result of matches) {
        addResultCard(result, false);
    }
}

function addResultCard(result, append = true) {
    if (!append) {
        // Already cleared in renderResults
    } else {
        // Remove "no results" placeholder
        const placeholder = resultsGrid.querySelector('.no-results');
        if (placeholder) placeholder.remove();
    }

    const card = document.createElement('div');
    card.className = 'result-card';
    card.innerHTML = `
        <img src="${result.dataUrl}" alt="${result.name}">
        <div class="result-score">${Math.round(result.score * 100)}%</div>
        <div class="result-info">
            <div class="result-name">${result.name}</div>
        </div>
    `;

    if (append) {
        resultsGrid.prepend(card);
    } else {
        resultsGrid.appendChild(card);
    }
}

function saveState() {
    const state = {
        version: 1,
        processedFiles: Array.from(processedFiles),
        results: results,
        threshold: threshold
    };

    try {
        localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
    } catch (e) {
        // localStorage might be full, try saving without dataUrls
        console.warn('localStorage full, saving without images');
        state.results = results.map(r => ({ ...r, dataUrl: null }));
        localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
    }
}

function loadState() {
    try {
        const saved = localStorage.getItem(STORAGE_KEY);
        if (!saved) return;

        const state = JSON.parse(saved);
        if (state.version !== 1) return;

        processedFiles = new Set(state.processedFiles || []);
        results = state.results || [];
        threshold = state.threshold || 50;

        thresholdInput.value = threshold;
        thresholdValue.textContent = `${threshold}%`;

        updateStats();
        renderResults();

        if (processedFiles.size > 0) {
            resumeBanner.classList.add('active');
            resumeCount.textContent = processedFiles.size;
        }

    } catch (e) {
        console.error('Error loading state:', e);
    }
}

// Folder selection via webkitdirectory input (works in Safari and Chrome)
function handleFolderSelect(e) {
    const allFiles = Array.from(e.target.files);

    // Filter to only image files
    files = allFiles.filter(f => {
        const name = f.name.toLowerCase();
        return IMAGE_EXTENSIONS.some(ext => name.endsWith(ext));
    });

    if (files.length === 0) {
        folderMonitor.classList.remove('active');
        fileCount.textContent = 'No images found in folder';
        return;
    }

    // Extract folder name from webkitRelativePath
    const firstPath = files[0].webkitRelativePath || files[0].name;
    const folderName = firstPath.split('/')[0];

    folderPath.textContent = folderName;
    folderMonitor.classList.add('active');
    fileCount.textContent = `${files.length} images in folder`;
    startBtn.disabled = files.length === 0 || !classifier;

    // Check for resumable state
    const savedFiles = files.map(f => f.name);
    const canResume = results.some(r => savedFiles.includes(r.name));

    if (canResume && processedFiles.size > 0) {
        resumeBanner.classList.add('active');
        resumeCount.textContent = processedFiles.size;
    } else {
        resumeBanner.classList.remove('active');
    }

    const newCount = files.filter(f => !processedFiles.has(f.name)).length;
    monitorStatus.textContent = `${newCount} unprocessed`;
}

function clearFolder() {
    files = [];
    folderInput.value = '';
    folderMonitor.classList.remove('active');
    folderPath.textContent = '';
    monitorStatus.textContent = '';
    fileCount.textContent = 'No files selected';
    startBtn.disabled = true;
}

// Start the app
init();
