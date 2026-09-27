// api.js - SupplySphere AI API Client & Utilities

const API_BASE = "/api";
const cache = {};
const CACHE_DURATION_MS = 15000; // 15 seconds cache for fast response

// Toast Notification System
export function showToast(message, type = 'info') {
    let container = document.getElementById('toast-container');
    if (!container) {
        container = document.createElement('div');
        container.id = 'toast-container';
        container.className = 'fixed bottom-5 right-5 z-50 flex flex-col gap-2 pointer-events-none';
        document.body.appendChild(container);
    }

    const toast = document.createElement('div');
    const colorClasses = {
        success: 'bg-green-700 text-white border-green-800',
        error: 'bg-red-700 text-white border-red-800',
        warning: 'bg-amber-600 text-white border-amber-700',
        info: 'bg-primary text-white border-primary-container'
    }[type] || 'bg-primary text-white border-primary-container';

    toast.className = `pointer-events-auto px-4 py-3 rounded-xl shadow-lg border text-xs font-medium flex items-center gap-2 transform transition-all duration-300 translate-y-4 opacity-0 ${colorClasses}`;
    
    const icon = {
        success: 'check_circle',
        error: 'error',
        warning: 'warning',
        info: 'info'
    }[type] || 'info';

    toast.innerHTML = `
        <span class="material-symbols-outlined text-sm">${icon}</span>
        <span>${escapeHtml(message)}</span>
    `;

    container.appendChild(toast);

    requestAnimationFrame(() => {
        toast.classList.remove('translate-y-4', 'opacity-0');
    });

    setTimeout(() => {
        toast.classList.add('opacity-0', 'translate-y-2');
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

async function fetchWithCache(url, cacheKey, forceRefresh = false) {
    if (!forceRefresh && cache[cacheKey] && (Date.now() - cache[cacheKey].timestamp < CACHE_DURATION_MS)) {
        return cache[cacheKey].data;
    }
    try {
        const res = await fetch(url);
        if (!res.ok) {
            const errData = await res.json().catch(() => ({}));
            throw new Error(errData.detail || `HTTP ${res.status}: Failed to load data`);
        }
        const data = await res.json();
        cache[cacheKey] = { data, timestamp: Date.now() };
        return data;
    } catch (err) {
        console.error(`API Error on ${url}:`, err);
        showToast(err.message || "Network error. Please check backend connection.", 'error');
        throw err;
    }
}

export async function fetchDashboard(forceRefresh = false) {
    return await fetchWithCache(`${API_BASE}/dashboard`, 'dashboard', forceRefresh);
}

export async function fetchForecast(forceRefresh = false) {
    return await fetchWithCache(`${API_BASE}/forecast`, 'forecast', forceRefresh);
}

export async function fetchAllRisk(forceRefresh = false) {
    return await fetchWithCache(`${API_BASE}/risk`, 'risk', forceRefresh);
}

export async function fetchProducts() {
    return await fetchWithCache(`${API_BASE}/products`, 'products');
}

export async function fetchInventoryPlan(productId, currentStock = null) {
    if (!productId) {
        showToast("Please select a product first.", "warning");
        return { error: "Please select a product." };
    }
    
    const bodyObj = { product_id: productId };
    if (currentStock !== null && currentStock !== undefined && !isNaN(currentStock)) {
        bodyObj.current_stock = parseFloat(currentStock);
    }

    try {
        const res = await fetch(`${API_BASE}/inventory`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(bodyObj)
        });
        if (!res.ok) {
            const errData = await res.json().catch(() => ({}));
            throw new Error(errData.detail || "Failed to calculate inventory plan");
        }
        return await res.json();
    } catch (err) {
        showToast(err.message, 'error');
        throw err;
    }
}

export async function askCopilot(message) {
    try {
        const res = await fetch(`${API_BASE}/copilot`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ message })
        });
        if (!res.ok) {
            const errData = await res.json().catch(() => ({}));
            throw new Error(errData.detail || "Copilot request failed");
        }
        return await res.json();
    } catch (err) {
        showToast(err.message, 'error');
        throw err;
    }
}

export async function generateReport(reportType) {
    try {
        const res = await fetch(`${API_BASE}/report`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ report_type: reportType })
        });
        if (!res.ok) {
            const errData = await res.json().catch(() => ({}));
            throw new Error(errData.detail || "Report generation failed");
        }
        return await res.json();
    } catch (err) {
        showToast(err.message, 'error');
        throw err;
    }
}

// Format numbers using Indian Locale (e.g. 100000 -> 1,00,000)
export function formatNumber(num) {
    if (num === null || num === undefined || isNaN(num)) return "--";
    return Number(num).toLocaleString('en-IN');
}

// Shared status -> fs-badge class mapping (was duplicated inline in
// inventory.html and supplier-risk.html). Keeping it in one place
// means the color mapping only ever needs to change once.
export function inventoryHealthBadgeClass(status) {
    if (status === 'Critical') return 'fs-badge-critical';
    if (status === 'Warning') return 'fs-badge-warning';
    return 'fs-badge-healthy';
}
export function riskLevelBadgeClass(level) {
    if (level === 'High') return 'fs-badge-high';
    if (level === 'Medium') return 'fs-badge-medium';
    return 'fs-badge-low';
}

// Animates a KPI number counting up from 0 (or from its current text)
// to targetValue on page load. Purely cosmetic - the final displayed
// value is always exactly targetValue, formatted with formatFn.
export function animateNumber(el, targetValue, { duration = 600, formatFn = formatNumber } = {}) {
    if (!el || targetValue === null || targetValue === undefined || isNaN(targetValue)) {
        if (el) el.textContent = formatFn(targetValue);
        return;
    }
    const start = performance.now();
    const from = 0;
    function tick(now) {
        const p = Math.min(1, (now - start) / duration);
        const eased = 1 - Math.pow(1 - p, 3); // ease-out cubic
        const current = from + (targetValue - from) * eased;
        el.textContent = formatFn(p < 1 ? current : targetValue);
        if (p < 1) requestAnimationFrame(tick);
    }
    requestAnimationFrame(tick);
}

// Renders model/report markdown text to sanitized HTML. marked.js and
// DOMPurify are loaded as CDN globals (window.marked / window.DOMPurify)
// by any page that uses this - both are configured to escape raw HTML
// from the source text, then every render is additionally passed
// through DOMPurify before being injected into the page.
// (Previously copy-pasted identically into copilot.html and reports.html.)
if (window.marked) {
    marked.setOptions({ breaks: true, gfm: true });
}
export function renderMarkdown(text) {
    const raw = window.marked ? marked.parse(text) : text;
    return window.DOMPurify ? DOMPurify.sanitize(raw) : raw;
}

