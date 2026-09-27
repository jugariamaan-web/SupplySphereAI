// components.js - SupplySphere AI shared navigation shell (sidebar + header).
// Design language: enterprise BI (Microsoft Fabric / Power BI / Azure Portal) -
// flat surfaces, hairline borders, restrained radii, a slim left-rail nav.

const NAV_ITEMS = [
    { href: "/", match: ["/", "/dashboard"], icon: "grid_view", label: "Dashboard" },
    { href: "/forecast", match: ["/forecast"], icon: "trending_up", label: "Demand Forecast" },
    { href: "/inventory", match: ["/inventory"], icon: "inventory_2", label: "Inventory Plan" },
    { href: "/supplier-risk", match: ["/supplier-risk"], icon: "warning", label: "Inventory Risk" },
    { href: "/copilot", match: ["/copilot"], icon: "auto_awesome", label: "SupplySphere Intelligence" },
    { href: "/reports", match: ["/reports"], icon: "assessment", label: "AI Reports" },
];

// A "skip navigation" link for keyboard/screen-reader users: without it,
// every page's first ~6 tab stops are sidebar nav items on every single
// navigation, before ever reaching the page's actual content. Visually
// hidden until focused (via .sr-only-focusable in theme.css).
function renderSkipLink() {
    return `<a href="#main-content-wrapper" class="sr-only-focusable">Skip to main content</a>`;
}

export function renderSidebar() {
    const currentPath = window.location.pathname;

    const items = NAV_ITEMS.map(item => {
        const isActive = item.match.includes(currentPath);
        return `
            <li>
                <a class="group flex items-center px-3 py-2.5 gap-3 rounded-md relative
                    transition-[background-color,color] duration-150 ease-out
                    ${isActive
                        ? 'bg-primary/[0.08] text-primary font-semibold'
                        : 'text-on-surface-variant hover:bg-surface-container hover:text-on-surface font-medium'}"
                   href="${item.href}" ${isActive ? 'aria-current="page"' : ''}>
                    <span class="absolute left-0 top-1.5 bottom-1.5 w-[3px] rounded-full bg-primary transition-opacity duration-150 ${isActive ? 'opacity-100' : 'opacity-0'}"></span>
                    <span class="material-symbols-outlined text-[19px] shrink-0 transition-transform duration-150 ease-out group-hover:scale-110 group-hover:-translate-y-px">${item.icon}</span>
                    <span class="font-body-sm text-[13.5px] leading-tight">${item.label}</span>
                </a>
            </li>`;
    }).join('');

    return `
    <nav class="h-screen w-60 fixed left-0 top-0 flex flex-col bg-surface-container-lowest border-r border-outline-variant z-50">
        <div class="h-16 flex items-center px-5 border-b border-outline-variant shrink-0">
            <div class="flex items-center gap-2.5">
                <div class="w-8 h-8 rounded-md bg-primary flex items-center justify-center text-on-primary shrink-0">
                    <span class="material-symbols-outlined text-[18px]">layers</span>
                </div>
                <div class="leading-tight">
                    <h1 class="font-title-md text-[14.5px] font-bold text-on-surface tracking-tight">SupplySphere AI</h1>
                    <p class="font-body-sm text-[10.5px] text-on-surface-variant tracking-wide uppercase">Inventory Intelligence</p>
                </div>
            </div>
        </div>
        <ul class="flex-1 flex flex-col gap-0.5 px-3 py-3 overflow-y-auto">
            ${items}
        </ul>
    </nav>
    `;
}

export function renderHeader(titlePlaceholder = "Search products, categories...") {
    return `
    <header class="flex items-center justify-between w-full h-16 px-6 sticky top-0 z-40 bg-surface-container-lowest/95 backdrop-blur-md border-b border-outline-variant">
        <div class="flex-1 flex items-center">
            <div class="relative w-72 sm:w-96 max-w-full">
                <span aria-hidden="true" class="material-symbols-outlined absolute left-3 top-1/2 -translate-y-1/2 text-on-surface-variant text-[18px] pointer-events-none">search</span>
                <input id="global-header-search" aria-label="${titlePlaceholder}" class="w-full pl-9 pr-3.5 py-[7px] bg-surface-container border border-transparent rounded-md text-[13px] focus:outline-none focus:bg-surface-container-lowest focus:border-primary focus:ring-1 focus:ring-primary transition-all duration-150 placeholder:text-on-surface-variant" placeholder="${titlePlaceholder}" type="text" />
            </div>
        </div>
        <div class="flex items-center gap-2.5">
            <div id="system-status-badge" class="px-2.5 py-1 bg-surface-container border border-outline-variant rounded-md text-[11.5px] font-semibold text-on-surface-variant flex items-center gap-1.5 transition-colors duration-200">
                <span class="w-1.5 h-1.5 rounded-full bg-slate-300"></span>
                <span>Checking system&hellip;</span>
            </div>
        </div>
    </header>
    `;
}

/**
 * Every DB query in the backend degrades gracefully (returns empty instead
 * of a 500) so one bad table doesn't break a whole page - but that also
 * means an unreachable/misconfigured database looks identical to "this
 * product has no data yet": every page just quietly shows 0s. This banner
 * turns that into an explicit, actionable message instead, using the
 * `database` block /api/health now returns (see backend/database.py's
 * check_connection()).
 */
function renderDbBanner(hint) {
    return `
    <div id="db-status-banner" class="mx-6 mt-4 px-4 py-3 bg-red-50 border border-red-200 rounded-md flex items-start gap-3 text-[12.5px] text-red-900">
        <span class="material-symbols-outlined text-red-600 text-[19px] shrink-0">database_off</span>
        <div>
            <p class="font-semibold">Database not connected - every page will show 0s/empty until this is fixed.</p>
            <p class="mt-0.5 text-red-800">${hint}</p>
        </div>
    </div>`;
}

async function checkSystemHealth() {
    const badge = document.getElementById('system-status-badge');
    try {
        const res = await fetch('/api/health');
        const data = await res.json();
        const dbConnected = data.database?.connected;

        if (badge) {
            badge.className = `px-2.5 py-1 rounded-md text-[11.5px] font-semibold flex items-center gap-1.5 border ${dbConnected ? 'bg-green-50 border-green-200 text-green-700' : 'bg-red-50 border-red-200 text-red-700'}`;
            badge.innerHTML = dbConnected
                ? '<span class="w-1.5 h-1.5 rounded-full bg-green-600 animate-pulse"></span><span>System Online</span>'
                : '<span class="w-1.5 h-1.5 rounded-full bg-red-500"></span><span>Database Offline</span>';
        }

        const wrapper = document.getElementById('main-content-wrapper');
        const existingBanner = document.getElementById('db-status-banner');
        if (!dbConnected && wrapper && !existingBanner) {
            wrapper.insertAdjacentHTML('afterbegin', renderDbBanner(data.database?.hint || 'Check your MySQL server and backend/.env credentials.'));
        } else if (dbConnected && existingBanner) {
            existingBanner.remove();
        }
    } catch {
        if (badge) {
            badge.className = 'px-2.5 py-1 bg-red-50 border border-red-200 rounded-md text-[11.5px] font-semibold text-red-700 flex items-center gap-1.5';
            badge.innerHTML = '<span class="w-1.5 h-1.5 rounded-full bg-red-500"></span><span>Backend Unreachable</span>';
        }
    }
}

export function initLayout(titlePlaceholder = "Search products, categories...") {
    if (document.getElementById('app-layout-injected')) return;

    const body = document.body;
    const mainContent = document.querySelector('main') || body.firstElementChild;

    if (!mainContent) return;

    body.className = "bg-background text-on-surface flex w-full min-h-screen";

    // Build the sidebar/header shell in a detached container first.
    // IMPORTANT: we then MOVE (appendChild) the real <main> node into it
    // instead of cloning it. cloneNode() + body.innerHTML= would silently
    // drop every event listener already attached to elements inside <main>
    // (e.g. a page's Send/Generate button wired up before this runs),
    // which was the cause of buttons appearing to do nothing.
    const shell = document.createElement('div');
    shell.innerHTML = `
        ${renderSkipLink()}
        ${renderSidebar()}
        <div class="ml-60 flex-1 flex flex-col min-h-screen w-[calc(100%-15rem)]">
            ${renderHeader(titlePlaceholder)}
            <div id="main-content-wrapper" tabindex="-1" class="flex-1 w-full flex flex-col min-h-0"></div>
        </div>
    `;

    shell.querySelector('#main-content-wrapper').appendChild(mainContent); // moves, doesn't clone

    body.innerHTML = "";
    while (shell.firstChild) {
        body.appendChild(shell.firstChild);
    }

    const marker = document.createElement('div');
    marker.id = 'app-layout-injected';
    marker.style.display = 'none';
    body.appendChild(marker);

    // Functional Global Header Search across active pages
    const headerSearchInput = document.getElementById('global-header-search');
    if (headerSearchInput) {
        headerSearchInput.addEventListener('input', (e) => {
            const query = e.target.value;

            // Forward input to active page search input element if present
            const pageSearchInput = document.getElementById('supplier-search') ||
                                     document.getElementById('inventory-search') ||
                                     document.getElementById('product-search');
            if (pageSearchInput) {
                pageSearchInput.value = query;
                pageSearchInput.dispatchEvent(new Event('input', { bubbles: true }));
            }

            // Broadcast custom search event for any subscriber page logic
            window.dispatchEvent(new CustomEvent('app-global-search', { detail: { query } }));
        });
    }

    checkSystemHealth();
}
