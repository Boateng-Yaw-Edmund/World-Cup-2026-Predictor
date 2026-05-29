const groups2026 = {
  A: ["Mexico", "South Africa", "Korea Republic", "Czech Republic"],
  B: ["Canada", "Bosnia and Herzegovina", "Qatar", "Switzerland"],
  C: ["Brazil", "Morocco", "Haiti", "Scotland"],
  D: ["USA", "Paraguay", "Australia", "Turkey"],
  E: ["Germany", "Curacao", "Cote d'Ivoire", "Ecuador"],
  F: ["Netherlands", "Japan", "Sweden", "Tunisia"],
  G: ["Belgium", "Egypt", "Iran", "New Zealand"],
  H: ["Spain", "Cabo Verde", "Saudi Arabia", "Uruguay"],
  I: ["France", "Senegal", "Norway", "Iraq"],
  J: ["Argentina", "Algeria", "Austria", "Jordan"],
  K: ["Portugal", "Uzbekistan", "Colombia", "DR Congo"],
  L: ["England", "Croatia", "Ghana", "Panama"],
};

const API_BASE = window.location.port === "8767" ? "http://127.0.0.1:8768" : "";
const STORAGE_KEY = "worldCup2026PredictorState";

const state = {
  results: null,
  search: "",
  runCount: 0,
  lastRun: null,
  seed: "-",
  backend: null,
  lastError: "",
  comparison: null,
  activeTab: "groups",
  expandedGroups: {},
};

const groupsGrid = document.querySelector("#groupsGrid");
const thirdsTable = document.querySelector("#thirdsTable");
const bracketGrid = document.querySelector("#bracketGrid");
const compareGrid = document.querySelector("#compareGrid");
const championName = document.querySelector("#championName");
const championPath = document.querySelector("#championPath");
const runCount = document.querySelector("#runCount");
const lastRun = document.querySelector("#lastRun");
const runSeed = document.querySelector("#runSeed");
const runStatus = document.querySelector("#runStatus");
const teamSearch = document.querySelector("#teamSearch");
const modelSelect = document.querySelector("#modelSelect");
const apiStatus = document.querySelector("#apiStatus");

async function simulate() {
  setRunStatus("Training...");
  const model = modelSelect.value;

  try {
    const response = await fetch(`${API_BASE}/api/simulate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model, include_group_matches: true }),
    });
    const payload = await response.json();

    if (!response.ok || !payload.ok) {
      throw new Error(payload.error || "Backend simulation failed.");
    }

    const simulation = payload.simulation;
    state.runCount += 1;
    state.lastRun = new Date();
    state.seed = simulation.model.toUpperCase();
    state.backend = simulation.status;
    state.lastError = "";
    state.results = {
      groupTables: simulation.groups,
      groupMatches: simulation.group_matches || {},
      thirds: simulation.thirds,
      bracket: simulation.bracket,
      champion: simulation.champion,
      source: "backend",
      model: simulation.model,
    };
    render();
    saveState();
    flashRunStatus();
  } catch (error) {
    state.lastError = error.message;
    setRunStatus("Setup needed", true);
    renderSetupState();
    await refreshBackendStatus();
  }
}

async function compareModels() {
  setRunStatus("Comparing...");

  try {
    const response = await fetch(`${API_BASE}/api/compare`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({}),
    });
    const payload = await response.json();

    if (!response.ok || !payload.ok) {
      throw new Error(payload.error || "Model comparison failed.");
    }

    state.comparison = payload.comparison;
    state.backend = payload.comparison.status;
    state.lastError = "";
    render();
    activateTab("compare");
    saveState();
    flashRunStatus();
  } catch (error) {
    state.lastError = error.message;
    setRunStatus("Setup needed", true);
    render();
    await refreshBackendStatus();
  }
}

function reset() {
  teamSearch.value = "";
  state.search = "";
  state.runCount = 0;
  state.lastRun = null;
  state.seed = "-";
  state.comparison = null;
  simulate();
}

function clearSavedState() {
  localStorage.removeItem(STORAGE_KEY);
  state.results = null;
  state.comparison = null;
  state.search = "";
  state.runCount = 0;
  state.lastRun = null;
  state.seed = "-";
  state.lastError = "";
  state.activeTab = "groups";
  teamSearch.value = "";
  activateTab("groups", { persist: false });
  render();
  flashRunStatus();
}

function renderGroups() {
  if (!state.results) {
    groupsGrid.innerHTML = setupMessage();
    return;
  }
  const { groupTables, groupMatches } = state.results;
  const query = state.search.trim().toLowerCase();

  groupsGrid.innerHTML = Object.entries(groupTables)
    .map(([group, table]) => {
      const hasSearch = query && table.some((row) => row.team.toLowerCase().includes(query));
      return `
        <article class="group-card ${query && !hasSearch ? "dimmed" : ""}">
          <h3>Group ${group}<span class="badge">${table[0].team}</span></h3>
          <table>
            <thead>
              <tr><th>#</th><th>Team</th><th>Pts</th><th>GD</th></tr>
            </thead>
            <tbody>
              ${table.map((row, index) => `
                <tr class="${query && row.team.toLowerCase().includes(query) ? "winner" : ""}">
                  <td><span class="seed">${index + 1}</span></td>
                  <td class="team-cell">${row.team}</td>
                  <td>${row.points}</td>
                  <td>${row.gd > 0 ? "+" : ""}${row.gd}</td>
                </tr>
              `).join("")}
            </tbody>
          </table>
          ${renderGroupMatches(group, groupMatches[group] || [])}
        </article>
      `;
    })
    .join("");
}

function renderGroupMatches(group, matches) {
  if (!matches.length) return "";
  const isExpanded = Boolean(state.expandedGroups[group]);

  return `
    <div class="group-fixtures">
      <button class="fixture-toggle" type="button" data-group="${group}" aria-expanded="${isExpanded}">
        <span>Fixtures</span>
        <strong>${isExpanded ? "Hide" : "Show"} 6 matches</strong>
      </button>
      <div class="fixture-list ${isExpanded ? "" : "hidden"}">
        ${matches.map((match) => `
          <article class="fixture">
            <div class="fixture-score">
              <span>${match.home}</span>
              <strong>${match.home_goals}-${match.away_goals}</strong>
              <span>${match.away}</span>
            </div>
            ${renderMatchDetails(match)}
          </article>
        `).join("")}
      </div>
    </div>
  `;
}

function renderThirds() {
  if (!state.results) {
    thirdsTable.innerHTML = "";
    return;
  }
  const { thirds } = state.results;
  thirdsTable.innerHTML = thirds
    .map((row, index) => `
      <tr>
        <td><span class="seed">${index + 1}</span></td>
        <td class="team-cell">${row.team}</td>
        <td>${row.group}</td>
        <td>${row.points}</td>
        <td>${row.gd > 0 ? "+" : ""}${row.gd}</td>
        <td class="${index < 8 ? "qualified" : "out"}">${index < 8 ? "Qualified" : "Eliminated"}</td>
      </tr>
    `)
    .join("");
}

function renderBracket() {
  if (!state.results) {
    bracketGrid.innerHTML = setupMessage();
    return;
  }
  const { bracket } = state.results;
  bracketGrid.innerHTML = bracket
    .map((round) => `
      <section class="round">
        <h3>${round.name}</h3>
        ${round.matches.map((match) => `
          <article class="match">
            <div class="match-row ${match.winner === match.home ? "winner" : ""}">
              <span>${match.home}</span><strong>${match.winner === match.home ? "W" : ""}</strong>
            </div>
            <div class="match-row ${match.winner === match.away ? "winner" : ""}">
              <span>${match.away}</span><strong>${match.winner === match.away ? "W" : ""}</strong>
            </div>
            ${renderMatchDetails(match)}
          </article>
        `).join("")}
      </section>
    `)
    .join("");
}

function renderCompare() {
  if (!state.comparison) {
    compareGrid.innerHTML = `
      <div class="empty-state">
        Click <strong>Compare Models</strong> to run Random Forest and XGBoost side by side.
      </div>
    `;
    return;
  }

  const comparison = state.comparison;
  const rf = comparison.models.rf;
  const xgb = comparison.models.xgb;
  const bracketDisagreements = comparison.bracket_disagreements || [];
  const groupDisagreements = comparison.group_disagreements || [];

  compareGrid.innerHTML = `
    <section class="compare-summary">
      ${renderModelSummary("Random Forest", rf)}
      ${renderModelSummary("XGBoost", xgb)}
      <article class="compare-card">
        <span class="label">Disagreements</span>
        <strong>${bracketDisagreements.length}</strong>
        <p>Knockout matches where the models pick different winners.</p>
      </article>
      <article class="compare-card">
        <span class="label">Group Differences</span>
        <strong>${groupDisagreements.length}</strong>
        <p>Groups where the projected top three order changes.</p>
      </article>
    </section>

    <section class="compare-section">
      <h3>Knockout Disagreements</h3>
      ${bracketDisagreements.length ? bracketDisagreements.map(renderBracketDisagreement).join("") : `
        <div class="empty-state compact-state">No knockout winner disagreements.</div>
      `}
    </section>

    <section class="compare-section">
      <h3>Group Table Disagreements</h3>
      ${groupDisagreements.length ? groupDisagreements.map(renderGroupDisagreement).join("") : `
        <div class="empty-state compact-state">No group top-three disagreements.</div>
      `}
    </section>
  `;
}

function renderModelSummary(label, model) {
  return `
    <article class="compare-card">
      <span class="label">${label}</span>
      <strong>${model.champion}</strong>
      <p>${model.final.home} vs ${model.final.away}</p>
    </article>
  `;
}

function renderBracketDisagreement(item) {
  return `
    <article class="disagreement">
      <div>
        <span class="label">${item.round} ${item.match}</span>
        <strong>${item.rf.home} vs ${item.rf.away}</strong>
      </div>
      <div class="split-picks">
        <span>RF: <strong>${item.rf.winner}</strong></span>
        <span>XGB: <strong>${item.xgb.winner}</strong></span>
      </div>
    </article>
  `;
}

function renderGroupDisagreement(item) {
  return `
    <article class="disagreement">
      <div>
        <span class="label">Group ${item.group}</span>
        <strong>${item.rf_winner} vs ${item.xgb_winner}</strong>
      </div>
      <div class="split-picks">
        <span>RF top 3: <strong>${item.rf_top_three.join(", ")}</strong></span>
        <span>XGB top 3: <strong>${item.xgb_top_three.join(", ")}</strong></span>
      </div>
    </article>
  `;
}

function renderMatchDetails(match) {
  const prediction = match.prediction;
  if (!prediction) return "";

  const probabilities = prediction.probabilities || {};
  const features = prediction.features || {};
  const eloDiff = Number(features.elo_diff || 0);
  const formDiff = Number(features.home_form || 0) - Number(features.away_form || 0);

  return `
    <div class="match-probs" aria-label="Prediction probabilities">
      <div>
        <span>${shortTeam(match.home)}</span>
        <strong>${formatPercent(probabilities.home_win)}</strong>
      </div>
      <div>
        <span>Draw</span>
        <strong>${formatPercent(probabilities.draw)}</strong>
      </div>
      <div>
        <span>${shortTeam(match.away)}</span>
        <strong>${formatPercent(probabilities.away_win)}</strong>
      </div>
    </div>
    <dl class="match-factors">
      <div>
        <dt>Elo diff</dt>
        <dd>${formatSigned(eloDiff)}</dd>
      </div>
      <div>
        <dt>Form diff</dt>
        <dd>${formatSigned(formDiff, 1)}</dd>
      </div>
    </dl>
  `;
}

function formatPercent(value) {
  const numeric = Number(value || 0);
  return `${Math.round(numeric * 100)}%`;
}

function formatSigned(value, digits = 0) {
  const numeric = Number(value || 0);
  const formatted = numeric.toFixed(digits);
  return numeric > 0 ? `+${formatted}` : formatted;
}

function shortTeam(team) {
  return team.length > 12 ? `${team.slice(0, 11)}.` : team;
}

function renderChampion() {
  if (!state.results) {
    championName.textContent = "-";
    championPath.textContent = state.lastError || "Start the backend API, install ML dependencies, and add the data files.";
    runCount.textContent = state.runCount;
    lastRun.textContent = "-";
    runSeed.textContent = state.seed;
    return;
  }
  const { champion, bracket } = state.results;
  const final = bracket[bracket.length - 1].matches[0];
  championName.textContent = champion;
  championPath.textContent = `${final.home} vs ${final.away} in the final`;
  runCount.textContent = state.runCount;
  lastRun.textContent = state.lastRun
    ? new Date(state.lastRun).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" })
    : "-";
  runSeed.textContent = state.seed;
}

function render() {
  renderChampion();
  renderGroups();
  bindGroupFixtureToggles();
  renderThirds();
  renderBracket();
  renderCompare();
  renderApiStatus();
}

function bindGroupFixtureToggles() {
  document.querySelectorAll(".fixture-toggle").forEach((button) => {
    button.addEventListener("click", () => {
      const group = button.dataset.group;
      state.expandedGroups[group] = !state.expandedGroups[group];
      renderGroups();
      bindGroupFixtureToggles();
      saveState();
    });
  });
}

function flashRunStatus() {
  setRunStatus("Updated");
  window.clearTimeout(flashRunStatus.timer);
  flashRunStatus.timer = window.setTimeout(() => {
    setRunStatus("Ready");
  }, 1200);
}

function setRunStatus(text, isWarning = false) {
  runStatus.textContent = text;
  runStatus.classList.toggle("active", text !== "Ready" && !isWarning);
  runStatus.classList.toggle("warning", isWarning);
}

function setupMessage() {
  if (state.backend?.ready && !state.lastError) {
    const cacheReady = state.backend.cache?.[modelSelect.value]?.valid;
    return `
      <div class="empty-state">
        Backend is ready${cacheReady ? " and the selected model cache is valid" : ""}.
        Click <strong>Run Simulation</strong> to generate the group tables and knockout bracket.
      </div>
    `;
  }

  return `
    <div class="empty-state">
      The UI is connected to the backend API now. To run the real model, start
      <strong>backend/server.py</strong>, install the packages in <strong>requirements.txt</strong>,
      and add <strong>data/results.csv</strong>, <strong>data/shootouts.csv</strong>, and
      <strong>data/eloratings.csv</strong>.
      ${state.lastError ? `<br><br><strong>Current backend response:</strong> ${state.lastError}` : ""}
    </div>
  `;
}

function renderSetupState() {
  state.results = null;
  render();
}

function renderApiStatus() {
  if (state.lastError) {
    apiStatus.textContent = state.lastError;
    return;
  }
  if (!state.backend) {
    apiStatus.textContent = "Backend not checked yet.";
    return;
  }
  const missingData = Object.entries(state.backend.data || {})
    .filter((entry) => !entry[1])
    .map((entry) => entry[0]);
  const missingDeps = Object.entries(state.backend.dependencies || {})
    .filter((entry) => !entry[1])
    .map((entry) => entry[0]);

  if (state.backend.ready) {
    const selectedModel = modelSelect.value;
    const selectedCache = state.backend.cache?.[selectedModel];
    const models = state.backend.available_models && state.backend.available_models.length
      ? state.backend.available_models.join(", ")
      : selectedCache?.valid
        ? `${selectedModel.toUpperCase()} cached`
        : "training on first run";
    apiStatus.textContent = `Ready. Models: ${models}.`;
    return;
  }

  const parts = [];
  if (missingData.length) parts.push(`Missing data: ${missingData.join(", ")}`);
  if (missingDeps.length) parts.push(`Missing packages: ${missingDeps.join(", ")}`);
  apiStatus.textContent = parts.join(". ") || "Backend is running but not ready.";
}

async function refreshBackendStatus() {
  try {
    const response = await fetch(`${API_BASE}/api/status`);
    const payload = await response.json();
    if (!response.ok || !payload.ok) throw new Error(payload.error || "Backend status failed.");
    state.backend = payload;
    if (!payload.ready && !state.lastError) {
      state.lastError = "Backend is running, but model setup is incomplete.";
    }
  } catch (error) {
    state.backend = null;
    state.lastError = "Backend API is not running on http://127.0.0.1:8768.";
  }
  render();
}

function activateTab(tabName, options = { persist: true }) {
  state.activeTab = tabName;
  document.querySelectorAll(".tab").forEach((item) => {
    item.classList.toggle("active", item.dataset.tab === tabName);
  });
  document.querySelectorAll(".view").forEach((view) => view.classList.remove("active"));
  document.querySelector(`#${tabName}View`).classList.add("active");
  if (options.persist) saveState();
}

document.querySelectorAll(".tab").forEach((tab) => {
  tab.addEventListener("click", () => activateTab(tab.dataset.tab));
});

teamSearch.addEventListener("input", (event) => {
  state.search = event.target.value;
  renderGroups();
  saveState();
});

document.querySelector("#simulateBtn").addEventListener("click", simulate);
document.querySelector("#compareBtn").addEventListener("click", compareModels);
document.querySelector("#resetBtn").addEventListener("click", reset);
document.querySelector("#clearSavedBtn").addEventListener("click", clearSavedState);

modelSelect.addEventListener("change", () => {
  saveState();
  renderApiStatus();
});

function saveState() {
  const persisted = {
    results: state.results,
    comparison: state.comparison,
    search: state.search,
    runCount: state.runCount,
    lastRun: state.lastRun ? new Date(state.lastRun).toISOString() : null,
    seed: state.seed,
    activeTab: state.activeTab,
    selectedModel: modelSelect.value,
    expandedGroups: state.expandedGroups,
  };
  localStorage.setItem(STORAGE_KEY, JSON.stringify(persisted));
}

function loadState() {
  const raw = localStorage.getItem(STORAGE_KEY);
  if (!raw) return;

  try {
    const persisted = JSON.parse(raw);
    state.results = persisted.results || null;
    state.comparison = persisted.comparison || null;
    state.search = persisted.search || "";
    state.runCount = Number(persisted.runCount || 0);
    state.lastRun = persisted.lastRun || null;
    state.seed = persisted.seed || "-";
    state.activeTab = persisted.activeTab || "groups";
    state.expandedGroups = persisted.expandedGroups || {};
    if (persisted.selectedModel) modelSelect.value = persisted.selectedModel;
    teamSearch.value = state.search;
  } catch (error) {
    localStorage.removeItem(STORAGE_KEY);
  }
}

loadState();
render();
activateTab(state.activeTab, { persist: false });
refreshBackendStatus();
