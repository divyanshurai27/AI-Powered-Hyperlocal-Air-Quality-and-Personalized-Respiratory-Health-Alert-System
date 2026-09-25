/* AeroGard web demo client. Talks only to the public /api/v1 endpoints. */
"use strict";

const API = "/api/v1";
const $ = (sel) => document.querySelector(sel);

const PLACES = [
  { name: "Koramangala", lat: 12.9352, lon: 77.6245 },
  { name: "Jayanagar", lat: 12.9250, lon: 77.5938 },
  { name: "Indiranagar", lat: 12.9719, lon: 77.6412 },
  { name: "Hebbal", lat: 13.0358, lon: 77.5970 },
  { name: "Peenya", lat: 13.0285, lon: 77.5197 },
  { name: "Electronic City", lat: 12.8452, lon: 77.6602 },
  { name: "Whitefield", lat: 12.9698, lon: 77.7500 },
];
const DEFAULT_OFFICE = { name: "MG Road office", lat: 12.9756, lon: 77.6066 };

const LABEL = { pm25: "PM<sub>2.5</sub>", pm10: "PM<sub>10</sub>", no2: "NO<sub>2</sub>", o3: "O<sub>3</sub>", so2: "SO<sub>2</sub>", co: "CO" };
const PLAIN = { pm25: "PM2.5", pm10: "PM10", no2: "NO2", o3: "O3", so2: "SO2", co: "CO" };
// NO2/SO2 absolute units on the source feed are unverified (see docs/data_and_methods.md).
const UNVERIFIED = new Set(["no2", "so2"]);

// India NAQI breakpoints (upper bound of each category, µg/m³; CO in mg/m³).
const NAQI = {
  pm25: [30, 60, 90, 120, 250],
  pm10: [50, 100, 250, 350, 430],
  o3: [50, 100, 168, 208, 748],
  co: [1, 2, 10, 17, 34],
};
const BANDS = [
  { name: "Good", color: "var(--good)", hex: "#00a65a" },
  { name: "Satisfactory", color: "var(--satisfactory)", hex: "#9acd32" },
  { name: "Moderate", color: "var(--moderate)", hex: "#f2c200" },
  { name: "Poor", color: "var(--poor)", hex: "#ff8c00" },
  { name: "Very poor", color: "var(--very-poor)", hex: "#e53935" },
  { name: "Severe", color: "var(--severe)", hex: "#8e1b1b" },
];
const LEVEL_COLOR = { good: "#00a65a", caution: "#8aa81f", limit: "#ff8c00", avoid: "#e53935" };
const LEVEL_NAME = { good: "Good", caution: "Take care", limit: "Limit outdoors", avoid: "Avoid outdoors" };
const ENV_COLOR = { indoor_home: "#4f8a73", indoor_other: "#7aa7c7", in_transit: "#e08a3c", outdoor: "#c9a227", unknown: "#9aa6a1" };
const ENV_NAME = { indoor_home: "Home", indoor_other: "Office", in_transit: "Commute", outdoor: "Outdoors", unknown: "Unknown" };

const state = {
  token: sessionStorage.getItem("ag_token"),
  email: sessionStorage.getItem("ag_email"),
  place: { ...PLACES[0], kind: "preset" },
  at: "2026-09-21T16:00:00Z",
  pollutant: "pm25",
  profile: null,
  stations: [],
  map: null,
  mapMode: "stations",
  city: { pollutant: "pm25", hour: 0, grid: null },
  layers: {},
  charts: {},
};

/* ---------------- API ---------------- */
async function api(path, { method = "GET", body, params } = {}) {
  const url = new URL(API + path, location.origin);
  Object.entries(params || {}).forEach(([k, v]) => v != null && url.searchParams.set(k, v));
  const res = await fetch(url, {
    method,
    headers: {
      "Content-Type": "application/json",
      ...(state.token ? { Authorization: `Bearer ${state.token}` } : {}),
    },
    body: body ? JSON.stringify(body) : undefined,
  });
  if (res.status === 204) return null;
  const data = await res.json().catch(() => ({}));
  if (res.status === 401 && state.token) signOut("Your session expired. Please sign in again.");
  if (!res.ok) {
    const d = data.detail || {};
    const err = new Error(d.message || `Request failed (${res.status})`);
    err.code = d.code;
    err.status = res.status;
    throw err;
  }
  return data;
}

const atParam = () => (state.at === "live" ? undefined : state.at);

/* ---------------- helpers ---------------- */
let toastTimer;
function toast(msg, isError = false) {
  const t = $("#toast");
  t.textContent = msg;
  t.className = "toast" + (isError ? " error" : "");
  t.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (t.hidden = true), 4200);
}

function bandIndex(pollutant, value) {
  const bp = NAQI[pollutant];
  if (!bp || value == null) return -1;
  const v = pollutant === "co" ? value / 1000 : value;
  const i = bp.findIndex((ub) => v <= ub);
  return i === -1 ? 5 : i;
}
const band = (p, v) => (bandIndex(p, v) < 0 ? null : BANDS[bandIndex(p, v)]);

const IST = { timeZone: "Asia/Kolkata" };
const fmtTime = (iso) => new Date(iso).toLocaleString("en-IN", { ...IST, day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" });
const fmtHour = (iso) => new Date(iso).toLocaleTimeString("en-IN", { ...IST, hour: "2-digit", minute: "2-digit" });
const fmtClock = (iso) => new Date(iso).toLocaleTimeString("en-IN", { ...IST, hour: "numeric", minute: "2-digit" }).replace(":00", "");
const istHour = (iso) => Number(new Date(iso).toLocaleString("en-GB", { ...IST, hour: "2-digit", hour12: false })) % 24;
const fmtWindow = (w) => `${fmtClock(w.start)} – ${fmtClock(w.end)}`;

/* ---------------- auth ---------------- */
let authMode = "login";
function showAuth() {
  $("#app-view").hidden = true;
  $("#auth-view").hidden = false;
}
function setAuthMode(mode) {
  authMode = mode;
  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t.dataset.mode === mode));
  $("#auth-submit").textContent = mode === "login" ? "Sign in" : "Create account";
  $("#auth-password").autocomplete = mode === "login" ? "current-password" : "new-password";
  $("#auth-error").textContent = "";
}
async function submitAuth(e) {
  e.preventDefault();
  const email = $("#auth-email").value.trim();
  const password = $("#auth-password").value;
  const btn = $("#auth-submit");
  $("#auth-error").textContent = "";
  btn.disabled = true;
  try {
    if (authMode === "register") {
      if (password.length < 8) throw new Error("Use at least 8 characters for the password.");
      await api("/auth/register", { method: "POST", body: { email, password } });
    }
    const tokens = await api("/auth/login", { method: "POST", body: { email, password } });
    state.token = tokens.access_token;
    state.email = email;
    sessionStorage.setItem("ag_token", state.token);
    sessionStorage.setItem("ag_email", email);
    await startApp();
  } catch (err) {
    $("#auth-error").textContent = err.message;
  } finally {
    btn.disabled = false;
  }
}
function signOut(message) {
  state.token = null;
  sessionStorage.removeItem("ag_token");
  sessionStorage.removeItem("ag_email");
  showAuth();
  if (message) $("#auth-error").textContent = message;
}

/* ---------------- app ---------------- */
async function startApp() {
  $("#auth-view").hidden = true;
  $("#app-view").hidden = false;
  $("#user-email").textContent = state.email || "";
  initMap();
  await Promise.all([loadStations(), loadProfile()]);
  // Each user opens on *their* area: saved home first, else the first preset.
  const home = savedPlace("home");
  if (home) state.place = home;
  renderLocationOptions();
  if (location.hash === "#city") state.mapMode = "city";
  await refreshAll();
  if (state.mapMode === "city") setMapMode("city");
}

async function refreshAll() {
  updateBanner();
  const jobs = [loadToday(), loadCurrent(), loadForecast(), loadExposure()];
  if (state.mapMode === "city") jobs.push(loadCity());
  await Promise.all(jobs);
}

function updateBanner() {
  const b = $("#banner");
  b.hidden = false;
  b.innerHTML =
    state.at === "live"
      ? "<b>Live mode.</b> The public CPCB feed on OpenAQ runs about 3 days behind, so live values usually show as <i>no fresh data</i>. That is deliberate: stale data is never shown as current."
      : `Showing the system as of <b>${fmtTime(state.at)} IST</b>, using real recorded data. Forecasts and guidance are computed as they would have been at that moment, with no later information.`;
}

/* ---------------- places ---------------- */
function savedPlace(kind) {
  const p = state.profile && state.profile[kind];
  return p ? { name: p.label, lat: p.latitude, lon: p.longitude, kind } : null;
}

function renderLocationOptions() {
  const sel = $("#loc-select");
  const opts = [];
  ["home", "work"].forEach((k) => {
    const p = savedPlace(k);
    if (p) opts.push(`<option value="${k}">${k === "home" ? "🏠" : "💼"} ${escapeHtml(p.name)}</option>`);
  });
  opts.push(...PLACES.map((p, i) => `<option value="p${i}">${p.name}</option>`));
  if (state.place.kind === "custom") opts.push(`<option value="custom">${escapeHtml(state.place.name)}</option>`);
  sel.innerHTML = opts.join("");
  sel.value = state.place.kind === "preset" ? `p${PLACES.findIndex((p) => p.name === state.place.name)}` : state.place.kind;
}

function setPlace(place) {
  state.place = place;
  renderLocationOptions();
  refreshAll();
}

async function saveAs(kind, place = state.place) {
  const label = kind === "home" ? `Home · ${place.name.replace(/^(Home|Office|Work) · /, "")}` : `Work · ${place.name.replace(/^(Home|Office|Work) · /, "")}`;
  try {
    state.profile = await api("/patients/me", {
      method: "PATCH",
      body: { [kind]: { label: label.slice(0, 80), latitude: place.lat, longitude: place.lon } },
    });
    renderProfilePlaces();
    state.place = savedPlace(kind);
    renderLocationOptions();
    toast(`Saved as your ${kind}. Guidance now opens here.`);
    refreshAll();
  } catch (err) {
    toast(err.message, true);
  }
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
}

/* ---------------- today for you ---------------- */
async function loadToday() {
  const body = $("#today-body");
  $("#today-place").textContent = state.place.name;
  body.innerHTML = `<span class="muted">Working out the best times today…</span>`;
  const params = { at: atParam() };
  if (state.place.kind === "home" || state.place.kind === "work") params.place = state.place.kind;
  else Object.assign(params, { lat: state.place.lat, lon: state.place.lon });
  try {
    const d = await api("/recommendations/today", { params });
    if (!d.available) {
      body.innerHTML = `<div class="notes"><b>No guidance right now.</b><span>${reasonText({ unavailable_reason: d.unavailable_reason })}</span></div>`;
      return;
    }
    const worstDiffers = d.worst_window && (!d.best_window || d.worst_window.start !== d.best_window.start);
    const windows = [
      d.best_window && `<div class="win best"><div class="k">Best time outdoors</div><div class="v">${fmtWindow(d.best_window)}</div></div>`,
      d.avoid_windows.length
        ? `<div class="win avoid"><div class="k">Avoid being outdoors</div><div class="v">${d.avoid_windows.map(fmtWindow).join(", ")}</div></div>`
        : worstDiffers && `<div class="win"><div class="k">Least good time</div><div class="v">${fmtWindow(d.worst_window)}</div></div>`,
    ].filter(Boolean);
    const strip = d.hours
      .map((h) => {
        const title = `${fmtHour(new Date(new Date(h.time) - 3600e3).toISOString())}–${fmtHour(h.time)}: ${LEVEL_NAME[h.level]} (${h.category}, ${PLAIN[h.dominant_pollutant]} ${h.values[h.dominant_pollutant]} µg/m³)`;
        return `<div class="h ${h.waking ? "" : "night"}" style="background:${LEVEL_COLOR[h.level]}" title="${escapeHtml(title)}"></div>`;
      })
      .join("");
    const labels = d.hours
      .filter((_, i) => i % 3 === 0)
      .map((h) => `<span>${fmtClock(new Date(new Date(h.time) - 3600e3).toISOString())}</span>`)
      .join("");
    const used = Object.keys(d.model_versions).map((p) => PLAIN[p]).join(", ");
    body.innerHTML = `
      <div class="today-top">
        <span class="level ${d.level}">${LEVEL_NAME[d.level]}</span>
        <span class="headline">${escapeHtml(d.headline)}</span>
      </div>
      <div class="windows">${windows.join("")}</div>
      <div><div class="strip">${strip}</div><div class="strip-labels">${labels}</div></div>
      <div class="notes">
        <span><b>Why:</b> ${PLAIN[d.worst_pollutant]} is forecast to reach <b>${d.worst_category}</b> on India’s NAQI. ${escapeHtml(d.health_note)}</span>
        <span><b>For you:</b> ${escapeHtml(d.personalisation_note)}${d.disease_type ? "" : " Add your condition in My profile for personalised thresholds."}</span>
        <span>Rule-based air-quality guidance from the ${used} forecast (${d.policy_version}). The personal risk model comes in Phase 4. ${escapeHtml(d.disclaimer)}</span>
      </div>`;
  } catch (err) {
    body.innerHTML = `<span class="muted">${escapeHtml(err.message)}</span>`;
  }
}

/* ---------------- current air ---------------- */
async function loadCurrent() {
  $("#hero-place").textContent = state.place.name;
  $("#hero-num").classList.add("skeleton");
  try {
    const d = await api("/air/current", { params: { lat: state.place.lat, lon: state.place.lon, at: atParam() } });
    const pm = d.pollutants.find((p) => p.pollutant === "pm25");
    $("#hero-num").classList.remove("skeleton");
    if (pm && pm.available) {
      $("#hero-num").textContent = pm.concentration.toFixed(0);
      const b = band("pm25", pm.concentration);
      $("#hero-band").textContent = b.name;
      $("#hero-band").style.background = b.color;
      $("#hero-meta").textContent = `${pm.method === "idw" ? `Blend of ${pm.contributions.length} stations` : "At a monitoring station"} · nearest ${pm.nearest_station_km.toFixed(1)} km · data ${Math.round(pm.data_age_minutes)} min old`;
    } else {
      $("#hero-num").textContent = "—";
      $("#hero-band").textContent = "Unavailable";
      $("#hero-band").style.background = "#9aa6a1";
      $("#hero-meta").textContent = pm ? reasonText(pm) : "";
    }
    renderPollutants(d.pollutants);
    renderContributions(pm);
  } catch (err) {
    $("#hero-num").classList.remove("skeleton");
    toast(err.message, true);
  }
}

function reasonText(p) {
  return (
    {
      no_fresh_data: "No fresh reading nearby (the public feed is about 3 days behind).",
      no_stations_in_radius: "No monitoring station within 10 km, so AeroGard won’t guess.",
      no_usable_data: "Nearby readings failed quality checks.",
      MODEL_NOT_AVAILABLE: "No forecast model is loaded.",
      PREDICTION_FAILED: "Nearby stations don’t have enough recent data to forecast from.",
    }[p.unavailable_reason] || p.unavailable_detail || "Unavailable"
  );
}

function renderPollutants(list) {
  const order = ["pm25", "pm10", "no2", "o3", "so2", "co"];
  $("#pollutants").innerHTML = order
    .map((k) => list.find((p) => p.pollutant === k))
    .filter(Boolean)
    .map((p) => {
      const b = p.available && !UNVERIFIED.has(p.pollutant) ? band(p.pollutant, p.concentration) : null;
      const val = p.available
        ? p.pollutant === "co"
          ? `${(p.concentration / 1000).toFixed(2)} <small>mg/m³</small>`
          : `${p.concentration.toFixed(1)} <small>µg/m³</small>`
        : `<small>unavailable</small>`;
      const note = UNVERIFIED.has(p.pollutant) && p.available ? `<div class="note">unit unverified</div>` : b ? `<div class="note">${b.name}</div>` : "";
      return `<div class="pchip">${b ? `<span class="dot" style="background:${b.color}"></span>` : ""}<div class="name">${LABEL[p.pollutant]}</div><div class="val">${val}</div>${note}</div>`;
    })
    .join("");
}

/* ---------------- map ---------------- */
function initMap() {
  if (state.map) return;
  state.map = L.map("map", { zoomControl: true }).setView([12.97, 77.6], 11);
  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", { maxZoom: 18, attribution: "© OpenStreetMap contributors" }).addTo(state.map);
  state.layers.city = L.layerGroup().addTo(state.map);
  state.layers.stations = L.layerGroup().addTo(state.map);
  state.layers.links = L.layerGroup().addTo(state.map);
  state.map.on("click", (e) => openSpotPopup(e.latlng));
  setTimeout(() => state.map.invalidateSize(), 50);
}

function openSpotPopup(latlng) {
  const spot = { name: `Spot ${latlng.lat.toFixed(3)}, ${latlng.lng.toFixed(3)}`, lat: latlng.lat, lon: latlng.lng, kind: "custom" };
  const el = document.createElement("div");
  el.innerHTML = `<b>${spot.name}</b><div class="popup-actions"><button data-a="check">Check air here</button><button data-a="home">Save as my home</button><button data-a="work">Save as my work</button></div>`;
  el.querySelector('[data-a="check"]').onclick = () => { state.map.closePopup(); setPlace(spot); };
  el.querySelector('[data-a="home"]').onclick = () => { state.map.closePopup(); saveAs("home", { ...spot, name: "Pinned spot" }); };
  el.querySelector('[data-a="work"]').onclick = () => { state.map.closePopup(); saveAs("work", { ...spot, name: "Pinned spot" }); };
  L.popup().setLatLng(latlng).setContent(el).openOn(state.map);
}

async function loadStations() {
  try {
    const all = await api("/air/stations");
    // Legacy station records from the old feed have no recent data; show active ones only.
    state.stations = all.filter((s) => s.last_observed_at && s.last_observed_at >= "2025-06-01");
    state.layers.stations.clearLayers();
    state.stations.forEach((s) => {
      L.circleMarker([s.latitude, s.longitude], { radius: 6, color: "#0b6e4f", weight: 2, fillColor: "#fff", fillOpacity: 1 })
        .bindTooltip(s.name.replace(/, Bengaluru.*$/, ""))
        .addTo(state.layers.stations);
    });
  } catch (err) {
    toast(err.message, true);
  }
}

function renderContributions(pm) {
  state.layers.links.clearLayers();
  const here = [state.place.lat, state.place.lon];
  L.marker(here).bindTooltip(state.place.name).addTo(state.layers.links);
  const box = $("#contrib");
  if (!pm || !pm.available) {
    box.innerHTML = `<span class="muted">${pm ? reasonText(pm) : ""}</span>`;
    if (state.mapMode === "stations") state.map.setView(here, 11);
    return;
  }
  const byId = Object.fromEntries(state.stations.map((s) => [s.id, s]));
  const pts = [here];
  pm.contributions.forEach((c) => {
    const s = byId[c.station_id];
    if (!s) return;
    pts.push([s.latitude, s.longitude]);
    if (state.mapMode === "stations") {
      L.polyline([here, [s.latitude, s.longitude]], { color: "#0b6e4f", weight: 1 + 6 * c.weight, opacity: 0.6 }).addTo(state.layers.links);
    }
  });
  if (state.mapMode === "stations") state.map.fitBounds(pts, { padding: [30, 30], maxZoom: 13 });
  box.innerHTML =
    `<div class="muted small">PM<sub>2.5</sub> here is a distance-weighted blend of:</div>` +
    pm.contributions
      .map((c) => {
        const s = byId[c.station_id];
        const name = s ? s.name.replace(/, Bengaluru.*$/, "") : c.station_id;
        return `<div class="row"><span>${name} · ${c.distance_km.toFixed(1)} km</span><span class="bar" style="width:${Math.max(8, c.weight * 120)}px"></span><span>${Math.round(c.weight * 100)}%</span></div>`;
      })
      .join("");
}

function setMapMode(mode) {
  state.mapMode = mode;
  document.querySelectorAll("#map-mode button").forEach((b) => b.classList.toggle("active", b.dataset.mode === mode));
  const city = mode === "city";
  $("#city-controls").hidden = !city;
  $("#city-legend").hidden = !city;
  $("#contrib").hidden = city;
  $("#map-title").textContent = city ? "Bengaluru air map" : "Monitoring stations";
  if (city) {
    state.layers.links.clearLayers();
    L.marker([state.place.lat, state.place.lon]).bindTooltip(state.place.name).addTo(state.layers.links);
    loadCity();
  } else {
    state.layers.city.clearLayers();
    loadCurrent();
  }
}

async function loadCity() {
  $("#city-hour-label").textContent = "Loading city grid…";
  try {
    state.city.grid = await api("/air/map", { params: { pollutant: state.city.pollutant, at: atParam() } });
    drawCity();
    const lats = state.city.grid.cells.map((c) => c.latitude);
    const lons = state.city.grid.cells.map((c) => c.longitude);
    if (lats.length) state.map.fitBounds([[Math.min(...lats), Math.min(...lons)], [Math.max(...lats), Math.max(...lons)]], { padding: [10, 10] });
  } catch (err) {
    $("#city-hour-label").textContent = "";
    toast(err.message, true);
  }
}

function drawCity() {
  const g = state.city.grid;
  state.layers.city.clearLayers();
  if (!g) return;
  const h = state.city.hour;
  const half = g.step_deg / 2;
  const p = g.pollutant;
  let shown = 0;
  g.cells.forEach((c) => {
    const v = h === 0 ? c.now : c.forecast ? c.forecast[h - 1] : null;
    if (v == null) return;
    shown++;
    const b = band(p, v);
    L.rectangle([[c.latitude - half, c.longitude - half], [c.latitude + half, c.longitude + half]], {
      stroke: false,
      fillColor: b ? b.hex : "#888",
      fillOpacity: 0.45,
      interactive: true,
    })
      .bindTooltip(`${PLAIN[p]} ${v.toFixed(1)} µg/m³ · ${b ? b.name : ""}`)
      .on("click", (e) => openSpotPopup(e.latlng))
      .addTo(state.layers.city);
  });
  const label =
    h === 0
      ? `Now · ${fmtTime(g.at)}`
      : g.origin
        ? `+${h} h · ${fmtTime(new Date(new Date(g.origin).getTime() + h * 3600e3).toISOString())}`
        : `+${h} h · no forecast model`;
  $("#city-hour-label").textContent = shown ? label : `${label} · no data`;
  $("#city-legend").innerHTML =
    BANDS.slice(0, 5).map((b) => `<span><i style="background:${b.hex}"></i>${b.name}</span>`).join("") +
    `<span><i style="background:transparent;border:1px dashed #9aa6a1"></i>No colour = no station within 10 km</span>`;
}

/* ---------------- forecast ---------------- */
async function loadForecast() {
  const meta = $("#forecast-meta");
  meta.textContent = "Loading forecast…";
  try {
    const d = await api("/air/forecast", { params: { lat: state.place.lat, lon: state.place.lon, pollutant: state.pollutant, at: atParam() } });
    if (!d.available) {
      drawForecast([], []);
      meta.textContent = `No forecast: ${reasonText(d)}`;
      return;
    }
    drawForecast(d.hours.map((h) => fmtHour(h.target_time)), d.hours.map((h) => h.concentration));
    const unver = UNVERIFIED.has(state.pollutant) ? " · absolute NO₂ level unverified" : "";
    meta.textContent = `Starts from the reading at ${fmtTime(d.origin)} IST · model ${d.model_version} (24 hourly models, LightGBM)${unver}`;
  } catch (err) {
    drawForecast([], []);
    meta.textContent = err.code === "MODEL_NOT_AVAILABLE" ? "No forecast model loaded for this pollutant." : err.message;
  }
}

function drawForecast(labels, values) {
  const max = Math.max(10, ...values) * 1.25;
  const bp = NAQI[state.pollutant];
  const cols = ["rgba(0,166,90,.08)", "rgba(154,205,50,.10)", "rgba(242,194,0,.12)", "rgba(255,140,0,.12)", "rgba(229,57,53,.12)"];
  const bands = !bp || UNVERIFIED.has(state.pollutant) ? [] : [0, ...bp].slice(0, -1).map((lo, i) => ({ lo, hi: bp[i], color: cols[i] })).filter((b) => b.lo < max);
  const bandPlugin = {
    id: "bands",
    beforeDraw(chart) {
      const { ctx, chartArea: a, scales: { y } } = chart;
      if (!a) return;
      bands.forEach((b) => {
        const top = y.getPixelForValue(Math.min(b.hi, y.max));
        ctx.fillStyle = b.color;
        ctx.fillRect(a.left, top, a.right - a.left, y.getPixelForValue(b.lo) - top);
      });
    },
  };
  if (state.charts.forecast) state.charts.forecast.destroy();
  state.charts.forecast = new Chart($("#forecast-chart"), {
    type: "line",
    data: { labels, datasets: [{ data: values, borderColor: "#0b6e4f", backgroundColor: "rgba(11,110,79,.12)", fill: true, tension: 0.3, pointRadius: 2, borderWidth: 2.5 }] },
    options: {
      maintainAspectRatio: false,
      animation: { duration: 300 },
      interaction: { mode: "index", intersect: false },
      scales: {
        y: { beginAtZero: true, suggestedMax: max, title: { display: true, text: "µg/m³" }, grid: { color: "#eef2f0" } },
        x: { grid: { display: false }, ticks: { maxTicksLimit: 8 } },
      },
      plugins: { legend: { display: false }, tooltip: { callbacks: { label: (c) => ` ${c.parsed.y.toFixed(1)} µg/m³` } } },
    },
    plugins: [bandPlugin],
  });
}

/* ---------------- exposure ---------------- */
function hourGrid(endIso, hours) {
  // AQ hours end at :30 UTC (whole IST hours). Latest hour end at or before `end`.
  const end = new Date(endIso);
  const anchor = new Date(end);
  anchor.setUTCMinutes(30, 0, 0);
  if (anchor > end) anchor.setUTCHours(anchor.getUTCHours() - 1);
  return Array.from({ length: hours }, (_, i) => new Date(anchor.getTime() - (hours - 1 - i) * 3600e3));
}

function scheduleFor(hourEnd, home, work) {
  const h = (istHour(hourEnd.toISOString()) + 23) % 24; // the hour this period covers
  const mid = { lat: (home.lat + work.lat) / 2, lon: (home.lon + work.lon) / 2 };
  if (h === 9 || h === 18) return { env: "in_transit", place: mid };
  if (h >= 10 && h < 18) return { env: "indoor_other", place: work };
  if (h === 19) return { env: "outdoor", place: home };
  return { env: "indoor_home", place: home };
}

async function simulateDay() {
  const btn = $("#simulate");
  btn.disabled = true;
  try {
    const profile = await api("/patients/me");
    if (profile.consent_status !== "granted") {
      toast("Tick the consent box in My profile and save, then simulate again.", true);
      return;
    }
    const end = state.at === "live" ? new Date().toISOString() : state.at;
    if (Date.now() - new Date(end).getTime() > 6.5 * 86400e3) {
      toast("Location history can only be recorded for the last 7 days. Switch “As of” to 21 Sep to simulate a day.", true);
      return;
    }
    const home = savedPlace("home") || state.place;
    const work = savedPlace("work") || DEFAULT_OFFICE;
    const fixes = hourGrid(end, 24).map((hourEnd) => {
      const s = scheduleFor(hourEnd, home, work);
      return { timestamp: new Date(hourEnd.getTime() - 20 * 60e3).toISOString(), latitude: s.place.lat, longitude: s.place.lon, activity_context: s.env };
    });
    const r = await api("/location", { method: "POST", body: { fixes } });
    toast(r.stored ? `Recorded a typical day: ${r.stored} location points (home, commute, office).` : "This day was already recorded; showing it again.");
    await loadExposure();
  } catch (err) {
    toast(err.message, true);
  } finally {
    btn.disabled = false;
  }
}

async function loadExposure() {
  const box = $("#exposure-summary");
  try {
    const d = await api("/exposure/history", { params: { pollutant: "pm25", hours: 24, at: atParam() } });
    const w24 = d.windows.find((w) => w.window_hours === 24);
    const covered = d.hours.filter((h) => h.exposure != null);
    if (!covered.length) {
      box.innerHTML = `<div class="stat" style="grid-column:1/-1"><div class="k">No location history for this period yet</div><div class="v" style="font-size:15px;font-weight:400">Press <b>Simulate my day</b> to record a typical day (home → commute → office → home).</div></div>`;
      drawExposure([]);
      return;
    }
    const ambientMean = covered.reduce((a, h) => a + h.ambient, 0) / covered.length;
    box.innerHTML = [
      ["24 h average exposure", `${w24.mean?.toFixed(1) ?? "—"} <small>µg/m³</small>`],
      ["Outdoor air average", `${ambientMean.toFixed(1)} <small>µg/m³</small>`],
      ["Peak hour", `${w24.maximum?.toFixed(1) ?? "—"} <small>µg/m³</small>`],
      ["Hours covered", `${Math.round(w24.coverage * 100)}%${w24.complete ? "" : " <small>(incomplete)</small>"}`],
    ].map(([k, v]) => `<div class="stat"><div class="k">${k}</div><div class="v">${v}</div></div>`).join("");
    drawExposure(d.hours);
  } catch (err) {
    if (err.code === "CONSENT_REQUIRED") {
      box.innerHTML = `<div class="stat" style="grid-column:1/-1"><div class="k">Consent needed</div><div class="v" style="font-size:15px;font-weight:400">Tick the consent box in <b>My profile</b> and save to enable exposure.</div></div>`;
      drawExposure([]);
    } else {
      toast(err.message, true);
    }
  }
}

function drawExposure(hours) {
  if (state.charts.exposure) state.charts.exposure.destroy();
  state.charts.exposure = new Chart($("#exposure-chart"), {
    data: {
      labels: hours.map((h) => fmtHour(h.hour)),
      datasets: [
        { type: "bar", label: "Your exposure", data: hours.map((h) => h.exposure), backgroundColor: hours.map((h) => ENV_COLOR[h.microenvironment] || ENV_COLOR.unknown), borderRadius: 4, order: 2 },
        { type: "line", label: "Outdoor air here", data: hours.map((h) => h.ambient), borderColor: "#16211d", borderDash: [5, 4], borderWidth: 1.5, pointRadius: 0, tension: 0.3, order: 1 },
      ],
    },
    options: {
      maintainAspectRatio: false,
      animation: { duration: 300 },
      interaction: { mode: "index", intersect: false },
      scales: {
        y: { beginAtZero: true, title: { display: true, text: "PM2.5 µg/m³" }, grid: { color: "#eef2f0" } },
        x: { grid: { display: false }, ticks: { maxTicksLimit: 8 } },
      },
      plugins: {
        legend: {
          labels: {
            generateLabels: () => [
              ...Object.entries(ENV_NAME).filter(([k]) => k !== "unknown").map(([k, n]) => ({ text: n, fillStyle: ENV_COLOR[k], strokeStyle: ENV_COLOR[k], lineWidth: 0 })),
              { text: "Outdoor air", strokeStyle: "#16211d", lineDash: [5, 4], lineWidth: 1.5, fillStyle: "transparent" },
            ],
          },
        },
        tooltip: {
          callbacks: {
            afterBody: (items) => {
              const h = hours[items[0].dataIndex];
              return h.exposure == null ? "No recent location" : `${ENV_NAME[h.microenvironment] || "Unknown"} × ${h.factor}`;
            },
          },
        },
      },
    },
  });
}

/* ---------------- profile ---------------- */
async function loadProfile() {
  try {
    state.profile = await api("/patients/me");
    const p = state.profile;
    const f = $("#profile-form");
    f.disease_type.value = p.disease_type || "";
    f.severity.value = p.severity || "";
    f.age.value = p.age ?? "";
    f.sex.value = p.sex || "";
    f.consent.checked = p.consent_status === "granted";
    $("#profile-status").textContent = p.profile_complete ? "Complete" : "Incomplete";
    renderProfilePlaces();
  } catch (err) {
    toast(err.message, true);
  }
}

function renderProfilePlaces() {
  ["home", "work"].forEach((k) => {
    const p = state.profile && state.profile[k];
    $(`#${k}-label`).textContent = p ? p.label : "Not set";
  });
}

async function saveProfile(e) {
  e.preventDefault();
  const f = e.target;
  const body = {
    disease_type: f.disease_type.value || null,
    severity: f.severity.value || null,
    age: f.age.value === "" ? null : Number(f.age.value),
    sex: f.sex.value || null,
    consent_status: f.consent.checked ? "granted" : "withdrawn",
  };
  try {
    state.profile = await api("/patients/me", { method: "PATCH", body });
    $("#profile-status").textContent = state.profile.profile_complete ? "Complete" : "Incomplete";
    toast("Profile saved. Guidance updated for your condition.");
    loadToday();
    loadExposure();
  } catch (err) {
    toast(err.message, true);
  }
}

/* ---------------- wiring ---------------- */
function init() {
  $("#loc-select").addEventListener("change", (e) => {
    const v = e.target.value;
    if (v === "home" || v === "work") setPlace(savedPlace(v));
    else if (v.startsWith("p")) setPlace({ ...PLACES[Number(v.slice(1))], kind: "preset" });
  });
  $("#time-select").addEventListener("change", (e) => {
    state.at = e.target.value;
    refreshAll();
  });
  $("#refresh").addEventListener("click", refreshAll);
  $("#logout").addEventListener("click", () => signOut());
  $("#simulate").addEventListener("click", simulateDay);
  $("#profile-form").addEventListener("submit", saveProfile);
  document.querySelectorAll("[data-set]").forEach((b) => b.addEventListener("click", () => saveAs(b.dataset.set)));
  document.querySelectorAll("#pollutant-seg button").forEach((b) =>
    b.addEventListener("click", () => {
      document.querySelectorAll("#pollutant-seg button").forEach((x) => x.classList.toggle("active", x === b));
      state.pollutant = b.dataset.p;
      loadForecast();
    }),
  );
  document.querySelectorAll("#map-mode button").forEach((b) => b.addEventListener("click", () => setMapMode(b.dataset.mode)));
  $("#city-pollutant").addEventListener("change", (e) => {
    state.city.pollutant = e.target.value;
    loadCity();
  });
  $("#city-hour").addEventListener("input", (e) => {
    state.city.hour = Number(e.target.value);
    drawCity();
  });
  document.querySelectorAll(".tab").forEach((t) => t.addEventListener("click", () => setAuthMode(t.dataset.mode)));
  $("#auth-form").addEventListener("submit", submitAuth);

  if (state.token) startApp().catch(() => signOut());
  else showAuth();
}

document.addEventListener("DOMContentLoaded", init);
