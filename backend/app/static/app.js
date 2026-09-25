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
const OFFICE = { name: "MG Road office", lat: 12.9756, lon: 77.6066 };

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
  { name: "Good", color: "var(--good)" },
  { name: "Satisfactory", color: "var(--satisfactory)" },
  { name: "Moderate", color: "var(--moderate)" },
  { name: "Poor", color: "var(--poor)" },
  { name: "Very poor", color: "var(--very-poor)" },
  { name: "Severe", color: "var(--severe)" },
];
const ENV_COLOR = { indoor_home: "#4f8a73", indoor_other: "#7aa7c7", in_transit: "#e08a3c", outdoor: "#c9a227", unknown: "#9aa6a1" };
const ENV_NAME = { indoor_home: "Home", indoor_other: "Office", in_transit: "Commute", outdoor: "Outdoors", unknown: "Unknown" };

const state = {
  token: sessionStorage.getItem("ag_token"),
  email: sessionStorage.getItem("ag_email"),
  place: PLACES[0],
  at: "2026-09-21T16:00:00Z",
  pollutant: "pm25",
  stations: [],
  map: null,
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
  if (res.status === 401 && state.token) {
    signOut("Your session expired. Please sign in again.");
  }
  if (!res.ok) {
    const d = data.detail || {};
    const err = new Error(d.message || `Request failed (${res.status})`);
    err.code = d.code;
    err.status = res.status;
    throw err;
  }
  return data;
}

function atParam() {
  return state.at === "live" ? undefined : state.at;
}

/* ---------------- UI helpers ---------------- */
let toastTimer;
function toast(msg, isError = false) {
  const t = $("#toast");
  t.textContent = msg;
  t.className = "toast" + (isError ? " error" : "");
  t.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (t.hidden = true), 4200);
}

function band(pollutant, value) {
  const bp = NAQI[pollutant];
  if (!bp || value == null) return null;
  const v = pollutant === "co" ? value / 1000 : value;
  const i = bp.findIndex((ub) => v <= ub);
  return BANDS[i === -1 ? 5 : i];
}

function fmtTime(iso) {
  return new Date(iso).toLocaleString("en-IN", {
    timeZone: "Asia/Kolkata", day: "numeric", month: "short", hour: "2-digit", minute: "2-digit",
  });
}
function fmtHour(iso) {
  return new Date(iso).toLocaleTimeString("en-IN", { timeZone: "Asia/Kolkata", hour: "2-digit", minute: "2-digit" });
}

/* ---------------- Auth ---------------- */
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

/* ---------------- App ---------------- */
async function startApp() {
  $("#auth-view").hidden = true;
  $("#app-view").hidden = false;
  $("#user-email").textContent = state.email || "";
  initMap();
  await Promise.all([loadStations(), loadProfile()]);
  await refreshAll();
}

async function refreshAll() {
  updateBanner();
  await Promise.all([loadCurrent(), loadForecast(), loadExposure()]);
}

function updateBanner() {
  const b = $("#banner");
  if (state.at === "live") {
    b.innerHTML = "<b>Live mode.</b> The public CPCB feed on OpenAQ runs about 3 days behind, so live values usually show as <i>no fresh data</i>. That is deliberate: stale data is never shown as current.";
    b.hidden = false;
  } else {
    b.innerHTML = `Showing the system as of <b>${fmtTime(state.at)} IST</b>, using real recorded data. Forecast and exposure are computed as they would have been at that moment, with no later information.`;
    b.hidden = false;
  }
}

/* ---- current air ---- */
async function loadCurrent() {
  $("#hero-place").textContent = state.place.name;
  $("#hero-num").classList.add("skeleton");
  try {
    const d = await api("/air/current", { params: { lat: state.place.lat, lon: state.place.lon, at: atParam() } });
    const byP = Object.fromEntries(d.pollutants.map((p) => [p.pollutant, p]));
    const pm = byP.pm25;
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
  return {
    no_fresh_data: "No fresh reading nearby (the public feed is behind).",
    no_stations_in_radius: "No monitoring station within 10 km, so AeroGard won’t guess.",
    no_usable_data: "Nearby readings failed quality checks.",
  }[p.unavailable_reason] || p.unavailable_detail || "Unavailable";
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

/* ---- map ---- */
function initMap() {
  if (state.map) return;
  state.map = L.map("map", { zoomControl: true }).setView([12.97, 77.6], 11);
  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 18,
    attribution: "© OpenStreetMap contributors",
  }).addTo(state.map);
  state.layers.stations = L.layerGroup().addTo(state.map);
  state.layers.links = L.layerGroup().addTo(state.map);
  state.map.on("click", (e) => {
    setPlace({ name: `Custom spot (${e.latlng.lat.toFixed(3)}, ${e.latlng.lng.toFixed(3)})`, lat: e.latlng.lat, lon: e.latlng.lng });
  });
  setTimeout(() => state.map.invalidateSize(), 50);
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
    state.map.setView(here, 11);
    return;
  }
  const byId = Object.fromEntries(state.stations.map((s) => [s.id, s]));
  const pts = [here];
  pm.contributions.forEach((c) => {
    const s = byId[c.station_id];
    if (!s) return;
    pts.push([s.latitude, s.longitude]);
    L.polyline([here, [s.latitude, s.longitude]], { color: "#0b6e4f", weight: 1 + 6 * c.weight, opacity: 0.6 }).addTo(state.layers.links);
  });
  state.map.fitBounds(pts, { padding: [30, 30], maxZoom: 13 });
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

/* ---- forecast ---- */
async function loadForecast() {
  const meta = $("#forecast-meta");
  meta.textContent = "Loading forecast…";
  try {
    const d = await api("/air/forecast", {
      params: { lat: state.place.lat, lon: state.place.lon, pollutant: state.pollutant, at: atParam() },
    });
    if (!d.available) {
      drawForecast([], []);
      meta.textContent = `No forecast: ${reasonText(d)}`;
      return;
    }
    const labels = d.hours.map((h) => fmtHour(h.target_time));
    const values = d.hours.map((h) => h.concentration);
    drawForecast(labels, values);
    const unver = UNVERIFIED.has(state.pollutant) ? " · absolute NO₂ level unverified" : "";
    meta.textContent = `Starts from the reading at ${fmtTime(d.origin)} IST · model ${d.model_version} (24 hourly models, LightGBM)${unver}`;
  } catch (err) {
    drawForecast([], []);
    meta.textContent = err.code === "MODEL_NOT_AVAILABLE" ? "No forecast model loaded for this pollutant." : err.message;
  }
}

function bandBackgrounds(pollutant, max) {
  // Coloured NAQI bands behind the forecast line (PM and O3 only).
  const bp = NAQI[pollutant];
  if (!bp || UNVERIFIED.has(pollutant)) return [];
  const edges = [0, ...bp];
  const cols = ["rgba(0,166,90,.08)", "rgba(154,205,50,.10)", "rgba(242,194,0,.12)", "rgba(255,140,0,.12)", "rgba(229,57,53,.12)"];
  return edges.slice(0, -1).map((lo, i) => ({ lo, hi: edges[i + 1], color: cols[i] })).filter((b) => b.lo < max);
}

function drawForecast(labels, values) {
  const ctx = $("#forecast-chart");
  const max = Math.max(10, ...values) * 1.25;
  const bands = bandBackgrounds(state.pollutant, max);
  const bandPlugin = {
    id: "bands",
    beforeDraw(chart) {
      const { ctx: c, chartArea: a, scales: { y } } = chart;
      if (!a) return;
      bands.forEach((b) => {
        const top = y.getPixelForValue(Math.min(b.hi, y.max));
        const bot = y.getPixelForValue(b.lo);
        c.fillStyle = b.color;
        c.fillRect(a.left, top, a.right - a.left, bot - top);
      });
    },
  };
  if (state.charts.forecast) state.charts.forecast.destroy();
  state.charts.forecast = new Chart(ctx, {
    type: "line",
    data: {
      labels,
      datasets: [{
        label: `${PLAIN[state.pollutant]} forecast (µg/m³)`,
        data: values,
        borderColor: "#0b6e4f",
        backgroundColor: "rgba(11,110,79,.12)",
        fill: true,
        tension: 0.3,
        pointRadius: 2,
        borderWidth: 2.5,
      }],
    },
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

/* ---- exposure ---- */
function hourGrid(endIso, hours) {
  // AQ hours end at :30 UTC (whole IST hours). Latest hour end at or before `end`.
  const end = new Date(endIso);
  const anchor = new Date(end);
  anchor.setUTCMinutes(30, 0, 0);
  if (anchor > end) anchor.setUTCHours(anchor.getUTCHours() - 1);
  return Array.from({ length: hours }, (_, i) => new Date(anchor.getTime() - (hours - 1 - i) * 3600e3));
}

function scheduleFor(date) {
  const h = (date.getUTCHours() + 5 + (date.getUTCMinutes() + 30 >= 60 ? 1 : 0)) % 24; // IST hour
  if (h >= 9 && h < 10) return { env: "in_transit", place: mid(state.place, OFFICE) };
  if (h >= 10 && h < 18) return { env: "indoor_other", place: OFFICE };
  if (h >= 18 && h < 19) return { env: "in_transit", place: mid(state.place, OFFICE) };
  if (h >= 19 && h < 20) return { env: "outdoor", place: state.place };
  return { env: "indoor_home", place: state.place };
}
const mid = (a, b) => ({ lat: (a.lat + b.lat) / 2, lon: (a.lon + b.lon) / 2 });

async function simulateDay() {
  const btn = $("#simulate");
  btn.disabled = true;
  try {
    const profile = await api("/patients/me");
    if (profile.consent_status !== "granted") {
      toast("Tick the consent box in My profile and save, then simulate again.", true);
      $("#profile-form [name=consent]").focus();
      return;
    }
    const end = state.at === "live" ? new Date().toISOString() : state.at;
    const fixes = hourGrid(end, 24).map((hourEnd) => {
      const s = scheduleFor(hourEnd);
      return {
        timestamp: new Date(hourEnd.getTime() - 20 * 60e3).toISOString(),
        latitude: s.place.lat,
        longitude: s.place.lon,
        activity_context: s.env,
      };
    });
    const r = await api("/location", { method: "POST", body: { fixes } });
    toast(r.stored ? `Recorded a typical day: ${r.stored} location points (home, commute, office).`
                   : "This day was already recorded; showing it again.");
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
    const hrs = d.hours;
    const covered = hrs.filter((h) => h.exposure != null);
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
    drawExposure(hrs);
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
  const ctx = $("#exposure-chart");
  if (state.charts.exposure) state.charts.exposure.destroy();
  state.charts.exposure = new Chart(ctx, {
    data: {
      labels: hours.map((h) => fmtHour(h.hour)),
      datasets: [
        {
          type: "bar",
          label: "Your exposure",
          data: hours.map((h) => h.exposure),
          backgroundColor: hours.map((h) => ENV_COLOR[h.microenvironment] || ENV_COLOR.unknown),
          borderRadius: 4,
          order: 2,
        },
        {
          type: "line",
          label: "Outdoor air here",
          data: hours.map((h) => h.ambient),
          borderColor: "#16211d",
          borderDash: [5, 4],
          borderWidth: 1.5,
          pointRadius: 0,
          tension: 0.3,
          order: 1,
        },
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

/* ---- profile ---- */
async function loadProfile() {
  try {
    const p = await api("/patients/me");
    const f = $("#profile-form");
    f.disease_type.value = p.disease_type || "";
    f.severity.value = p.severity || "";
    f.age.value = p.age ?? "";
    f.sex.value = p.sex || "";
    f.consent.checked = p.consent_status === "granted";
    $("#profile-status").textContent = p.profile_complete ? "Complete" : "Incomplete";
  } catch (err) {
    toast(err.message, true);
  }
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
    const p = await api("/patients/me", { method: "PATCH", body });
    $("#profile-status").textContent = p.profile_complete ? "Complete" : "Incomplete";
    toast("Profile saved.");
    loadExposure();
  } catch (err) {
    toast(err.message, true);
  }
}

/* ---------------- wiring ---------------- */
function setPlace(place) {
  state.place = place;
  const sel = $("#loc-select");
  if (!PLACES.includes(place)) {
    let opt = sel.querySelector("option[data-custom]");
    if (!opt) {
      opt = document.createElement("option");
      opt.dataset.custom = "1";
      sel.appendChild(opt);
    }
    opt.value = "custom";
    opt.textContent = place.name;
    sel.value = "custom";
  }
  refreshAll();
}

function init() {
  const sel = $("#loc-select");
  sel.innerHTML = PLACES.map((p, i) => `<option value="${i}">${p.name}</option>`).join("");
  sel.addEventListener("change", () => sel.value !== "custom" && setPlace(PLACES[Number(sel.value)]));
  $("#time-select").addEventListener("change", (e) => {
    state.at = e.target.value;
    refreshAll();
  });
  $("#refresh").addEventListener("click", refreshAll);
  $("#logout").addEventListener("click", async () => {
    signOut();
  });
  $("#simulate").addEventListener("click", simulateDay);
  $("#profile-form").addEventListener("submit", saveProfile);
  document.querySelectorAll("#pollutant-seg button").forEach((b) =>
    b.addEventListener("click", () => {
      document.querySelectorAll("#pollutant-seg button").forEach((x) => x.classList.toggle("active", x === b));
      state.pollutant = b.dataset.p;
      loadForecast();
    }),
  );
  document.querySelectorAll(".tab").forEach((t) => t.addEventListener("click", () => setAuthMode(t.dataset.mode)));
  $("#auth-form").addEventListener("submit", submitAuth);

  if (state.token) {
    startApp().catch(() => signOut());
  } else {
    showAuth();
  }
}

document.addEventListener("DOMContentLoaded", init);
