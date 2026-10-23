// Usage: node capture.js <outDir> <t1,t2,...|all> — renders video.html frames via headless Chrome CDP.
const { spawn } = require("child_process");
const fs = require("fs");
const path = require("path");

const CHROME = "C:/Program Files/Google/Chrome/Application/chrome.exe";
const FPS = 30, DUR = 21;
const out = path.resolve(process.argv[2] || "frames");
const which = process.argv[3] || "all";
fs.mkdirSync(out, { recursive: true });
const page = "file:///" + path.resolve(__dirname, "video.html").replace(/\\/g, "/");
const profile = path.resolve(__dirname, "chrome-profile");

const chrome = spawn(CHROME, ["--headless=new", "--remote-debugging-port=9333", `--user-data-dir=${profile}`,
  "--window-size=1920,1080", "--hide-scrollbars", "--force-device-scale-factor=1", "--allow-file-access-from-files", "about:blank"], { stdio: "ignore" });

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
(async () => {
  let targets;
  for (let i = 0; i < 50; i++) {
    try { targets = await (await fetch("http://127.0.0.1:9333/json")).json(); break; } catch { await sleep(200); }
  }
  const tgt = targets.find((t) => t.type === "page");
  const ws = new WebSocket(tgt.webSocketDebuggerUrl);
  await new Promise((r) => (ws.onopen = r));
  let id = 0; const pending = new Map();
  ws.onmessage = (m) => { const d = JSON.parse(m.data); if (d.id && pending.has(d.id)) { pending.get(d.id)(d); pending.delete(d.id); } };
  const send = (method, params = {}) => new Promise((r) => { const i = ++id; pending.set(i, r); ws.send(JSON.stringify({ id: i, method, params })); });
  const evalJs = async (expr) => { const r = await send("Runtime.evaluate", { expression: expr, awaitPromise: true, returnByValue: true }); return r.result && r.result.result && r.result.result.value; };

  await send("Emulation.setDeviceMetricsOverride", { width: 1920, height: 1080, deviceScaleFactor: 1, mobile: false });
  await send("Page.enable");
  await send("Page.navigate", { url: page });
  for (let i = 0; i < 100; i++) { if (await evalJs("window.ready ? window.ready.then(()=>true) : false")) break; await sleep(100); }

  const times = which === "all" ? Array.from({ length: FPS * DUR }, (_, i) => i / FPS) : which.split(",").map(Number);
  for (let i = 0; i < times.length; i++) {
    await evalJs(`render(${times[i]}); new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r))).then(()=>1)`);
    const shot = await send("Page.captureScreenshot", { format: "png" });
    const name = which === "all" ? `f${String(i).padStart(4, "0")}.png` : `t${times[i].toFixed(2)}.png`;
    fs.writeFileSync(path.join(out, name), Buffer.from(shot.result.data, "base64"));
    if (i % 60 === 0) console.log("frame", i, "/", times.length);
  }
  ws.close(); chrome.kill(); process.exit(0);
})().catch((e) => { console.error(e); chrome.kill(); process.exit(1); });
