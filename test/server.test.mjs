// Offline checks of who the program answers. Run: node --test
//
// Starts server/server.mjs on a spare port (no model download, no reading) and
// sends it the requests a web page, a local probe, or a misaddressed client
// could make. Each one must be refused before any work starts.
import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import http from "node:http";
import net from "node:net";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const PORT = 4790 + 100 + Math.floor(Math.random() * 500);
const HOST = `127.0.0.1:${PORT}`;
const EXT = "chrome-extension://abcdefghijklmnopabcdefghijklmnop";

let child;

function request({ method = "GET", path: p = "/", headers = {}, body = null } = {}) {
  return new Promise((resolve, reject) => {
    const req = http.request({ host: "127.0.0.1", port: PORT, method, path: p, headers }, (res) => {
      let data = "";
      res.on("data", (d) => (data += d));
      res.on("end", () => resolve({ status: res.statusCode, headers: res.headers, body: data }));
    });
    req.on("error", reject);
    if (body !== null) req.write(body);
    req.end();
  });
}

// A raw request line, for what a browser would never send but a local port scanner might.
function raw(text) {
  return new Promise((resolve, reject) => {
    const sock = net.connect(PORT, "127.0.0.1", () => sock.write(text));
    let data = "";
    sock.on("data", (d) => (data += d));
    sock.on("end", () => resolve(data));
    sock.on("close", () => resolve(data));
    sock.on("error", reject);
    setTimeout(() => sock.destroy(), 3000);
  });
}

test.before(async () => {
  child = spawn(process.execPath, [path.join(root, "server", "server.mjs")], {
    cwd: root,
    env: { ...process.env, SPONSORSKIP_PORT: String(PORT), SPONSORSKIP_NO_MODELS: "1" },
    windowsHide: true,
  });
  await new Promise((resolve, reject) => {
    let out = "";
    child.stdout.on("data", (d) => {
      out += d;
      if (out.includes(`http://127.0.0.1:${PORT}`)) resolve();
    });
    child.on("exit", (code) => reject(new Error(`server exited early (${code})`)));
    setTimeout(() => reject(new Error("server did not start in 10 s")), 10_000);
  });
});

test.after(() => child?.kill());

test("a GET without the x-sponsorskip header is refused", async () => {
  const r = await request({ path: "/ai-config" });
  assert.equal(r.status, 403);
});

test("the extension's header opens the door", async () => {
  const r = await request({ path: "/ai-config", headers: { "x-sponsorskip": "1" } });
  assert.equal(r.status, 200);
  assert.ok(JSON.parse(r.body));
});

test("a request addressed to another host name is refused, header or not (DNS rebinding)", async () => {
  const r = await request({ path: "/ai-config", headers: { "x-sponsorskip": "1", host: `evil.com:${PORT}` } });
  assert.equal(r.status, 403);
  const ok = await request({ path: "/ai-config", headers: { "x-sponsorskip": "1", host: `localhost:${PORT}` } });
  assert.equal(ok.status, 200);
});

test("a settings POST from a web page origin is refused; an extension origin is answered", async () => {
  const web = await request({ method: "POST", path: "/ai-config", body: "{}",
    headers: { "x-sponsorskip": "1", origin: "http://example.com", "content-type": "application/json" } });
  assert.equal(web.status, 403);
  // An unknown route: past the origin gate it answers 404, and the CORS header names the extension.
  // (Not /ai-test or /ai-config: those would run the AI self-test or write the settings file.)
  const ext = await request({ method: "POST", path: "/nothing-here", body: "{}",
    headers: { "x-sponsorskip": "1", origin: EXT, "content-type": "application/json" } });
  assert.equal(ext.status, 404);
  assert.equal(ext.headers["access-control-allow-origin"], EXT);
});

test("a preflight is answered for an extension origin only", async () => {
  const ext = await request({ method: "OPTIONS", path: "/analyze/dQw4w9WgXcQ", headers: { origin: EXT } });
  assert.equal(ext.status, 204);
  const web = await request({ method: "OPTIONS", path: "/analyze/dQw4w9WgXcQ", headers: { origin: "http://example.com" } });
  assert.equal(web.status, 403);
});

test("a bad video id is refused before any work", async () => {
  for (const p of ["/quick/short", "/analyze/not-a-video-id!", "/progress/../../etc"]) {
    const r = await request({ path: p, headers: { "x-sponsorskip": "1" } });
    assert.ok(r.status === 400 || r.status === 404, `${p} -> ${r.status}`);
  }
});

test("a malformed request line does not take the program down", async () => {
  const reply = await raw(`GET http://a:b:c/ HTTP/1.1\r\nHost: ${HOST}\r\nx-sponsorskip: 1\r\nConnection: close\r\n\r\n`);
  assert.match(reply, /^HTTP\/1\.1 4\d\d/);
  const after = await request({ path: "/ai-config", headers: { "x-sponsorskip": "1" } });
  assert.equal(after.status, 200);
});

test("/ai-ask needs the per-start token", async () => {
  const r = await request({ method: "POST", path: "/ai-ask", body: "{}", headers: { "content-type": "application/json" } });
  assert.equal(r.status, 403);
});

test("a correction is taken from the extension only, and only in its known shape", async () => {
  const body = JSON.stringify({ videoId: "abc", kind: "missed", start: 10, end: 5 });
  const headers = { "x-sponsorskip": "1", "content-type": "application/json" };
  const page = await request({ method: "POST", path: "/correction", headers: { ...headers, origin: "https://www.youtube.com" }, body });
  assert.equal(page.status, 403);
  // Bad id, end before start: refused before anything is written.
  const bad = await request({ method: "POST", path: "/correction", headers: { ...headers, origin: EXT }, body });
  assert.equal(bad.status, 400);
});
