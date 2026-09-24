/**
 * The level-1 model files, fetched once so they come with the program (0.9.0).
 *
 * models.json at the repo root names the release asset: {version, url, sha256, bytes}. At startup,
 * if marker/data/production/.models-version does not say that version, the asset is downloaded,
 * checked against its sha256, and unpacked into marker/data/production/ with the system tar. Until
 * then a level-1 reading fails as it always has without the files and falls back to SponsorBlock.
 * The files are CC BY-NC-SA 4.0 (NOTICE.md). No models.json, or no network: nothing happens.
 */
import crypto from "node:crypto";
import { spawn } from "node:child_process";
import fs from "node:fs";
import path from "node:path";

const status = { state: "unknown", note: "" };
export const modelStatus = () => ({ ...status });

function readManifest(root) {
  try {
    const m = JSON.parse(fs.readFileSync(path.join(root, "models.json"), "utf8"));
    return m.version && m.url && /^[0-9a-f]{64}$/.test(m.sha256) ? m : null;
  } catch {
    return null;
  }
}

// Windows ships bsdtar in System32. Prefer it by full path: a GNU tar earlier on PATH (Git's) reads a
// drive letter as a remote host. Everything is also passed relative to cwd, so no path has a colon.
function tarCommand() {
  if (process.platform !== "win32") return "tar";
  const system = path.join(process.env.SystemRoot || "C:\\Windows", "System32", "tar.exe");
  return fs.existsSync(system) ? system : "tar";
}

function untar(file, cwd) {
  return new Promise((resolve, reject) => {
    const child = spawn(tarCommand(), ["-xzf", file], { cwd, windowsHide: true });
    let err = "";
    child.stderr.on("data", (d) => (err += d));
    child.on("error", reject);
    child.on("close", (code) => (code === 0 ? resolve() : reject(new Error(`tar exited ${code}: ${err.trim().slice(0, 200)}`))));
  });
}

async function download(url, dest, expectedBytes, log) {
  const res = await fetch(url, { redirect: "follow" });
  if (!res.ok || !res.body) throw new Error(`HTTP ${res.status}`);
  const hash = crypto.createHash("sha256");
  const out = fs.createWriteStream(dest);
  let got = 0;
  let nextLog = 0.25;
  for await (const chunk of res.body) {
    hash.update(chunk);
    got += chunk.length;
    if (!out.write(chunk)) await new Promise((r) => out.once("drain", r));
    if (expectedBytes && got / expectedBytes >= nextLog) {
      log("models", `${Math.round((got / expectedBytes) * 100)}% downloaded`);
      nextLog += 0.25;
    }
  }
  await new Promise((resolve, reject) => out.end((e) => (e ? reject(e) : resolve())));
  return { sha256: hash.digest("hex"), bytes: got };
}

/** Make sure the level-1 files are present. Resolves either way; never throws. */
export async function ensureModels(root, log) {
  const manifest = readManifest(root);
  if (!manifest) return Object.assign(status, { state: "no-manifest", note: "models.json missing or incomplete" });
  const dir = path.join(root, "marker", "data", "production");
  const stamp = path.join(dir, ".models-version");
  const have = fs.existsSync(stamp) ? fs.readFileSync(stamp, "utf8").trim() : "";
  if (have === manifest.version) return Object.assign(status, { state: "ready", note: manifest.version });

  fs.mkdirSync(dir, { recursive: true });
  const partName = ".download.tar.gz";
  const part = path.join(dir, partName);
  Object.assign(status, { state: "downloading", note: `${manifest.version}, ${Math.round((manifest.bytes || 0) / 1e6)} MB` });
  log("models", `fetching ${manifest.version} (${status.note.split(", ")[1]}) for level 1`);
  try {
    const { sha256, bytes } = await download(manifest.url, part, manifest.bytes, log);
    if (sha256 !== manifest.sha256) throw new Error(`checksum mismatch (${bytes} bytes, sha256 ${sha256.slice(0, 12)}...)`);
    await untar(partName, dir);
    fs.writeFileSync(stamp, manifest.version + "\n");
    log("models", `${manifest.version} ready`);
    return Object.assign(status, { state: "ready", note: manifest.version });
  } catch (e) {
    log("models", `FAILED: ${e.message}; level 1 falls back to SponsorBlock until the next start`);
    return Object.assign(status, { state: "failed", note: e.message });
  } finally {
    fs.rmSync(part, { force: true });
  }
}
