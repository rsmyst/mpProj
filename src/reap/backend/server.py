"""Tiny dependency-free HTTP backend for the live stick-model demo.

Endpoints
    GET  /                 demo page (live MJPEG + buttons)
    GET  /stream.mjpg      annotated stick-model video stream
    GET  /snapshot.jpg     latest annotated frame
    GET  /api/pose         latest COCO-17 joints + angles + rep state (JSON)
    POST /api/frame        push a raw JPEG frame, get pose JSON back
    POST /api/speak        {"text": "Good rep"} -> local TTS
    POST /api/mode         {"mode": "mp" | "yolo" | "hybrid"}
    POST /api/stick        {"stick_only": true}
    POST /api/reset        reset the current exercise set
"""
from __future__ import annotations

import base64
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from reap.backend.controller import DemoController
from reap.backend.service import FrameOutput

INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>R.E.A.P live stick-model demo</title>
<style>
  :root { color-scheme: dark; --bg:#0b1220; --card:#111c31; --line:#233553;
          --ink:#e8eefc; --muted:#8da2c4; --blue:#4f8cff; --green:#31d68a;
          --amber:#f4b740; }
  * { box-sizing: border-box; }
  body { margin:0; min-height:100vh; background:radial-gradient(1200px 700px at 20% 0%,#16294a,#0b1220 60%);
         color:var(--ink); font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, sans-serif; }
  header { display:flex; align-items:baseline; gap:14px; padding:22px 28px 8px; }
  h1 { margin:0; font-size:clamp(20px,2.4vw,30px); letter-spacing:.4px; }
  h1 span { color:var(--blue); }
  header p { margin:0; color:var(--muted); font-size:14px; }
  main { display:grid; grid-template-columns:minmax(0,1.75fr) minmax(310px,.95fr);
         gap:18px; padding:14px 28px 28px; align-items:start; }
  .card { background:var(--card); border:1px solid var(--line); border-radius:16px;
          box-shadow:0 18px 50px rgba(0,0,0,.28); overflow:hidden; }
  .video { position:relative; min-height:280px; display:flex; align-items:center;
           justify-content:center; background:#050a13; }
  .video img { width:100%; height:auto; display:block; }
  .badge { position:absolute; top:12px; left:12px; background:rgba(5,10,19,.72);
           border:1px solid var(--line); color:var(--ink); padding:6px 10px;
           border-radius:999px; font-size:12px; letter-spacing:.3px; }
  .panel { padding:18px; display:flex; flex-direction:column; gap:16px; }
  .stats { display:grid; grid-template-columns:1fr 1fr; gap:10px; }
  .stat { background:#0c1628; border:1px solid var(--line); border-radius:12px;
          padding:10px 12px; }
  .stat label { display:block; color:var(--muted); font-size:11px; text-transform:uppercase;
                letter-spacing:.8px; margin-bottom:4px; }
  .stat b { font-size:18px; font-variant-numeric:tabular-nums; }
  .angles { color:var(--muted); font-size:13px; line-height:1.55; }
  .row { display:flex; flex-wrap:wrap; gap:8px; }
  button { border:1px solid var(--line); background:#172642; color:var(--ink);
           border-radius:10px; padding:9px 12px; font:inherit; font-size:13px;
           cursor:pointer; transition:.15s transform,.15s background,.15s border; }
  button:hover { background:#1e3154; border-color:#35517f; }
  button:active { transform:translateY(1px); }
  button.primary { background:var(--blue); border-color:var(--blue); color:#06122a;
                   font-weight:700; }
  button.good { border-color:#1d7a52; color:#bff5d9; }
  button.warn { border-color:#8a5c14; color:#ffe0a6; }
  button.active { background:#20406c; border-color:var(--blue); }
  input[type=text] { flex:1; min-width:130px; background:#0c1628; color:var(--ink);
                     border:1px solid var(--line); border-radius:10px; padding:9px 11px;
                     font:inherit; font-size:13px; }
  .label { color:var(--muted); font-size:12px; text-transform:uppercase;
           letter-spacing:.8px; }
  .cue { background:#0c1628; border:1px solid var(--line); border-radius:12px;
         padding:11px 12px; font-size:13px; color:#b9cdf0; min-height:44px; }
  .cue b { color:var(--green); }
  .foot { grid-column:1 / -1; color:var(--muted); font-size:12px; padding:0 4px; }
  .foot code { color:#b9cdf0; }
  .toggle { display:flex; align-items:center; gap:8px; color:var(--muted); font-size:13px; }
  @media (max-width:900px) { main { grid-template-columns:1fr; padding:12px; }
                             header { padding:18px 12px 4px; } }
</style>
</head>
<body>
<header>
  <h1>R.E.A.P <span>live stick-model demo</span></h1>
  <p>MediaPipe / YOLO-Pose &rarr; COCO-17 skeleton &rarr; spoken cues</p>
</header>
<main>
  <section class="card video">
    <img id="stream" src="/stream.mjpg" alt="live stick model">
    <div class="badge" id="badge">connecting…</div>
  </section>

  <aside class="card panel">
    <div class="stats">
      <div class="stat"><label>pose source</label><b id="source">—</b></div>
      <div class="stat"><label>fps</label><b id="fps">—</b></div>
      <div class="stat"><label>reps</label><b id="reps">0</b></div>
      <div class="stat"><label>good / fix</label><b id="quality">0 / 0</b></div>
    </div>
    <div class="angles" id="angles">joint angles will appear here</div>

    <div>
      <div class="label" style="margin-bottom:8px">pose model</div>
      <div class="row" id="modes">
        <button data-mode="mp" onclick="setMode('mp')">MediaPipe</button>
        <button data-mode="yolo" onclick="setMode('yolo')">YOLO-Pose</button>
        <button data-mode="hybrid" onclick="setMode('hybrid')">Hybrid</button>
      </div>
    </div>

    <div>
      <div class="label" style="margin-bottom:8px">speak a demo cue</div>
      <div class="row">
        <button class="primary" onclick="nextPhrase()">Speak demo phrase</button>
        <button class="good" onclick="speak('Good rep - nice and controlled.')">Good rep</button>
        <button class="warn" onclick="speak('Bend a little deeper for a full rep.')">Bend deeper</button>
        <button onclick="speak('I cannot see your joints clearly. Face the camera.')">Can't see me</button>
      </div>
      <div class="row" style="margin-top:8px">
        <input id="text" type="text" maxlength="240" placeholder="Type words to speak…"
               onkeydown="if(event.key==='Enter'){speakInput()}">
        <button onclick="speakInput()">Speak</button>
      </div>
    </div>

    <div class="cue">last cue: <b id="lastcue">—</b></div>

    <div class="row" style="justify-content:space-between; align-items:center">
      <label class="toggle"><input id="stick" type="checkbox" onchange="setStick()"> stick-only view</label>
      <button onclick="resetSession()">Reset set</button>
    </div>
  </aside>

  <div class="foot">
    Local TTS via Gradium Phonon-31M, with Piper/espeak-ng fallbacks.
    Try the voice directly: <code>uv run reap-speak "Good rep!"</code>.
  </div>
</main>

<script>
const PHRASES = [
  "Hello! I am R.E.A.P, your physio buddy. I will watch your form and speak up each rep.",
  "Good rep - nice and controlled. Keep your chest up.",
  "Bend a little deeper for a full rep, then stand tall.",
  "Ease up there - protect your knees and stay in the safe range.",
  "I cannot see your joints clearly. Face the camera and step back a little."
];
let phraseIndex = 0;

async function postJSON(path, body) {
  try {
    const r = await fetch(path, {method:'POST', headers:{'Content-Type':'application/json'},
                                 body:JSON.stringify(body)});
    return await r.json();
  } catch (e) { setBadge('backend offline'); return null; }
}
async function speak(text) {
  const out = await postJSON('/api/speak', {text, force:true});
  if (out && out.ok) { document.getElementById('lastcue').textContent = text; }
}
function speakInput() {
  const el = document.getElementById('text');
  const text = el.value.trim();
  if (text) { speak(text); el.value = ''; }
}
function nextPhrase() {
  speak(PHRASES[phraseIndex % PHRASES.length]);
  phraseIndex += 1;
}
async function setMode(mode) {
  await postJSON('/api/mode', {mode});
  paintModes(mode);
}
function paintModes(mode) {
  document.querySelectorAll('#modes button').forEach(b => {
    b.classList.toggle('active', b.dataset.mode === mode);
  });
}
async function setStick() {
  const value = document.getElementById('stick').checked;
  await postJSON('/api/stick', {stick_only:value});
}
async function resetSession() {
  await postJSON('/api/reset', {});
  setBadge('set reset');
}
function setBadge(text) { document.getElementById('badge').textContent = text; }
function fmt(v, digits=0) { return (v === null || v === undefined) ? '—' : Number(v).toFixed(digits); }

async function poll() {
  try {
    const r = await fetch('/api/pose', {cache:'no-store'});
    const d = await r.json();
    if (!d.ready) { setBadge('waiting for video…'); return; }
    setBadge(d.pose.present ? 'tracking · ' + d.pose.source : 'no person detected');
    document.getElementById('source').textContent = d.pose.source;
    document.getElementById('fps').textContent = fmt(d.fps, 1);
    const s = d.session || {};
    document.getElementById('reps').textContent = s.reps ?? 0;
    document.getElementById('quality').textContent =
      (s.good ?? 0) + ' / ' + (s.needs_fix ?? 0);
    const a = d.angles || {};
    document.getElementById('angles').textContent =
      'knee L/R ' + fmt(a.left_knee) + '° / ' + fmt(a.right_knee) + '°' +
      '   ·   hip L/R ' + fmt(a.left_hip) + '° / ' + fmt(a.right_hip) + '°' +
      '   ·   confidence ' + fmt(d.pose.avg_confidence, 2);
    if (d.last_spoken) document.getElementById('lastcue').textContent = d.last_spoken;
    paintModes(d.mode);
    document.getElementById('stick').checked = !!d.stick_only;
  } catch (e) { setBadge('backend offline'); }
}
setInterval(poll, 500); poll();
</script>
</body>
</html>
"""


class DemoHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, addr: tuple[str, int], controller: DemoController,
                 verbose: bool = False) -> None:
        self.controller = controller
        self.verbose = verbose
        super().__init__(addr, DemoRequestHandler)


class DemoRequestHandler(BaseHTTPRequestHandler):
    server: DemoHTTPServer
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args) -> None:  # noqa: A003
        if getattr(self.server, "verbose", False):
            super().log_message(fmt, *args)

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self._cors()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        if path in ("/", "/index.html"):
            self._send_html(INDEX_HTML)
        elif path == "/stream.mjpg":
            self._stream_mjpeg()
        elif path == "/snapshot.jpg":
            data = self.server.controller.snapshot()
            if not data:
                self._send_json({"error": "no frame yet"}, status=503)
            else:
                self._send_bytes(data, "image/jpeg")
        elif path == "/api/pose":
            self._send_json(self.server.controller.pose_payload())
        elif path == "/api/status":
            payload = self.server.controller.pose_payload()
            self._send_json({
                "ready": payload.get("ready", False),
                "mode": payload.get("mode"),
                "stick_only": payload.get("stick_only"),
                "fps": payload.get("fps"),
            })
        elif path == "/api/health":
            self._send_json({"status": "ok"})
        elif path == "/favicon.ico":
            self.send_response(204)
            self.send_header("Content-Length", "0")
            self.end_headers()
        else:
            self._send_json({"error": "not found"}, status=404)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/api/frame":
            self._ingest_frame(parse_qs(parsed.query))
            return
        try:
            body = self._read_json()
        except ValueError as exc:
            self._send_json({"error": str(exc)}, status=400)
            return

        controller = self.server.controller
        try:
            if path == "/api/speak":
                text = str(body.get("text", "")).strip()
                if not text:
                    self._send_json({"error": "text is required"}, status=400)
                    return
                ok = controller.speak(text[:300], force=bool(body.get("force", True)))
                self._send_json({"ok": ok, "text": text[:300]})
            elif path == "/api/mode":
                mode = str(body.get("mode", ""))
                controller.set_mode(mode)
                self._send_json({"ok": True, "mode": mode})
            elif path == "/api/stick":
                value = bool(body.get("stick_only", body.get("value", False)))
                controller.set_stick_only(value)
                self._send_json({"ok": True, "stick_only": value})
            elif path == "/api/reset":
                controller.reset_session()
                self._send_json({"ok": True})
            else:
                self._send_json({"error": "not found"}, status=404)
        except (ValueError, RuntimeError) as exc:
            self._send_json({"error": str(exc)}, status=400)

    def _ingest_frame(self, query: dict[str, list[str]]) -> None:
        length = int(self.headers.get("Content-Length", "0") or 0)
        if length <= 0:
            self._send_json({"error": "empty frame body"}, status=400)
            return
        if length > 16 * 1024 * 1024:
            self._send_json({"error": "frame too large"}, status=413)
            return
        data = self.rfile.read(length)
        result = self.server.controller.handle_jpeg(data)
        if result is None:
            self._send_json({"error": "could not decode JPEG"}, status=400)
            return
        out, jpeg = result
        payload = self._payload(out)
        if query.get("image", ["0"])[0] in ("1", "true", "yes"):
            payload["jpeg_b64"] = base64.b64encode(jpeg).decode("ascii")
        self._send_json(payload)

    def _payload(self, out: FrameOutput) -> dict:
        controller = self.server.controller
        payload = out.to_dict(fps=controller.fps, mode=controller.service.mode,
                              stick_only=controller.stick_only)
        payload["ready"] = True
        return payload

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0") or 0)
        if length <= 0:
            return {}
        if length > 1024 * 1024:
            raise ValueError("request body too large")
        raw = self.rfile.read(length)
        try:
            data = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("invalid JSON body") from exc
        if not isinstance(data, dict):
            raise ValueError("JSON body must be an object")
        return data

    def _send_html(self, html: str) -> None:
        self._send_bytes(html.encode("utf-8"), "text/html; charset=utf-8")

    def _send_json(self, obj: dict, status: int = 200) -> None:
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self._cors()
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_bytes(self, data: bytes, content_type: str) -> None:
        self.send_response(200)
        self._cors()
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _cors(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _stream_mjpeg(self) -> None:
        self.send_response(200)
        self._cors()
        self.send_header("Content-Type",
                         "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-cache, private")
        self.send_header("Pragma", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        try:
            while True:
                jpeg = self.server.controller.snapshot()
                if jpeg:
                    head = (b"--frame\r\nContent-Type: image/jpeg\r\n"
                            b"Content-Length: " + str(len(jpeg)).encode() +
                            b"\r\n\r\n")
                    self.wfile.write(head)
                    self.wfile.write(jpeg)
                    self.wfile.write(b"\r\n")
                self.wfile.flush()
                time.sleep(0.033)
        except (BrokenPipeError, ConnectionResetError, OSError):
            return
