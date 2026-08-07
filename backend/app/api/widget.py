import json
import re

from fastapi import APIRouter, Query
from fastapi.responses import HTMLResponse, Response

router = APIRouter(tags=["Chat Widget"])


# ------------------------------------------------------------------
# The loader: the only thing that ever touches the host page's DOM
# directly. It creates ONE <iframe> pointed back at this same backend
# (GET /widget below) and keeps it collapsed to a small bubble-sized
# box until the visitor opens it — the actual chat UI, its CSS, and its
# JS all live inside that iframe's document, isolated from the host
# page. The iframe posts a postMessage back here to resize itself
# between "bubble" and "open panel" states.
# ------------------------------------------------------------------

WIDGET_LOADER_JS = r"""
(function () {
  var thisScript = document.currentScript;

  var projectId = thisScript.getAttribute("data-project-id");
  var backendUrl = (thisScript.getAttribute("data-backend-url") || "").replace(/\/$/, "");
  var primaryColor = thisScript.getAttribute("data-primary-color") || "";

  if (!projectId || !backendUrl) {
    console.error("[chat-widget] data-project-id and data-backend-url are required.");
    return;
  }

  var BUBBLE_SIZE = "76px";
  var OPEN_WIDTH = "370px";
  var OPEN_HEIGHT = "600px";

  var params = new URLSearchParams({ project_id: projectId });
  if (primaryColor) params.set("primary_color", primaryColor);

  var iframe = document.createElement("iframe");
  iframe.src = backendUrl + "/widget?" + params.toString();
  iframe.title = "Chat widget";
  iframe.style.cssText =
    "position:fixed;bottom:16px;right:16px;border:none;background:transparent;" +
    "z-index:999999;width:" + BUBBLE_SIZE + ";height:" + BUBBLE_SIZE + ";" +
    "max-width:calc(100vw - 24px);max-height:calc(100vh - 24px);" +
    "transition:width .15s ease,height .15s ease;color-scheme:light;";
  iframe.setAttribute("allowtransparency", "true");

  document.body.appendChild(iframe);

  window.addEventListener("message", function (event) {
    if (!event.data || event.data.source !== "chat-widget" || event.data.projectId !== projectId) return;

    if (event.data.type === "resize") {
      iframe.style.width = event.data.open ? OPEN_WIDTH : BUBBLE_SIZE;
      iframe.style.height = event.data.open ? OPEN_HEIGHT : BUBBLE_SIZE;
    }
  });
})();
"""


@router.get("/widget-loader.js")
async def widget_loader_js():
    """
    The one-line embed script (paste before </body> on any site). Reads
    its own data-* attributes and injects a single iframe pointed at
    GET /widget below — no other DOM access on the host page.
    """

    return Response(content=WIDGET_LOADER_JS, media_type="application/javascript")


# ------------------------------------------------------------------
# The widget itself: a self-contained page (bubble + chat panel) meant
# to be loaded inside the iframe the loader script creates. All CSS/JS
# here is scoped to this document only. It talks to POST /chat on the
# same origin (no CORS needed since it's the same backend), and posts
# {source:'chat-widget', type:'resize', open, projectId} to the parent
# window so the loader's <iframe> element can resize around it.
#
# Two placeholder tokens (__PRIMARY_COLOR__, __PROJECT_ID_JSON__) are
# substituted via plain str.replace() below — NOT str.format() — so
# every other brace in this CSS/JS is written normally, unescaped.
# ------------------------------------------------------------------

_WIDGET_PAGE_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Chat</title>
<style>
  :root{ --cw-primary: __PRIMARY_COLOR__; }
  *{ box-sizing:border-box; }
  html,body{ margin:0; padding:0; height:100%; background:transparent; overflow:hidden;
    font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif; }
  #cw-root{ position:relative; width:100%; height:100%; }
  #cw-bubble{ position:absolute; bottom:0; right:0; width:76px; height:76px; border-radius:50%; background:var(--cw-primary);
    color:#fff; border:none; cursor:pointer; box-shadow:0 4px 14px rgba(0,0,0,.25);
    font-size:26px; display:flex; align-items:center; justify-content:center; }
  #cw-panel{ display:none; width:100%; height:100%; background:#fff; border-radius:14px;
    box-shadow:0 10px 40px rgba(0,0,0,.25); flex-direction:column; overflow:hidden; }
  #cw-panel.open{ display:flex; }
  #cw-bubble.hidden{ display:none; }
  #cw-header{ background:var(--cw-primary); color:#fff; padding:12px 14px; font-size:14px;
    font-weight:600; display:flex; justify-content:space-between; align-items:center; flex-shrink:0; }
  #cw-close{ background:none; border:none; color:#fff; font-size:18px; cursor:pointer; line-height:1; }
  #cw-log{ flex:1; overflow-y:auto; padding:12px; display:flex; flex-direction:column; gap:8px; background:#f7f7f8; }
  .cw-msg{ max-width:85%; padding:8px 11px; border-radius:10px; font-size:13px; line-height:1.45; white-space:pre-wrap; }
  .cw-msg.user{ align-self:flex-end; background:var(--cw-primary); color:#fff; }
  .cw-msg.bot{ align-self:flex-start; background:#fff; border:1px solid #e5e5e5; color:#111; }
  #cw-inputrow{ display:flex; gap:6px; padding:10px; border-top:1px solid #eee; background:#fff; flex-shrink:0; }
  #cw-input{ flex:1; border:1px solid #ddd; border-radius:8px; padding:8px 10px; font-size:13px; outline:none; }
  #cw-send{ background:var(--cw-primary); color:#fff; border:none; border-radius:8px; padding:0 14px; cursor:pointer; font-size:13px; }
  #cw-send:disabled{ opacity:.5; cursor:not-allowed; }
</style>
</head>
<body>
<div id="cw-root">
  <button id="cw-bubble" aria-label="Open chat">&#128172;</button>
  <div id="cw-panel">
    <div id="cw-header"><span>Chat with us</span><button id="cw-close" aria-label="Close chat">&times;</button></div>
    <div id="cw-log"></div>
    <div id="cw-inputrow">
      <input id="cw-input" type="text" placeholder="Ask a question..." />
      <button id="cw-send" type="button">Send</button>
    </div>
  </div>
</div>
<script>
(function () {
  var projectId = __PROJECT_ID_JSON__;
  var sessionId = (window.crypto && crypto.randomUUID && crypto.randomUUID())
    || ("session-" + Date.now() + "-" + Math.random().toString(16).slice(2));

  var bubble = document.getElementById("cw-bubble");
  var panel = document.getElementById("cw-panel");
  var log = document.getElementById("cw-log");
  var input = document.getElementById("cw-input");
  var sendBtn = document.getElementById("cw-send");
  var closeBtn = document.getElementById("cw-close");

  function notifyParent(open) {
    window.parent.postMessage({ source: "chat-widget", type: "resize", open: open, projectId: projectId }, "*");
  }

  function setOpen(open) {
    panel.classList.toggle("open", open);
    bubble.classList.toggle("hidden", open);
    notifyParent(open);
    if (open) input.focus();
  }

  bubble.addEventListener("click", function () { setOpen(true); });
  closeBtn.addEventListener("click", function () { setOpen(false); });

  function addMessage(text, who) {
    var el = document.createElement("div");
    el.className = "cw-msg " + who;
    el.textContent = text;
    log.appendChild(el);
    log.scrollTop = log.scrollHeight;
  }

  async function send() {
    var text = input.value.trim();
    if (!text) return;

    addMessage(text, "user");
    input.value = "";
    sendBtn.disabled = true;

    var thinking = document.createElement("div");
    thinking.className = "cw-msg bot";
    thinking.textContent = "...";
    log.appendChild(thinking);
    log.scrollTop = log.scrollHeight;

    try {
      var res = await fetch("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ project_id: projectId, session_id: sessionId, user_query: text }),
      });
      var data = await res.json();
      thinking.remove();
      addMessage(res.ok ? data.answer : ("Error: " + (data.detail || "request failed")), "bot");
    } catch (err) {
      thinking.remove();
      addMessage("Couldn't reach the server. Please try again shortly.", "bot");
    } finally {
      sendBtn.disabled = false;
    }
  }

  sendBtn.addEventListener("click", send);
  input.addEventListener("keydown", function (e) {
    if (e.key === "Enter") { e.preventDefault(); send(); }
  });
})();
</script>
</body>
</html>
"""


@router.get("/widget", response_class=HTMLResponse)
async def widget_page(
    project_id: str = Query(..., description="Project to chat against."),
    primary_color: str = Query(default="#4f46e5"),
):
    """
    The widget's actual UI, meant to be loaded inside an iframe (see
    widget-loader.js above). Self-contained: no build step, talks to
    POST /chat on this same origin.
    """

    safe_color = (
        primary_color
        if re.fullmatch(r"#[0-9a-fA-F]{3}|#[0-9a-fA-F]{6}", primary_color)
        else "#4f46e5"
    )

    html = (
        _WIDGET_PAGE_TEMPLATE
        .replace("__PRIMARY_COLOR__", safe_color)
        .replace("__PROJECT_ID_JSON__", json.dumps(project_id))
    )

    return HTMLResponse(content=html)