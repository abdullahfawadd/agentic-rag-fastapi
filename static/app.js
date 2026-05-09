const messages = document.querySelector("#messages");
const form = document.querySelector("#chatForm");
const input = document.querySelector("#questionInput");
const sendBtn = document.querySelector("#sendBtn");
const healthBtn = document.querySelector("#healthBtn");
const ingestBtn = document.querySelector("#ingestBtn");
const logsBtn = document.querySelector("#logsBtn");
const apiStatus = document.querySelector("#apiStatus");
const vectorCount = document.querySelector("#vectorCount");
const namespaceName = document.querySelector("#namespaceName");

const state = {
  busy: false,
};

function bootIcons() {
  if (window.lucide) {
    window.lucide.createIcons();
  }
}

function scrollToBottom() {
  messages.scrollTop = messages.scrollHeight;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function setBusy(isBusy) {
  state.busy = isBusy;
  sendBtn.disabled = isBusy;
  input.disabled = isBusy;
}

function addMessage(role, text, payload = {}) {
  const article = document.createElement("article");
  article.className = `message ${role}`;

  const avatar = document.createElement("div");
  avatar.className = "avatar";
  avatar.textContent = role === "user" ? "ME" : "AI";

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.innerHTML = `<p>${escapeHtml(text)}</p>`;

  if (payload.toolsUsed?.length) {
    const meta = document.createElement("div");
    meta.className = "meta-row";
    payload.toolsUsed.forEach((tool) => {
      const pill = document.createElement("span");
      pill.className = "tool-pill";
      pill.textContent = tool;
      meta.appendChild(pill);
    });
    bubble.appendChild(meta);
  }

  if (payload.trace?.length) {
    bubble.appendChild(renderTrace(payload.trace));
  }

  if (payload.citations?.length) {
    bubble.appendChild(renderCitations(payload.citations));
  }

  article.appendChild(avatar);
  article.appendChild(bubble);
  messages.appendChild(article);
  scrollToBottom();
  return article;
}

function renderTrace(trace) {
  const details = document.createElement("details");
  details.className = "tool-trace";
  details.open = true;
  details.innerHTML = `<summary>Agent Tool Trace (${trace.length})</summary>`;

  trace.forEach((item, index) => {
    const node = document.createElement("div");
    node.className = "trace-item";
    const inputValue = JSON.stringify(item.input ?? {}, null, 2);
    node.innerHTML = `
      <strong>${index + 1}. ${escapeHtml(item.tool)}</strong>
      <code>${escapeHtml(inputValue)}</code>
      <p>${escapeHtml(item.output_preview ?? "")}</p>
    `;
    details.appendChild(node);
  });

  return details;
}

function renderCitations(citations) {
  const wrapper = document.createElement("div");
  wrapper.className = "citation-list";
  citations.forEach((citation) => {
    const node = document.createElement("div");
    node.className = "citation-card";
    const score = citation.score ? ` · score ${citation.score}` : "";
    node.innerHTML = `
      <strong>${escapeHtml(citation.source)} · page ${escapeHtml(citation.page)}${escapeHtml(score)}</strong>
      <p>${escapeHtml(citation.snippet)}</p>
    `;
    wrapper.appendChild(node);
  });
  return wrapper;
}

function autoGrow() {
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, 180)}px`;
}

async function fetchJson(url, options = {}) {
  const response = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload.detail || `Request failed with ${response.status}`);
  }
  return payload;
}

async function runHealth(showMessage = true) {
  try {
    const health = await fetchJson("/health");
    apiStatus.textContent = health.status === "ok" ? "Online" : "Check";
    vectorCount.textContent = health.pinecone_vectors ?? 0;
    namespaceName.textContent = health.pinecone_namespace ?? "ai_agents_pdf";
    if (showMessage) {
      addMessage(
        "assistant",
        `Health check passed.\nVectors: ${health.pinecone_vectors}\nModel: ${health.llm_model}\nEmbeddings: ${health.embedding_model}\nWeb search: ${health.web_search_enabled ? "enabled" : "disabled"}`,
      );
    }
  } catch (error) {
    apiStatus.textContent = "Offline";
    if (showMessage) {
      addMessage("assistant", `Health check failed: ${error.message}`);
    }
  }
}

async function ingestPdf() {
  if (state.busy) return;
  setBusy(true);
  const thinking = addMessage("assistant", "Ingesting the AI agents PDF into Pinecone...");
  try {
    const result = await fetchJson("/ingest", { method: "POST" });
    thinking.remove();
    addMessage(
      "assistant",
      `Ingestion complete.\nChunks ingested: ${result.chunks_ingested}\nPages indexed: ${result.pages_indexed}\nNamespace: ${result.namespace}`,
    );
    await runHealth(false);
  } catch (error) {
    thinking.remove();
    addMessage("assistant", `Ingestion failed: ${error.message}`);
  } finally {
    setBusy(false);
  }
}

async function showLogs() {
  try {
    const payload = await fetchJson("/logs?limit=50");
    if (!payload.lines.length) {
      addMessage("assistant", "No tool calls have been logged yet. Run a few queries first.");
      return;
    }
    addMessage("assistant", `Recent tool log entries:\n${payload.lines.join("\n")}`);
  } catch (error) {
    addMessage("assistant", `Could not load logs: ${error.message}`);
  }
}

async function askQuestion(question) {
  if (!question.trim() || state.busy) return;
  addMessage("user", question.trim());
  input.value = "";
  autoGrow();
  setBusy(true);
  const thinking = addMessage("assistant", "Thinking, selecting tools, and building the answer...");
  thinking.querySelector(".bubble").classList.add("thinking");

  try {
    const payload = await fetchJson("/query", {
      method: "POST",
      body: JSON.stringify({ question }),
    });
    thinking.remove();
    addMessage("assistant", payload.answer, {
      toolsUsed: payload.tools_used,
      trace: payload.tool_trace,
      citations: payload.citations,
    });
    await runHealth(false);
  } catch (error) {
    thinking.remove();
    addMessage("assistant", `Query failed: ${error.message}`);
  } finally {
    setBusy(false);
    input.focus();
  }
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  askQuestion(input.value);
});

input.addEventListener("input", autoGrow);

input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    askQuestion(input.value);
  }
});

document.querySelectorAll("[data-prompt]").forEach((button) => {
  button.addEventListener("click", () => {
    input.value = button.dataset.prompt;
    autoGrow();
    input.focus();
  });
});

healthBtn.addEventListener("click", () => runHealth(true));
ingestBtn.addEventListener("click", ingestPdf);
logsBtn.addEventListener("click", showLogs);

bootIcons();
runHealth(false);
