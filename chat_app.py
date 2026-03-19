"""
chat_app.py — Asistente Brightspace · usuarios finales
Solo chat, sin configuración. Usa componentes nativos de Streamlit.

Repo (raíz):
  ├── asistente_rag_project_v3.py
  ├── chat_app.py
  ├── requirements.txt
  ├── .python-version
  ├── raw/
  ├── index/
  └── eval/

Secrets de Streamlit Cloud → OPENAI_API_KEY = "sk-..."
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st

# ── Rutas ────────────────────────────────────────────────────────────

def _resolve_root() -> Path:
    script_dir = Path(__file__).parent.resolve()
    for c in [script_dir, script_dir.parent, Path.cwd()]:
        if (c / "asistente_rag_project_v3.py").exists():
            return c
    return script_dir

PROJECT_ROOT = _resolve_root()
INDEX_DIR    = PROJECT_ROOT / "index"
LOG_DIR      = PROJECT_ROOT / "logs"

for _d in [str(PROJECT_ROOT), str(Path(__file__).parent)]:
    if _d not in sys.path:
        sys.path.insert(0, _d)

# ── Carga del motor RAG (cacheado) ───────────────────────────────────

@st.cache_resource(show_spinner=False)
def _load_engine():
    import traceback
    try:
        import asistente_rag_project_v3 as rag
        from sentence_transformers import SentenceTransformer
        cfg = rag.Settings(
            index_dir   = str(INDEX_DIR),
            log_dir     = str(LOG_DIR),
            use_llm     = True,
            llm_api_key = os.getenv("OPENAI_API_KEY", ""),
            llm_model   = "gpt-4o-mini",
            top_k       = 5,
            min_sim     = 0.35,
        )
        index, meta = rag.load_index(cfg)
        embedder     = SentenceTransformer(cfg.embed_model)
        return rag, cfg, index, meta, embedder, None
    except Exception:
        return None, None, None, None, None, traceback.format_exc()

# ── Página ───────────────────────────────────────────────────────────

st.set_page_config(
    page_title = "Asistente Brightspace",
    page_icon  = "🎓",
    layout     = "centered",
    initial_sidebar_state = "collapsed",
)

# CSS — solo estilos visuales, sin HTML personalizado en burbujas
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Lora:wght@600&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background: #FAFAF8;
}

/* Ocultar header nativo */
header[data-testid="stHeader"] { display: none; }
section[data-testid="stSidebar"] { display: none; }
.block-container { padding-top: 1.5rem !important; max-width: 740px !important; }

/* Avatar del asistente */
[data-testid="stChatMessageAvatarAssistant"] {
    background: #1a3a2e !important;
    border-radius: 10px !important;
}

/* Burbuja del asistente */
[data-testid="stChatMessageContentAssistant"] {
    background: #ffffff;
    border: 1px solid #e8e4dc;
    border-radius: 4px 16px 16px 16px;
    padding: 0.85rem 1.1rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    font-size: 0.95rem;
    line-height: 1.75;
    color: #1a1a1a;
}

/* Burbuja del usuario */
[data-testid="stChatMessageContentUser"] {
    background: #1a3a2e;
    color: #f0ede6 !important;
    border-radius: 16px 4px 16px 16px;
    padding: 0.75rem 1.1rem;
    font-size: 0.95rem;
    line-height: 1.6;
    box-shadow: 0 2px 8px rgba(26,58,46,0.15);
}
[data-testid="stChatMessageContentUser"] p { color: #f0ede6 !important; }

/* Input */
[data-testid="stChatInput"] {
    border: 1px solid #e0dbd1 !important;
    border-radius: 12px !important;
    background: #ffffff !important;
}

/* Botones de sugerencias */
.stButton button {
    background: #ffffff;
    border: 1px solid #ddd9cf;
    border-radius: 10px;
    color: #3a3a3a;
    font-size: 0.85rem;
    padding: 0.55rem 0.9rem;
    transition: all 0.15s;
    text-align: left;
    white-space: normal;
    height: auto;
    line-height: 1.4;
}
.stButton button:hover {
    border-color: #1a3a2e;
    color: #1a3a2e;
    background: #f2faf6;
}

/* Pills de fuentes */
.source-pills {
    margin-top: 0.5rem;
    display: flex; flex-wrap: wrap; gap: 0.4rem;
}
.source-pill {
    background: #f2efe8; border: 1px solid #ddd9cf;
    border-radius: 20px; padding: 2px 10px;
    font-size: 0.72rem; color: #5a5a5a;
    display: inline-block;
}
</style>
""", unsafe_allow_html=True)

# ── Estado ───────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []
if "engine_ready" not in st.session_state:
    st.session_state.engine_ready = False

# ── Header ───────────────────────────────────────────────────────────

st.markdown(
    "<h2 style='font-family:Lora,serif;font-size:1.5rem;color:#1a1a1a;"
    "margin:0 0 0.2rem;letter-spacing:-0.02em;'>🎓 Asistente Brightspace</h2>"
    "<p style='color:#6b6b6b;font-size:0.88rem;margin:0 0 1.2rem;'>"
    "Resuelvo dudas sobre la plataforma · Respuestas basadas en la documentación oficial</p>",
    unsafe_allow_html=True,
)

st.divider()

# ── Carga del motor ──────────────────────────────────────────────────

if not st.session_state.engine_ready:
    with st.spinner("Iniciando asistente…"):
        rag, cfg, index, meta, embedder, err = _load_engine()
    if err:
        st.error(f"No se pudo iniciar el asistente.\n\n```\n{err}\n```")
        st.stop()
    st.session_state.update({
        "engine_ready": True,
        "rag": rag, "cfg": cfg,
        "index": index, "meta": meta, "embedder": embedder,
    })

rag      = st.session_state["rag"]
cfg      = st.session_state["cfg"]
index    = st.session_state["index"]
meta     = st.session_state["meta"]
embedder = st.session_state["embedder"]

# ── Sugerencias (solo cuando el chat está vacío) ─────────────────────

SUGGESTIONS = [
    "¿Cómo activo mi curso para los estudiantes?",
    "¿Cómo exporto las calificaciones?",
    "¿Cómo configuro los intentos de un cuestionario?",
    "No puedo iniciar sesión, ¿qué hago?",
]

if not st.session_state.messages:
    st.caption("PREGUNTAS FRECUENTES")
    c1, c2 = st.columns(2)
    for i, sug in enumerate(SUGGESTIONS):
        col = c1 if i % 2 == 0 else c2
        if col.button(sug, key=f"sug_{i}"):
            st.session_state.messages.append({"role": "user", "content": sug})
            st.rerun()
    st.markdown("<br>", unsafe_allow_html=True)

# ── Historial ────────────────────────────────────────────────────────

def _clean_title(raw: str) -> str:
    t = raw.replace("_", " ").replace("-Brightspace", "")
    if "__" in t:
        t = t[:t.rfind("__")]
    return t.replace(".txt", "").replace(".xlsx", "").strip()

for msg in st.session_state.messages:
    role = msg["role"]

    with st.chat_message(role, avatar="🎓" if role == "assistant" else "👤"):
        st.markdown(msg["content"])

        # Fuentes (solo en mensajes del asistente)
        if role == "assistant":
            sources = msg.get("sources", [])
            routed  = msg.get("routed_to_human", False)

            if routed:
                st.warning("No encontré evidencia suficiente. Te recomiendo contactar a soporte.")

            elif sources:
                seen, pills = set(), []
                for s in sources:
                    t = _clean_title(s["title"])
                    if t and t not in seen and len(pills) < 4:
                        seen.add(t)
                        pills.append(t)
                if pills:
                    pills_html = "".join(
                        f'<span class="source-pill">📄 {t}</span>' for t in pills
                    )
                    st.markdown(
                        f'<div class="source-pills">{pills_html}</div>',
                        unsafe_allow_html=True,
                    )

# ── Input ────────────────────────────────────────────────────────────

query = st.chat_input("Escribe tu pregunta sobre Brightspace…")

if query:
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("assistant", avatar="🎓"):
        with st.spinner(""):
            try:
                result = rag.rag_answer(cfg, query, embedder, index, meta)
            except Exception as e:
                result = {
                    "answer":          f"Ocurrió un error. Por favor intenta de nuevo.\n\n`{e}`",
                    "routed_to_human": True,
                    "sources":         [],
                    "confidence":      0,
                    "latency_s":       0,
                    "has_citations":   False,
                }

        st.markdown(result["answer"])

        if result.get("routed_to_human"):
            st.warning("No encontré evidencia suficiente. Te recomiendo contactar a soporte.")
        elif result.get("sources"):
            seen, pills = set(), []
            for s in result["sources"]:
                t = _clean_title(s["title"])
                if t and t not in seen and len(pills) < 4:
                    seen.add(t)
                    pills.append(t)
            if pills:
                pills_html = "".join(
                    f'<span class="source-pill">📄 {t}</span>' for t in pills
                )
                st.markdown(
                    f'<div class="source-pills">{pills_html}</div>',
                    unsafe_allow_html=True,
                )

    st.session_state.messages.append({
        "role":            "assistant",
        "content":         result["answer"],
        "routed_to_human": result["routed_to_human"],
        "sources":         result["sources"],
        "confidence":      result["confidence"],
        "latency_s":       result["latency_s"],
    })
    st.rerun()
