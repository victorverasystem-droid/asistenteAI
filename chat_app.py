"""
chat_app.py — Asistente Brightspace para usuarios finales
Solo chat, sin configuración. Todo se carga automáticamente.

Estructura del repo (igual que streamlit_app.py):
  ├── asistente_rag_project_v3.py
  ├── chat_app.py               ← este archivo
  ├── requirements.txt
  ├── raw/
  ├── index/
  └── eval/

Secrets de Streamlit Cloud (Settings → Secrets):
  OPENAI_API_KEY = "sk-..."
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st

# ══════════════════════════════════════════════════════════════════════
# Rutas — misma lógica robusta que streamlit_app.py
# ══════════════════════════════════════════════════════════════════════

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

# ══════════════════════════════════════════════════════════════════════
# Carga del índice y modelo (una sola vez, cacheado)
# ══════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def _load_engine():
    import traceback
    try:
        import asistente_rag_project_v3 as rag
        from sentence_transformers import SentenceTransformer

        cfg = rag.Settings(
            index_dir    = str(INDEX_DIR),
            log_dir      = str(LOG_DIR),
            use_llm      = True,
            llm_api_key  = os.getenv("OPENAI_API_KEY", ""),
            llm_model    = "gpt-4o-mini",
            top_k        = 5,
            min_sim      = 0.35,
        )
        index, meta = rag.load_index(cfg)
        embedder     = SentenceTransformer(cfg.embed_model)
        return rag, cfg, index, meta, embedder, None
    except Exception as e:
        return None, None, None, None, None, traceback.format_exc()

# ══════════════════════════════════════════════════════════════════════
# Página
# ══════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Asistente Brightspace",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Lora:ital,wght@0,400;0,600;1,400&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Reset & base ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background: #FAFAF8;
    color: #1a1a1a;
}

/* ── Ocultar header de Streamlit y barra lateral ── */
header[data-testid="stHeader"] { display: none; }
section[data-testid="stSidebar"] { display: none; }
.block-container { padding-top: 2rem !important; max-width: 760px !important; }

/* ── Header de la app ── */
.chat-header {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
    border-bottom: 1px solid #e8e4dc;
    margin-bottom: 1.5rem;
}
.chat-header .logo {
    width: 52px; height: 52px;
    background: #1a3a2e;
    border-radius: 14px;
    display: inline-flex; align-items: center; justify-content: center;
    font-size: 1.6rem; margin-bottom: 0.8rem;
    box-shadow: 0 4px 16px rgba(26,58,46,0.18);
}
.chat-header h1 {
    font-family: 'Lora', serif;
    font-size: 1.6rem; font-weight: 600;
    color: #1a1a1a; margin: 0 0 0.3rem;
    letter-spacing: -0.02em;
}
.chat-header p {
    color: #6b6b6b; font-size: 0.9rem;
    font-weight: 300; margin: 0;
}

/* ── Burbuja usuario ── */
.bubble-user-wrap {
    display: flex; justify-content: flex-end;
    margin: 0.9rem 0;
}
.bubble-user {
    background: #1a3a2e;
    color: #f0ede6;
    border-radius: 18px 18px 4px 18px;
    padding: 0.75rem 1.1rem;
    max-width: 78%;
    font-size: 0.95rem; line-height: 1.6;
    box-shadow: 0 2px 8px rgba(26,58,46,0.15);
}

/* ── Burbuja asistente ── */
.bubble-assistant-wrap {
    display: flex; align-items: flex-start; gap: 0.7rem;
    margin: 0.9rem 0;
}
.bubble-avatar {
    width: 32px; height: 32px; flex-shrink: 0;
    background: #1a3a2e; border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1rem; margin-top: 2px;
}
.bubble-assistant {
    background: #ffffff;
    border: 1px solid #e8e4dc;
    border-radius: 4px 18px 18px 18px;
    padding: 0.75rem 1.1rem;
    max-width: 84%;
    font-size: 0.95rem; line-height: 1.7;
    color: #1a1a1a;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    white-space: pre-wrap;
}

/* ── Pill de fuentes ── */
.sources-row {
    display: flex; flex-wrap: wrap; gap: 0.4rem;
    margin-top: 0.6rem; margin-left: 2.5rem;
}
.source-pill {
    background: #f2efe8; border: 1px solid #ddd9cf;
    border-radius: 20px; padding: 3px 10px;
    font-size: 0.72rem; color: #5a5a5a;
    cursor: default;
}
.source-pill:hover { background: #e8e4dc; }

/* ── Badge derivado ── */
.badge-routed {
    display: inline-block;
    background: #fff3cd; border: 1px solid #f0c040;
    color: #7a5800; border-radius: 6px;
    padding: 3px 10px; font-size: 0.75rem;
    margin-bottom: 0.4rem;
}

/* ── Indicador de carga ── */
.typing-indicator {
    display: flex; align-items: center; gap: 0.7rem;
    margin: 0.9rem 0;
}
.typing-dots span {
    display: inline-block; width: 7px; height: 7px;
    background: #1a3a2e; border-radius: 50%;
    animation: bounce 1.2s infinite;
}
.typing-dots span:nth-child(2) { animation-delay: 0.2s; }
.typing-dots span:nth-child(3) { animation-delay: 0.4s; }
@keyframes bounce {
    0%, 60%, 100% { transform: translateY(0); opacity: 0.4; }
    30% { transform: translateY(-6px); opacity: 1; }
}

/* ── Sugerencias de preguntas ── */
.suggestions { margin: 1rem 0 0.5rem; }
.suggestions p {
    font-size: 0.78rem; color: #9a9a9a;
    text-transform: uppercase; letter-spacing: 0.08em;
    margin-bottom: 0.5rem;
}

/* ── Input de chat ── */
.stChatInputContainer {
    border-top: 1px solid #e8e4dc !important;
    background: #FAFAF8 !important;
    padding: 0.8rem 0 !important;
}

/* ── Error box ── */
.error-box {
    background: #fff5f5; border: 1px solid #fca5a5;
    border-radius: 10px; padding: 1rem 1.2rem;
    color: #7f1d1d; font-size: 0.88rem;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# Estado de sesión
# ══════════════════════════════════════════════════════════════════════

if "messages" not in st.session_state:
    st.session_state.messages = []
if "engine_ready" not in st.session_state:
    st.session_state.engine_ready = False

# ══════════════════════════════════════════════════════════════════════
# Header
# ══════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="chat-header">
  <div class="logo">🎓</div>
  <h1>Asistente Brightspace</h1>
  <p>Resuelvo dudas sobre el uso de la plataforma · Respuestas basadas en la documentación oficial</p>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# Carga del motor RAG (con spinner elegante)
# ══════════════════════════════════════════════════════════════════════

if not st.session_state.engine_ready:
    with st.spinner("Iniciando asistente…"):
        rag, cfg, index, meta, embedder, err = _load_engine()

    if err:
        st.markdown(f'<div class="error-box">⚠️ No se pudo iniciar el asistente.<br><br><code>{err}</code></div>',
                    unsafe_allow_html=True)
        st.stop()

    st.session_state.update({
        "engine_ready": True,
        "rag": rag, "cfg": cfg,
        "index": index, "meta": meta, "embedder": embedder,
    })

rag      = st.session_state.get("rag")
cfg      = st.session_state.get("cfg")
index    = st.session_state.get("index")
meta     = st.session_state.get("meta")
embedder = st.session_state.get("embedder")

# ══════════════════════════════════════════════════════════════════════
# Sugerencias (solo cuando no hay mensajes)
# ══════════════════════════════════════════════════════════════════════

SUGGESTIONS = [
    "¿Cómo activo mi curso para que los estudiantes lo vean?",
    "¿Cómo exporto las calificaciones?",
    "¿Cómo configuro los intentos de un cuestionario?",
    "No puedo iniciar sesión, ¿qué hago?",
]

if not st.session_state.messages:
    st.markdown('<div class="suggestions"><p>Preguntas frecuentes</p></div>', unsafe_allow_html=True)
    cols = st.columns(2)
    for i, sug in enumerate(SUGGESTIONS):
        if cols[i % 2].button(sug, key=f"sug_{i}", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": sug})
            st.rerun()

# ══════════════════════════════════════════════════════════════════════
# Historial de mensajes
# ══════════════════════════════════════════════════════════════════════

for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"""
<div class="bubble-user-wrap">
  <div class="bubble-user">{msg["content"]}</div>
</div>""", unsafe_allow_html=True)

    else:
        routed  = msg.get("routed_to_human", False)
        sources = msg.get("sources", [])

        badge = '<div class="badge-routed">⚠️ No encontré evidencia suficiente — te recomiendo contactar a soporte.</div>' if routed else ""

        st.markdown(f"""
<div class="bubble-assistant-wrap">
  <div class="bubble-avatar">🎓</div>
  <div>
    {badge}
    <div class="bubble-assistant">{msg["content"]}</div>
  </div>
</div>""", unsafe_allow_html=True)

        # Pills de fuentes (solo los títulos únicos, máximo 4)
        if sources and not routed:
            seen, pills = set(), []
            for s in sources:
                title = s["title"].replace("_", " ").replace("-Brightspace", "").strip()
                # Limpiar hash al final del nombre de archivo .txt
                if "__" in title:
                    title = title[:title.rfind("__")]
                title = title.replace(".txt","").replace(".xlsx","").strip()
                if title not in seen and len(pills) < 4:
                    seen.add(title)
                    pills.append(title)
            if pills:
                pills_html = "".join(f'<span class="source-pill">📄 {t}</span>' for t in pills)
                st.markdown(f'<div class="sources-row">{pills_html}</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# Input de chat
# ══════════════════════════════════════════════════════════════════════

query = st.chat_input("Escribe tu pregunta sobre Brightspace…")

if query:
    st.session_state.messages.append({"role": "user", "content": query})

    with st.spinner(""):
        try:
            result = rag.rag_answer(cfg, query, embedder, index, meta)
        except Exception as e:
            result = {
                "answer": f"Ocurrió un error al procesar tu pregunta. Por favor intenta de nuevo.\n\n_{e}_",
                "routed_to_human": True,
                "sources": [],
                "confidence": 0,
                "latency_s": 0,
                "has_citations": False,
            }

    st.session_state.messages.append({
        "role":            "assistant",
        "content":         result["answer"],
        "routed_to_human": result["routed_to_human"],
        "sources":         result["sources"],
        "confidence":      result["confidence"],
        "latency_s":       result["latency_s"],
    })
    st.rerun()
