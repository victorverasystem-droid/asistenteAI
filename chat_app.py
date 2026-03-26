"""
chat_app.py — Asistente Brightspace · usuarios finales
Solo chat, sin configuración. Usa componentes nativos de Streamlit.

Mejoras:
  - Semáforo de confianza (🔴🟡🟢) con barra visual animada
  - Fuentes con enlaces clicables (href al documento)
  - Badge de latencia
  - Diseño refinado con tipografía Lora + DM Sans

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

# ── Detector de saludos y despedidas ────────────────────────────────

import re as _re
import random as _random

# Palabras clave de saludo al inicio del mensaje
_GREET_START = _re.compile(
    r"^\s*(hola|buenas?|buenos?\s+(días?|tardes?|noches?)|saludos?|hey|hi|hello|buen\s+día)",
    _re.IGNORECASE | _re.UNICODE,
)

# Frases que son SOLO saludo/cortesía (sin contenido real)
_GREET_ONLY = _re.compile(
    r"^\s*("
    r"hola|buenas?|buenos?\s+(días?|tardes?|noches?)|saludos?|hey|hi|hello|buen\s+día|"
    r"qué\s+tal|cómo\s+estás?|como\s+estás?|cómo\s+te\s+va|todo\s+bien|"
    r"hola[,\s]+cómo\s+estás?|hola[,\s]+como\s+estás?|"
    r"buenas[,\s]+cómo\s+estás?|buenas[,\s]+como\s+estás?|"
    r"hola[,\s]+qué\s+tal|buenas[,\s]+qué\s+tal"
    r")[!?¡¿.,\s]*$",
    _re.IGNORECASE | _re.UNICODE,
)

_BYE_ONLY = _re.compile(
    r"^\s*("
    r"adios?|adiós?|hasta\s+(luego|pronto|mañana|la\s+vista)|"
    r"chao|chau|bye|goodbye|nos\s+vemos?|hasta\s+pronto|"
    r"fue\s+un\s+placer|un\s+placer|que\s+tengas?\s+(buen|buena)"
    r")[!?¡¿.,\s]*$",
    _re.IGNORECASE | _re.UNICODE,
)

_THANKS_ONLY = _re.compile(
    r"^\s*("
    r"muchas?\s+gracias?|gracias?|thanks?|thank\s+you|de\s+nada|"
    r"te\s+lo\s+agradezco|muy\s+amable|excelente[,\s]+gracias?"
    r")[!?¡¿.,\s]*$",
    _re.IGNORECASE | _re.UNICODE,
)

# Palabras que indican que hay una pregunta real sobre la plataforma
_HAS_REAL_QUERY = _re.compile(
    r"(cómo|como|cuándo|cuando|dónde|donde|qué|que|cuál|cual|puedo|puede|"
    r"necesito|quiero|ayuda|problema|error|configurar|crear|exportar|importar|"
    r"activar|calificaci|cuestionario|curso|aula|usuario|login|acceso|brightspace)",
    _re.IGNORECASE | _re.UNICODE,
)

_GREET_REPLIES = [
    "¡Hola! 👋 Soy el asistente de Brightspace. ¿En qué puedo ayudarte hoy?",
    "¡Buenas! Estoy aquí para ayudarte con cualquier duda sobre Brightspace. ¿Qué necesitas?",
    "¡Hola! ¿Tienes alguna pregunta sobre la plataforma? Con gusto te ayudo. 😊",
    "¡Bienvenido/a! Puedo ayudarte con cursos, calificaciones, cuestionarios y más. ¿Por dónde empezamos?",
]

_BYE_REPLIES = [
    "¡Hasta luego! Fue un gusto ayudarte. Si tienes más dudas sobre Brightspace, aquí estaré. 👋",
    "¡Cuídate mucho! Cualquier otra consulta sobre la plataforma, no dudes en escribir. 😊",
    "¡Hasta pronto! Espero haber sido de ayuda. 🎓",
    "Con mucho gusto. ¡Que tengas un excelente día! Si necesitas algo más, aquí estoy. 👋",
]

_THANKS_REPLIES = [
    "¡De nada! 😊 Si tienes más preguntas sobre Brightspace, con gusto te ayudo.",
    "¡Un placer! Para eso estoy. ¿Hay algo más en lo que pueda ayudarte?",
    "¡Me alegra haber podido ayudar! Escríbeme cuando necesites. 🎓",
]

def _is_small_talk(text: str) -> str | None:
    """
    Retorna la categoría ('greet', 'bye', 'thanks') si el mensaje es
    puro small talk sin contenido real sobre Brightspace.
    Retorna None si debe procesarse con el RAG.

    Lógica:
    1. Si coincide exactamente con patrón de saludo/despedida/gracias → small talk.
    2. Si empieza con saludo pero también tiene palabras clave de consulta real → RAG.
    """
    t = text.strip()

    # Agradecimientos puros
    if _THANKS_ONLY.match(t):
        return "thanks"

    # Despedidas puras
    if _BYE_ONLY.match(t):
        return "bye"

    # Saludos puros (sin pregunta real adjunta)
    if _GREET_ONLY.match(t):
        return "greet"

    # Mensaje que empieza con saludo pero contiene consulta real → RAG
    # (ej: "hola, ¿cómo exporto las calificaciones?")
    if _GREET_START.match(t) and _HAS_REAL_QUERY.search(t):
        return None

    # Mensaje corto (<= 6 palabras) que empieza con saludo y no tiene
    # palabras de consulta → probablemente es un saludo informal
    if _GREET_START.match(t) and len(t.split()) <= 6 and not _HAS_REAL_QUERY.search(t):
        return "greet"

    return None

def _small_talk_result(kind: str) -> dict:
    """Construye un resultado sintético (sin RAG) para small talk."""
    if kind == "greet":
        answer = _random.choice(_GREET_REPLIES)
    elif kind == "thanks":
        answer = _random.choice(_THANKS_REPLIES)
    else:
        answer = _random.choice(_BYE_REPLIES)
    return {
        "answer":          answer,
        "routed_to_human": False,
        "sources":         [],
        "confidence":      1.0,   # confianza máxima, pero no se muestra
        "latency_s":       0.0,
        "has_citations":   False,
        "is_small_talk":   True,  # flag para suprimir semáforo y fuentes
    }


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

# ── Helpers: semáforo y fuentes ──────────────────────────────────────

def _confidence_badge(confidence: float, latency_s: float) -> str:
    """
    Devuelve HTML con:
      - Semáforo de color según nivel de confianza
      - Barra de progreso animada
      - Etiqueta textual + latencia
    Umbrales: alto >= 0.70 | medio >= 0.45 | bajo < 0.45
    """
    pct = min(max(int(confidence * 100), 0), 100)

    if confidence >= 0.70:
        dot_color  = "#22c55e"   # verde
        bar_color  = "#22c55e"
        label      = "Alta confianza"
        text_color = "#15803d"
        bg_color   = "#f0fdf4"
        border_col = "#bbf7d0"
    elif confidence >= 0.45:
        dot_color  = "#f59e0b"   # ambar
        bar_color  = "#f59e0b"
        label      = "Confianza media"
        text_color = "#b45309"
        bg_color   = "#fffbeb"
        border_col = "#fde68a"
    else:
        dot_color  = "#ef4444"   # rojo
        bar_color  = "#ef4444"
        label      = "Confianza baja"
        text_color = "#b91c1c"
        bg_color   = "#fef2f2"
        border_col = "#fecaca"

    lat_str = f"{latency_s:.1f}s" if latency_s else ""

    return f"""
<div style="
    display:flex; align-items:center; gap:0.7rem;
    background:{bg_color}; border:1px solid {border_col};
    border-radius:10px; padding:0.45rem 0.75rem;
    margin-top:0.65rem; width:fit-content; max-width:100%;
">
  <span style="
      width:11px; height:11px; border-radius:50%;
      background:{dot_color};
      box-shadow: 0 0 0 3px {dot_color}33;
      flex-shrink:0;
  "></span>

  <div style="flex:1; min-width:120px;">
    <div style="
        display:flex; justify-content:space-between; align-items:center;
        margin-bottom:3px;
    ">
      <span style="font-size:0.73rem; font-weight:600; color:{text_color}; letter-spacing:0.01em;">
          {label}
      </span>
      <span style="font-size:0.70rem; color:#9ca3af; margin-left:0.5rem;">
          {pct}%{"  &middot;  " + lat_str if lat_str else ""}
      </span>
    </div>
    <div style="
        height:4px; border-radius:4px;
        background:#e5e7eb; overflow:hidden; width:180px;
    ">
      <div style="
          height:100%; border-radius:4px;
          width:{pct}%; background:{bar_color};
          transition: width 0.6s cubic-bezier(.4,0,.2,1);
      "></div>
    </div>
  </div>
</div>
"""


def _sources_html(sources: list[dict]) -> str:
    """
    Devuelve HTML con pills de fuentes.
    - Si la fuente tiene 'url' o 'link' HTTP  → pill azul clicable con icono externo.
    - Si la fuente tiene 'path' local          → pill violeta (no clicable en browser).
    - Sin URL                                   → pill gris neutro.
    """
    seen, pills = set(), []
    for s in sources:
        raw_title = s.get("title", "")
        t = raw_title.replace("_", " ").replace("-Brightspace", "")
        if "__" in t:
            t = t[:t.rfind("__")]
        t = t.replace(".txt", "").replace(".xlsx", "").strip()

        if not t or t in seen or len(pills) >= 5:
            continue
        seen.add(t)

        url = s.get("url") or s.get("link") or s.get("path") or ""

        if url and url.startswith("http"):
            pill = (
                f'<a href="{url}" target="_blank" rel="noopener noreferrer" '
                f'class="source-pill source-pill--link" title="Abrir: {t}">'
                f'<span>&#128196;</span> {t} '
                f'<span class="pill-ext-icon">&#8599;</span></a>'
            )
        elif url:
            pill = (
                f'<span class="source-pill source-pill--local" title="Archivo local: {url}">'
                f'&#128193; {t}</span>'
            )
        else:
            pill = f'<span class="source-pill">&#128196; {t}</span>'

        pills.append(pill)

    if not pills:
        return ""

    pills_html = "".join(pills)
    return (
        '<div style="margin-top:0.5rem;">'
        '<span style="font-size:0.70rem; color:#9ca3af; letter-spacing:0.04em;'
        'text-transform:uppercase; font-weight:600;">Fuentes</span>'
        f'<div class="source-pills" style="margin-top:0.3rem;">{pills_html}</div>'
        '</div>'
    )

# ── Página ───────────────────────────────────────────────────────────

st.set_page_config(
    page_title = "Asistente Brightspace",
    page_icon  = "🎓",
    layout     = "centered",
    initial_sidebar_state = "collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Lora:wght@600&family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;1,400&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background: #FAFAF8;
}

header[data-testid="stHeader"] { display: none; }
section[data-testid="stSidebar"] { display: none; }
.block-container { padding-top: 1.5rem !important; max-width: 740px !important; }

[data-testid="stChatMessageAvatarAssistant"] {
    background: #1a3a2e !important;
    border-radius: 10px !important;
}

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

[data-testid="stChatInput"] {
    border: 1px solid #e0dbd1 !important;
    border-radius: 12px !important;
    background: #ffffff !important;
}

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

/* ── Pills de fuentes ── */
.source-pills {
    display: flex;
    flex-wrap: wrap;
    gap: 0.4rem;
}

.source-pill {
    border-radius: 20px;
    padding: 3px 11px;
    font-size: 0.72rem;
    display: inline-flex;
    align-items: center;
    gap: 0.25rem;
    font-family: 'DM Sans', sans-serif;
    font-weight: 500;
    transition: all 0.15s ease;
    text-decoration: none !important;
}

/* Sin enlace */
.source-pill:not(a) {
    background: #f2efe8;
    border: 1px solid #ddd9cf;
    color: #5a5a5a;
    cursor: default;
}

/* Con enlace HTTP */
a.source-pill--link {
    background: #edf6ff;
    border: 1px solid #bfdbfe;
    color: #1d4ed8 !important;
    cursor: pointer;
}
a.source-pill--link:hover {
    background: #dbeafe;
    border-color: #93c5fd;
    color: #1e40af !important;
    text-decoration: none !important;
    transform: translateY(-1px);
    box-shadow: 0 2px 6px rgba(29,78,216,0.12);
}
.pill-ext-icon {
    font-size: 0.65rem;
    opacity: 0.7;
}

/* Ruta local */
span.source-pill--local {
    background: #f5f3ff;
    border: 1px solid #ddd6fe;
    color: #6d28d9;
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

for msg in st.session_state.messages:
    role = msg["role"]

    with st.chat_message(role, avatar="🎓" if role == "assistant" else "👤"):
        st.markdown(msg["content"])

        if role == "assistant":
            routed     = msg.get("routed_to_human", False)
            latency_s  = msg.get("latency_s", 0.0)
            is_small   = msg.get("is_small_talk", False)

            # Nunca mostrar semáforo ni fuentes si no hay evidencia real
            if routed or is_small:
                confidence = 0.0
                sources    = []
            else:
                confidence = msg.get("confidence", 0.0)
                sources    = msg.get("sources", [])

            if routed:
                st.warning("No encontré evidencia suficiente. Te recomiendo contactar a soporte.")
            elif not is_small:
                if confidence is not None and confidence > 0:
                    st.markdown(
                        _confidence_badge(confidence, latency_s),
                        unsafe_allow_html=True,
                    )
                if sources:
                    src_html = _sources_html(sources)
                    if src_html:
                        st.markdown(src_html, unsafe_allow_html=True)

# ── Input ────────────────────────────────────────────────────────────

query = st.chat_input("Escribe tu pregunta sobre Brightspace…")

if query:
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("assistant", avatar="🎓"):
        with st.spinner(""):
            # ── Detector de small talk (saludo / despedida / gracias) ──
            small_talk_kind = _is_small_talk(query)
            if small_talk_kind:
                result = _small_talk_result(small_talk_kind)
            else:
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

        routed        = result.get("routed_to_human", False)
        latency_s     = result.get("latency_s", 0.0)
        is_small_talk = result.get("is_small_talk", False)

        # Si no hay evidencia real o es small talk: limpiar confidence y fuentes
        if routed or is_small_talk:
            confidence = 0.0
            sources    = []
        else:
            confidence = result.get("confidence", 0.0)
            sources    = result.get("sources", [])

        st.markdown(result["answer"])

        if routed:
            st.warning("No encontré evidencia suficiente. Te recomiendo contactar a soporte.")
        elif not is_small_talk:
            if confidence is not None and confidence > 0:
                st.markdown(
                    _confidence_badge(confidence, latency_s),
                    unsafe_allow_html=True,
                )
            if sources:
                src_html = _sources_html(sources)
                if src_html:
                    st.markdown(src_html, unsafe_allow_html=True)

    st.session_state.messages.append({
        "role":            "assistant",
        "content":         result["answer"],
        "routed_to_human": routed,
        "sources":         sources,
        "confidence":      confidence,
        "latency_s":       latency_s,
        "is_small_talk":   result.get("is_small_talk", False),
    })
    st.rerun()
