"""
streamlit_app.py  —  UI para asistente_rag_project_v3
Ejecutar: streamlit run streamlit_app.py
"""

import streamlit as st
import os, sys, json, time, asyncio, tempfile, shutil
from pathlib import Path

# ── Agrega el directorio actual al path para importar el módulo RAG ──
sys.path.insert(0, os.path.dirname(__file__))

# ── Importación lazy del módulo RAG ──
@st.cache_resource(show_spinner="Cargando módulo RAG…")
def import_rag():
    import asistente_rag_project_v3 as rag
    return rag

# ══════════════════════════════════════════
# Configuración de página
# ══════════════════════════════════════════
st.set_page_config(
    page_title="Asistente RAG",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════
# CSS personalizado
# ══════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0f1117;
    border-right: 1px solid #2a2d3a;
}
section[data-testid="stSidebar"] * {
    color: #e0e0e0 !important;
}
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] .stTextInput label {
    color: #8892a0 !important;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* Header banner */
.rag-header {
    background: linear-gradient(135deg, #0f1117 0%, #1a1f2e 60%, #0d1b2a 100%);
    border: 1px solid #2a2d3a;
    border-radius: 10px;
    padding: 1.4rem 2rem;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    gap: 1rem;
}
.rag-header h1 {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.6rem;
    color: #58a6ff;
    margin: 0;
    letter-spacing: -0.02em;
}
.rag-header p {
    margin: 0;
    color: #6e7681;
    font-size: 0.88rem;
}

/* Metric cards */
.metric-row {
    display: flex;
    gap: 0.8rem;
    margin: 1rem 0;
    flex-wrap: wrap;
}
.metric-card {
    background: #161b22;
    border: 1px solid #2a2d3a;
    border-radius: 8px;
    padding: 0.6rem 1rem;
    flex: 1;
    min-width: 120px;
    text-align: center;
}
.metric-card .val {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.4rem;
    font-weight: 600;
    color: #58a6ff;
}
.metric-card .lbl {
    font-size: 0.72rem;
    color: #6e7681;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

/* Answer box */
.answer-box {
    background: #0d1117;
    border-left: 3px solid #58a6ff;
    border-radius: 0 8px 8px 0;
    padding: 1.2rem 1.5rem;
    margin: 0.8rem 0;
    color: #c9d1d9;
    line-height: 1.7;
    white-space: pre-wrap;
    font-size: 0.95rem;
}

/* Source cards */
.source-card {
    background: #161b22;
    border: 1px solid #2a2d3a;
    border-radius: 6px;
    padding: 0.75rem 1rem;
    margin: 0.4rem 0;
    transition: border-color 0.2s;
}
.source-card:hover { border-color: #58a6ff44; }
.source-card .src-title {
    font-weight: 600;
    color: #c9d1d9;
    font-size: 0.88rem;
}
.source-card .src-meta {
    color: #6e7681;
    font-size: 0.78rem;
    font-family: 'IBM Plex Mono', monospace;
    margin-top: 0.2rem;
}
.source-card .src-snippet {
    color: #8892a0;
    font-size: 0.82rem;
    margin-top: 0.4rem;
    line-height: 1.5;
    border-top: 1px solid #2a2d3a;
    padding-top: 0.4rem;
}
.badge-routed { background:#da3633; color:#fff; border-radius:4px; padding:2px 8px; font-size:0.7rem; }
.badge-ok     { background:#238636; color:#fff; border-radius:4px; padding:2px 8px; font-size:0.7rem; }
.badge-warn   { background:#9e6a03; color:#fff; border-radius:4px; padding:2px 8px; font-size:0.7rem; }

/* Chat history bubbles */
.bubble-user {
    background: #1f2d3d;
    border-radius: 12px 12px 2px 12px;
    padding: 0.7rem 1rem;
    margin: 0.5rem 0 0.5rem auto;
    max-width: 75%;
    color: #c9d1d9;
    font-size: 0.92rem;
    text-align: right;
}
.bubble-assistant {
    background: #161b22;
    border: 1px solid #2a2d3a;
    border-radius: 12px 12px 12px 2px;
    padding: 0.7rem 1rem;
    margin: 0.5rem 0;
    max-width: 85%;
    color: #c9d1d9;
    font-size: 0.92rem;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════
# Estado de sesión
# ══════════════════════════════════════════
for key, default in {
    "messages": [],
    "index_loaded": False,
    "index": None,
    "meta": None,
    "embedder": None,
    "settings": None,
    "build_log": "",
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ══════════════════════════════════════════
# Sidebar — Configuración
# ══════════════════════════════════════════
with st.sidebar:
    st.markdown("### ⚙️ Configuración RAG")

    st.markdown("**LLM**")
    llm_base_url = st.text_input("Base URL", value="https://api.openai.com/v1", key="llm_url")
    llm_api_key  = st.text_input("API Key", type="password", key="llm_key")
    llm_model    = st.text_input("Modelo", value="gpt-4o-mini", key="llm_mod")
    use_llm      = st.toggle("Usar LLM", value=True)

    st.divider()
    st.markdown("**Recuperación**")
    top_k   = st.slider("Top-K fuentes", 1, 10, 5)
    min_sim = st.slider("Similitud mínima", 0.1, 0.9, 0.35, 0.05)

    st.divider()
    st.markdown("**Índice**")
    index_dir = st.text_input("Directorio índice", value="./data/index")
    raw_dir   = st.text_input("Directorio datos raw", value="./data/raw")

    st.divider()
    st.markdown("**Subir documentos**")
    uploaded_files = st.file_uploader(
        "PDF / DOCX / TXT / HTML / XLSX / CSV / JSON",
        accept_multiple_files=True,
        type=["pdf","docx","txt","html","htm","xlsx","xls","csv","json"],
    )

    col1, col2 = st.columns(2)
    btn_save  = col1.button("💾 Guardar", use_container_width=True)
    btn_build = col2.button("🔨 Indexar", use_container_width=True)
    btn_load  = st.button("⚡ Cargar índice", use_container_width=True)

    if btn_save and uploaded_files:
        os.makedirs(raw_dir, exist_ok=True)
        saved = 0
        for f in uploaded_files:
            dest = os.path.join(raw_dir, f.name)
            with open(dest, "wb") as out:
                out.write(f.read())
            saved += 1
        st.success(f"✅ {saved} archivo(s) guardado(s) en `{raw_dir}`")

    if btn_build:
        rag = import_rag()
        cfg = rag.Settings(
            raw_dir=raw_dir,
            index_dir=index_dir,
            use_llm=use_llm,
            llm_base_url=llm_base_url,
            llm_api_key=llm_api_key or os.getenv("OPENAI_API_KEY",""),
            llm_model=llm_model,
            top_k=top_k,
            min_sim=min_sim,
        )
        with st.spinner("Construyendo índice FAISS…"):
            try:
                _, _, n = rag.build_faiss_index(cfg)
                st.success(f"✅ Índice construido — {n} chunks")
            except Exception as e:
                st.error(f"Error al indexar: {e}")

    if btn_load:
        rag = import_rag()
        cfg = rag.Settings(
            raw_dir=raw_dir,
            index_dir=index_dir,
            use_llm=use_llm,
            llm_base_url=llm_base_url,
            llm_api_key=llm_api_key or os.getenv("OPENAI_API_KEY",""),
            llm_model=llm_model,
            top_k=top_k,
            min_sim=min_sim,
        )
        with st.spinner("Cargando índice y modelo de embeddings…"):
            try:
                from sentence_transformers import SentenceTransformer
                idx, meta = rag.load_index(cfg)
                embedder = SentenceTransformer(cfg.embed_model)
                st.session_state.update({
                    "index_loaded": True,
                    "index": idx,
                    "meta": meta,
                    "embedder": embedder,
                    "settings": cfg,
                })
                st.success(f"✅ Índice cargado — {len(meta)} chunks")
            except Exception as e:
                st.error(f"Error: {e}")

    st.divider()
    if st.button("🗑️ Limpiar chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ══════════════════════════════════════════
# Main — Header
# ══════════════════════════════════════════
st.markdown("""
<div class="rag-header">
  <div>
    <h1>🔍 Asistente RAG</h1>
    <p>Recuperación aumentada con generación · FAISS + multilingual-e5 · anti-alucinación</p>
  </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════
# Pestañas
# ══════════════════════════════════════════
tab_chat, tab_sources, tab_about = st.tabs(["💬 Chat", "📚 Fuentes recuperadas", "ℹ️ Acerca de"])

# ── TAB CHAT ──────────────────────────────
with tab_chat:
    status_col, _ = st.columns([3,1])
    with status_col:
        if st.session_state.index_loaded:
            n_chunks = len(st.session_state.meta)
            st.markdown(f'<span class="badge-ok">● Índice activo — {n_chunks} chunks</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="badge-warn">● Sin índice · Carga uno desde el panel lateral</span>', unsafe_allow_html=True)

    st.markdown("---")

    # Historial de mensajes
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f'<div class="bubble-user">{msg["content"]}</div>', unsafe_allow_html=True)
        else:
            with st.container():
                answer = msg["content"]
                conf   = msg.get("confidence", 0)
                routed = msg.get("routed_to_human", False)
                lat    = msg.get("latency_s", 0)

                badge = '<span class="badge-routed">↗ Derivado a humano</span>' if routed else '<span class="badge-ok">✓ Respondido</span>'
                st.markdown(f'{badge} &nbsp; confianza <b>{conf:.0%}</b> &nbsp; ⏱ {lat:.2f}s', unsafe_allow_html=True)
                st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)

                if msg.get("sources"):
                    with st.expander(f"📎 {len(msg['sources'])} fuentes utilizadas"):
                        for i, src in enumerate(msg["sources"], 1):
                            t    = src["meta"].get("type","")
                            cat  = src["meta"].get("categoria","")
                            meta_str = f"type={t}" + (f" | cat={cat}" if cat else "")
                            st.markdown(f"""
<div class="source-card">
  <div class="src-title">[{i}] {src['title']} › {src['section']}</div>
  <div class="src-meta">score={src['score']:.3f} | {meta_str}</div>
  <div class="src-snippet">{src['snippet']}</div>
</div>
""", unsafe_allow_html=True)

    # Input
    query = st.chat_input("Escribe tu pregunta…", disabled=not st.session_state.index_loaded)

    if query:
        st.session_state.messages.append({"role": "user", "content": query})

        rag = import_rag()
        cfg = st.session_state.settings
        # Actualizar settings dinámicos desde sidebar
        cfg.top_k   = top_k
        cfg.min_sim = min_sim
        cfg.use_llm = use_llm
        cfg.llm_api_key = llm_api_key or os.getenv("OPENAI_API_KEY","")

        with st.spinner("Buscando respuesta…"):
            try:
                result = rag.rag_answer(
                    cfg, query,
                    st.session_state.embedder,
                    st.session_state.index,
                    st.session_state.meta,
                )
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "confidence": result["confidence"],
                    "routed_to_human": result["routed_to_human"],
                    "latency_s": result["latency_s"],
                    "has_citations": result["has_citations"],
                    "sources": result["sources"],
                })
            except Exception as e:
                st.error(f"Error en RAG: {e}")

        st.rerun()

# ── TAB FUENTES ───────────────────────────
with tab_sources:
    if not st.session_state.index_loaded:
        st.info("Carga un índice primero para explorar las fuentes.")
    else:
        meta_list = st.session_state.meta
        st.markdown(f"**{len(meta_list)} chunks** indexados en total.")

        # Filtros
        fc1, fc2 = st.columns(2)
        types = sorted(set(m["meta"].get("type","") for m in meta_list if m["meta"].get("type","")))
        ftype = fc1.selectbox("Tipo de documento", ["(todos)"] + types)
        fsearch = fc2.text_input("🔍 Buscar en título/texto")

        filtered = meta_list
        if ftype != "(todos)":
            filtered = [m for m in filtered if m["meta"].get("type","") == ftype]
        if fsearch:
            q = fsearch.lower()
            filtered = [m for m in filtered if q in m["title"].lower() or q in m["text"].lower()]

        st.markdown(f"Mostrando **{min(50, len(filtered))}** de {len(filtered)} resultados.")
        for item in filtered[:50]:
            t   = item["meta"].get("type","")
            cat = item["meta"].get("categoria","")
            with st.expander(f"[{t}] {item['title']} › {item['section']}" + (f" — {cat}" if cat else "")):
                st.text(item["text"][:600] + ("…" if len(item["text"]) > 600 else ""))

# ── TAB ABOUT ────────────────────────────
with tab_about:
    st.markdown("""
## Asistente RAG — Pipeline completo

Este asistente implementa un sistema de **Retrieval-Augmented Generation** con:

| Componente | Detalle |
|---|---|
| Embeddings | `intfloat/multilingual-e5-base` (multilingual) |
| Vector store | FAISS `IndexFlatIP` (producto interno) |
| LLM | OpenAI-compatible (gpt-4o-mini por defecto) |
| Anti-alucinación | Guardrail `MIN_SIM` — derivación a humano si score < umbral |
| Formatos soportados | PDF, DOCX, TXT, HTML, XLSX (FAQ), CSV/JSON (tickets) |

### Flujo de trabajo

1. **Subir documentos** → panel lateral → guardar en `raw_dir`
2. **Indexar** → botón 🔨 genera el índice FAISS
3. **Cargar índice** → botón ⚡ carga modelo + índice en memoria
4. **Chatear** → el chat recupera chunks relevantes y genera respuestas citadas

### Variables de entorno

```bash
OPENAI_API_KEY=sk-...    # O ingresa la key en el panel lateral
```
    """)
