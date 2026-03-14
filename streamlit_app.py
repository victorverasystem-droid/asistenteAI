"""
streamlit_app.py  —  Asistente Virtual RAG · Brightspace

Estructura de carpetas compatible con el notebook Colab Y con Streamlit Cloud:

  Repositorio GitHub (raíz = PROJECT_ROOT)
  ├── asistente_rag_project_v3.py   ← módulo RAG principal  ← DEBE ESTAR AQUÍ
  ├── streamlit_app.py              ← esta interfaz
  ├── requirements.txt
  ├── raw/          ← documentos de conocimiento (subidos desde la UI)
  ├── index/        ← índice FAISS persistente
  ├── logs/         ← events.jsonl
  └── eval/         ← intents_priorizado.json · dataset · urls_master.xlsx

  Uso local (equivalente al notebook Colab):
    streamlit run streamlit_app.py

  Streamlit Cloud:
    - Sube todos los archivos a la raíz del repo en GitHub
    - Añade OPENAI_API_KEY en Settings → Secrets

Variables de entorno:
  OPENAI_API_KEY       API Key del LLM
  RAG_PROJECT_ROOT     Override de la raíz (opcional, avanzado)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import streamlit as st

# ══════════════════════════════════════════════════════════════════════
# Resolución de rutas
# Funciona en: Streamlit Cloud, local, Colab-like con scripts/
# ══════════════════════════════════════════════════════════════════════

RAG_MODULE_NAME = "asistente_rag_project_v3.py"

def _resolve_project_root() -> Path:
    """
    Busca la carpeta que CONTIENE asistente_rag_project_v3.py en este orden:
    1. Env var RAG_PROJECT_ROOT  (override explícito)
    2. CLI --project-root
    3. Directorio del propio streamlit_app.py  (Streamlit Cloud: repo raíz)
    4. Directorio padre si streamlit_app.py está en scripts/
    5. cwd
    6. cwd/RAG_Brightspace
    La carpeta que realmente contiene el módulo RAG gana siempre.
    """
    # Candidatos base (orden de preferencia)
    candidates: list[Path] = []

    env = os.getenv("RAG_PROJECT_ROOT", "").strip()
    if env:
        candidates.append(Path(env).expanduser().resolve())

    try:
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--project-root", default=None)
        known, _ = parser.parse_known_args()
        if known.project_root:
            candidates.append(Path(known.project_root).expanduser().resolve())
    except Exception:
        pass

    script_dir = Path(__file__).parent.resolve()
    candidates.append(script_dir)                          # mismo dir que streamlit_app.py
    if script_dir.name == "scripts":
        candidates.append(script_dir.parent)               # repo raíz si está en scripts/
    candidates.append(Path.cwd())
    candidates.append(Path.cwd() / "RAG_Brightspace")

    # La primera que contenga el módulo RAG gana
    for c in candidates:
        if (c / RAG_MODULE_NAME).exists():
            return c

    # Si no encontramos el módulo, devolvemos el directorio del script
    # (el error se mostrará luego con instrucciones claras)
    return script_dir


PROJECT_ROOT = _resolve_project_root()
RAW_DIR    = PROJECT_ROOT / "raw"
INDEX_DIR  = PROJECT_ROOT / "index"
LOG_DIR    = PROJECT_ROOT / "logs"
EVAL_DIR   = PROJECT_ROOT / "eval"
SCRIPT_DIR = PROJECT_ROOT  # en Streamlit Cloud scripts/ == raíz del repo

# Garantizar que PROJECT_ROOT esté en sys.path para importar el módulo RAG
for _d in [str(PROJECT_ROOT), str(Path(__file__).parent)]:
    if _d not in sys.path:
        sys.path.insert(0, _d)

# ══════════════════════════════════════════════════════════════════════
# Importación lazy del módulo RAG
# ══════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Cargando módulo RAG…")
def import_rag():
    import traceback, importlib
    # Verificar si el archivo físicamente existe antes de importar
    rag_file = PROJECT_ROOT / "asistente_rag_project_v3.py"
    if not rag_file.exists():
        # Buscar en todo sys.path como diagnóstico
        found_in = [p for p in sys.path if (Path(p) / "asistente_rag_project_v3.py").exists()]
        st.error(
            f"### Archivo no encontrado\n\n"
            f"Buscado en: `{PROJECT_ROOT}`\n\n"
            f"Encontrado en sys.path: {found_in}\n\n"
            "Sube `asistente_rag_project_v3.py` a la raiz del repositorio."
        )
        st.stop()
    try:
        import asistente_rag_project_v3 as rag
        return rag
    except Exception as e:
        st.error(
            f"### Error al importar el módulo RAG\n\n"
            f"**Archivo:** `{rag_file}`\n\n"
            f"**Error:** `{type(e).__name__}: {e}`\n\n"
            f"```\n{traceback.format_exc()}\n```"
        )
        st.stop()

# ══════════════════════════════════════════════════════════════════════
# Página
# ══════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Asistente Brightspace · RAG",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Diagnóstico siempre visible (ayuda a depurar en Streamlit Cloud) ──────────
with st.sidebar.expander("🔧 Diagnóstico de rutas", expanded=False):
    _rag_file = PROJECT_ROOT / "asistente_rag_project_v3.py"
    st.markdown(f"**PROJECT_ROOT:** `{PROJECT_ROOT}`")
    st.markdown(f"**Módulo RAG existe:** `{_rag_file.exists()}`")
    st.markdown(f"**`__file__`:** `{Path(__file__).resolve()}`")
    st.markdown(f"**cwd:** `{Path.cwd()}`")
    _hits = [p for p in sys.path if (Path(p) / "asistente_rag_project_v3.py").exists()]
    st.markdown(f"**Encontrado en sys.path:** `{_hits}`")
    st.markdown("**sys.path:**")
    st.code("\n".join(sys.path[:12]))


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }

section[data-testid="stSidebar"] { background:#0f1117; border-right:1px solid #21262d; }
section[data-testid="stSidebar"] * { color:#e0e0e0 !important; }
section[data-testid="stSidebar"] label {
    color:#8892a0 !important; font-size:0.76rem;
    text-transform:uppercase; letter-spacing:0.07em;
}

.rag-header {
    background:linear-gradient(135deg,#0d1117 0%,#161b22 70%,#0d1b2a 100%);
    border:1px solid #21262d; border-radius:10px;
    padding:1.3rem 2rem; margin-bottom:1.2rem;
    display:flex; align-items:center; gap:1.2rem;
}
.rag-header h1 { font-family:'IBM Plex Mono',monospace; font-size:1.5rem; color:#58a6ff; margin:0; }
.rag-header p  { margin:0; color:#6e7681; font-size:0.86rem; }

.answer-box {
    background:#0d1117; border-left:3px solid #58a6ff;
    border-radius:0 8px 8px 0; padding:1.1rem 1.4rem;
    margin:0.6rem 0; color:#c9d1d9; line-height:1.75;
    white-space:pre-wrap; font-size:0.94rem;
}
.source-card {
    background:#161b22; border:1px solid #21262d;
    border-radius:6px; padding:0.7rem 1rem; margin:0.35rem 0;
    transition: border-color 0.2s;
}
.source-card:hover { border-color:#58a6ff55; }
.src-title { font-weight:600; color:#c9d1d9; font-size:0.87rem; }
.src-meta  { color:#6e7681; font-size:0.76rem; font-family:'IBM Plex Mono',monospace; margin-top:2px; }
.src-snip  { color:#8892a0; font-size:0.81rem; margin-top:6px; border-top:1px solid #21262d; padding-top:6px; line-height:1.5; }

.badge-ok     { background:#238636; color:#fff; border-radius:4px; padding:2px 8px; font-size:0.7rem; }
.badge-warn   { background:#9e6a03; color:#fff; border-radius:4px; padding:2px 8px; font-size:0.7rem; }
.badge-routed { background:#da3633; color:#fff; border-radius:4px; padding:2px 8px; font-size:0.7rem; }

.bubble-user {
    background:#1f2d3d; border-radius:12px 12px 2px 12px;
    padding:0.65rem 1rem; margin:0.45rem 0 0.45rem auto;
    max-width:75%; color:#c9d1d9; font-size:0.91rem; text-align:right;
}
.bubble-assistant {
    background:#161b22; border:1px solid #21262d;
    border-radius:12px 12px 12px 2px; padding:0.65rem 1rem;
    margin:0.45rem 0; max-width:88%; color:#c9d1d9; font-size:0.91rem;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# Estado de sesión
# ══════════════════════════════════════════════════════════════════════

for k, v in {
    "messages":     [],
    "index_loaded": False,
    "faiss_index":  None,
    "meta":         None,
    "embedder":     None,
    "settings":     None,
    "eval_metrics": None,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ══════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════

def _dir_count(d: Path) -> str:
    if not d.exists(): return "—"
    return str(sum(1 for f in d.iterdir() if f.is_file()))

def _index_exists() -> bool:
    return (INDEX_DIR / "faiss.index").exists() and (INDEX_DIR / "chunks_meta.jsonl").exists()

def _make_settings(rag, **overrides):
    kw = dict(
        raw_dir      = str(RAW_DIR),
        index_dir    = str(INDEX_DIR),
        log_dir      = str(LOG_DIR),
        use_llm      = st.session_state.get("use_llm", True),
        llm_base_url = st.session_state.get("llm_url", "https://api.openai.com/v1"),
        llm_api_key  = st.session_state.get("llm_key", "") or os.getenv("OPENAI_API_KEY", ""),
        llm_model    = st.session_state.get("llm_model", "gpt-4o-mini"),
        top_k        = st.session_state.get("top_k", 5),
        min_sim      = st.session_state.get("min_sim", 0.35),
    )
    kw.update(overrides)
    return rag.Settings(**kw)

# ══════════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### 🎓 RAG · Brightspace")

    with st.expander("📁 Carpetas del proyecto", expanded=False):
        for label, d in [("raw/", RAW_DIR), ("index/", INDEX_DIR),
                          ("logs/", LOG_DIR), ("eval/", EVAL_DIR), ("scripts/", SCRIPT_DIR)]:
            icon = "✅" if d.exists() else "❌"
            st.markdown(f"{icon} `{label}` — {_dir_count(d)} archivos")
        st.caption(f"Raíz: `{PROJECT_ROOT}`")
        if st.button("🏗️ Crear carpetas faltantes", width='stretch'):
            for d in [RAW_DIR, INDEX_DIR, LOG_DIR, EVAL_DIR, SCRIPT_DIR]:
                d.mkdir(parents=True, exist_ok=True)
            st.success("Carpetas creadas.")
            st.rerun()

    st.divider()
    st.markdown("**LLM**")
    st.text_input("Base URL", value="https://api.openai.com/v1", key="llm_url")
    st.text_input("API Key", type="password", key="llm_key",
                  placeholder="✓ en env" if os.getenv("OPENAI_API_KEY") else "sk-…")
    st.text_input("Modelo", value="gpt-4o-mini", key="llm_model")
    st.toggle("Usar LLM", value=True, key="use_llm")

    st.divider()
    st.markdown("**Recuperación**")
    st.slider("Top-K fuentes", 1, 10, 5, key="top_k")
    st.slider("Similitud mínima", 0.10, 0.90, 0.35, 0.05, key="min_sim")

    st.divider()
    st.markdown("**Subir a `raw/`**")
    up_raw = st.file_uploader(
        "PDF / DOCX / TXT / HTML / XLSX / CSV / JSON",
        accept_multiple_files=True,
        type=["pdf","docx","txt","html","htm","xlsx","xls","csv","json"],
        key="up_raw",
    )
    if st.button("💾 Guardar en raw/", width='stretch') and up_raw:
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        for f in up_raw:
            (RAW_DIR / f.name).write_bytes(f.read())
        st.success(f"✅ {len(up_raw)} archivo(s) → `raw/`")

    st.markdown("**Subir a `eval/`**")
    up_eval = st.file_uploader(
        "intents JSON · dataset JSONL · urls Excel",
        accept_multiple_files=True,
        type=["json","jsonl","xlsx","xls","csv"],
        key="up_eval",
    )
    if st.button("💾 Guardar en eval/", width='stretch') and up_eval:
        EVAL_DIR.mkdir(parents=True, exist_ok=True)
        for f in up_eval:
            (EVAL_DIR / f.name).write_bytes(f.read())
        st.success(f"✅ {len(up_eval)} archivo(s) → `eval/`")

    st.divider()
    c1, c2 = st.columns(2)
    btn_build = c1.button("🔨 Indexar", width='stretch')
    btn_load  = c2.button("⚡ Cargar",  width='stretch')

    if btn_build:
        rag = import_rag()
        with st.spinner("Construyendo índice FAISS…"):
            try:
                _, _, n = rag.build_faiss_index(_make_settings(rag))
                st.success(f"✅ {n} chunks indexados")
            except Exception as e:
                st.error(f"Error: {e}")

    if btn_load:
        rag = import_rag()
        with st.spinner("Cargando índice y modelo…"):
            try:
                from sentence_transformers import SentenceTransformer
                cfg = _make_settings(rag)
                idx, meta = rag.load_index(cfg)
                embedder  = SentenceTransformer(cfg.embed_model)
                st.session_state.update({
                    "index_loaded": True,
                    "faiss_index": idx,
                    "meta": meta,
                    "embedder": embedder,
                    "settings": cfg,
                })
                st.success(f"✅ {len(meta)} chunks en memoria")
            except Exception as e:
                st.error(f"Error: {e}")

    st.divider()
    if st.button("🗑️ Limpiar chat", width='stretch'):
        st.session_state.messages = []
        st.rerun()

# ══════════════════════════════════════════════════════════════════════
# Header principal
# ══════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="rag-header">
  <div>
    <h1>🎓 Asistente Virtual · Brightspace</h1>
    <p>RAG con FAISS · multilingual-e5-base · guardrail anti-alucinación · catálogo de intents</p>
  </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# Tabs
# ══════════════════════════════════════════════════════════════════════

tab_chat, tab_eval, tab_sources, tab_crawl, tab_about = st.tabs([
    "💬 Chat", "📊 Evaluación", "📚 Índice", "🌐 Crawl web", "ℹ️ Acerca de",
])

# ─────────────────────────────────────────────────────────────────────
# CHAT
# ─────────────────────────────────────────────────────────────────────
with tab_chat:
    if st.session_state.index_loaded:
        n = len(st.session_state.meta)
        st.markdown(f'<span class="badge-ok">● Índice activo — {n} chunks</span>', unsafe_allow_html=True)
    elif _index_exists():
        st.markdown('<span class="badge-warn">● Índice en disco · pulsa ⚡ Cargar</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge-warn">● Sin índice · sube docs a raw/ y pulsa 🔨 Indexar</span>', unsafe_allow_html=True)

    st.markdown("---")

    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f'<div class="bubble-user">{msg["content"]}</div>', unsafe_allow_html=True)
        else:
            conf   = msg.get("confidence", 0)
            routed = msg.get("routed_to_human", False)
            lat    = msg.get("latency_s", 0)
            badge  = '<span class="badge-routed">↗ Derivado a humano</span>' if routed \
                     else '<span class="badge-ok">✓ Respondido</span>'
            st.markdown(f'{badge} &nbsp; confianza <b>{conf:.0%}</b> &nbsp; ⏱ {lat:.2f}s', unsafe_allow_html=True)
            st.markdown(f'<div class="answer-box">{msg["content"]}</div>', unsafe_allow_html=True)

            if msg.get("sources"):
                with st.expander(f"📎 {len(msg['sources'])} fuentes"):
                    for i, src in enumerate(msg["sources"], 1):
                        t   = src["meta"].get("type","")
                        cat = src["meta"].get("categoria","")
                        st.markdown(f"""
<div class="source-card">
  <div class="src-title">[{i}] {src['title']} › {src['section']}</div>
  <div class="src-meta">score={src['score']:.3f} | type={t}{f' | cat={cat}' if cat else ''}</div>
  <div class="src-snip">{src['snippet']}</div>
</div>""", unsafe_allow_html=True)

    query = st.chat_input(
        "Escribe tu pregunta sobre Brightspace…",
        disabled=not st.session_state.index_loaded,
    )

    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        rag = import_rag()
        cfg = st.session_state.settings
        cfg.top_k       = st.session_state.top_k
        cfg.min_sim     = st.session_state.min_sim
        cfg.use_llm     = st.session_state.use_llm
        cfg.llm_api_key = st.session_state.llm_key or os.getenv("OPENAI_API_KEY", "")

        with st.spinner("Buscando respuesta…"):
            try:
                result = rag.rag_answer(cfg, query,
                                        st.session_state.embedder,
                                        st.session_state.faiss_index,
                                        st.session_state.meta)
                st.session_state.messages.append({
                    "role":            "assistant",
                    "content":         result["answer"],
                    "confidence":      result["confidence"],
                    "routed_to_human": result["routed_to_human"],
                    "latency_s":       result["latency_s"],
                    "has_citations":   result["has_citations"],
                    "sources":         result["sources"],
                })
            except Exception as e:
                st.error(f"Error en RAG: {e}")
        st.rerun()

# ─────────────────────────────────────────────────────────────────────
# EVALUACIÓN
# ─────────────────────────────────────────────────────────────────────
with tab_eval:
    st.markdown("### 📊 Evaluación con métricas de intención")

    eval_path    = EVAL_DIR / "dataset_intents_sintetico_v1.jsonl"
    intents_path = EVAL_DIR / "intents_priorizado.json"
    out_csv      = EVAL_DIR / "eval_results.csv"
    metrics_path = EVAL_DIR / "eval_results_metrics.json"

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Archivos requeridos en `eval/`**")
        for label, p in [("dataset_intents_sintetico_v1.jsonl", eval_path),
                          ("intents_priorizado.json", intents_path)]:
            st.markdown(f"{'✅' if p.exists() else '❌'} `{label}`")
    with c2:
        st.markdown("**Resultados**")
        for label, p in [("eval_results.csv", out_csv),
                          ("eval_results_metrics.json", metrics_path)]:
            st.markdown(f"{'✅' if p.exists() else '—'} `{label}`")

    st.markdown("---")

    ec1, ec2, ec3 = st.columns([2,1,1])
    custom_ds   = ec1.text_input("Dataset personalizado (nombre en eval/)", placeholder="otro.jsonl")
    use_llm_ev  = ec2.toggle("Usar LLM", value=False, key="eval_use_llm",
                              help="Desactiva para evaluar solo retrieval+intents (más rápido, sin costo)")
    max_rows_ev = ec3.number_input("Max filas (0=todas)", value=0, min_value=0, step=50)

    if st.button("▶️ Ejecutar evaluación", width='stretch',
                 disabled=not (eval_path.exists() or custom_ds)):
        rag = import_rag()
        cfg = _make_settings(rag, use_llm=use_llm_ev)
        ds  = EVAL_DIR / custom_ds if custom_ds else eval_path
        intents_arg = str(intents_path) if intents_path.exists() else ""
        max_r = int(max_rows_ev) if max_rows_ev > 0 else None

        with st.spinner("Evaluando… (puede tomar varios minutos con el dataset completo)"):
            try:
                result = rag.run_eval(cfg, str(ds),
                                      out_csv=str(out_csv),
                                      intents_json=intents_arg,
                                      max_rows=max_r)
                st.session_state.eval_metrics = result.get("metrics", result)
                st.success("✅ Evaluación completada")
            except Exception as e:
                st.error(f"Error: {e}")

    # Cargar métricas guardadas
    if metrics_path.exists() and st.session_state.eval_metrics is None:
        try:
            with open(metrics_path, encoding="utf-8") as f:
                loaded = json.load(f)
            st.session_state.eval_metrics = loaded.get("metrics", loaded)
        except Exception:
            pass

    if st.session_state.eval_metrics:
        m = st.session_state.eval_metrics
        st.markdown("#### Métricas")
        cols = st.columns(7)
        cards = [
            ("Intent Acc",    f"{m.get('intent_accuracy',0):.1%}"),
            ("Confianza",     f"{m.get('mean_confidence',0):.1%}"),
            ("Citas",         f"{m.get('citation_rate',0):.1%}"),
            ("Derivados",     f"{m.get('routed_rate',0):.1%}"),
            ("Lat p50",       f"{m.get('p50_latency_s',0):.3f}s"),
            ("Lat p95",       f"{m.get('p95_latency_s',0):.3f}s"),
            ("Evaluados",     str(m.get('n_intent_labeled',0))),
        ]
        for col, (lbl, val) in zip(cols, cards):
            col.metric(lbl, val)

        # Notas igual que en el notebook
        if m.get("citation_rate", 0) == 0.0:
            st.info("ℹ️ `citation_rate = 0.0` — esperado en modo sin LLM. Activa **Usar LLM** para medir citas reales.")
        if m.get("n_intent_labeled", 0) == 0 or m.get("intent_accuracy") is None:
            st.warning("⚠️ `intent_accuracy = None` — verifica que el JSONL tenga `{\"label\": {\"intent_id\": \"...\"}}`")

    if out_csv.exists():
        with st.expander("📄 Ver eval_results.csv", expanded=False):
            try:
                import pandas as pd
                df = pd.read_csv(out_csv)
                st.dataframe(df, width='stretch')
                st.download_button("⬇️ Descargar CSV",
                                   df.to_csv(index=False).encode("utf-8"),
                                   "eval_results.csv", "text/csv")
            except Exception as e:
                st.error(f"Error al leer CSV: {e}")

# ─────────────────────────────────────────────────────────────────────
# ÍNDICE / FUENTES
# ─────────────────────────────────────────────────────────────────────
with tab_sources:
    if not st.session_state.index_loaded:
        st.info("Carga el índice (⚡ Cargar en el panel lateral) para explorar los chunks.")
    else:
        meta_list = st.session_state.meta
        st.markdown(f"**{len(meta_list)} chunks** indexados en total.")

        fc1, fc2, fc3 = st.columns([1,1,2])
        types   = sorted({m["meta"].get("type","") for m in meta_list if m["meta"].get("type","")})
        ftype   = fc1.selectbox("Tipo", ["(todos)"] + types)
        fcat    = fc2.text_input("Categoría")
        fsearch = fc3.text_input("🔍 Buscar en título / texto")

        filtered = meta_list
        if ftype != "(todos)":
            filtered = [m for m in filtered if m["meta"].get("type","") == ftype]
        if fcat:
            q = fcat.lower()
            filtered = [m for m in filtered if q in m["meta"].get("categoria","").lower()]
        if fsearch:
            q = fsearch.lower()
            filtered = [m for m in filtered if q in m["title"].lower() or q in m["text"].lower()]

        st.markdown(f"Mostrando **{min(60, len(filtered))}** de {len(filtered)} resultados.")
        for item in filtered[:60]:
            t   = item["meta"].get("type","")
            cat = item["meta"].get("categoria","")
            tid = item["meta"].get("ticket_id","")
            lbl = f"[{t}] {item['title']} › {item['section']}"
            if cat: lbl += f" — {cat}"
            if tid: lbl += f" — ticket #{tid}"
            with st.expander(lbl):
                st.text(item["text"][:700] + ("…" if len(item["text"]) > 700 else ""))

# ─────────────────────────────────────────────────────────────────────
# CRAWL WEB
# ─────────────────────────────────────────────────────────────────────
with tab_crawl:
    st.markdown("### 🌐 Crawl web gobernado por Excel")
    st.markdown(
        "Descarga páginas web desde `eval/urls_master.xlsx` y guarda los `.txt` en `raw/`. "
        "El Excel de estado actualizado se escribe en `eval/urls_master_updated.xlsx`."
    )

    excel_options = sorted([f.name for f in EVAL_DIR.glob("*.xlsx")]) if EVAL_DIR.exists() else []
    selected_excel = st.selectbox(
        "Excel de URLs (en eval/)",
        options=excel_options or ["urls_master.xlsx"],
    )
    excel_path = EVAL_DIR / selected_excel
    out_excel  = EVAL_DIR / "urls_master_updated.xlsx"

    wc1, wc2 = st.columns(2)
    min_chars = wc1.number_input("Mínimo de chars por página", value=80, min_value=50, step=50)
    wc2.info("💡 Instala `trafilatura` para mejor extracción de contenido principal.")

    if st.button('▶️ Iniciar crawl', width='stretch', disabled=not excel_path.exists()):
        rag = import_rag()
        cfg = _make_settings(rag, min_chars=int(min_chars))
        with st.spinner("Crawleando…"):
            try:
                stats = rag.process_urls_from_excel(cfg, str(excel_path), out_excel=str(out_excel))
                st.success(
                    f"✅ ok={stats['ok']} | fail={stats['fail']} | "
                    f"sin_cambio={stats['skipped_no_change']} | baja_calidad={stats['skipped_low_quality']}"
                )
                st.caption(f"Excel actualizado: `{out_excel}`")
            except Exception as e:
                st.error(f"Error: {e}")
    elif not excel_path.exists():
        st.warning(f"No encontré `{excel_path}`. Sube el Excel en el panel lateral → **eval/**.")

    st.markdown("---")
    st.markdown("#### Descubrir URLs desde HUB")
    hub_url      = st.text_input("URL del HUB", placeholder="https://kb.brightspace.com/")
    domain_allow = st.text_input("Dominio permitido (opcional)", placeholder="kb.brightspace.com")
    disc_excel   = st.selectbox("Excel destino (en eval/)", options=excel_options or ["urls_master.xlsx"], key="disc_xl")

    if st.button("🔍 Descubrir URLs", disabled=not hub_url):
        rag = import_rag()
        import requests
        session = requests.Session()
        session.headers["User-Agent"] = "Mozilla/5.0"
        with st.spinner("Descubriendo…"):
            try:
                urls  = rag.discover_urls_from_hub(session, hub_url, domain_allow=domain_allow or None)
                added = rag.add_to_pending(str(EVAL_DIR / disc_excel), urls, hub_url)
                st.success(f"Descubiertas: {len(urls)} | Nuevas en pending_discovered: {added}")
            except Exception as e:
                st.error(f"Error: {e}")

# ─────────────────────────────────────────────────────────────────────
# ACERCA DE
# ─────────────────────────────────────────────────────────────────────
with tab_about:
    st.markdown(f"""
## Asistente Virtual RAG · Brightspace

Interfaz Streamlit que replica la estructura del notebook Colab en entorno local/servidor.

### Estructura del repositorio GitHub

```
asistenteai/                          ← raíz del repo (PROJECT_ROOT resuelto = {PROJECT_ROOT})
├── asistente_rag_project_v3.py       ← REQUERIDO · módulo RAG principal
├── streamlit_app.py                  ← esta interfaz
├── requirements.txt                  ← dependencias
├── raw/                              ← documentos de conocimiento
│   ├── KnowledgeBase.xlsx            ← FAQ (columnas: pregunta, respuesta, categoria)
│   ├── *.txt                         ← páginas web crawleadas de Brightspace KB
│   └── ...                           ← PDF, DOCX, HTML, CSV-tickets, JSON-tickets
├── index/                            ← índice FAISS persistente
│   ├── faiss.index
│   └── chunks_meta.jsonl
├── logs/
│   └── events.jsonl
└── eval/
    ├── intents_priorizado.json            ← catálogo de intents
    ├── dataset_intents_sintetico_v1.jsonl ← dataset de evaluación
    ├── urls_master.xlsx                   ← URLs para crawl
    ├── urls_master_updated.xlsx
    ├── eval_results.csv
    └── eval_results_metrics.json
```

> **Streamlit Cloud:** `asistente_rag_project_v3.py` y `streamlit_app.py`
> deben estar en la **raíz del repositorio**. Los archivos `raw/`, `index/` y `eval/`
> también viven en la raíz — no en una subcarpeta `scripts/`.

### Flujo de trabajo

| Paso | Acción |
|------|--------|
| 1 | Sube documentos → **raw/** |
| 2 | 🔨 **Indexar** — construye FAISS en index/ |
| 3 | ⚡ **Cargar** — carga índice + modelo en memoria |
| 4 | **Chat** — recupera chunks y genera respuestas citadas |
| 5 | (Opcional) **Crawl** desde eval/urls_master.xlsx → raw/ |
| 6 | (Opcional) **Evaluación** con intents_priorizado.json |

### Componentes

| Componente | Detalle |
|---|---|
| Embeddings | `intfloat/multilingual-e5-base` (1.1 GB, multilingüe) |
| Vector store | FAISS `IndexFlatIP` (producto interno normalizado) |
| Anti-alucinación | Guardrail `MIN_SIM` — deriva a humano si score < umbral |
| Intents | Clasificación semántica vs. catálogo JSON |
| LLM | OpenAI-compatible · `gpt-4o-mini` · temperatura 0.2 |

### Variables de entorno

```bash
OPENAI_API_KEY=sk-...
RAG_PROJECT_ROOT=/ruta/a/RAG_Brightspace   # opcional
```

### Métricas de referencia (producción Feb 2026)

- 266 documentos → **2184 chunks**
- Intent accuracy: **87.5%** (400 queries)
- Confianza media: **86.1%** | Latencia p50/p95: **115 ms / 175 ms** (sin LLM)
    """)
    st.caption(f"Raíz del proyecto resuelta: `{PROJECT_ROOT}`")
    if not PROJECT_ROOT.exists():
        st.warning("La raíz no existe aún — pulsa **Crear carpetas faltantes** en el panel lateral.")
