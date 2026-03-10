#!/usr/bin/env python3
"""
asistente_rag_project_v3.py  (Todo-en-uno, pipeline completo)

Incluye:
1) Ingesta web gobernada por Excel (crawler) + descubrimiento desde HUB
2) Indexación RAG desde un directorio "raw" con múltiples formatos:
   - PDF (.pdf)      -> texto por página (pypdf)
   - Word (.docx)    -> párrafos (python-docx)
   - HTML local (.html/.htm) -> texto limpio (BeautifulSoup)
   - TXT (.txt)      -> texto directo
   - FAQ en Excel (.xlsx/.xls) con columnas: pregunta, respuesta, categoria
       * Cada fila = 1 documento (Q/A) con meta.categoria
   - Tickets exportados (CSV/JSON) (best-effort):
       * CSV: detecta columnas típicas (id, subject/title, description/body, category, created_at, status)
       * JSON: lista de objetos o dict con key "tickets"
       * Cada ticket = 1 documento con metadatos (ticket_id, categoria, created_at, status)
3) Chat RAG (CLI) con guardrail anti-alucinación (MIN_SIM) y generación con LLM OpenAI-compatible (opcional)
4) Evaluación con métricas (baseline de intención + citas + derivación + latencia + recall@k opcional)

Requisitos (pip):
  pip install numpy pandas openpyxl requests httpx tqdm beautifulsoup4 lxml \
              sentence-transformers faiss-cpu pypdf python-docx
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse

import numpy as np
import pandas as pd
import requests
import faiss
import httpx
from bs4 import BeautifulSoup

# Opcional (mejor extractor de 'contenido principal' para sitios ruidosos)
try:
    import trafilatura
except Exception:
    trafilatura = None

from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from pypdf import PdfReader
from docx import Document as DocxDocument

# =========================
# Config
# =========================

@dataclass
class Settings:
    # LLM OpenAI-compatible
    llm_base_url: str = "https://api.openai.com/v1"
    llm_api_key: str = ""
    llm_model: str = "gpt-4o-mini"
    use_llm: bool = True

    # RAG
    #embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embed_model: str = "intfloat/multilingual-e5-base"
    top_k: int = 5
    min_sim: float = 0.35
    chunk_size: int = 900
    chunk_overlap: int = 150
    min_chunk_chars: int = 80

    # Paths
    raw_dir: str = "./data/raw"
    index_dir: str = "./data/index"
    log_dir: str = "./data/logs"

    # Crawling
    timeout_s: int = 30
    min_chars: int = 500
    sleep_min: float = 0.6
    sleep_max: float = 1.4

# =========================
# Utils
# =========================

CITE_PATTERN = re.compile(r"\[\d+\]")

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def safe_filename(s: str, max_len: int = 120) -> str:
    s = (s or "doc").strip()
    s = re.sub(r"[^\w\-]+", "_", s, flags=re.UNICODE).strip("_")
    return (s[:max_len] or "doc")

def clean_text_basic(t: str) -> str:
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()

def log_event(settings: Settings, event: Dict[str, Any]) -> None:
    ensure_dir(settings.log_dir)
    path = os.path.join(settings.log_dir, "events.jsonl")
    event["ts"] = time.time()
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")

# =========================
# 1) Web ingestion (Excel-governed) + HUB discovery
# =========================

def normalize_active(v: Any) -> bool:
    s = str(v).strip().lower()
    return s in {"true", "1", "yes", "si", "sí", "y"}

def clean_html_to_text(html: str) -> Tuple[str, str]:
    """
    Extrae texto de HTML.
    - 1er intento: trafilatura (si está instalada) para obtener el "main content"
    - fallback: BeautifulSoup (más simple)
    """
    # title (rápido con BS4)
    soup0 = BeautifulSoup(html, "lxml")
    title = soup0.title.get_text(" ", strip=True) if soup0.title else "sin_titulo"

    # Intento 1: trafilatura (mejor para KBs)
    if trafilatura is not None:
        try:
            extracted = trafilatura.extract(html, include_comments=False, include_tables=False)
            if extracted and len(extracted.strip()) > 200:
                return title, clean_text_basic(extracted)
        except Exception:
            pass

    # Fallback: BeautifulSoup (no eliminar contenedores a ciegas; puede matar el body en algunos temas)
    soup = soup0
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    # Priorizar <main> si existe
    main = soup.find("main")
    if main is not None:
        text = main.get_text("\n", strip=True)
    else:
        text = soup.get_text("\n", strip=True)

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return title, "\n".join(lines)

def fetch_text(session: requests.Session, url: str, timeout: int) -> Tuple[str, str, str, str]:
    r = session.get(url, timeout=timeout, allow_redirects=True)
    r.raise_for_status()
    html = r.text
    title, text = clean_html_to_text(html)
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return title, text, h, html

def save_doc_txt(raw_dir: str, url: str, title: str, text: str, content_hash: str) -> str:
    ensure_dir(raw_dir)
    short = content_hash[:10]
    fname = f"{safe_filename(title)}__{short}.txt"
    path = os.path.join(raw_dir, fname)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"URL: {url}\nTITLE: {title}\nHASH: {content_hash}\nTYPE: web\n\n")
        f.write(text)
    return path

def discover_urls_from_hub(session: requests.Session, hub_url: str, domain_allow: Optional[str] = None) -> List[str]:
    r = session.get(hub_url, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "lxml")

    found = []
    for a in soup.select("a[href]"):
        href = a["href"].strip()
        if "/kb/" not in href:
            continue
        url = urljoin(hub_url, href)
        if domain_allow and urlparse(url).netloc != domain_allow:
            continue
        found.append(url)

    seen = set()
    urls = []
    for u in found:
        if u not in seen:
            seen.add(u)
            urls.append(u)
    return urls

def read_excel_urls(excel_path: str) -> pd.DataFrame:
    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"No existe el Excel: {excel_path}")
    df = pd.read_excel(excel_path, sheet_name="urls").fillna("")
    if "url" not in df.columns:
        raise ValueError("La hoja `urls` debe incluir columna `url`.")
    if "active" not in df.columns:
        df["active"] = True
    df["url"] = df["url"].astype(str).str.strip()
    df["active"] = df["active"].apply(normalize_active)

    for col in ["source","area","priority","crawl_freq_days","last_crawled","last_hash","status","notes","title","saved_path"]:
        if col not in df.columns:
            df[col] = ""
    return df

def write_excel_preserve_pending(excel_path: str, df_urls: pd.DataFrame, pending_df: Optional[pd.DataFrame] = None) -> None:
    if pending_df is None and os.path.exists(excel_path):
        xls = pd.ExcelFile(excel_path)
        if "pending_discovered" in xls.sheet_names:
            pending_df = pd.read_excel(excel_path, sheet_name="pending_discovered").fillna("")
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        df_urls.to_excel(writer, sheet_name="urls", index=False)
        if pending_df is not None:
            pending_df.to_excel(writer, sheet_name="pending_discovered", index=False)

def add_to_pending(excel_path: str, discovered_urls: List[str], hub_url: str) -> int:
    if os.path.exists(excel_path):
        xls = pd.ExcelFile(excel_path)
        df_urls = pd.read_excel(excel_path, sheet_name="urls").fillna("") if "urls" in xls.sheet_names else pd.DataFrame(columns=["url","active"])
        df_pending = pd.read_excel(excel_path, sheet_name="pending_discovered").fillna("") if "pending_discovered" in xls.sheet_names else pd.DataFrame(columns=["url","discovered_at","hub_url","notes"])
    else:
        df_urls = pd.DataFrame(columns=["url","active","source","area","priority","crawl_freq_days","last_crawled","last_hash","status","notes","title","saved_path"])
        df_pending = pd.DataFrame(columns=["url","discovered_at","hub_url","notes"])

    existing_urls = set(df_urls.get("url", pd.Series([], dtype=str)).astype(str).str.strip().tolist())
    existing_pending = set(df_pending.get("url", pd.Series([], dtype=str)).astype(str).str.strip().tolist())

    now = pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    rows = []
    for u in discovered_urls:
        u = str(u).strip()
        if not u.startswith("http"):
            continue
        if u in existing_urls or u in existing_pending:
            continue
        rows.append({"url": u, "discovered_at": now, "hub_url": hub_url, "notes": ""})

    if rows:
        df_pending = pd.concat([df_pending, pd.DataFrame(rows)], ignore_index=True)

    write_excel_preserve_pending(excel_path, df_urls, df_pending)
    return len(rows)

def process_urls_from_excel(settings: Settings, excel_path: str, out_excel: Optional[str] = None) -> Dict[str, int]:
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "es-ES,es;q=0.9,en;q=0.8",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
    })

    df = read_excel_urls(excel_path)
    df_active = df[df["active"] & df["url"].str.startswith("http")].copy()

    stats = {"ok": 0, "fail": 0, "skipped_no_change": 0, "skipped_low_quality": 0}
    now = pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    ensure_dir(settings.raw_dir)

    for idx, row in tqdm(df_active.iterrows(), total=len(df_active), desc="Crawling"):
        url = row["url"]
        prev_hash = str(row.get("last_hash", "")).strip()

        try:
            title, text, h, raw_html = fetch_text(session, url, timeout=settings.timeout_s)
            df.loc[idx, "last_crawled"] = now
            df.loc[idx, "title"] = title

            # Heurística: detectar páginas "vacías" / bloqueo suave (200 pero sin contenido)
            low = len(text) < settings.min_chars
            if low:
                # Guarda HTML crudo para diagnóstico
                debug_dir = os.path.join(settings.raw_dir, "_debug_html")
                ensure_dir(debug_dir)
                short = h[:10]
                dbg_name = f"{safe_filename(title)}__{short}.html"
                dbg_path = os.path.join(debug_dir, dbg_name)
                try:
                    with open(dbg_path, "w", encoding="utf-8", errors="ignore") as f:
                        f.write(raw_html)
                    df.loc[idx, "notes"] = (str(df.loc[idx, "notes"]) + f" | debug_html={dbg_path}").strip(" |")
                except Exception:
                    pass

                # Señales comunes de bloqueo/JS
                lower = (raw_html or "").lower()
                if any(s in lower for s in ["enable javascript", "captcha", "access denied", "cloudflare", "akamai", "bot", "cookies"]):
                    df.loc[idx, "status"] = "SKIPPED_LOW_QUALITY_POSSIBLE_BLOCK"
                else:
                    df.loc[idx, "status"] = "SKIPPED_LOW_QUALITY"
                stats["skipped_low_quality"] += 1
                continue


            if prev_hash and prev_hash == h:
                df.loc[idx, "status"] = "SKIPPED_NO_CHANGE"
                stats["skipped_no_change"] += 1
                continue

            saved_path = save_doc_txt(settings.raw_dir, url, title, text, h)
            df.loc[idx, "status"] = "OK"
            df.loc[idx, "last_hash"] = h
            df.loc[idx, "saved_path"] = saved_path
            stats["ok"] += 1

        except Exception as e:
            df.loc[idx, "status"] = f"FAIL: {e}"
            df.loc[idx, "last_crawled"] = now
            stats["fail"] += 1

        time.sleep(random.uniform(settings.sleep_min, settings.sleep_max))

    out_path = out_excel or excel_path
    write_excel_preserve_pending(out_path, df)

    log_event(settings, {"type": "crawl", "excel": excel_path, "out_excel": out_path, "stats": stats})
    return stats


# =========================
# 2) Multi-format ingestion for indexing
# =========================

SUPPORTED_EXTS = {".txt",".pdf",".docx",".html",".htm",".xlsx",".xls",".csv",".json"}

def iter_source_files(raw_dir: str) -> List[str]:
    paths: List[str] = []
    for root, _, files in os.walk(raw_dir):
        for fn in files:
            ext = os.path.splitext(fn)[1].lower()
            if ext in SUPPORTED_EXTS:
                paths.append(os.path.join(root, fn))
    return sorted(paths)

def read_pdf(path: str) -> str:
    reader = PdfReader(path)
    parts = []
    for page in reader.pages:
        parts.append(page.extract_text() or "")
    return "\n".join(parts)

def read_docx(path: str) -> str:
    doc = DocxDocument(path)
    return "\n".join(p.text for p in doc.paragraphs if p.text is not None)

def read_html_file(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        html = f.read()
    _, text = clean_html_to_text(html)
    return text

def read_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    return df

def detect_faq_schema(df: pd.DataFrame) -> bool:
    cols = set(df.columns)
    return {"pregunta","respuesta"}.issubset(cols)


def _read_csv_best_effort(path: str) -> pd.DataFrame:
    """Lee CSV tolerante: autodetecta separador, soporta BOM, y salta líneas malas."""
    # Intento 1: autodetección de separador
    try:
        return pd.read_csv(path, engine="python", sep=None, encoding="utf-8-sig", on_bad_lines="skip").fillna("")
    except Exception:
        pass
    # Intento 2: separador ';' (muy común en ES/LatAm)
    try:
        return pd.read_csv(path, engine="python", sep=";", encoding="utf-8-sig", on_bad_lines="skip").fillna("")
    except Exception:
        pass
    # Intento 3: latin-1
    return pd.read_csv(path, engine="python", sep=None, encoding="latin-1", on_bad_lines="skip").fillna("")

def ingest_faq_csv(path: str) -> List[Dict[str, Any]]:
    """
    Lee FAQ en CSV con columnas: pregunta, respuesta, categoria (opcional).
    Soporta separador ';' y BOM.
    Cada fila -> un documento FAQ.
    """
    docs: List[Dict[str, Any]] = []
    df = _read_csv_best_effort(path)
    df = normalize_columns(df)

    # Aceptar variantes de columna (con/sin tilde o en inglés)
    col_p = None
    col_r = None
    col_c = None
    for cand in ["pregunta", "question", "preguntas"]:
        if cand in df.columns:
            col_p = cand
            break
    for cand in ["respuesta", "answer", "respuestas"]:
        if cand in df.columns:
            col_r = cand
            break
    for cand in ["categoria", "categoría", "category", "cat"]:
        if cand in df.columns:
            col_c = cand
            break

    if not (col_p and col_r):
        return docs

TICKET_COL_CANDIDATES = {
    "id": ["id","ticket_id","case_id","numero","nro","number"],
    "subject": ["subject","title","asunto","tema","summary"],
    "description": ["description","body","detalle","details","request","request_text","contenido"],
    "category": ["category","categoria","type","tipo","area","queue"],
    "created_at": ["created_at","created","fecha","fecha_creacion","date","createddate"],
    "status": ["status","estado"],
    "resolution": ["resolution","solucion","respuesta","comentario","comment","reply"],
}

def pick_col(df, candidates):
    cols = set(df.columns)
    for c in candidates:
        if c in cols:
            return c
    return None

def ingest_tickets_csv(path: str) -> List[Dict[str, Any]]:
    """
    Best-effort: cada fila = ticket.
    Soporta CSV con separador automático, ';', BOM, encoding latin-1.
    Si el CSV no parece export de tickets, retorna [] sin lanzar excepción.
    """
    docs: List[Dict[str, Any]] = []

    def _read_csv_safe(enc=None, sep=None):
        return pd.read_csv(path, encoding=enc, sep=sep, engine="python",
                           on_bad_lines="skip").fillna("")

    try:
        df = pd.read_csv(path).fillna("")
    except Exception:
        try:
            df = _read_csv_safe(enc=None, sep=None)
        except Exception:
            try:
                df = _read_csv_safe(enc="latin-1", sep=None)
            except Exception:
                try:
                    df = _read_csv_safe(enc=None, sep=";")
                except Exception:
                    try:
                        df = _read_csv_safe(enc="latin-1", sep=";")
                    except Exception:
                        return docs

    df = normalize_columns(df)

    col_id      = pick_col(df, TICKET_COL_CANDIDATES["id"])
    col_subject = pick_col(df, TICKET_COL_CANDIDATES["subject"])
    col_desc    = pick_col(df, TICKET_COL_CANDIDATES["description"])
    col_cat     = pick_col(df, TICKET_COL_CANDIDATES["category"])
    col_created = pick_col(df, TICKET_COL_CANDIDATES["created_at"])
    col_status  = pick_col(df, TICKET_COL_CANDIDATES["status"])
    col_res     = pick_col(df, TICKET_COL_CANDIDATES["resolution"])

    if not (col_desc or col_subject):
        return docs

    title = os.path.basename(path)
    for i, row in df.iterrows():
        ticket_id = str(row.get(col_id, f"{i+1}")).strip() if col_id else str(i+1)
        subject   = str(row.get(col_subject, "")).strip()  if col_subject else ""
        desc      = str(row.get(col_desc,    "")).strip()  if col_desc    else ""
        cat       = str(row.get(col_cat,     "")).strip()  if col_cat     else ""
        created   = str(row.get(col_created, "")).strip()  if col_created else ""
        status    = str(row.get(col_status,  "")).strip()  if col_status  else ""
        res       = str(row.get(col_res,     "")).strip()  if col_res     else ""

        if len(desc) < 50 and len(subject) < 10:
            continue

        text = "TICKET\n"
        if subject: text += f"Asunto: {subject}\n"
        if cat:     text += f"Categoría: {cat}\n"
        if created: text += f"Creado: {created}\n"
        if status:  text += f"Estado: {status}\n"
        text += f"\nDescripción:\n{desc}\n"
        if res:     text += f"\nResolución/Respuesta:\n{res}\n"

        docs.append({
            "doc_id": f"ticket::{title}::{ticket_id}",
            "title": title,
            "text": text,
            "meta": {
                "type": "ticket",
                "source_file": path,
                "ticket_id": ticket_id,
                "categoria": cat,
                "created_at": created,
                "status": status,
            }
        })
    return docs

def ingest_tickets_json(path: str) -> List[Dict[str, Any]]:
    docs: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        data = json.load(f)

    if isinstance(data, dict) and "tickets" in data and isinstance(data["tickets"], list):
        items = data["tickets"]
    elif isinstance(data, list):
        items = data
    else:
        return docs

    df = pd.DataFrame(items).fillna("")
    df = normalize_columns(df)

    col_id = pick_col(df, TICKET_COL_CANDIDATES["id"])
    col_subject = pick_col(df, TICKET_COL_CANDIDATES["subject"])
    col_desc = pick_col(df, TICKET_COL_CANDIDATES["description"])
    col_cat = pick_col(df, TICKET_COL_CANDIDATES["category"])
    col_created = pick_col(df, TICKET_COL_CANDIDATES["created_at"])
    col_status = pick_col(df, TICKET_COL_CANDIDATES["status"])
    col_res = pick_col(df, TICKET_COL_CANDIDATES["resolution"])

    if not (col_desc or col_subject):
        return docs

    title = os.path.basename(path)
    for i, row in df.iterrows():
        ticket_id = str(row.get(col_id, f"{i+1}")).strip() if col_id else str(i+1)
        subject = str(row.get(col_subject, "")).strip() if col_subject else ""
        desc = str(row.get(col_desc, "")).strip() if col_desc else ""
        cat = str(row.get(col_cat, "")).strip() if col_cat else ""
        created = str(row.get(col_created, "")).strip() if col_created else ""
        status = str(row.get(col_status, "")).strip() if col_status else ""
        res = str(row.get(col_res, "")).strip() if col_res else ""

        if len(desc) < 50 and len(subject) < 10:
            continue

        text = "TICKET\n"
        if subject:
            text += f"Asunto: {subject}\n"
        if cat:
            text += f"Categoría: {cat}\n"
        if created:
            text += f"Creado: {created}\n"
        if status:
            text += f"Estado: {status}\n"
        text += f"\nDescripción:\n{desc}\n"
        if res:
            text += f"\nResolución/Respuesta:\n{res}\n"

        docs.append({
            "doc_id": f"ticket::{title}::{ticket_id}",
            "title": title,
            "text": text,
            "meta": {
                "type": "ticket",
                "source_file": path,
                "ticket_id": ticket_id,
                "categoria": cat,
                "created_at": created,
                "status": status,
            }
        })
    return docs

def ingest_generic_file(path: str) -> List[Dict[str, Any]]:
    ext = os.path.splitext(path)[1].lower()
    title = os.path.basename(path)

    if ext == ".pdf":
        text = clean_text_basic(read_pdf(path))
        if len(text) < 20:
            return []
        return [{
            "doc_id": f"pdf::{title}",
            "title": title,
            "text": text,
            "meta": {"type": "pdf", "source_file": path},
        }]

    if ext == ".docx":
        text = clean_text_basic(read_docx(path))
        if len(text) < 20:
            return []
        return [{
            "doc_id": f"docx::{title}",
            "title": title,
            "text": text,
            "meta": {"type": "docx", "source_file": path},
        }]

    if ext in {".html", ".htm"}:
        text = clean_text_basic(read_html_file(path))
        if len(text) < 20:
            return []
        return [{
            "doc_id": f"html::{title}",
            "title": title,
            "text": text,
            "meta": {"type": "html", "source_file": path},
        }]

    if ext == ".txt":
        text = clean_text_basic(read_txt(path))
        if len(text) < 20:
            return []
        return [{
            "doc_id": f"txt::{title}",
            "title": title,
            "text": text,
            "meta": {"type": "txt", "source_file": path},
        }]

    if ext in {".xlsx", ".xls"}:
        docs = ingest_faq_excel(path)
        return docs

    if ext == ".csv":
        # Primero intentar como FAQ (pregunta/respuesta/categoria)
        docs = ingest_faq_csv(path)
        if docs:
            return docs
        # Si no es FAQ, intentar como export de tickets
        return ingest_tickets_csv(path)

    if ext == ".json":
        return ingest_tickets_json(path)

    return []


# =========================
# 3) Build FAISS index
# =========================

def chunk_text(text: str, chunk_size: int, overlap: int, min_chars: int) -> List[str]:
    """Chunker simple con umbral mínimo.

    Para FAQs/tickets, los textos suelen ser cortos; por eso el umbral por defecto es bajo.
    """
    chunks: List[str] = []
    i = 0
    text = text.strip()
    if len(text) <= chunk_size:
        return [text] if len(text) >= min_chars else []
    while i < len(text):
        ch = text[i:i+chunk_size].strip()
        if len(ch) >= min_chars:
            chunks.append(ch)
        i += (chunk_size - overlap)
    return chunks

def build_faiss_index(settings: Settings) -> Tuple[str, str, int]:
    ensure_dir(settings.index_dir)
    index_path = os.path.join(settings.index_dir, "faiss.index")
    meta_path = os.path.join(settings.index_dir, "chunks_meta.jsonl")

    files = iter_source_files(settings.raw_dir)
    if not files:
        raise RuntimeError(f"No hay archivos soportados en {settings.raw_dir}")

    embedder = SentenceTransformer(settings.embed_model)

    all_vectors: List[np.ndarray] = []
    all_meta: List[Dict[str, Any]] = []

    for path in tqdm(files, desc="Ingest+Index"):
        try:
            docs = ingest_generic_file(path)
        except Exception as e:
            # No detenemos el build por un archivo malformado
            log_event(settings, {"type":"ingest_skip", "path": path, "error": str(e)})
            continue
        for doc in docs:
            text = doc["text"]
            if len(text) < 20:
                continue

            chunks = chunk_text(text, settings.chunk_size, settings.chunk_overlap, settings.min_chunk_chars)
            if not chunks:
                continue

            vecs = embedder.encode(chunks, normalize_embeddings=True)
            vecs = np.asarray(vecs, dtype="float32")
            all_vectors.append(vecs)

            for j, chunk in enumerate(chunks, 1):
                all_meta.append({
                    "source_id": f"{doc['doc_id']}::chunk_{j}",
                    "title": doc["title"],
                    "section": f"chunk_{j}",
                    "text": chunk,
                    "meta": doc.get("meta", {}),
                })

    if not all_vectors:
        raise RuntimeError("No se generaron chunks útiles para indexar.")

    X = np.vstack(all_vectors)
    dim = X.shape[1]

    index = faiss.IndexFlatIP(dim)
    index.add(X)
    faiss.write_index(index, index_path)

    with open(meta_path, "w", encoding="utf-8") as f:
        for item in all_meta:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    log_event(settings, {"type": "build_index", "files": len(files), "chunks": len(all_meta), "index_dir": settings.index_dir})
    return index_path, meta_path, len(all_meta)


# =========================
# 4) RAG engine
# =========================

def load_index(settings: Settings):
    index_path = os.path.join(settings.index_dir, "faiss.index")
    meta_path = os.path.join(settings.index_dir, "chunks_meta.jsonl")
    if not (os.path.exists(index_path) and os.path.exists(meta_path)):
        raise FileNotFoundError("No existe índice. Ejecuta build-index primero.")
    index = faiss.read_index(index_path)
    meta = []
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            meta.append(json.loads(line))
    return index, meta

def build_prompt(settings: Settings, query: str, retrieved: List[Tuple[float, Dict[str, Any]]]) -> Tuple[str, float, bool]:
    if not retrieved:
        return (
            f"Pregunta: {query}\n\nNo hay evidencia disponible.\nResponde: 'No tengo evidencia suficiente' y sugiere canal de soporte.",
            0.0,
            True,
        )

    max_score = max(s for s, _ in retrieved)
    if max_score < settings.min_sim:
        return (
            f"Pregunta: {query}\n\nEVIDENCIA INSUFICIENTE (score={max_score:.2f}).\n"
            "Responde: 'No tengo evidencia suficiente' y sugiere qué documento/proceso consultar o a quién escalar.",
            float(max_score),
            True,
        )

    ctx = []
    for i, (score, item) in enumerate(retrieved, 1):
        ctx.append(f"[{i}] (score={score:.2f}) {item['title']} > {item['section']}\n{item['text']}\n")

    prompt = (
        f"Pregunta del usuario:\n{query}\n\n"
        "Evidencia disponible (usa SOLO esto para responder y cita [n]):\n\n"
        + "\n".join(ctx)
        + "\nInstrucciones:\n"
          "1) Respuesta directa.\n"
          "2) Pasos si aplica.\n"
          "3) 'Fuentes' con [n].\n"
          "4) Si falta evidencia para algo, dilo explícitamente.\n"
    )
    return prompt, float(max_score), False

async def call_llm(settings: Settings, prompt: str) -> str:
    if not settings.llm_api_key:
        raise RuntimeError("LLM API key vacío. Usa --llm-api-key o env OPENAI_API_KEY.")

    url = settings.llm_base_url.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {settings.llm_api_key}"}
    payload = {
        "model": settings.llm_model,
        "messages": [
            {"role": "system", "content": "Eres un asistente institucional. Responde solo con evidencia. Si no hay evidencia suficiente, dilo y deriva."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post(url, headers=headers, json=payload)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

def rag_answer(settings: Settings, query: str, embedder: SentenceTransformer, index, meta) -> Dict[str, Any]:
    t0 = time.time()

    qv = embedder.encode([query], normalize_embeddings=True)
    qv = np.asarray(qv, dtype="float32")

    scores, idxs = index.search(qv, settings.top_k)
    retrieved: List[Tuple[float, Dict[str, Any]]] = []
    for score, idx in zip(scores[0].tolist(), idxs[0].tolist()):
        if idx == -1:
            continue
        retrieved.append((float(score), meta[idx]))

    prompt, confidence, routed = build_prompt(settings, query, retrieved)

    if settings.use_llm:
        answer = asyncio.run(call_llm(settings, prompt))
    else:
        answer = "No tengo evidencia suficiente." if not retrieved else ("Evidencia (extracto):\n\n" + retrieved[0][1]["text"][:1000])

    latency = time.time() - t0

    sources = [{
        "source_id": item["source_id"],
        "title": item["title"],
        "section": item["section"],
        "score": float(score),
        "snippet": item["text"][:350],
        "meta": item.get("meta", {}),
    } for score, item in retrieved]

    out = {
        "query": query,
        "answer": answer,
        "confidence": float(confidence),
        "routed_to_human": bool(routed),
        "latency_s": float(latency),
        "has_citations": bool(CITE_PATTERN.search(answer)),
        "sources": sources,
    }

    log_event(settings, {
        "type": "chat",
        "query": query,
        "confidence": out["confidence"],
        "routed_to_human": out["routed_to_human"],
        "latency_s": out["latency_s"],
        "has_citations": out["has_citations"],
        "sources": [{"title": s["title"], "score": s["score"], "type": s["meta"].get("type","")} for s in sources],
    })

    return out


# =========================
# 5) Intent baseline + evaluación
# =========================

INTENTS: List[Tuple[str, str]] = [
    ("crear_aula", "Creación de cursos/aulas virtuales para un periodo académico"),
    ("replicar_master", "Replicación o copia de contenido desde master course"),
    ("asignar_docente", "Asignación de docentes y roles en un curso"),
    ("cambiar_rol", "Cambio de rol de usuario en el aula"),
    ("copiar_contenido", "Copiar/importar contenido entre cursos o periodos"),
    ("recuperar_contenido", "Recuperar actividades o contenidos eliminados"),
    ("visibilidad_curso", "Disponibilidad, fechas y visibilidad del curso para estudiantes"),
    ("cierre_curso", "Cierre, fin de periodo y acceso posterior"),
    ("config_cuestionario", "Configuración de cuestionarios, intentos, disponibilidad"),
    ("restablecer_intento", "Restablecer o desbloquear intentos de evaluación"),
    ("tiempo_adicional", "Conceder tiempo adicional o excepciones a estudiantes"),
    ("exportar_notas", "Exportar libro de calificaciones y reportes de notas"),
    ("sync_banner", "Sincronización de calificaciones con sistema académico (Banner u otro)"),
    ("rubricas", "Crear o asociar rúbricas en evaluaciones"),
    ("evaluacion_final", "Configuración y ponderación de evaluación final"),
    ("problemas_login", "Problemas de acceso, login, credenciales o cuenta bloqueada"),
    ("importar_usuarios", "Carga masiva de usuarios, errores de importación"),
    ("integracion_smowl", "Integración y uso de proctoring Smowl"),
    ("integracion_turnitin", "Integración y uso de Turnitin / DraftCoach"),
    ("limite_archivos", "Errores por tamaño/límite de archivos o formatos"),
    ("navegadores", "Compatibilidad de navegadores y requisitos técnicos"),
    ("fechas_periodo", "Fechas oficiales de inicio/fin de periodo, cronograma"),
    ("extension_fechas", "Solicitudes de extensión de plazos en actividades"),
    ("cambio_paralelo", "Cambios de paralelo o movimiento de estudiantes"),
    ("certificados", "Emisión y descarga de certificados"),
    ("eliminar_usuario", "Eliminación/baja de usuario y políticas de datos"),
    ("estructura_aula", "Buenas prácticas de organización del aula virtual"),
    ("indicadores_participacion", "Indicadores/analytics de participación estudiantil"),
    ("reportes_actividad", "Reportes de actividad y acceso"),
    ("capacitacion_docente", "Capacitaciones, guías y soporte a docentes"),
]


def load_intents_from_json(intents_json_path: str) -> List[Tuple[str, str]]:
    """
    Lee intents desde un JSON con estructura:
      {
        "intent_catalog": [
          {"intent_id":"...", "display_name":"...", "description":"...", "category":"...", "utterances":[...]},
          ...
        ]
      }
    Retorna List[(intent_id, text_ancla_para_embedding)].
    """
    if not intents_json_path:
        return []

    if not os.path.exists(intents_json_path):
        raise FileNotFoundError(f"No existe el archivo de intents: {intents_json_path}")

    with open(intents_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    catalog = data.get("intent_catalog", [])
    if not isinstance(catalog, list) or not catalog:
        raise ValueError("El JSON de intents no contiene `intent_catalog` o está vacío.")

    intents: List[Tuple[str, str]] = []
    for it in catalog:
        intent_id = str(it.get("intent_id", "")).strip()
        display = str(it.get("display_name", "")).strip()
        desc = str(it.get("description", "")).strip()
        cat = str(it.get("category", "")).strip()

        if not intent_id:
            continue

        utterances = it.get("utterances", []) or []
        if isinstance(utterances, list):
            utt_text = " | ".join([str(u).strip() for u in utterances if str(u).strip()])
        else:
            utt_text = str(utterances).strip()

        # Texto ancla para embeddings: nombre + descripción + utterances + categoría
        text = " | ".join([x for x in [display, desc, utt_text, f"categoria={cat}" if cat else ""] if x])
        intents.append((intent_id, text))

    if not intents:
        raise ValueError("No se pudieron construir intents desde el JSON (revisa campos intent_id).")

    return intents
def load_eval_rows(path: str, max_rows: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Soporta:
    - CSV con columnas: query, intent, expected_source_title, must_have_citations
    - JSONL con campos:
        text
        label.intent_id
      (y opcionalmente qa.split para filtrar)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"No existe el dataset: {path}")

    ext = os.path.splitext(path)[1].lower()
    rows: List[Dict[str, Any]] = []

    if ext == ".csv":
        df = pd.read_csv(path).fillna("")
        if max_rows:
            df = df.head(max_rows)
        for _, r in df.iterrows():
            rows.append({
                "query": str(r.get("query", "")).strip(),
                "intent_true": str(r.get("intent", "")).strip(),
                "expected_source_title": str(r.get("expected_source_title", "")).strip(),
                "must_have_citations": str(r.get("must_have_citations", "yes")).strip().lower(),
            })
        return rows

    if ext == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                text = str(obj.get("text", "")).strip()
                # Soporta {"label":{"intent_id":...}} (anidado) y {"intent_id":...} o {"intent":...} (plano)
                intent_true = (
                    str((obj.get("label") or {}).get("intent_id", "")).strip()
                    or str(obj.get("intent_id", "")).strip()
                    or str(obj.get("intent", "")).strip()
                )
                if not text:
                    continue
                rows.append({
                    "query": text,
                    "intent_true": intent_true,
                    "expected_source_title": str(obj.get("expected_source_title", "")).strip(),
                    "must_have_citations": str(obj.get("must_have_citations", "no")).strip().lower(),
                })
                if max_rows and len(rows) >= max_rows:
                    break
        return rows

    raise ValueError("Formato no soportado. Usa .csv o .jsonl")



def build_intent_matrix(embedder: SentenceTransformer, intents: List[Tuple[str, str]]) -> Tuple[List[str], np.ndarray]:
    if not intents:
        raise ValueError("Lista de intents vacía. Provee --intents-json o define INTENTS.")
    texts = [f"{intent_id}: {desc}" for intent_id, desc in intents]
    vecs = embedder.encode(texts, normalize_embeddings=True)
    vecs = np.asarray(vecs, dtype="float32")
    names = [intent_id for intent_id, _ in intents]
    return names, vecs


def predict_intent(embedder: SentenceTransformer, intent_names: List[str], intent_vecs: np.ndarray, query: str) -> Tuple[str, float]:
    qv = embedder.encode([query], normalize_embeddings=True)
    qv = np.asarray(qv, dtype="float32")[0]
    sims = intent_vecs @ qv
    idx = int(np.argmax(sims))
    return intent_names[idx], float(sims[idx])

def run_eval(settings: Settings, eval_path: str, out_csv: str, intents_json: str = "", max_rows: Optional[int] = None) -> Dict[str, Any]:
    rows_in = load_eval_rows(eval_path, max_rows=max_rows)


    index, meta = load_index(settings)
    embedder = SentenceTransformer(settings.embed_model)

    intents = load_intents_from_json(intents_json) if intents_json else INTENTS
    intent_names, intent_vecs = build_intent_matrix(embedder, intents)
    rows = []
    for r in tqdm(rows_in, total=len(rows_in), desc="Evaluating"):
        query = str(r.get("query","")).strip()
        if not query:
            continue

        intent_true = str(r.get("intent_true","")).strip()
        expected_title = str(r.get("expected_source_title","")).strip()
        must_cite = str(r.get("must_have_citations","yes")).strip().lower() in {"yes","si","sí","true","1"}

        t0 = time.time()
        qv = embedder.encode([query], normalize_embeddings=True)
        qv = np.asarray(qv, dtype="float32")

        scores, idxs = index.search(qv, settings.top_k)
        retrieved: List[Tuple[float, Dict[str, Any]]] = []
        for score, idx in zip(scores[0].tolist(), idxs[0].tolist()):
            if idx == -1:
                continue
            retrieved.append((float(score), meta[idx]))

        prompt, conf, routed = build_prompt(settings, query, retrieved)

        if settings.use_llm:
            try:
                answer = asyncio.run(call_llm(settings, prompt))
            except Exception as e:
                answer = f"[ERROR LLM] {e}"
        else:
            answer = "No hay evidencia suficiente." if not retrieved else retrieved[0][1]["text"][:800]

        latency = time.time() - t0

        intent_pred, intent_sim = predict_intent(embedder, intent_names, intent_vecs, query)

        has_citations = bool(CITE_PATTERN.search(answer))
        citation_ok = (has_citations if must_cite and not routed else True)

        titles = [it["title"] for _, it in retrieved]
        recall_at_k = None
        if expected_title:
            recall_at_k = 1.0 if expected_title in titles else 0.0

        rows.append({
            "query": query,
            "intent_true": intent_true,
            "intent_pred": intent_pred,
            "intent_sim": intent_sim,
            "intent_correct": (intent_true == intent_pred) if intent_true else None,
            "confidence": conf,
            "routed": routed,
            "latency_s": latency,
            "has_citations": has_citations,
            "citation_ok": citation_ok,
            "top1_title": titles[0] if titles else "",
            "topk_titles": " | ".join(titles),
            "recall_at_k": recall_at_k,
        })

    res = pd.DataFrame(rows)
    ensure_dir(os.path.dirname(out_csv) or ".")
    res.to_csv(out_csv, index=False, encoding="utf-8")

    metrics: Dict[str, Any] = {}
    if res.empty:
        metrics["error"] = "No hay resultados."
        return {"metrics": metrics, "results_rows": 0}

    mask_int = res["intent_correct"].notna()
    metrics["intent_accuracy"] = float(res.loc[mask_int, "intent_correct"].mean()) if mask_int.any() else None
    metrics["n_intent_labeled"] = int(mask_int.sum())

    metrics["citation_rate"] = float(res["has_citations"].mean())
    metrics["citation_ok_rate"] = float(res["citation_ok"].mean())
    metrics["routed_rate"] = float(res["routed"].mean())
    metrics["mean_confidence"] = float(res["confidence"].mean())
    metrics["p50_latency_s"] = float(res["latency_s"].quantile(0.50))
    metrics["p95_latency_s"] = float(res["latency_s"].quantile(0.95))

    if res["recall_at_k"].notna().any():
        metrics["recall_at_k"] = float(res["recall_at_k"].dropna().mean())
        metrics["n_recall_labeled"] = int(res["recall_at_k"].notna().sum())
    else:
        metrics["recall_at_k"] = None
        metrics["n_recall_labeled"] = 0

    metrics_path = os.path.splitext(out_csv)[0] + "_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    log_event(settings, {"type":"eval", "eval_path": eval_path, "out_csv": out_csv, "metrics": metrics})

    return {"metrics": metrics, "results_rows": int(len(res))}


# =========================
# CLI wiring
# =========================

def apply_overrides(settings: Settings, args: argparse.Namespace) -> Settings:
    # LLM
    if getattr(args, "llm_base_url", None):
        settings.llm_base_url = args.llm_base_url
    if getattr(args, "llm_api_key", None):
        settings.llm_api_key = args.llm_api_key
    if not settings.llm_api_key:
        settings.llm_api_key = os.environ.get("OPENAI_API_KEY", "")

    if getattr(args, "llm_model", None):
        settings.llm_model = args.llm_model
    if getattr(args, "use_llm", None) is not None:
        settings.use_llm = args.use_llm

    # paths
    if getattr(args, "raw_dir", None):
        settings.raw_dir = args.raw_dir
    if getattr(args, "index_dir", None):
        settings.index_dir = args.index_dir
    if getattr(args, "log_dir", None):
        settings.log_dir = args.log_dir

    # rag
    for k in ["embed_model","top_k","min_sim","chunk_size","chunk_overlap"]:
        if getattr(args, k, None) is not None:
            setattr(settings, k, getattr(args, k))

    # crawling
    for k in ["timeout_s","min_chars","sleep_min","sleep_max"]:
        if getattr(args, k, None) is not None:
            setattr(settings, k, getattr(args, k))

    return settings

def main():
    parser = argparse.ArgumentParser(prog="asistente_rag_project_v3.py", description="Proyecto completo RAG en un solo .py (pipeline completo)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # discover
    p_disc = sub.add_parser("discover", help="Descubre URLs desde HUB y agrega a pending_discovered")
    p_disc.add_argument("--excel", required=True)
    p_disc.add_argument("--hub-url", required=True)
    p_disc.add_argument("--domain-allow", default="")

    # crawl
    p_crawl = sub.add_parser("crawl", help="Descarga URLs activas desde Excel (incremental) y guarda TXT en raw_dir")
    p_crawl.add_argument("--excel", required=True)
    p_crawl.add_argument("--out-excel", default="")
    p_crawl.add_argument("--raw-dir", default=None)
    p_crawl.add_argument("--log-dir", default=None)
    p_crawl.add_argument("--timeout-s", type=int, default=None)
    p_crawl.add_argument("--min-chars", type=int, default=None)
    p_crawl.add_argument("--sleep-min", type=float, default=None)
    p_crawl.add_argument("--sleep-max", type=float, default=None)

    # build-index
    p_bi = sub.add_parser("build-index", help="Construye índice FAISS desde raw_dir (PDF/DOCX/HTML/TXT/FAQ Excel/Tickets CSV/JSON)")
    p_bi.add_argument("--raw-dir", default=None)
    p_bi.add_argument("--index-dir", default=None)
    p_bi.add_argument("--log-dir", default=None)
    p_bi.add_argument("--embed-model", default=None)
    p_bi.add_argument("--chunk-size", type=int, default=None)
    p_bi.add_argument("--chunk-overlap", type=int, default=None)

    # chat
    p_chat = sub.add_parser("chat", help="Chat CLI (una pregunta)")
    p_chat.add_argument("--query", required=True)
    p_chat.add_argument("--index-dir", default=None)
    p_chat.add_argument("--raw-dir", default=None)
    p_chat.add_argument("--log-dir", default=None)
    p_chat.add_argument("--embed-model", default=None)
    p_chat.add_argument("--top-k", type=int, default=None)
    p_chat.add_argument("--min-sim", type=float, default=None)
    p_chat.add_argument("--use-llm", action=argparse.BooleanOptionalAction, default=None)
    p_chat.add_argument("--llm-base-url", default=None)
    p_chat.add_argument("--llm-api-key", default=None)
    p_chat.add_argument("--llm-model", default=None)

    # eval
    p_eval = sub.add_parser("eval", help="Evaluación con dataset CSV y métricas")
    p_eval.add_argument("--eval-path", required=True)
    p_eval.add_argument("--out-csv", default="eval_results.csv")
    p_eval.add_argument("--max-rows", type=int, default=None)
    p_eval.add_argument("--intents-json", default="", help="Ruta a intents_priorizado.json")
    p_eval.add_argument("--index-dir", default=None)
    p_eval.add_argument("--raw-dir", default=None)
    p_eval.add_argument("--log-dir", default=None)
    p_eval.add_argument("--embed-model", default=None)
    p_eval.add_argument("--top-k", type=int, default=None)
    p_eval.add_argument("--min-sim", type=float, default=None)
    p_eval.add_argument("--use-llm", action=argparse.BooleanOptionalAction, default=None)
    p_eval.add_argument("--llm-base-url", default=None)
    p_eval.add_argument("--llm-api-key", default=None)
    p_eval.add_argument("--llm-model", default=None)

    args = parser.parse_args()
    settings = apply_overrides(Settings(), args)

    ensure_dir(settings.raw_dir)
    ensure_dir(settings.index_dir)
    ensure_dir(settings.log_dir)

    if args.cmd == "discover":
        session = requests.Session()
        session.headers.update({
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "es-ES,es;q=0.9,en;q=0.8",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
        })
        domain = args.domain_allow.strip() or None
        urls = discover_urls_from_hub(session, args.hub_url, domain_allow=domain)
        added = add_to_pending(args.excel, urls, args.hub_url)
        print(f"Descubiertas: {len(urls)} | Agregadas a pending_discovered: {added}")
        log_event(settings, {"type":"discover", "hub_url": args.hub_url, "found": len(urls), "added": added})
        return

    if args.cmd == "crawl":
        if args.raw_dir is not None:
            settings.raw_dir = args.raw_dir
        if args.log_dir is not None:
            settings.log_dir = args.log_dir
        if args.timeout_s is not None:
            settings.timeout_s = args.timeout_s
        if args.min_chars is not None:
            settings.min_chars = args.min_chars
        if args.sleep_min is not None:
            settings.sleep_min = args.sleep_min
        if args.sleep_max is not None:
            settings.sleep_max = args.sleep_max

        out_excel = args.out_excel.strip() or None
        stats = process_urls_from_excel(settings, args.excel, out_excel=out_excel)
        print("=== RESUMEN CRAWL ===")
        for k, v in stats.items():
            print(f"{k}: {v}")
        return

    if args.cmd == "build-index":
        if args.raw_dir is not None:
            settings.raw_dir = args.raw_dir
        if args.index_dir is not None:
            settings.index_dir = args.index_dir
        if args.log_dir is not None:
            settings.log_dir = args.log_dir
        if args.embed_model is not None:
            settings.embed_model = args.embed_model
        if args.chunk_size is not None:
            settings.chunk_size = args.chunk_size
        if args.chunk_overlap is not None:
            settings.chunk_overlap = args.chunk_overlap

        index_path, meta_path, n = build_faiss_index(settings)
        print("✅ Índice construido")
        print("faiss:", index_path)
        print("meta :", meta_path)
        print("chunks:", n)
        return

    if args.cmd == "chat":
        if args.index_dir is not None:
            settings.index_dir = args.index_dir
        if args.raw_dir is not None:
            settings.raw_dir = args.raw_dir
        if args.log_dir is not None:
            settings.log_dir = args.log_dir
        if args.embed_model is not None:
            settings.embed_model = args.embed_model
        if args.top_k is not None:
            settings.top_k = args.top_k
        if args.min_sim is not None:
            settings.min_sim = args.min_sim
        if args.use_llm is not None:
            settings.use_llm = args.use_llm
        if args.llm_base_url is not None:
            settings.llm_base_url = args.llm_base_url
        if args.llm_api_key is not None:
            settings.llm_api_key = args.llm_api_key
        if args.llm_model is not None:
            settings.llm_model = args.llm_model

        index, meta = load_index(settings)
        embedder = SentenceTransformer(settings.embed_model)
        out = rag_answer(settings, args.query, embedder, index, meta)
        print("Q:", out["query"])
        print(f"confidence={out['confidence']:.2f} | routed={out['routed_to_human']} | latency={out['latency_s']:.2f}s | citations={out['has_citations']}")
        print("\nA:\n", out["answer"])
        print("\nFuentes:")
        for i, s in enumerate(out["sources"], 1):
            t = s["meta"].get("type","")
            cat = s["meta"].get("categoria","")
            extra = f" | categoria={cat}" if cat else ""
            print(f"  [{i}] score={s['score']:.2f} | {s['title']} > {s['section']} | type={t}{extra}")
        return

    if args.cmd == "eval":
        if args.index_dir is not None:
            settings.index_dir = args.index_dir
        if args.raw_dir is not None:
            settings.raw_dir = args.raw_dir
        if args.log_dir is not None:
            settings.log_dir = args.log_dir
        if args.embed_model is not None:
            settings.embed_model = args.embed_model
        if args.top_k is not None:
            settings.top_k = args.top_k
        if args.min_sim is not None:
            settings.min_sim = args.min_sim
        if args.use_llm is not None:
            settings.use_llm = args.use_llm
        if args.llm_base_url is not None:
            settings.llm_base_url = args.llm_base_url
        if args.llm_api_key is not None:
            settings.llm_api_key = args.llm_api_key
        if args.llm_model is not None:
            settings.llm_model = args.llm_model

        result = run_eval(settings, args.eval_path, out_csv=args.out_csv, intents_json=getattr(args, "intents_json", ""), max_rows=args.max_rows)
        print("✅ Evaluación terminada")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

if __name__ == "__main__":
    main()