#!/usr/bin/env python3
"""
cross_validation_rag.py
=======================
Validación cruzada para el clasificador de intents del proyecto RAG.

Estrategias disponibles:
  1. stratified-kfold   → K folds estratificados por intent (recomendado)
  2. actor              → cada actor como fold de test (docente/estudiante/admin/soporte)
  3. channel            → cada canal como fold de test (chat/portal/telefono/correo)

Uso:
  python cross_validation_rag.py \\
    --jsonl  dataset_intents_sintetico_v1.jsonl \\
    --intents intents_priorizado.json \\
    --embed-model sentence-transformers/all-MiniLM-L6-v2 \\
    --strategy stratified-kfold \\
    --k 5 \\
    --out-dir cv_results/

Requisitos:
  pip install sentence-transformers numpy pandas scikit-learn tqdm
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm


# ══════════════════════════════════════════════════════════════════════════════
# 1. Estructuras de datos
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Record:
    record_id: str
    text: str
    intent_id: str
    category: str
    actor: str
    channel: str
    split: str          # split original del dataset (train/val/test)
    confidence: float

@dataclass
class FoldResult:
    fold_id: str        # "fold_1", "actor_docente", "channel_chat", etc.
    strategy: str
    n_train: int
    n_test: int
    accuracy: float
    per_intent: Dict[str, Dict]   # intent_id → {correct, total, precision, recall}
    confusion: Dict[str, Dict]    # {true_intent: {pred_intent: count}}
    mean_sim_correct: float
    mean_sim_wrong: float
    latency_s: float
    rows: List[Dict]              # filas individuales para exportar a CSV


# ══════════════════════════════════════════════════════════════════════════════
# 2. Carga de datos
# ══════════════════════════════════════════════════════════════════════════════

def load_records(jsonl_path: str) -> List[Record]:
    records = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            label  = obj.get("label") or {}
            source = obj.get("source") or {}
            qa     = obj.get("qa") or {}

            text = str(obj.get("text", "")).strip()
            intent_id = (
                str(label.get("intent_id", "")).strip()
                or str(obj.get("intent_id", "")).strip()
                or str(obj.get("intent", "")).strip()
            )
            if not text or not intent_id:
                continue

            records.append(Record(
                record_id  = str(obj.get("record_id", f"rec_{len(records)}")),
                text       = text,
                intent_id  = intent_id,
                category   = str(label.get("category", "")).strip(),
                actor      = str(label.get("actor", "")).strip(),
                channel    = str(source.get("channel", "")).strip(),
                split      = str(qa.get("split", "unknown")).strip(),
                confidence = float(label.get("confidence", 1.0)),
            ))
    return records


def load_intents(intents_path: str) -> List[Tuple[str, str]]:
    """Retorna List[(intent_id, texto_ancla_para_embedding)]."""
    with open(intents_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    catalog = data.get("intent_catalog", [])
    intents = []
    for it in catalog:
        intent_id = str(it.get("intent_id", "")).strip()
        if not intent_id:
            continue
        display  = str(it.get("display_name", "")).strip()
        desc     = str(it.get("description", "")).strip()
        cat      = str(it.get("category", "")).strip()
        utts     = it.get("utterances", []) or []
        utt_text = " | ".join(str(u).strip() for u in utts if str(u).strip())
        text     = " | ".join(x for x in [display, desc, utt_text,
                              f"categoria={cat}" if cat else ""] if x)
        intents.append((intent_id, text))
    return intents


# ══════════════════════════════════════════════════════════════════════════════
# 3. Clasificador de intents (embedding + coseno)
# ══════════════════════════════════════════════════════════════════════════════

class IntentClassifier:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer
        print(f"  Cargando modelo: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.intent_names: List[str] = []
        self.intent_vecs: Optional[np.ndarray] = None

    def fit(self, intents: List[Tuple[str, str]]) -> None:
        """Construye la matriz de embeddings del catálogo de intents."""
        self.intent_names = [iid for iid, _ in intents]
        texts = [f"{iid}: {desc}" for iid, desc in intents]
        vecs = self.model.encode(texts, normalize_embeddings=True,
                                  show_progress_bar=False)
        self.intent_vecs = np.asarray(vecs, dtype="float32")

    def predict(self, texts: List[str]) -> Tuple[List[str], List[float]]:
        """Predice intent para cada texto. Retorna (preds, sims)."""
        assert self.intent_vecs is not None, "Llama a fit() primero."
        qvecs = self.model.encode(texts, normalize_embeddings=True,
                                   show_progress_bar=False)
        qvecs = np.asarray(qvecs, dtype="float32")
        sims  = qvecs @ self.intent_vecs.T      # (N, n_intents)
        idxs  = sims.argmax(axis=1)
        preds = [self.intent_names[i] for i in idxs]
        scores = [float(sims[n, i]) for n, i in enumerate(idxs)]
        return preds, scores


# ══════════════════════════════════════════════════════════════════════════════
# 4. Estrategias de partición (folds)
# ══════════════════════════════════════════════════════════════════════════════

def stratified_kfold_splits(
    records: List[Record], k: int, seed: int = 42
) -> List[Tuple[List[Record], List[Record]]]:
    """
    Genera K pares (train, test) con distribución proporcional por intent.
    Garantiza al menos 1 muestra de cada intent en cada fold de test.
    """
    rng = random.Random(seed)
    by_intent: Dict[str, List[Record]] = defaultdict(list)
    for r in records:
        by_intent[r.intent_id].append(r)

    # Asignar cada registro a un fold de forma round-robin dentro de su intent
    folds: List[List[Record]] = [[] for _ in range(k)]
    for recs in by_intent.values():
        shuffled = recs[:]
        rng.shuffle(shuffled)
        for i, r in enumerate(shuffled):
            folds[i % k].append(r)

    splits = []
    for test_fold_i in range(k):
        test  = folds[test_fold_i]
        train = [r for i, fold in enumerate(folds) if i != test_fold_i for r in fold]
        splits.append((train, test))
    return splits


def group_splits(
    records: List[Record], group_field: str
) -> List[Tuple[str, List[Record], List[Record]]]:
    """
    Genera un fold por cada valor único del campo group_field.
    Test = registros de ese grupo. Train = todos los demás.
    Retorna List[(grupo, train, test)].
    """
    groups: Dict[str, List[Record]] = defaultdict(list)
    for r in records:
        val = getattr(r, group_field, "") or "unknown"
        groups[val].append(r)

    result = []
    for group_val, test_recs in sorted(groups.items()):
        train_recs = [r for r in records if getattr(r, group_field, "") != group_val]
        result.append((group_val, train_recs, test_recs))
    return result


# ══════════════════════════════════════════════════════════════════════════════
# 5. Evaluación de un fold
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_fold(
    fold_id: str,
    strategy: str,
    train: List[Record],
    test: List[Record],
    classifier: IntentClassifier,
    intents: List[Tuple[str, str]],
) -> FoldResult:
    """
    Nota: en clasificación por embedding sobre catálogo fijo, el conjunto
    'train' no se usa para re-entrenar el modelo (que es pre-entrenado),
    pero SÍ se usa para:
      - Verificar cobertura de intents representados
      - En versiones futuras: fine-tuning o umbral adaptativo por fold
    """
    t0 = time.time()

    # Filtrar intents que tienen al menos 1 muestra en train
    train_intents = {r.intent_id for r in train}
    active_intents = [(iid, desc) for iid, desc in intents if iid in train_intents]

    if not active_intents:
        active_intents = intents   # fallback: usar todos

    classifier.fit(active_intents)

    texts = [r.text for r in test]
    preds, sims = classifier.predict(texts)

    latency = time.time() - t0

    # ── Métricas globales ───────────────────────────────────────────────────
    correct_mask = [p == r.intent_id for p, r in zip(preds, test)]
    accuracy = sum(correct_mask) / len(correct_mask) if correct_mask else 0.0

    sim_correct = [s for s, c in zip(sims, correct_mask) if c]
    sim_wrong   = [s for s, c in zip(sims, correct_mask) if not c]
    mean_sim_correct = float(np.mean(sim_correct)) if sim_correct else 0.0
    mean_sim_wrong   = float(np.mean(sim_wrong))   if sim_wrong   else 0.0

    # ── Métricas por intent ─────────────────────────────────────────────────
    tp_per: Dict[str, int] = defaultdict(int)
    total_true: Dict[str, int] = defaultdict(int)
    total_pred: Dict[str, int] = defaultdict(int)
    confusion: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for r, pred, sim in zip(test, preds, sims):
        total_true[r.intent_id] += 1
        total_pred[pred] += 1
        confusion[r.intent_id][pred] += 1
        if pred == r.intent_id:
            tp_per[r.intent_id] += 1

    all_intents = sorted(set(list(total_true.keys()) + list(total_pred.keys())))
    per_intent = {}
    for iid in all_intents:
        tp = tp_per.get(iid, 0)
        n_true = total_true.get(iid, 0)
        n_pred = total_pred.get(iid, 0)
        precision = tp / n_pred  if n_pred  > 0 else 0.0
        recall    = tp / n_true  if n_true  > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)
        per_intent[iid] = {
            "true_positive": tp,
            "n_true": n_true,
            "n_pred": n_pred,
            "precision": round(precision, 4),
            "recall":    round(recall, 4),
            "f1":        round(f1, 4),
        }

    # ── Filas individuales para CSV ─────────────────────────────────────────
    rows = []
    for r, pred, sim, ok in zip(test, preds, sims, correct_mask):
        rows.append({
            "fold_id":      fold_id,
            "record_id":    r.record_id,
            "text":         r.text,
            "intent_true":  r.intent_id,
            "intent_pred":  pred,
            "sim":          round(float(sim), 4),
            "correct":      ok,
            "actor":        r.actor,
            "channel":      r.channel,
            "split_orig":   r.split,
        })

    return FoldResult(
        fold_id         = fold_id,
        strategy        = strategy,
        n_train         = len(train),
        n_test          = len(test),
        accuracy        = round(accuracy, 4),
        per_intent      = per_intent,
        confusion       = {k: dict(v) for k, v in confusion.items()},
        mean_sim_correct= round(mean_sim_correct, 4),
        mean_sim_wrong  = round(mean_sim_wrong, 4),
        latency_s       = round(latency, 3),
        rows            = rows,
    )


# ══════════════════════════════════════════════════════════════════════════════
# 6. Agregación de resultados
# ══════════════════════════════════════════════════════════════════════════════

def aggregate_results(fold_results: List[FoldResult]) -> Dict[str, Any]:
    accuracies = [f.accuracy for f in fold_results]
    mean_acc   = float(np.mean(accuracies))
    std_acc    = float(np.std(accuracies))
    ci_95      = 1.96 * std_acc / (len(accuracies) ** 0.5)

    sim_c = [f.mean_sim_correct for f in fold_results]
    sim_w = [f.mean_sim_wrong   for f in fold_results]

    # Macro-avg F1 por intent (promediando sobre folds)
    all_intents = set()
    for fr in fold_results:
        all_intents.update(fr.per_intent.keys())

    per_intent_agg = {}
    for iid in sorted(all_intents):
        f1s  = [fr.per_intent[iid]["f1"]        for fr in fold_results if iid in fr.per_intent]
        recs = [fr.per_intent[iid]["recall"]     for fr in fold_results if iid in fr.per_intent]
        pres = [fr.per_intent[iid]["precision"]  for fr in fold_results if iid in fr.per_intent]
        per_intent_agg[iid] = {
            "mean_f1":        round(float(np.mean(f1s)),  4) if f1s  else None,
            "mean_precision": round(float(np.mean(pres)), 4) if pres else None,
            "mean_recall":    round(float(np.mean(recs)), 4) if recs else None,
            "std_f1":         round(float(np.std(f1s)),   4) if f1s  else None,
        }

    macro_f1 = float(np.mean([v["mean_f1"] for v in per_intent_agg.values()
                               if v["mean_f1"] is not None]))

    # Intents más problemáticos
    worst = sorted(
        [(iid, v["mean_f1"]) for iid, v in per_intent_agg.items()
         if v["mean_f1"] is not None],
        key=lambda x: x[1]
    )[:10]

    return {
        "n_folds":          len(fold_results),
        "mean_accuracy":    round(mean_acc, 4),
        "std_accuracy":     round(std_acc, 4),
        "ci_95_accuracy":   round(ci_95, 4),
        "min_accuracy":     round(min(accuracies), 4),
        "max_accuracy":     round(max(accuracies), 4),
        "macro_f1":         round(macro_f1, 4),
        "mean_sim_correct": round(float(np.mean(sim_c)), 4),
        "mean_sim_wrong":   round(float(np.mean(sim_w)), 4),
        "per_fold": [
            {
                "fold_id":  f.fold_id,
                "accuracy": f.accuracy,
                "n_test":   f.n_test,
                "n_train":  f.n_train,
            }
            for f in fold_results
        ],
        "per_intent": per_intent_agg,
        "worst_10_intents": [
            {"intent_id": iid, "mean_f1": round(f1, 4)}
            for iid, f1 in worst
        ],
    }


# ══════════════════════════════════════════════════════════════════════════════
# 7. Guardado de resultados
# ══════════════════════════════════════════════════════════════════════════════

def save_results(
    out_dir: str,
    fold_results: List[FoldResult],
    summary: Dict[str, Any],
    strategy: str,
    k: Optional[int],
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    tag = f"{strategy}_k{k}" if strategy == "stratified-kfold" else strategy

    # 1) Resumen JSON
    summary_path = os.path.join(out_dir, f"cv_summary_{tag}.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"  Resumen guardado  : {summary_path}")

    # 2) CSV con todas las predicciones individuales
    all_rows = [row for fr in fold_results for row in fr.rows]
    rows_path = os.path.join(out_dir, f"cv_predictions_{tag}.csv")
    pd.DataFrame(all_rows).to_csv(rows_path, index=False, encoding="utf-8")
    print(f"  Predicciones CSV  : {rows_path}")

    # 3) CSV de métricas por intent
    intent_rows = []
    for iid, vals in summary["per_intent"].items():
        intent_rows.append({"intent_id": iid, **vals})
    intent_path = os.path.join(out_dir, f"cv_per_intent_{tag}.csv")
    pd.DataFrame(intent_rows).sort_values("mean_f1").to_csv(
        intent_path, index=False, encoding="utf-8"
    )
    print(f"  Por intent CSV    : {intent_path}")

    # 4) Matriz de confusión global (suma sobre todos los folds)
    confusion_global: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for fr in fold_results:
        for true_int, pred_dict in fr.confusion.items():
            for pred_int, count in pred_dict.items():
                confusion_global[true_int][pred_int] += count

    all_labels = sorted(confusion_global.keys())
    conf_matrix = pd.DataFrame(
        [[confusion_global[t].get(p, 0) for p in all_labels] for t in all_labels],
        index=all_labels, columns=all_labels
    )
    conf_path = os.path.join(out_dir, f"cv_confusion_{tag}.csv")
    conf_matrix.to_csv(conf_path, encoding="utf-8")
    print(f"  Confusión CSV     : {conf_path}")


# ══════════════════════════════════════════════════════════════════════════════
# 8. CLI principal
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        prog="cross_validation_rag.py",
        description="Validación cruzada del clasificador de intents RAG",
    )
    parser.add_argument("--jsonl",        required=True,  help="Ruta al dataset .jsonl")
    parser.add_argument("--intents",      required=True,  help="Ruta a intents_priorizado.json")
    parser.add_argument("--embed-model",  default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--strategy",     default="stratified-kfold",
                        choices=["stratified-kfold", "actor", "channel"])
    parser.add_argument("--k",            type=int, default=5,
                        help="Número de folds (solo para stratified-kfold)")
    parser.add_argument("--seed",         type=int, default=42)
    parser.add_argument("--out-dir",      default="cv_results")
    args = parser.parse_args()

    print("=" * 60)
    print("  VALIDACIÓN CRUZADA — Clasificador de Intents RAG")
    print("=" * 60)
    print(f"  Dataset   : {args.jsonl}")
    print(f"  Intents   : {args.intents}")
    print(f"  Modelo    : {args.embed_model}")
    print(f"  Estrategia: {args.strategy}" + (f"  K={args.k}" if args.strategy == "stratified-kfold" else ""))
    print(f"  Salida    : {args.out_dir}")
    print()

    # ── Cargar datos ──────────────────────────────────────────────────────────
    print("Cargando datos...")
    records = load_records(args.jsonl)
    intents = load_intents(args.intents)
    print(f"  Registros cargados: {len(records)}")
    print(f"  Intents en catálogo: {len(intents)}")

    intent_counts = Counter(r.intent_id for r in records)
    unlabeled = [iid for iid, _ in intents if iid not in intent_counts]
    if unlabeled:
        print(f"  ⚠️  Intents sin muestras en el dataset: {unlabeled}")

    # ── Inicializar clasificador ───────────────────────────────────────────────
    print("\nInicializando clasificador...")
    clf = IntentClassifier(args.embed_model)

    # ── Generar folds ─────────────────────────────────────────────────────────
    print(f"\nGenerando folds ({args.strategy})...")
    fold_results: List[FoldResult] = []

    if args.strategy == "stratified-kfold":
        splits = stratified_kfold_splits(records, k=args.k, seed=args.seed)
        for fold_i, (train, test) in enumerate(
            tqdm(splits, desc="Evaluando folds"), start=1
        ):
            fr = evaluate_fold(
                fold_id  = f"fold_{fold_i}",
                strategy = args.strategy,
                train    = train,
                test     = test,
                classifier = clf,
                intents  = intents,
            )
            fold_results.append(fr)
            print(f"  Fold {fold_i}/{args.k}  "
                  f"train={fr.n_train}  test={fr.n_test}  "
                  f"accuracy={fr.accuracy:.4f}  "
                  f"sim_ok={fr.mean_sim_correct:.3f}  "
                  f"sim_fail={fr.mean_sim_wrong:.3f}")

    elif args.strategy in ("actor", "channel"):
        group_field = args.strategy  # "actor" o "channel"
        group_folds = group_splits(records, group_field)
        for group_val, train, test in tqdm(group_folds, desc="Evaluando grupos"):
            if not test:
                continue
            fr = evaluate_fold(
                fold_id  = f"{group_field}_{group_val}",
                strategy = args.strategy,
                train    = train,
                test     = test,
                classifier = clf,
                intents  = intents,
            )
            fold_results.append(fr)
            print(f"  {group_field}={group_val:<15}  "
                  f"train={fr.n_train}  test={fr.n_test}  "
                  f"accuracy={fr.accuracy:.4f}")

    # ── Agregar y guardar ──────────────────────────────────────────────────────
    print("\nAgregando resultados...")
    summary = aggregate_results(fold_results)

    print()
    print("=" * 60)
    print("  RESUMEN FINAL")
    print("=" * 60)
    print(f"  Folds evaluados    : {summary['n_folds']}")
    print(f"  Accuracy media     : {summary['mean_accuracy']:.4f} ± {summary['std_accuracy']:.4f}")
    print(f"  IC 95%             : ± {summary['ci_95_accuracy']:.4f}")
    print(f"  Rango              : [{summary['min_accuracy']:.4f}, {summary['max_accuracy']:.4f}]")
    print(f"  Macro F1           : {summary['macro_f1']:.4f}")
    print(f"  Sim media correctas: {summary['mean_sim_correct']:.4f}")
    print(f"  Sim media errores  : {summary['mean_sim_wrong']:.4f}")
    print()
    print("  10 intents con menor F1:")
    for item in summary["worst_10_intents"]:
        print(f"    {item['intent_id']:<45} F1={item['mean_f1']:.4f}")

    print("\nGuardando archivos de resultados...")
    save_results(
        out_dir      = args.out_dir,
        fold_results = fold_results,
        summary      = summary,
        strategy     = args.strategy,
        k            = args.k if args.strategy == "stratified-kfold" else None,
    )
    print("\n✅ Validación cruzada completada.")


if __name__ == "__main__":
    main()
