"""Обучение классификатора OCR-ошибок для правила S каскада.

Запуск (однократно, перед postprocess.py; из корня репозитория):
    python experiments/train_model.py

Читает data/train_predictions.jsonl + data/train_labels.jsonl и
reference/subdivisions.csv из корня репозитория, обучает HistGradientBoosting
на признаках errfeatures и сохраняет артефакт:
    model/ocr_error_classifier.pkl
(модель + обученные триграммные LM + список признаков + метаданные).
После этого postprocess.py работает без train-данных и запускается быстрее.

Методология: чтобы признаки обучающих строк соответствовали условиям test
(значения, которых модель «не видела»), OOV-признаки считаются с
leave-one-doc-out — триграммы GT самого документа вычитаются из корпуса
(lm_sub). Лексиконные признаки исключены: на test значения не встречались
в словаре train (медиана сходства 0.67 против 1.0), и признак «нет поддержки
в словаре» ложно трактовался как ошибка.
"""
import csv
import json
import os
import pickle
import sys
from collections import defaultdict
from datetime import datetime

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)          # корень репозитория (postprocess.py, data/, model/)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "lib"))
import errfeatures
import mrzlib
import postprocess as pp
from errfeatures import TEXT_FIELDS

MODEL_DIR = os.path.join(ROOT, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "ocr_error_classifier.pkl")


def build_training_rows(records, builder):
    """Признаковые строки + метки; lm_sub = GT этого же документа (LOODO)."""
    rows, y = [], []
    for label_rec, pred_rec in records:
        gt_fields, viz_fields = label_rec["fields"], pred_rec["viz"]
        cc = mrzlib.cross_check(viz_fields, pred_rec.get("mrz"))
        mfeat = errfeatures.mrz_doc_features(cc)
        qfeat = errfeatures.quality_features(pred_rec)
        for field in TEXT_FIELDS:
            viz = viz_fields.get(field)
            if viz is None:
                continue
            feats = pp.build_row(builder, field, viz, mfeat, qfeat, None,
                                 lm_sub=[gt_fields[field]])
            text = feats.pop("__text__")
            rows.append(feats)
            y.append(0 if pp.norm(text) == pp.norm(gt_fields[field]) else 1)
    return rows, np.array(y)


def main():
    data_dir = os.path.join(ROOT, "data")
    train_records = list(zip(
        pp.load_jsonl(os.path.join(data_dir, "train_labels.jsonl")),
        pp.load_jsonl(os.path.join(data_dir, "train_predictions.jsonl"))))

    # корпус триграммной LM — из GT train
    corpus = defaultdict(list)
    for label_rec, _ in train_records:
        for field in TEXT_FIELDS:
            corpus[field].append(label_rec["fields"][field])
    extra_lexicon = {"Issue_organization_ru": []}
    subdiv = os.path.join(ROOT, "reference", "subdivisions.csv")
    if os.path.exists(subdiv):
        with open(subdiv, encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                extra_lexicon["Issue_organization_ru"].append(row["name"])
    builder = errfeatures.TextFeatureBuilder().fit(corpus, extra_lexicon).strip_lexicon()

    rows, y = build_training_rows(train_records, builder)
    keys = sorted(set().union(*(r.keys() for r in rows)))
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1,
                                         random_state=0)
    clf.fit(np.array([[r.get(k, 0.0) for k in keys] for r in rows]), y)

    meta = {
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "train_docs": len(train_records),
        "train_rows": len(rows),
        "positives": int(y.sum()),
        "n_features": len(keys),
        "sklearn": __import__("sklearn").__version__,
        "threshold_ml": pp.ML_T,
        "note": "LM-признаки обучающих строк считались leave-one-doc-out; "
                "лексиконные признаки исключены (сдвиг распределения на test).",
    }
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(MODEL_PATH, "wb") as fh:
        pickle.dump({"model": clf, "builder": builder, "keys": keys,
                     "meta": meta}, fh)
    print(f"модель сохранена: {MODEL_PATH} "
          f"({os.path.getsize(MODEL_PATH) / 1e6:.1f} MB)")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
