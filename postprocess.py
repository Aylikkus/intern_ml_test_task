"""Постобработка OCR паспорта РФ: решение accept/reject по каждому документу.

Запуск:
    python train_model.py        # однократно: обучает и сохраняет модель
    python postprocess.py data/test_predictions.jsonl answers.jsonl

Модель классификатора OCR-ошибок хранится в model/ocr_error_classifier.pkl
и загружается при старте; train-данные для postprocess не нужны.
Данные лежат в репозитории: data/ (train_predictions.jsonl, train_labels.jsonl,
test_predictions.jsonl) и reference/subdivisions.csv — рядом со скриптом.
Если рядом со входным файлом есть своя папка с train-данными, она тоже
подойдёт (проверяется несколько вариантов расположения).
Работает на CPU, без интернета. Зависимости: scikit-learn, numpy.

Каскад (порог подобран на train, см. REPORT.txt):
  C  min char_conf поля < 0.92                      -> reject
  M  у какого-то поля нет предсказания              -> reject
  Q  glare > 0.15 или sharpness < 125               -> reject
  F  нарушение формата поля (цифры в ФИО, даты,
     коды, допуск по алфавиту)                      -> reject
  R1 код подразделения отсутствует в справочнике    -> reject
  X1 дата рождения не совпала с MRZ                 -> reject (43 TP / 0 FP)
  X3 номер лицензии не совпал с MRZ (MRZ цел)       -> reject (56 TP / 0 FP)
  X4 дата выдачи / код подразделения не совпали
     с MRZ (MRZ цел)                                -> reject (69 TP / 1 FP)
  V1 дата выдачи > 01.10.2026, V2 возраст при выдаче
     < 14, V3 выдача раньше рождения                -> reject (33 TP / 0 FP)
  S  классификатор OCR-ошибок даёт p > 0.5 хотя бы
     для одного текстового поля                     -> reject (~79 TP / ~2 FP)
MRZ — НЕ истина в последней инстанции: ошибки виз. зоны и MRZ частично
совпадают, поэтому используется только факт РАСХОЖДЕНИЯ.
"""
import csv
import json
import os
import pickle
import re
import sys
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mrzlib
import errfeatures
from errfeatures import TEXT_FIELDS

CHAR_CONF_T = 0.92
GLARE_MAX = 0.15
SHARPNESS_MIN = 125
ML_T = 0.5
REF_DATE = datetime(2026, 10, 1)

FIELDS = ["Last_name_ru", "First_name_ru", "Middle_name_ru", "Sex_ru",
          "Birth_date", "Birth_place_ru", "Issue_date",
          "Issue_organisation_code", "Licence_number", "Issue_organization_ru"]
FMT_NAME = ("Last_name_ru", "First_name_ru", "Middle_name_ru")
FMT_DATE = ("Birth_date", "Issue_date")
FMT_TEXT_RE = re.compile(r"[A-Za-zА-ЯЁа-яё\s.\d,()№/'\"’-]*\Z")


def norm(s):
    return " ".join(str(s or "").split()).upper()


def viz_text(entry):
    if entry is None:
        return None
    return " ".join(l.get("text", "") for l in entry.get("lines", [])).strip()


def load_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def old_rule_flags(viz, pred_fields, quality):
    """C, M, Q, F, R1 — правила первой версии каскада."""
    min_conf = None
    missing = False
    for f in FIELDS:
        v = viz.get(f)
        if v is None:
            missing = True
            continue
        for line in v.get("lines", []):
            for c in line.get("char_conf", []):
                if c is not None and (min_conf is None or c < min_conf):
                    min_conf = c
    fmt_bad = False
    for f in FIELDS:
        v = norm(pred_fields[f])
        if not v:
            continue
        if f in FMT_NAME:
            fmt_bad |= re.search(r"\d", v) is not None
        elif f == "Sex_ru":
            fmt_bad |= v not in ("МУЖ.", "ЖЕН.")
        elif f in FMT_DATE:
            try:
                datetime.strptime(v, "%d.%m.%Y")
            except ValueError:
                fmt_bad = True
        elif f == "Issue_organisation_code":
            fmt_bad |= re.fullmatch(r"[\d-]*", v) is None
        elif f == "Licence_number":
            fmt_bad |= re.fullmatch(r"[\d\s]*", v) is None
        else:
            fmt_bad |= FMT_TEXT_RE.fullmatch(v) is None
    code = norm(pred_fields["Issue_organisation_code"])
    return {
        "C": min_conf is not None and min_conf < CHAR_CONF_T,
        "M": missing,
        "Q": (quality.get("glare", 0) > GLARE_MAX)
             or (quality.get("sharpness", 10 ** 9) < SHARPNESS_MIN),
        "F": fmt_bad,
        "R1": bool(code) and code not in codes_ref,
    }


def mrz_flag(viz, mrz):
    """X1/X3/X4: расхождения виз. зоны и MRZ; X3/X4 только при целом MRZ."""
    cc = mrzlib.cross_check(viz, mrz)
    clean = cc["mrz_ok"] and cc["cd_fail_count"] == 0
    return (cc["birth_mismatch"] is True
            or (clean and cc["licence_mismatch"] is True)
            or (clean and cc["issue_date_mismatch"] is True)
            or (clean and cc["org_code_mismatch"] is True))


def validity_flag(pred_fields):
    """V1/V2/V3 — консервативная проверка правдоподобия дат."""
    try:
        b = datetime.strptime(pred_fields["Birth_date"], "%d.%m.%Y")
        i = datetime.strptime(pred_fields["Issue_date"], "%d.%m.%Y")
        return i > REF_DATE or (i - b).days / 365.25 < 14 or i < b
    except (ValueError, TypeError):
        return False


# ------------------------------------------------------------- classifier --

def build_row(builder, field, viz, mfeat, qfeat, keys_cache, lm_sub=None):
    feats, text = builder.transform(field, viz, lm_sub=lm_sub)
    feats.update(mfeat)
    feats.update(qfeat)
    feats["__text__"] = text
    return feats


def vec(row, keys):
    return [row.get(k, 0.0) for k in keys]


def main(inp_path, out_path):
    here = os.path.dirname(os.path.abspath(__file__))
    # справочник подразделений: сначала рядом со скриптом (reference/),
    # затем относительно входного файла
    subdiv = os.path.join(here, "reference", "subdivisions.csv")
    if not os.path.exists(subdiv):
        data_dir = os.path.dirname(os.path.abspath(inp_path))
        subdiv = os.path.join(data_dir, "..", "reference", "subdivisions.csv")

    global codes_ref
    codes_ref = set()
    if os.path.exists(subdiv):
        with open(subdiv, encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                codes_ref.add(row["code"].strip())

    # --- 1. классификатор OCR-ошибок: загрузка обученной модели
    model_path = os.path.join(here, "model", "ocr_error_classifier.pkl")
    if not os.path.exists(model_path):
        sys.exit(f"не найдена модель {model_path} — сначала запустите train_model.py")
    with open(model_path, "rb") as fh:
        artifact = pickle.load(fh)
    clf, builder, keys = artifact["model"], artifact["builder"], artifact["keys"]
    print(f"модель загружена: {model_path} "
          f"(обучена {artifact['meta'].get('trained_at', '?')})", file=sys.stderr)

    # --- 2. каскад на test
    n_docs = n_accept = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for doc_i, pred_rec in enumerate(load_jsonl(inp_path)):
            n_docs += 1
            viz = pred_rec.get("viz") or {}
            pred_fields = {f: viz_text(viz.get(f)) for f in FIELDS}
            quality = pred_rec.get("quality") or {}

            old = old_rule_flags(viz, pred_fields, quality)
            reject = any(old.values()) \
                or mrz_flag(viz, pred_rec.get("mrz")) \
                or validity_flag(pred_fields)

            if not reject:
                # S: классификатор OCR-ошибок по текстовым полям
                cc = mrzlib.cross_check(viz, pred_rec.get("mrz"))
                mfeat = errfeatures.mrz_doc_features(cc)
                qfeat = errfeatures.quality_features(pred_rec)
                p_max = 0.0
                for field in TEXT_FIELDS:
                    viz_f = viz.get(field)
                    if viz_f is None:
                        continue
                    row = build_row(builder, field, viz_f, mfeat, qfeat, keys)
                    row.pop("__text__")
                    p_max = max(p_max, float(clf.predict_proba(
                        np.array([vec(row, keys)]))[0, 1]))
                reject = p_max > ML_T

            rec = {"doc_id": pred_rec["doc_id"],
                   "decision": "reject" if reject else "accept"}
            if not reject:
                rec["fields"] = {f: v for f, v in pred_fields.items() if v is not None}
                n_accept += 1
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"документов: {n_docs}, принято: {n_accept}, отклонено: {n_docs - n_accept}",
          file=sys.stderr)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("использование: python postprocess.py <test_predictions.jsonl> <answers.jsonl>")
    main(sys.argv[1], sys.argv[2])
