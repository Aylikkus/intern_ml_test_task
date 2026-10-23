"""Features for the per-field OCR-error classifier (string-level).

Goal: given the OCR text of ONE field value, predict P(the value contains an
OCR error). Labels are free: compare the prediction with GT on train.
Applies to the free-text fields only (dates/code/sex/licence are fully
covered by deterministic rules):
    Last_name_ru, First_name_ru, Middle_name_ru, Birth_place_ru,
    Issue_organization_ru

Feature groups:
  1. char-confidence shape of the field (min/mean, fractions below thresholds)
  2. format plausibility (digits/Latin in a name, odd punctuation, length)
  3. character trigram language model fitted on train GT of the SAME field
     (OOV-trigram rate + mean trigram log-frequency) - catches substitutions
     like СМИРНОВ->СМИРНСВ even when char_conf is high
  4. doc-level MRZ cross-check features (fio_score, cd_fail_count, ...)
  5. doc-level image quality (glare, sharpness, doc_px, source)

NOTE: lexicon-similarity features (trigram-Dice vs train-GT vocabulary) were
tried and REMOVED: they are toxic under distribution shift. On train the
value's own GT sits in the vocabulary (median sim 1.0), on test values are
unseen (median 0.67), so the model turns 'no lexicon support' into 'error'
and rejects almost every correct test document. The method lexicon_sim is
kept for ad-hoc analysis only.
"""
import math
import re
from collections import Counter, defaultdict

TEXT_FIELDS = ('Last_name_ru', 'First_name_ru', 'Middle_name_ru',
               'Birth_place_ru', 'Issue_organization_ru')


def char_conf_features(viz_entry):
    """(text, confs, feats) for one viz field; confs drop None separators."""
    lines = (viz_entry or {}).get('lines', [])
    text = ' '.join(l.get('text', '') for l in lines).strip()
    confs = [c for l in lines for c in l.get('char_conf', []) if c is not None]
    n = max(1, len(confs))
    feats = {
        'n_chars': len(text),
        'n_lines': len(lines),
        'min_conf': min(confs) if confs else 0.0,
        'mean_conf': sum(confs) / n if confs else 0.0,
        'frac_below_92': sum(c < 0.92 for c in confs) / n,
        'frac_below_95': sum(c < 0.95 for c in confs) / n,
        'frac_below_99': sum(c < 0.99 for c in confs) / n,
    }
    return text, confs, feats


def _norm(s):
    return ' '.join(str(s or '').split()).upper()


def _trigrams(s):
    s = f' {s} '
    return [s[i:i + 3] for i in range(len(s) - 2)]


class TextFeatureBuilder:
    """Fits trigram LMs and lexicons per field; transforms one field value."""

    def fit(self, corpus_by_field, extra_lexicon_by_field=None):
        """corpus_by_field: {field: iterable of GT strings}."""
        self.grams = {}
        self.gram_total = {}
        self.lexicon = {}
        self.index = {}
        for f in TEXT_FIELDS:
            texts = [_norm(t) for t in corpus_by_field.get(f, ()) if t]
            cnt = Counter()
            for t in texts:
                cnt.update(_trigrams(t))
            self.grams[f] = cnt
            self.gram_total[f] = sum(cnt.values()) or 1
            vocab = list(dict.fromkeys(texts))
            for extra in (extra_lexicon_by_field or {}).get(f, ()):  # e.g. subdivisions.csv
                v = _norm(extra)
                if v and len(v) >= 4:
                    vocab.append(v)
            entries = [(v, set(_trigrams(v))) for v in vocab]
            self.lexicon[f] = entries
            inv = defaultdict(list)
            for i, (v, gs) in enumerate(entries):
                for g in gs:
                    inv[g].append(i)
            self.index[f] = inv
        return self

    def lm_features(self, field, text, sub=None):
        """OOV-trigram rate and mean log-frequency for one value.

        sub: iterable of the SAME doc's own GT values for this field; their
        trigrams are subtracted from the corpus first (leave-one-doc-out), so
        that a training row's features match the 'unseen value' conditions of
        test time. Without this, every correct train value has oov3_rate == 0
        and the model learns a rule that does not transfer to unseen names.
        """
        t = _norm(text)
        tri = _trigrams(t)
        if not tri:
            return {'oov3_rate': 0.0, 'mean_logfreq3': 0.0}
        cnt = self.grams[field]
        total = self.gram_total[field]
        if sub:
            sub_cnt = Counter()
            for s in sub:
                if s:
                    sub_cnt.update(_trigrams(_norm(s)))
            cnt = cnt - sub_cnt
            total = max(1, total - sum(sub_cnt.values()))
        oov = sum(1 for g in tri if cnt.get(g, 0) == 0)
        mean_log = sum(math.log((cnt.get(g, 0) + 1) / total) for g in tri) / len(tri)
        return {'oov3_rate': oov / len(tri), 'mean_logfreq3': mean_log}

    def lexicon_sim(self, field, text, probe=12, top_k=40):
        """Max trigram-Dice similarity to lexicon entries.

        Candidates are fetched via the RAREST trigrams of the value only
        (short posting lists), which keeps this fast without hurting recall
        much: a near-match shares even the rare trigrams.
        """
        t = _norm(text)
        gs = set(_trigrams(t))
        if not gs:
            return 0.0
        inv = self.index[field]
        probe_grams = sorted(gs, key=lambda g: len(inv.get(g, ())))[:probe]
        cand = Counter()
        for g in probe_grams:
            for i in inv.get(g, ()):
                cand[i] += 1
        best = 0.0
        for i, _ in cand.most_common(top_k):
            ref = self.lexicon[field][i][1]
            inter = len(gs & ref)
            dice = 2 * inter / (len(gs) + len(ref)) if (gs or ref) else 0.0
            best = max(best, dice)
        return best

    @staticmethod
    def format_features(field, text):
        t = _norm(text)
        is_name = field in TEXT_FIELDS[:3]
        allowed = re.compile(r"[А-ЯЁ\s.-]*\Z") if is_name else \
            re.compile(r"[A-ZА-ЯЁ0-9\s.,()№/'\"’-]*\Z")
        return {
            'has_digit': float(any(c.isdigit() for c in t)),
            'has_latin': float(any('A' <= c <= 'Z' for c in t)),
            'charset_bad': float(t != '' and allowed.fullmatch(t) is None),
            'n_dots': t.count('.'),
        }

    def strip_lexicon(self):
        """Удаляет словари из состояния (нужны только для lexicon_sim,
        который не входит в признаки). Вызывать перед сохранением модели,
        чтобы pickle был компактным. После этого lexicon_sim недоступен."""
        self.lexicon = {}
        self.index = {}
        return self

    def transform(self, field, viz_entry, lm_sub=None):
        """Feature dict for one viz field value.

        lexicon_sim is deliberately NOT included: on train the value's own GT
        sits in the lexicon (median similarity 1.0), but on test values are
        unseen (median 0.67), which makes the feature toxic under distribution
        shift. Kept as a method for ad-hoc analysis only.
        lm_sub: pass the doc's own GT value of this field when building
        TRAINING features (leave-one-doc-out LM, see lm_features).
        """
        text, confs, feats = char_conf_features(viz_entry)
        out = dict(feats)
        out.update({f'field={field}': 1.0 for f in TEXT_FIELDS if f != field})
        out.update(self.lm_features(field, text, sub=lm_sub))
        out.update(self.format_features(field, text))
        return out, text


def mrz_doc_features(cc):
    """Doc-level MRZ cross-check features from mrzlib.cross_check output."""
    return {
        'mrz_ok': float(cc['mrz_ok'] is True),
        'mrz_missing': float(cc['mrz_ok'] is None),
        'birth_mismatch': float(cc['birth_mismatch'] is True),
        'sex_mismatch': float(cc['sex_mismatch'] is True),
        'licence_mismatch': float(cc['licence_mismatch'] is True),
        'issue_date_mismatch': float(cc['issue_date_mismatch'] is True),
        'org_code_mismatch': float(cc['org_code_mismatch'] is True),
        'fio_score': cc['fio_score'] if cc['fio_score'] is not None else 1.0,
        'cd_fail_count': float(cc['cd_fail_count'] or 0),
    }


def quality_features(rec):
    q = rec.get('quality') or {}
    src = rec.get('source') or '?'
    feats = {
        'glare': float(q.get('glare') or 0.0),
        'sharpness': float(q.get('sharpness') or 0.0),
        'doc_px': float(q.get('doc_px') or 0.0),
    }
    for s in ('web', 'mobile', 'mobile_v2'):
        feats[f'source={s}'] = float(src == s)
    return feats
