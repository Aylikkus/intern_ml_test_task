"""MRZ (machine-readable zone) parsing and cross-validation for RF passport OCR.

MRZ layout (verified empirically on train, 2758 docs with 2 MRZ lines):

Line 1 (44 chars): PNRUS<SURNAME<<GIVEN<MIDDLE<<<<...<<
Line 2 (44 chars):
    [0:9)   doc number = licence series[:3] + licence number (6 digits)
    [9]     check digit of doc number          (verifies on 99.6% of train)
    [10:13) nationality 'RUS'
    [13:19) birth date yymmdd
    [19]    check digit of birth date          (99.7%)
    [20]    sex 'M'/'F'                        (100% consistent with GT on train)
    [21:29) filler ('<' + noise digits)
    [29:35) issue date yymmdd
    [35:41) issue organisation code (6 digits, no dash)
    [41]    (padding digit)
    [28:42) 'personal number' field per ICAO TD1
    [42]    check digit of [28:42)             (97.9%)
    [43]    composite check digit over [0:10)+[13:20)+[21:43)   (95.3%)

Transliteration (derived by aligning MRZ line 1 with GT FIO on train):
    А→A Б→B В→V Г→G Д→D Е→E Ж→J З→Z И→I Й→Q К→K Л→L М→M Н→N О→O П→P Р→R
    С→S Т→T У→U Ф→F Х→H Ц→C Ч→3 Ш→4 Щ→W Ъ→X Ы→Y Ь→9 Э→6 Ю→7 Я→8 Ё→E
    (leetspeak digits; rare OCR-noise alts in the MRZ itself, e.g. А→4 И→1,
    are deliberately NOT accepted so that a corrupted MRZ lowers the
    similarity score instead of hiding an error).

Principle: MRZ is NOT ground truth (it has its own OCR errors, some with
char_conf 1.0). Cross-checks only ever flag DISAGREEMENT between the visual
zone and the MRZ, never overwrite a value.
"""
import re
from difflib import SequenceMatcher

MRZ_LEN = 44
_WEIGHTS = (7, 3, 1)


def icao_cd(s: str) -> str:
    """ICAO 9303 check digit: weights 7,3,1; '<'=0, digits as-is, A=10..Z=35."""
    total = 0
    for i, c in enumerate(s):
        if c == '<':
            v = 0
        elif c.isdigit():
            v = int(c)
        elif 'A' <= c.upper() <= 'Z':
            v = ord(c.upper()) - 55
        else:
            v = 0
        total += v * _WEIGHTS[i % 3]
    return str(total % 10)


def _clean(text: str) -> str:
    return text.replace(' ', '').upper()


# ---------------------------------------------------------------- line 2 ----

_LINE2_RE = re.compile(r'(\d{9})(\d)([A-Z<]{3})(\d{6})(\d)([MF])(.*)')


def parse_line2(text: str) -> dict | None:
    """Parse MRZ line 2. Returns None if the strict TD1 structure is broken."""
    t = _clean(text)
    if len(t) < MRZ_LEN:
        return None
    t = t[:MRZ_LEN]
    m = _LINE2_RE.fullmatch(t)
    if not m:
        return None
    num, num_cd, nat, birth, birth_cd, sex, _tail = m.groups()
    return {
        'doc_num': num, 'doc_num_cd': num_cd, 'nationality': nat,
        'birth': birth, 'birth_cd': birth_cd, 'sex': sex,
        'issue_date': t[29:35], 'org_code': t[35:41],
        'doc_num_cd_ok': icao_cd(num) == num_cd,
        'birth_cd_ok': icao_cd(birth) == birth_cd,
        'personal_cd_ok': icao_cd(t[28:42]) == t[42],
        'composite_cd_ok': icao_cd(t[0:10] + t[13:20] + t[21:43]) == t[43],
    }


# ---------------------------------------------------------------- line 1 ----

_LINE1_RE = re.compile(r'^P[A-Z<\d]?[A-Z]{3}(.*)$')


def parse_line1(text: str) -> dict | None:
    """Parse MRZ line 1 into name parts. Returns None if structure is broken."""
    t = _clean(text)
    m = _LINE1_RE.match(t)
    if not m:
        return None
    parts = [p for p in re.split(r'<+', m.group(1)) if p]
    if not parts:
        return None
    return {'parts': parts, 'joined': ''.join(parts)}


# -------------------------------------------------------- transliteration ---

TRANSLIT = {
    'А': 'A', 'Б': 'B', 'В': 'V', 'Г': 'G', 'Д': 'D', 'Е': 'E', 'Ж': 'J',
    'З': 'Z', 'И': 'I', 'Й': 'Q', 'К': 'K', 'Л': 'L', 'М': 'M', 'Н': 'N',
    'О': 'O', 'П': 'P', 'Р': 'R', 'С': 'S', 'Т': 'T', 'У': 'U', 'Ф': 'F',
    'Х': 'H', 'Ц': 'C', 'Ч': '3', 'Ш': '4', 'Щ': 'W', 'Ъ': 'X', 'Ы': 'Y',
    'Ь': '9', 'Э': '6', 'Ю': '7', 'Я': '8', 'Ё': 'E',
}


def translit(s: str) -> str:
    """Cyrillic -> expected MRZ Latin. Unknown chars pass through unchanged."""
    return ''.join(TRANSLIT.get(c, c) for c in s.upper())


def fio_similarity(gt_fio: str, mrz_line1) -> float | None:
    """Similarity (0..1) between Cyrillic FIO and the MRZ line-1 name.

    mrz_line1: raw line-1 text or the dict returned by parse_line1.
    Compares transliterated FIO with the concatenated MRZ name parts, so
    part-boundary shifts ('<' noise) do not matter.
    """
    l1 = mrz_line1 if isinstance(mrz_line1, dict) else (
        parse_line1(mrz_line1) if mrz_line1 else None)
    if not l1 or not gt_fio:
        return None
    expected = translit(gt_fio.replace(' ', ''))
    if not expected:
        return None
    return SequenceMatcher(None, expected, l1['joined'], autojunk=False).ratio()


# ---------------------------------------------------------- cross-checking --

def _parse_ddmmyyyy(text) -> str | None:
    """'10.06.2010' -> '100610' (yymmdd); None if not a plausible date."""
    if not text:
        return None
    m = re.fullmatch(r'(\d{2})\.(\d{2})\.(\d{4})', str(text).strip())
    if not m:
        return None
    dd, mm, yy = m.groups()
    return yy[-2:] + mm + dd


def _viz_text(entry) -> str | None:
    """Join OCR lines of one viz field into plain text; None if field absent."""
    if entry is None:
        return None
    if isinstance(entry, dict):
        return ' '.join(l.get('text', '') for l in entry.get('lines', [])).strip()
    return str(entry).strip()


def cross_check(viz_fields: dict, mrz,
                fio_fields=('Last_name_ru', 'First_name_ru', 'Middle_name_ru')) -> dict:
    """Compare visual-zone fields with the MRZ. Returns a feature dict.

    Every *_mismatch value is True / False / None; None = 'cannot evaluate'
    (no MRZ, MRZ parse failure, or the viz field missing/unparseable).
    """
    out = {
        'mrz_ok': None,
        'birth_mismatch': None, 'sex_mismatch': None, 'licence_mismatch': None,
        'issue_date_mismatch': None, 'org_code_mismatch': None,
        'fio_score': None, 'cd_fail_count': None,
    }
    if not mrz:
        return out
    lines = mrz.get('lines', []) if isinstance(mrz, dict) else []
    if len(lines) != 2:
        return out
    l2 = parse_line2(lines[1].get('text', ''))
    l1 = parse_line1(lines[0].get('text', ''))
    out['mrz_ok'] = l2 is not None
    if l2 is None:
        return out

    viz_birth = _parse_ddmmyyyy(_viz_text(viz_fields.get('Birth_date')))
    out['birth_mismatch'] = viz_birth is not None and viz_birth != l2['birth']

    sex_viz = (_viz_text(viz_fields.get('Sex_ru')) or '').upper()
    expect_sex = 'M' if sex_viz.startswith('МУЖ') else (
        'F' if sex_viz.startswith('ЖЕН') else None)
    out['sex_mismatch'] = expect_sex is not None and expect_sex != l2['sex']

    lic_digits = re.sub(r'\D', '', _viz_text(viz_fields.get('Licence_number')) or '')
    if len(lic_digits) == 10:
        out['licence_mismatch'] = l2['doc_num'] != lic_digits[:3] + lic_digits[4:]

    viz_issue = _parse_ddmmyyyy(_viz_text(viz_fields.get('Issue_date')))
    out['issue_date_mismatch'] = viz_issue is not None and viz_issue != l2['issue_date']

    code_digits = re.sub(r'\D', '', _viz_text(viz_fields.get('Issue_organisation_code')) or '')
    if len(code_digits) == 6:
        out['org_code_mismatch'] = l2['org_code'] != code_digits

    fio_text = ' '.join(filter(None, (_viz_text(viz_fields.get(f)) for f in fio_fields)))
    if fio_text.strip():
        out['fio_score'] = fio_similarity(fio_text, l1)

    out['cd_fail_count'] = sum(1 for k in ('doc_num_cd_ok', 'birth_cd_ok',
                                           'personal_cd_ok', 'composite_cd_ok')
                               if not l2[k])
    return out
