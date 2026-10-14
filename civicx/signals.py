"""Rule-based information extraction for urgency: duration, impact, urgency keywords, severity lexicon.
Works on the *normalised* text so it is language aware but does not need stemming."""
import math
import re

from .lexicon import EN_IMPACT, EN_SEVERE, EN_URGENT, MR_IMPACT, MR_SEVERE, MR_URGENT

_EN_NUM = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
           "nine": 9, "ten": 10, "couple": 2, "few": 3}
_MR_NUM = {"एक": 1, "एका": 1, "दोन": 2, "तीन": 3, "चार": 4, "पाच": 5, "सहा": 6, "सात": 7, "आठ": 8, "नऊ": 9, "दहा": 10}
_UNIT_DAYS_EN = {"hour": 1 / 24, "day": 1, "week": 7, "month": 30}
_UNIT_DAYS_MR = {"तास": 1 / 24, "दिवस": 1, "आठवड": 7, "महिन": 30}

_EN_DUR = re.compile(r"\b(\d+|" + "|".join(_EN_NUM) + r")\s*(hour|day|week|month)s?\b")
_MR_DUR = re.compile(r"(\d+|" + "|".join(_MR_NUM) + r")\s*(तास|दिवस|आठवड|महिन)")
_EN_SINCE = re.compile(r"since (morning|yesterday|last night|monday|tuesday|wednesday|thursday|friday|saturday|sunday)|from last night|last night|whole day|all day")
_MR_SINCE = re.compile(r"सकाळपासून|काल रात्रीपासून|कालपासून|रात्रीपासून|सोमवारपासून|दिवसभर")


def extract_duration_days(norm: str, lang: str) -> float:
    """Longest duration mentioned (in days). 'since morning' style phrases count as 1 day."""
    best = 0.0
    if lang == "mr":
        for n, unit in _MR_DUR.findall(norm):
            val = int(n) if n.isdigit() else _MR_NUM.get(n, 1)
            best = max(best, val * _UNIT_DAYS_MR[unit])
        if best == 0 and _MR_SINCE.search(norm):
            best = 1.0
        # Marathi complaints sometimes contain English durations too
        for n, unit in _EN_DUR.findall(norm):
            val = int(n) if n.isdigit() else _EN_NUM.get(n, 1)
            best = max(best, val * _UNIT_DAYS_EN[unit])
    else:
        for n, unit in _EN_DUR.findall(norm):
            val = int(n) if n.isdigit() else _EN_NUM.get(n, 1)
            best = max(best, val * _UNIT_DAYS_EN[unit])
        if best == 0 and _EN_SINCE.search(norm):
            best = 1.0
    return best


FEATURE_NAMES = ["log_duration", "urgent_kw", "severe_count", "hospital", "children_elderly", "school",
                 "public_safety", "negation"]


# "near Sion hospital" / "सायन रुग्णालयाजवळ" is a landmark, not an affected hospital -> blank it out before matching impact
_LANDMARK = re.compile(r"\b(near|opposite|outside|behind|beside)\s+(\w+\s+){0,2}(hospital|school|college|clinic)\b"
                       r"|(\S+\s+)?(रुग्णालय|शाळ|महाविद्यालय|दवाखान)\S*\s*(जवळ|समोर|बाहेर)\S*")


def extract_signals(norm: str, lang: str) -> dict:
    norm_all = norm
    norm = _LANDMARK.sub(" ", norm)
    imp = MR_IMPACT if lang == "mr" else EN_IMPACT
    urgent = MR_URGENT if lang == "mr" else EN_URGENT
    severe = MR_SEVERE if lang == "mr" else EN_SEVERE
    days = extract_duration_days(norm, lang)
    sig = {
        "duration_days": days,
        "urgent_kw": int(bool(re.search(urgent, norm))),
        "severe_words": [w for w in severe if w in norm_all],
        "negation": int(bool(re.search(r"\b(no|not|nor|never)\b|नाही|नाहीत|नसतात|नये", norm))),
    }
    for k, pat in imp.items():
        sig[k] = int(bool(re.search(pat, norm)))
    return sig


def signal_vector(sig: dict) -> list:
    return [math.log1p(sig["duration_days"]), sig["urgent_kw"], len(sig["severe_words"]), sig["hospital"],
            sig["children_elderly"], sig["school"], sig["public_safety"], sig["negation"]]


def explain(sig: dict, lang: str = "en") -> list:
    """Human-readable reasons behind the urgency score (explainability for the dashboard)."""
    r = []
    if sig["duration_days"] >= 1:
        d = sig["duration_days"]
        r.append(("Duration ≈ %g day(s)" % round(d, 1)) if lang == "en" else ("कालावधी ≈ %g दिवस" % round(d, 1)))
    labels = {"hospital": ("Hospital / patients affected", "रुग्णालय / रुग्णांवर परिणाम"),
              "children_elderly": ("Children / elderly affected", "लहान मुले / वृद्धांना त्रास"),
              "school": ("Students / school affected", "विद्यार्थी / शाळेवर परिणाम"),
              "public_safety": ("Public-safety hazard", "सार्वजनिक सुरक्षेला धोका")}
    for k, (en, mr) in labels.items():
        if sig.get(k):
            r.append(en if lang == "en" else mr)
    if sig["urgent_kw"]:
        r.append("Urgent / emergency wording" if lang == "en" else "तातडीचा उल्लेख")
    if sig["severe_words"]:
        r.append(("Severity cues: " if lang == "en" else "गंभीरतेचे शब्द: ") + ", ".join(sig["severe_words"][:4]))
    return r
