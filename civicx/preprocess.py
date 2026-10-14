"""Language-specific NLP preprocessing for English and Marathi.

Pipeline per language:
  1. Unicode normalisation (NFC, ZWJ/ZWNJ removal, digit mapping, chandrabindu/anusvara unification)
  2. Tokenisation (script-aware regex; the Devanagari danda '।' is treated as sentence punctuation)
  3. Stop-word removal (negations such as 'no', 'not', 'नाही' are deliberately KEPT)
  4. Morphological reduction
        English: irregular-lemma table + rule based light stemmer
        Marathi: dictionary lemmatiser for frequent verb forms + longest-suffix-match stemmer
                 for case markers/postpositions (मध्ये, ात, ाला, ांच्या ...) and verb endings
  5. Language independent 'concept tokens' + gazetteer-based location tag (see lexicon.py)
"""
import re
import unicodedata
from dataclasses import dataclass, field

from .lexicon import (CONCEPTS, EN_STOPWORDS, LOCATIONS, MR_STOPWORDS)

DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
_DEV = re.compile(r"[\u0900-\u097F]")
_LAT = re.compile(r"[A-Za-z]")
_TOKEN = re.compile(r"[\u0900-\u097F]+|[a-z0-9]+")


# ------------------------------------------------------------------ language identification
def detect_language(text: str) -> str:
    """'mr' if the message is mostly Devanagari, else 'en' (script-ratio heuristic).
    Hindi/Marathi share a script; the app only supports Marathi so Devanagari => Marathi."""
    dev, lat = len(_DEV.findall(text)), len(_LAT.findall(text))
    if dev == 0 and lat == 0:
        return "en"
    return "mr" if dev >= lat else "en"


def normalize(text: str, lang: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\u200d", "").replace("\u200c", "").replace("\u00a0", " ")
    text = text.translate(DEVANAGARI_DIGITS)
    if lang == "mr":
        text = text.replace("ँ", "ं").replace("़", "")          # unify chandrabindu; drop nukta
        text = text.replace("ऱ", "र").replace("ळ", "ळ")
        text = re.sub(r"[।॥]", " . ", text)
    text = text.lower()
    text = re.sub(r"n['’]t\b", " not", text)                # hasn't -> has not (keep the negation!)
    text = re.sub(r"https?://\S+|\S+@\S+", " ", text)
    text = re.sub(r"[^\w\s\u0900-\u097F]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


# ------------------------------------------------------------------ English
_EN_IRREGULAR = {"children": "child", "people": "person", "men": "man", "women": "woman", "was": "be", "were": "be",
                 "been": "be", "has": "have", "had": "have", "did": "do", "done": "do", "gone": "go", "went": "go",
                 "broke": "break", "broken": "break", "stuck": "stick", "fell": "fall", "fallen": "fall",
                 "burnt": "burn", "dogs": "dog", "buses": "bus", "leaves": "leaf", "ran": "run", "came": "come",
                 "cut": "cut", "cuts": "cut", "flooded": "flood", "flooding": "flood", "flooded": "flood"}


def stem_en(w: str) -> str:
    if w in _EN_IRREGULAR:
        return _EN_IRREGULAR[w]
    if len(w) <= 3 or w.isdigit():
        return w
    if w.endswith("ies") and len(w) > 4:
        w = w[:-3] + "y"
    elif w.endswith("sses"):
        w = w[:-2]
    elif w.endswith("s") and not w.endswith(("ss", "us", "is")):
        w = w[:-1]
    for suf in ("ing", "ed"):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            w = w[: -len(suf)]
            break
    if w.endswith("e") and len(w) > 3:
        w = w[:-1]
    if len(w) > 3 and w[-1] == w[-2] and w[-1] not in "lsz":
        w = w[:-1]
    return w


# ------------------------------------------------------------------ Marathi
# Frequent inflected forms -> lemma (dictionary lemmatiser; checked BEFORE suffix stripping)
_MR_LEMMA = {
    "उचलला": "उचल", "उचलले": "उचल", "उचलली": "उचल", "उचलण्यात": "उचल", "उचलत": "उचल",
    "फुटली": "फुट", "फुटले": "फुट", "फुटला": "फुट", "फुटलेली": "फुट", "फुटलेल्या": "फुट",
    "वाहत": "वाह", "वाहते": "वाह", "वाहतो": "वाह", "वाहतात": "वाह",
    "तुंबला": "तुंब", "तुंबले": "तुंब", "तुंबली": "तुंब", "तुंबलेले": "तुंब", "तुंबलेला": "तुंब",
    "साचले": "साच", "साचली": "साच", "साचला": "साच", "साचत": "साच", "साचते": "साच", "साचतो": "साच",
    "उखडला": "उखड", "उखडली": "उखड", "उखडले": "उखड", "पडले": "पड", "पडला": "पड", "पडली": "पड", "पडून": "पड",
    "जाते": "जा", "जातो": "जा", "जाणे": "जा", "गेला": "जा", "गेले": "जा", "गेली": "जा",
    "येते": "ये", "येतो": "ये", "येत": "ये", "आलेला": "ये", "आलेली": "ये", "आले": "ये", "आली": "ये", "आला": "ये",
    "बंद": "बंद", "चालू": "चालू", "राहतात": "राह", "राहते": "राह", "करतात": "कर", "करत": "कर", "केले": "कर",
    "पसरत": "पसर", "पसरले": "पसर", "वाढली": "वाढ", "वाढले": "वाढ", "वाढला": "वाढ",
    "पोहोचली": "पोहोच", "अडकली": "अडक", "अडकले": "अडक", "अडकला": "अडक", "शिरले": "शिर", "शिरली": "शिर",
    "उडत": "उड", "जाळला": "जाळ", "जाळली": "जाळ", "जाळले": "जाळ", "जाळत": "जाळ", "जाळला": "जाळ",
    "झालेली": "हो", "झालेले": "हो", "झालेला": "हो", "झाली": "हो", "झाले": "हो", "झाला": "हो", "होत": "हो",
    "वाकला": "वाक", "वाकली": "वाक", "वाया": "वाया", "तुटून": "तुट", "पडलेला": "पड", "नकार": "नकार",
    "सोडले": "सोड", "सोडली": "सोड", "खराब": "खराब", "दिसतात": "दिस", "आढळले": "आढळ", "धावतात": "धाव",
    "मुले": "मुल", "मुलांना": "मुल", "मुलांनी": "मुल", "मुलांचे": "मुल", "मुलांच्या": "मुल", "मुलं": "मुल",
    "रस्ता": "रस्त", "रस्त्यावर": "रस्त", "रस्त्यावरील": "रस्त", "रस्त्याचे": "रस्त", "रस्त्यात": "रस्त",
    "रस्त्याजवळ": "रस्त", "रस्त्यांवर": "रस्त", "रस्त्याला": "रस्त", "रस्त्याची": "रस्त", "रस्त्याचा": "रस्त",
    "चालवतो": "चालव", "येणार": "ये", "मिळत": "मिळ", "मिळते": "मिळ", "मिळाले": "मिळ", "उपलब्ध": "उपलब्ध",
}
# case markers / postpositions / verb endings, longest match first, applied once (iteratively for stacked suffixes)
_MR_SUFFIXES = sorted("""
मध्येच मध्ये ामध्ये ांमध्ये च्यामुळे ामुळे मुळे ांच्यामुळे ापासून पासून ांपासून ांच्या ाच्या च्या चा ची चे
ांना ांनी ांवर ांत ांचे ांची ांचा ाचे ाची ाचा ाला ाने ावर ाकडे ाकडून कडून कडे ासाठी साठी
ात ेत ीत ूत ीच्या ीचे ीची ीचा ीने ीला ीवर ीपासून ेला ेली ेले ेने ेचे ेची ेचा
लेला लेली लेले लेल्या ल्या ला ली ले णारा णारी णारे ताना ल्यावर ावर वर ना नी ने त ही च
येथील येथे इथे ातील तील चं च्या
""".split(), key=len, reverse=True)
_MR_MIN_STEM = 3


def stem_mr(w: str) -> str:
    """Longest-suffix-match stemmer for Marathi (rule based, in the spirit of Lightweight Marathi stemmers)."""
    if w in _MR_LEMMA:
        return _MR_LEMMA[w]
    if len(w) <= 3 or not _DEV.search(w):
        return w
    for _ in range(2):                       # allow two stacked suffixes (e.g. ...ांच्या + त)
        for suf in _MR_SUFFIXES:
            if w.endswith(suf) and len(w) - len(suf) >= _MR_MIN_STEM:
                w = w[: -len(suf)]
                break
        else:
            break
        if w in _MR_LEMMA:
            return _MR_LEMMA[w]
    if w.endswith("्य") and len(w) - 2 >= _MR_MIN_STEM:      # oblique form of aa-ending nouns: कुर्ल्य <- कुर्ला
        w = w[:-2]
    stripped = w.rstrip("ािीुूेैोौ")          # trailing matras -> uniform stem (पाणी/पाण्या share 'पाण')
    return stripped if len(stripped) >= _MR_MIN_STEM else w


# ------------------------------------------------------------------ concept tokens & location
def concept_tokens(norm_text: str) -> list:
    out = []
    for name, (en, mr) in CONCEPTS.items():
        if any(re.search(r"\b" + re.escape(p), norm_text) for p in en) or any(p in norm_text for p in mr):
            out.append("cpt_" + name.lower())
    return out


def extract_location(norm_text: str):
    """Gazetteer-based Named-Entity-Recognition for Mumbai localities (English + Marathi)."""
    best = None
    for canon, variants in LOCATIONS.items():
        for v in [canon.lower()] + variants:
            i = norm_text.find(v)
            if i >= 0 and (best is None or len(v) > best[1]):
                best = (canon, len(v))
    return best[0] if best else None


# ------------------------------------------------------------------ public API
@dataclass
class Processed:
    raw: str
    lang: str
    normalized: str
    tokens: list                      # stemmed content tokens
    concepts: list = field(default_factory=list)
    location: str = None

    @property
    def text(self):                   # string fed to TF-IDF (stemmed tokens)
        return " ".join(self.tokens)

    @property
    def semantic_text(self):          # tokens + concept tokens + location tag (used for duplicate detection)
        loc = ["loc_" + self.location.lower()] if self.location else []
        return " ".join(self.tokens + self.concepts * 2 + loc * 2)


def preprocess(text: str, lang: str = None) -> Processed:
    lang = lang or detect_language(text)
    norm = normalize(text, lang)
    toks = _TOKEN.findall(norm)
    if lang == "mr":
        # Marathi complaints often contain English loan-words (BMC, pothole, bus) -> stem those the English way
        out = []
        for t in toks:
            if _DEV.search(t):
                if t in MR_STOPWORDS:
                    continue
                out.append(stem_mr(t))
            elif t not in EN_STOPWORDS:
                out.append(stem_en(t))
        toks = [t for t in out if t]
    else:
        toks = [stem_en(t) for t in toks if t not in EN_STOPWORDS]
    return Processed(text, lang, norm, toks, concept_tokens(norm), extract_location(norm))
