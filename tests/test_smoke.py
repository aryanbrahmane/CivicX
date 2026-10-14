"""Run:  python -m tests.test_smoke   (after python train.py)"""
import os, tempfile
from civicx.preprocess import preprocess, detect_language, stem_mr, stem_en
from civicx.signals import extract_duration_days


def test_language_and_stemming():
    assert detect_language("पाणी नाही") == "mr" and detect_language("no water") == "en"
    assert stem_mr("रस्त्यावर") == stem_mr("रस्ता") == "रस्त"
    assert stem_mr("दिवसांपासून") == "दिवस"
    assert stem_en("potholes") == stem_en("pothole")
    p = preprocess("अंधेरीमध्ये ५ दिवसांपासून पाणी नाही")
    assert p.lang == "mr" and p.location == "Andheri" and "5" in p.tokens and "नाह" in p.tokens


def test_duration():
    assert extract_duration_days("no water for 3 days", "en") == 3
    assert extract_duration_days("गेल्या 2 आठवड्यांपासून", "mr") == 14


def test_pipeline():
    from civicx.pipeline import CivicX
    cx = CivicX(db_path=os.path.join(tempfile.mkdtemp(), "t.db"))
    a = cx.submit("Water supply stopped in Andheri for 4 days, children suffering")
    b = cx.submit("Andheri mein water supply band hai, no water for 4 days")     # romanised mix, still same issue
    c = cx.submit("कुर्ल्यात रस्त्यावर मोठे खड्डे पडले आहेत")
    d = cx.submit("तात्काळ मदत! ICU मध्ये वीज गेली आहे")
    assert a["category"] == "Water Supply" and c["category"] == "Roads" and c["lang"] == "mr"
    assert d["priority"] in ("High", "Critical")
    assert a["ticket_no"].startswith("CX-") and "Ticket ID" in a["response"] and "तक्रार क्रमांक" in c["response"]
    assert not a["is_duplicate"] and b["is_duplicate"]


if __name__ == "__main__":
    for f in (test_language_and_stemming, test_duration, test_pipeline):
        f(); print("PASS", f.__name__)
