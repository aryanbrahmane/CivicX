"""Gazetteer (Mumbai localities), bilingual concept lexicon, stopwords."""

# canonical English name -> Marathi surface variants (stems that survive Marathi inflection,
# e.g. कुर्ला -> कुर्ल्यात, वांद्रे -> वांद्र्यात)
LOCATIONS = {
    "Andheri": ["अंधेरी"], "Bandra": ["वांद्रे", "वांद्र्"], "Dadar": ["दादर"],
    "Kurla": ["कुर्ला", "कुर्ल्"], "Borivali": ["बोरिवली"], "Ghatkopar": ["घाटकोपर"],
    "Mulund": ["मुलुंड"], "Chembur": ["चेंबूर"], "Malad": ["मालाड"], "Goregaon": ["गोरेगाव"],
    "Sion": ["सायन"], "Worli": ["वरळी"], "Powai": ["पवई"], "Colaba": ["कुलाबा", "कुलाब्"],
    "Vikhroli": ["विक्रोळी"], "Dharavi": ["धारावी"], "Juhu": ["जुहू"], "Versova": ["वर्सोवा", "वर्सोव्"],
    "Kandivali": ["कांदिवली"], "Byculla": ["भायखळा", "भायखळ्"], "Wadala": ["वडाळा", "वडाळ्"],
    "Parel": ["परळ"], "Govandi": ["गोवंडी"], "Mankhurd": ["मानखुर्द"], "Bhandup": ["भांडुप"],
}
LOC_MR_MAIN = {k: v[0] for k, v in LOCATIONS.items()}

# ---------------------------------------------------------------- stopwords
EN_STOPWORDS = set("""a an the is are was were be been being am do does did has have had having of at by for with about
against between into through during before after above below to from up down in out on off over under again further then
once here there when where why how all any both each few more most other some such no nor not only own same so than too very
can will just should now i me my we our you your he she it its they them their this that these those and but if or because as
until while please kindly sir madam hello hi dear""".split()) - {"no", "not", "nor"}

MR_STOPWORDS = set("""आहे आहेत आहोत होते होता होती होत असे असून असतो असते असल्याने आणि व किंवा पण परंतु तर की का
हा ही हे या त्या तो ती ते तेथे येथे इथे तिथे मी आम्ही तू तुम्ही आपण माझा माझी माझे आमचा आमची आमचे आपला आपली आपले
त्याचा त्याची त्याचे तिचा तिचे म्हणून म्हणजे साठी पासून पर्यंत मध्ये वर खाली आत बाहेर जवळ कडे मात्र सुद्धा देखील
फक्त खूप अगदी जरा कृपया माननीय साहेब नमस्कार महोदय विनंती करत करा करावी करावे केले केली केल्या गेले गेली
हवे हवी हवा आता काल आज उद्या एक तसेच अशी असा अशा येत जात होत राहिले राहिली गेल्या ने ना ला चा ची चे नी त ही""".split())

# negative/severity cue words (sentiment-like lexicon used as features + for explanation)
EN_SEVERE = ["urgent", "urgently", "emergency", "dangerous", "danger", "fire", "risk", "flooded", "flooding", "accident",
             "burst", "live wire", "unsafe", "suffering", "critical", "immediately", "serious", "life", "death", "sparking",
             "attack", "attacking", "stuck", "worst", "terrible"]
MR_SEVERE = ["तातडी", "आणीबाणी", "धोका", "धोकादायक", "आग", "अपघात", "फुटली", "फुटले", "असुरक्षित", "त्रास", "गंभीर",
             "जीव", "ठिणग्या", "हल्ला", "अडकली", "अडकले", "भयंकर", "प्रचंड", "शिरले", "तुटून"]

EN_IMPACT = {
    "hospital": r"hospital|icu|clinic|dispensary|ambulance|patient",
    "children_elderly": r"child|children|kids|elderly|old people|senior citizen|infant|pregnant",
    "school": r"school|student|college|exam",
    "public_safety": r"accident|fire|live wire|sparking|open manhole|electrocut|attack",
}
MR_IMPACT = {
    "hospital": r"रुग्णालय|दवाखान|आयसीयू|icu|hospital|रुग्णवाहिका|रुग्ण",
    "children_elderly": r"मुले|मुलां|लहान|वृद्ध|ज्येष्ठ|गर्भवती|बालक",
    "school": r"शाळ|विद्यार्थी|महाविद्यालय|परीक्षा",
    "public_safety": r"अपघात|आग|तार तुटून|उघडे|ठिणग्या|हल्ला|विजेचा धक्का",
}
EN_URGENT = r"urgent|urgently|emergency|immediately|asap|at once|right away"
MR_URGENT = r"तातडी|तात्काळ|तत्काळ|आणीबाणी|ताबडतोब|त्वरित|लगेच|urgent|emergency"

# ---------------------------------------------------------------- bilingual concept lexicon
# concept -> (english lemma prefixes, marathi stem prefixes). Used to build language-independent
# "concept tokens" so an English and a Marathi complaint about the same problem look alike (offline
# stand-in for a multilingual sentence encoder).
CONCEPTS = {
    "WATER":    (["water", "tap", "pipeline", "tanker", "supply"], ["पाणी", "नळ", "जलवाहिनी", "टँकर"]),
    "PRESSURE": (["pressure"], ["दाब"]),
    "DIRTY":    (["dirty", "smelly", "contaminat", "muddy"], ["घाणेरडे", "दुर्गंधीयुक्त", "गढूळ"]),
    "BURST":    (["burst", "leak", "wasted"], ["फुट", "गळ", "वाया"]),
    "BILL":     (["bill", "meter", "charge"], ["बिल", "मीटर"]),
    "POWER":    (["power", "electric", "voltage", "transformer", "wire", "outage"], ["वीज", "ट्रान्सफॉर्मर", "व्होल्टेज", "तार"]),
    "CUT":      (["cut", "outage", "off", "stopped", "stop"], ["जाते", "बंद", "खंडित"]),
    "SPARK":    (["spark", "fire"], ["ठिणग्या", "आग"]),
    "ROAD":     (["road", "street", "lane"], ["रस्त", "रस्ता", "गल्ली"]),
    "POTHOLE":  (["pothole", "broken", "damaged", "digging"], ["खड्ड", "उखड", "खराब", "खोदकाम", "खण"]),
    "FOOTPATH": (["footpath", "pavement"], ["पदपथ"]),
    "MANHOLE":  (["manhole"], ["मॅनहोल"]),
    "SPEEDBRK": (["speed breaker"], ["गतिरोधक"]),
    "GARBAGE":  (["garbage", "waste", "trash", "dump", "sweeping"], ["कचर", "झाडलोट"]),
    "TOILET":   (["toilet"], ["शौचालय"]),
    "ANIMAL":   (["animal", "dog", "dogs", "stray"], ["प्राणी", "कुत्र", "भटक"]),
    "SMOKE":    (["smoke", "burning"], ["धूर", "जाळ"]),
    "SEWAGE":   (["sewage", "gutter", "drain", "nala", "chocked", "choked", "blocked"], ["गटार", "नाला", "तुंब"]),
    "LOGGING":  (["logging", "waterlog", "flood", "stuck"], ["साचले", "साचत", "साचल", "शिरले", "अडकली", "अडकले"]),
    "MOSQUITO": (["mosquito", "fogging", "dengue", "malaria"], ["डास", "फवारणी", "डेंग्यू", "मलेरिया"]),
    "STREETLT": (["light", "lamp", "dark", "flicker"], ["पथदिव", "अंधार", "दिवा", "दिवे"]),
    "POLE":     (["pole", "leaning"], ["खांब", "वाकला"]),
    "BUS":      (["bus", "best"], ["बस", "बेस्ट"]),
    "TRAIN":    (["train", "station", "escalator", "lift"], ["लोकल", "स्थानक", "जिना", "गाड्या"]),
    "AUTO":     (["auto", "rickshaw", "taxi"], ["रिक्षा"]),
    "LATE":     (["late", "delay"], ["उशिर"]),
    "RUDE":     (["rude", "rash"], ["उद्धट", "बेदरकार"]),
    "HOSPITAL": (["hospital", "dispensary", "doctor", "medicine", "ambulance"], ["रुग्णालय", "दवाखान", "डॉक्टर", "औषध", "रुग्णवाहिका"]),
    "CROWD":    (["crowd"], ["गर्दी"]),
    "SHELTER":  (["shelter"], ["निवारा"]),
    "MISSING":  (["not available", "unavailable", "no medicines", "refuse"], ["उपलब्ध नाही", "नकार"]),
    "NOTDONE":  (["not done", "not collected", "not cleaned", "not arrived", "not sprayed", "not working"],
                 ["उचलला गेला नाही", "झालेली नाही", "आलेला नाही", "झालेले नाही", "होत नाही", "फवारणी झालेली नाही"]),
}
