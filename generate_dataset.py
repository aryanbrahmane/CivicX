"""Synthetic-but-realistic Mumbai civic complaint corpus (English + Marathi + code-mixed).

No public labelled Marathi civic-complaint dataset exists, so we build one:
  * 8 categories x 6 problem types x 2 paraphrases x 2 languages, Mumbai localities (gazetteer)
  * priority label derived from  base severity of the problem + duration + impact + urgency cues
  * realism noise: typos, Devanagari digits, glued suffixes ("दादरमध्ये"), code-mixing, label noise
Run:  python generate_dataset.py
Replace / extend with real data (e.g. MyBMC 'CRM' exports, Mumbai open-data 311 style logs, Kaggle tickets)
by keeping the same CSV columns.
"""
import random
import pandas as pd
from civicx.config import DATASET_PATH
from civicx.lexicon import LOC_MR_MAIN

random.seed(42)

# (base_severity 0-3, [english paraphrases], [marathi paraphrases])
T = {
"Water Supply": [
 (1, ["There is no water supply in {loc} since morning.", "No water is coming in the taps at {loc}, the supply has stopped."],
     ["{loc} मध्ये सकाळपासून पाणीपुरवठा बंद आहे.", "{loc} येथे नळाला पाणी येत नाही, पाणीपुरवठा बंद झाला आहे."]),
 (0, ["Water pressure is very low in our building in {loc}.", "Very weak water pressure in the taps at {loc}."],
     ["{loc} येथे आमच्या इमारतीत पाण्याचा दाब खूप कमी आहे.", "{loc} मध्ये नळाला पाण्याचा दाब अतिशय कमी आहे."]),
 (2, ["Dirty and smelly water is coming from the taps in {loc}.", "Muddy contaminated water is supplied in {loc}, it is not fit to drink."],
     ["{loc} मध्ये नळातून घाणेरडे आणि दुर्गंधीयुक्त पाणी येत आहे.", "{loc} मध्ये गढूळ आणि दूषित पाणी पुरवले जात आहे, ते पिण्यायोग्य नाही."]),
 (2, ["Main water pipeline burst near {loc} station and water is being wasted on the road.", "A big water pipe is leaking near {loc}, thousands of litres are wasted."],
     ["{loc} जवळ मुख्य जलवाहिनी फुटली असून रस्त्यावर पाणी वाया जात आहे.", "{loc} जवळ पाण्याची मोठी पाईप गळत आहे, हजारो लिटर पाणी वाया जात आहे."]),
 (1, ["The water tanker has not arrived in {loc}.", "The scheduled water tanker did not come to our society in {loc}."],
     ["{loc} मध्ये पाण्याचा टँकर आलेला नाही.", "{loc} मधील आमच्या सोसायटीत ठरलेला पाण्याचा टँकर आला नाही."]),
 (0, ["The water bill received for my flat in {loc} is wrong.", "Incorrect water meter reading and excess water charges in {loc}."],
     ["{loc} मधील माझ्या फ्लॅटचे पाणी बिल चुकीचे आले आहे.", "{loc} मध्ये पाण्याचे मीटर रीडिंग चुकीचे असून जास्त पाणी शुल्क लावले आहे."]),
],
"Electricity": [
 (1, ["Frequent power cuts in {loc} every evening.", "Electricity keeps going off again and again in {loc}."],
     ["{loc} मध्ये दररोज संध्याकाळी वारंवार वीज जाते.", "{loc} मध्ये वीज सतत जात आहे, पुरवठा खंडित होतो."]),
 (0, ["My electricity bill for {loc} is incorrect and far too high.", "Wrong electricity meter reading, excess bill charged in {loc}."],
     ["{loc} येथील माझे वीज बिल चुकीचे आणि खूप जास्त आले आहे.", "{loc} मध्ये वीज मीटर रीडिंग चुकीचे असून जास्त बिल लावले आहे."]),
 (3, ["Transformer is sparking near {loc} market, there is fire risk.", "Sparks and smoke from the electric transformer in {loc}, danger of fire."],
     ["{loc} बाजाराजवळ ट्रान्सफॉर्मरमधून ठिणग्या उडत आहेत, आगीचा धोका आहे.", "{loc} मध्ये विजेच्या ट्रान्सफॉर्मरमधून ठिणग्या आणि धूर येत आहे, आग लागण्याचा धोका आहे."]),
 (2, ["Power outage in the entire {loc} area since last night.", "There is no electricity in the whole {loc} locality from last night."],
     ["काल रात्रीपासून संपूर्ण {loc} परिसरात वीज नाही.", "{loc} परिसरात काल रात्रीपासून वीजपुरवठा पूर्णपणे बंद आहे."]),
 (3, ["A live electric wire has fallen on the road in {loc}.", "Electric cable snapped and is lying on the street at {loc}, live wire."],
     ["{loc} येथे रस्त्यावर विजेची तार तुटून पडली आहे.", "{loc} मध्ये वीजवाहक तार तुटून रस्त्यावर पडली आहे, विजेचा धक्का लागण्याची शक्यता आहे."]),
 (1, ["Voltage fluctuation is damaging our appliances in {loc}.", "Very low voltage at night in {loc}, fans and fridge are not working properly."],
     ["{loc} मध्ये व्होल्टेजमध्ये चढ-उतार होत असल्याने उपकरणे खराब होत आहेत.", "{loc} मध्ये रात्री व्होल्टेज खूप कमी असते, पंखे आणि फ्रीज नीट चालत नाहीत."]),
],
"Roads": [
 (1, ["Big potholes on the road near {loc} are causing traffic jams.", "The road at {loc} is full of deep potholes."],
     ["{loc} जवळील रस्त्यावर मोठे खड्डे पडले असून वाहतूक कोंडी होत आहे.", "{loc} येथील रस्ता खड्ड्यांनी भरला आहे."]),
 (1, ["The road near {loc} is broken and needs repair.", "Badly damaged road at {loc}, repair work is required."],
     ["{loc} जवळील रस्ता उखडला असून दुरुस्तीची गरज आहे.", "{loc} येथील रस्ता खूप खराब झाला आहे, दुरुस्ती करावी."]),
 (3, ["An open manhole without a cover on the road in {loc}.", "Manhole cover is missing on the main road at {loc}, people can fall in."],
     ["{loc} मध्ये रस्त्यावर मॅनहोलचे झाकण उघडे आहे.", "{loc} मधील मुख्य रस्त्यावर मॅनहोलचे झाकण गायब आहे, लोक आत पडू शकतात."]),
 (1, ["The footpath in {loc} is damaged and unsafe for walking.", "Pavement at {loc} is broken and full of debris."],
     ["{loc} मधील पदपथ खराब झाला असून चालण्यासाठी असुरक्षित आहे.", "{loc} येथील पदपथ तुटला असून तिथे राडारोडा पडला आहे."]),
 (1, ["Road digging work has been left incomplete in {loc}.", "Trenches dug on the road at {loc} are left open after the work."],
     ["{loc} मध्ये रस्त्याचे खोदकाम अर्धवट सोडले आहे.", "{loc} येथे रस्ता खणून ठेवला असून काम पूर्ण झालेले नाही."]),
 (0, ["Please install a speed breaker on the road near {loc} junction.", "Request to build a speed breaker at {loc}."],
     ["{loc} चौकाजवळ गतिरोधक बसवावा ही विनंती.", "{loc} येथे गतिरोधक बांधण्याची विनंती आहे."]),
],
"Sanitation": [
 (1, ["Garbage has not been collected in {loc}.", "The garbage van does not come to our lane in {loc}."],
     ["{loc} मध्ये कचरा उचलला गेला नाही.", "{loc} मधील आमच्या गल्लीत कचरा गाडी येत नाही."]),
 (2, ["Garbage dump near {loc} market is overflowing and stinking.", "Overflowing garbage bins at {loc} with a terrible smell."],
     ["{loc} बाजाराजवळ कचराकुंडी ओसंडून वाहत असून प्रचंड दुर्गंधी आहे.", "{loc} मध्ये कचऱ्याचे डबे भरून वाहत आहेत, भयंकर दुर्गंधी पसरली आहे."]),
 (1, ["The public toilet in {loc} is dirty and never cleaned.", "Public toilets at {loc} are unclean and have no water."],
     ["{loc} मधील सार्वजनिक शौचालय अस्वच्छ असून साफसफाई होत नाही.", "{loc} येथील सार्वजनिक शौचालयात घाण आणि पाण्याचा अभाव आहे."]),
 (2, ["A dead animal is lying on the road in {loc} and smells.", "Carcass of a dead dog lying near {loc} station."],
     ["{loc} मध्ये रस्त्यावर मृत प्राणी पडला असून दुर्गंधी येत आहे.", "{loc} स्थानकाजवळ मृत कुत्रा पडलेला आहे."]),
 (2, ["Garbage is being burnt in the open in {loc} and smoke is spreading.", "People burn waste openly in {loc}, the smoke makes it hard to breathe."],
     ["{loc} मध्ये उघड्यावर कचरा जाळला जात असून धूर पसरत आहे.", "{loc} मध्ये उघड्यावर कचरा जाळला जातो, धुरामुळे श्वास घेणे कठीण झाले आहे."]),
 (0, ["Road sweeping is not done regularly in {loc}.", "The lanes in {loc} are not swept and cleaned properly."],
     ["{loc} मध्ये रस्त्यांची नियमित झाडलोट होत नाही.", "{loc} मधील गल्ल्यांची नीट साफसफाई केली जात नाही."]),
],
"Drainage & Flooding": [
 (2, ["Sewage water is overflowing on the road in {loc}.", "Gutter water is flowing on the street at {loc}."],
     ["{loc} मध्ये रस्त्यावर गटाराचे पाणी वाहत आहे.", "{loc} येथे गटारातील पाणी रस्त्यावर वाहत आहे."]),
 (2, ["The drain is blocked in {loc} and water gets logged after every rain.", "Choked nala near {loc}, rain water is not draining."],
     ["{loc} मध्ये नाला तुंबला असून प्रत्येक पावसानंतर पाणी साचत आहे.", "{loc} जवळील नाला तुंबला आहे, पावसाचे पाणी वाहून जात नाही."]),
 (3, ["Water logging in the {loc} subway, vehicles are stuck.", "Heavy waterlogging on the road at {loc}, cars and buses are stuck."],
     ["{loc} सबवेमध्ये पाणी साचले असून वाहने अडकली आहेत.", "{loc} येथे रस्त्यावर प्रचंड पाणी साचले आहे, गाड्या आणि बस अडकल्या आहेत."]),
 (1, ["A choked gutter near {loc} is causing a foul smell.", "The open gutter in {loc} is uncovered and smelly."],
     ["{loc} जवळील गटार तुंबले असून दुर्गंधी पसरली आहे.", "{loc} मधील गटार उघडे असून दुर्गंधी येत आहे."]),
 (1, ["Nala cleaning before the monsoon has not been done in {loc}.", "Pre-monsoon drain desilting is pending in {loc}."],
     ["{loc} मध्ये पावसाळ्यापूर्वी नालेसफाई झालेली नाही.", "{loc} येथे पावसाळ्यापूर्वीची नाल्यातील गाळ काढण्याची कामे प्रलंबित आहेत."]),
 (3, ["Sewage water has entered the houses in {loc}.", "Dirty drain water is entering our homes in {loc}."],
     ["{loc} मध्ये घरांमध्ये गटाराचे पाणी शिरले आहे.", "{loc} मध्ये घाण पाणी आमच्या घरात शिरत आहे."]),
],
"Street Lighting": [
 (1, ["The street light is not working near {loc} circle.", "The street lamp at {loc} chowk has stopped working."],
     ["{loc} चौकाजवळील पथदिवा बंद आहे.", "{loc} चौकातील पथदिवा बंद पडला आहे."]),
 (1, ["All street lights in the lane at {loc} are off and it is dark at night.", "The whole road in {loc} is in darkness at night as the lights are not working."],
     ["{loc} मधील गल्लीतील सर्व पथदिवे बंद असून रात्री अंधार असतो.", "{loc} मधील संपूर्ण रस्त्यावर पथदिवे बंद असल्याने रात्री अंधार असतो."]),
 (2, ["A street light pole is leaning dangerously in {loc}.", "The rusted street light pole at {loc} may fall any time."],
     ["{loc} मध्ये पथदिव्याचा खांब धोकादायकरित्या वाकला आहे.", "{loc} येथील गंजलेला पथदिव्याचा खांब कधीही पडू शकतो."]),
 (0, ["Street lights stay switched on during the day in {loc}.", "Lights on the road at {loc} burn all day, wasting energy."],
     ["{loc} मध्ये दिवसाही पथदिवे चालू राहतात.", "{loc} येथील रस्त्यावरील दिवे दिवसभर चालू असतात."]),
 (0, ["The street light near {loc} bus stop keeps flickering.", "There is a flickering lamp on the road at {loc}."],
     ["{loc} बस थांब्याजवळील पथदिवा सतत लुकलुकतो.", "{loc} येथील रस्त्यावरील दिवा सतत लुकलुकतो."]),
 (1, ["No lights in the public park at {loc}, it is unsafe after sunset.", "The park lights at {loc} are broken and the place is dark."],
     ["{loc} मधील उद्यानात दिवे नाहीत, सूर्यास्तानंतर असुरक्षित वाटते.", "{loc} येथील उद्यानातील दिवे तुटले असून परिसरात अंधार आहे."]),
],
"Public Transport": [
 (0, ["The BEST bus on the route near {loc} is always very late.", "Buses on the {loc} route come after a long wait every day."],
     ["{loc} मार्गावरील बेस्ट बस नेहमी खूप उशिरा येते.", "{loc} मार्गावरील बसेस रोज खूप वेळ वाट पाहिल्यानंतर येतात."]),
 (1, ["Local trains are delayed daily at {loc} station.", "Train services at {loc} station run late every day."],
     ["{loc} स्थानकात लोकल गाड्या रोज उशिरा धावतात.", "{loc} स्थानकावर रोज लोकल सेवा उशिरा सुरू असते."]),
 (0, ["Auto rickshaw drivers refuse to run on meter at {loc}.", "Rickshaw drivers at {loc} refuse the meter fare and demand extra money."],
     ["{loc} येथे रिक्षाचालक मीटरने येण्यास नकार देतात.", "{loc} येथे रिक्षाचालक मीटरप्रमाणे भाडे नाकारतात आणि जास्त पैसे मागतात."]),
 (0, ["There is no shelter at the bus stop in {loc} and the buses are overcrowded.", "The bus stop at {loc} has no shed or seating and gets very crowded."],
     ["{loc} येथील बस थांब्यावर निवारा नाही आणि बसमध्ये खूप गर्दी असते.", "{loc} येथील बस थांब्यावर शेड आणि बसण्याची व्यवस्था नाही, खूप गर्दी होते."]),
 (1, ["The BEST bus driver was rude and drove rashly at {loc}.", "Rash driving and rude behaviour by a bus conductor near {loc}."],
     ["{loc} मध्ये बेस्ट बसचालक उद्धट वागला आणि बेदरकारपणे बस चालवली.", "{loc} जवळ बस वाहकाचे वर्तन उद्धट होते आणि बस बेदरकारपणे चालवली जात होती."]),
 (1, ["The escalator at {loc} station is broken and not working.", "The lift at {loc} station is out of order."],
     ["{loc} स्थानकावरील सरकता जिना बंद आहे.", "{loc} स्थानकावरील लिफ्ट बंद पडली आहे."]),
],
"Public Health": [
 (1, ["Mosquito breeding is increasing in {loc} and no fogging has been done.", "No pest control or fogging done in {loc} despite many mosquitoes."],
     ["{loc} मध्ये डासांची पैदास वाढली असून फवारणी झालेली नाही.", "{loc} मध्ये खूप डास असूनही फवारणी केली जात नाही."]),
 (2, ["Dengue cases have been reported in our society in {loc}.", "Several people in {loc} are down with dengue and malaria."],
     ["{loc} मधील आमच्या सोसायटीत डेंग्यूचे रुग्ण आढळले आहेत.", "{loc} मध्ये अनेक जणांना डेंग्यू आणि मलेरिया झाला आहे."]),
 (2, ["The municipal hospital in {loc} has run out of medicines.", "No medicines are available at the BMC hospital in {loc}."],
     ["{loc} मधील महापालिका रुग्णालयात औषधे संपली आहेत.", "{loc} मधील मनपा रुग्णालयात औषधे उपलब्ध नाहीत."]),
 (1, ["The doctor is never available at the dispensary in {loc}.", "The BMC dispensary at {loc} remains closed and no doctor is present."],
     ["{loc} मधील दवाखान्यात डॉक्टर उपलब्ध नसतात.", "{loc} मधील मनपा दवाखाना बंद असतो आणि डॉक्टर उपस्थित नसतात."]),
 (2, ["Stray dogs are attacking people in {loc}.", "A pack of stray dogs is biting pedestrians at {loc}."],
     ["{loc} मध्ये भटके कुत्रे लोकांवर हल्ला करत आहेत.", "{loc} येथे भटक्या कुत्र्यांचा टोळका पादचाऱ्यांना चावतो आहे."]),
 (3, ["The ambulance did not reach on time in {loc}.", "An ambulance was delayed by more than an hour in {loc}."],
     ["{loc} मध्ये रुग्णवाहिका वेळेवर पोहोचली नाही.", "{loc} मध्ये रुग्णवाहिका एका तासापेक्षा जास्त उशिरा आली."]),
],
}

OPEN_EN = ["", "", "", "Sir, ", "Hello BMC, ", "Dear Sir/Madam, ", "Kindly note: ", "Complaint: "]
OPEN_MR = ["", "", "", "माननीय साहेब, ", "नमस्कार, ", "महोदय, ", "विनंती: ", "तक्रार: "]
DUR_EN = ["The problem has continued for {n}.", "It has been like this since the last {n}.", "We are facing this for {n} now."]
DUR_MR = ["ही समस्या गेल्या {n}पासून सुरू आहे.", "गेल्या {n}पासून आम्ही हा त्रास सहन करत आहोत.", "{n} झाले तरी काहीच कार्यवाही नाही."]
# (english, marathi, bonus)
IMPACT = [
 ("Children and elderly people are suffering.", "लहान मुले आणि वृद्ध नागरिकांना खूप त्रास होत आहे.", 1),
 ("Pregnant women and senior citizens are facing great difficulty.", "गर्भवती महिला आणि ज्येष्ठ नागरिकांना मोठी अडचण होत आहे.", 1),
 ("Students are unable to go to school.", "विद्यार्थ्यांना शाळेत जाता येत नाही.", 1),
 ("A nearby hospital is affected badly.", "जवळच्या रुग्णालयावर गंभीर परिणाम होत आहे.", 2),
 ("Patients in the hospital ICU are at risk.", "रुग्णालयातील आयसीयूमधील रुग्ण धोक्यात आहेत.", 2),
 ("There is a big risk of accident.", "अपघाताचा मोठा धोका आहे.", 1),
]
URGENT = [("Please act urgently!", "कृपया तातडीने कार्यवाही करा!"), ("This is an emergency.", "ही आणीबाणीची परिस्थिती आहे."),
          ("Fix it immediately.", "ताबडतोब दुरुस्ती करा.")]
CALM = [("Kindly look into it when possible.", "कृपया योग्य ती दखल घ्यावी."), ("Just informing the department.", "फक्त विभागाच्या माहितीसाठी कळवत आहे.")]
CODE_MIX = {"पाणीपुरवठा": "water supply", "पथदिवा": "street light", "पथदिवे": "street lights", "कचरा": "garbage",
            "खड्डे": "potholes", "बिल": "bill", "रस्त्यावर": "road वर", "बस थांब्या": "bus stop"}
MR_DIGITS = str.maketrans("0123456789", "०१२३४५६७८९")


def typo(s):
    words = s.split()
    for _ in range(random.choice([1, 1, 2])):
        i = random.randrange(len(words))
        w = words[i]
        if len(w) > 4 and w.isascii():
            j = random.randrange(1, len(w) - 2)
            words[i] = w[:j] + w[j + 1] + w[j] + w[j + 2:]
    return " ".join(words)


def build(n_per_variant=22):
    rows, uid = [], 0
    locs = list(LOC_MR_MAIN.items())
    for cat, tpls in T.items():
        for ti, (base, en_vars, mr_vars) in enumerate(tpls):
            for lang, variants in (("en", en_vars), ("mr", mr_vars)):
                for vi, v in enumerate(variants):
                    for _ in range(n_per_variant):
                        loc_en, loc_mr = random.choice(locs)
                        loc = loc_en if lang == "en" else loc_mr
                        s = v.format(loc=loc)
                        if lang == "mr" and random.random() < 0.3:
                            s = s.replace(f"{loc} मध्ये", f"{loc}मध्ये")
                        # ---- duration
                        days = None
                        if random.random() < 0.45:
                            days = random.choice([1, 2, 3, 3, 4, 5, 5, 7, 7, 10, 14, 15, 20, 30])
                            weeks = days % 7 == 0 and random.random() < 0.5
                            if lang == "en":
                                unit = f"{days // 7} weeks" if weeks and days // 7 > 1 else (f"{days} days" if not weeks else "1 week")
                                s += " " + random.choice(DUR_EN).format(n=unit)
                            else:
                                num = str(days // 7 if weeks else days)
                                if random.random() < .5: num = num.translate(MR_DIGITS)
                                unit = f"{num} आठवड्यां" if weeks else f"{num} दिवसां"
                                s += " " + random.choice(DUR_MR).format(n=unit)
                            if weeks and days // 7 == 1 and lang == "mr":
                                pass
                        bonus = 0
                        if days:
                            bonus += (days >= 7) + (days >= 20)
                            if base == 0:
                                bonus = min(bonus, 1)
                        # ---- impact / urgency cues
                        if random.random() < 0.28:
                            e, m, b = random.choice(IMPACT)
                            s += " " + (e if lang == "en" else m)
                            bonus += b
                        r = random.random()
                        if r < 0.12:
                            e, m = random.choice(URGENT)
                            s += " " + (e if lang == "en" else m)
                            bonus += 1
                        elif r < 0.30:
                            e, m = random.choice(CALM)
                            s += " " + (e if lang == "en" else m)
                        s = (random.choice(OPEN_EN) if lang == "en" else random.choice(OPEN_MR)) + s
                        style = lang
                        if lang == "en" and random.random() < 0.15:
                            s = typo(s)
                        if lang == "en" and random.random() < 0.08:
                            s = s.lower()
                        if lang == "mr" and random.random() < 0.15:
                            for k, val in CODE_MIX.items():
                                if k in s:
                                    s = s.replace(k, val, 1); style = "mr-en"; break
                        pr = min(3, base + bonus)
                        if random.random() < 0.04:                       # label noise
                            pr = max(0, min(3, pr + random.choice([-1, 1])))
                        uid += 1
                        rows.append(dict(id=uid, text=s, lang=lang, style=style, category=cat,
                                         priority=["Low", "Medium", "High", "Critical"][pr], location=loc_en,
                                         issue_id=f"{cat}|{ti}|{loc_en}", tpl=ti, variant=vi, duration_days=days or 0))
    df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
    return df


if __name__ == "__main__":
    df = build()
    DATASET_PATH.parent.mkdir(exist_ok=True)
    df.to_csv(DATASET_PATH, index=False, encoding="utf-8-sig")
    print(f"saved {len(df)} rows -> {DATASET_PATH}")
    print(df.groupby(["lang", "priority"]).size().unstack(fill_value=0))
    print(df.category.value_counts())
    print(df.sample(6, random_state=1)[["text", "category", "priority"]].to_string())
