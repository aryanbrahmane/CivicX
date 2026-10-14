from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
DB_PATH = ROOT / "civicx.db"
DATASET_PATH = DATA_DIR / "mumbai_complaints.csv"

CATEGORIES = [
    "Water Supply", "Electricity", "Roads", "Sanitation",
    "Drainage & Flooding", "Street Lighting", "Public Transport", "Public Health",
]
# Marathi display names (used by the dashboard / auto-replies)
CATEGORY_MR = {
    "Water Supply": "पाणीपुरवठा", "Electricity": "वीजपुरवठा", "Roads": "रस्ते",
    "Sanitation": "स्वच्छता", "Drainage & Flooding": "गटार व पूरस्थिती",
    "Street Lighting": "पथदिवे", "Public Transport": "सार्वजनिक वाहतूक",
    "Public Health": "सार्वजनिक आरोग्य",
}
DEPARTMENTS = {
    "Water Supply": "BMC Hydraulic Engineer Dept.",
    "Electricity": "BEST / Utility Power Distribution",
    "Roads": "BMC Roads & Traffic Dept.",
    "Sanitation": "BMC Solid Waste Management Dept.",
    "Drainage & Flooding": "BMC Storm Water Drains / Sewerage Dept.",
    "Street Lighting": "BMC Street Lighting Dept.",
    "Public Transport": "BEST Undertaking / Transport Authority",
    "Public Health": "BMC Public Health Dept.",
}
DEPARTMENTS_MR = {
    "Water Supply": "बृहन्मुंबई महानगरपालिका - जल अभियंता विभाग",
    "Electricity": "बेस्ट / वीज वितरण विभाग",
    "Roads": "मनपा रस्ते व वाहतूक विभाग",
    "Sanitation": "मनपा घनकचरा व्यवस्थापन विभाग",
    "Drainage & Flooding": "मनपा पर्जन्य जलवाहिनी / मलनिःसारण विभाग",
    "Street Lighting": "मनपा पथदिवे विभाग",
    "Public Transport": "बेस्ट उपक्रम / परिवहन प्राधिकरण",
    "Public Health": "मनपा सार्वजनिक आरोग्य विभाग",
}
PRIORITIES = ["Low", "Medium", "High", "Critical"]
PRIORITY_MR = {"Low": "कमी", "Medium": "मध्यम", "High": "उच्च", "Critical": "अत्यंत तातडीचे"}
# expected resolution time (hours) per priority
SLA_HOURS = {"Low": 96, "Medium": 72, "High": 48, "Critical": 6}
