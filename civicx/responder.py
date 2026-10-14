"""Module 4 - template-based, bilingual response generation (replies in the language of the complaint)."""
from datetime import datetime, timedelta

from .lexicon import LOC_MR_MAIN
from .config import CATEGORY_MR, DEPARTMENTS, DEPARTMENTS_MR, PRIORITY_MR, SLA_HOURS

_EN = {
    "Critical": "This has been marked CRITICAL and escalated for immediate action. If anyone is in danger, please also call the BMC helpline 1916 or 112.",
    "High": "This has been marked HIGH priority and assigned for action on a priority basis.",
    "Medium": "This has been registered and will be attended to in the regular queue.",
    "Low": "This has been registered and will be attended to in due course.",
}
_MR = {
    "Critical": "ही तक्रार 'अत्यंत तातडीची' म्हणून नोंदवून तत्काळ कार्यवाहीसाठी पाठवली आहे. कोणाच्या जीविताला धोका असल्यास कृपया बीएमसी हेल्पलाइन 1916 किंवा 112 वर संपर्क साधा.",
    "High": "ही तक्रार 'उच्च' प्राधान्याची म्हणून नोंदवून प्राधान्याने कार्यवाहीसाठी पाठवली आहे.",
    "Medium": "ही तक्रार नोंदवली असून नियमित प्रक्रियेनुसार तिची दखल घेतली जाईल.",
    "Low": "ही तक्रार नोंदवली असून योग्य वेळी तिची दखल घेतली जाईल.",
}


def _eta(hours):
    return f"{hours} hours" if hours < 48 else f"{hours // 24} days"


def generate_response(lang, ticket_no, category, priority, location=None, duplicate_of=None, dup_count=0, created=None):
    hours = SLA_HOURS[priority]
    created = created or datetime.now()
    eta_dt = (created + timedelta(hours=hours)).strftime("%d %b %Y, %I:%M %p")
    if lang == "mr":
        loc = f" ({LOC_MR_MAIN.get(location, location)} परिसर)" if location else ""
        eta = f"{hours} तास" if hours < 48 else f"{hours // 24} दिवस"
        msg = (f"नमस्कार, तक्रार नोंदवल्याबद्दल धन्यवाद.\n\n"
               f"तक्रार क्रमांक: {ticket_no}\nवर्ग: {CATEGORY_MR[category]}{loc}\nप्राधान्य: {PRIORITY_MR[priority]}\n"
               f"संबंधित विभाग: {DEPARTMENTS_MR[category]}\n\n{_MR[priority]}\n")
        if duplicate_of:
            msg += (f"\nयाच समस्येबाबत आधीच तक्रार ({duplicate_of}) नोंदवलेली आहे; तुमची तक्रार त्याच्याशी जोडली आहे "
                    f"(एकूण {dup_count} नागरिकांनी हीच समस्या नोंदवली आहे).\n")
        return msg + f"\nअपेक्षित निराकरण: {eta} (सुमारे {eta_dt} पर्यंत)."
    loc = f" ({location})" if location else ""
    msg = (f"Thank you for reporting this issue.\n\n"
           f"Ticket ID: {ticket_no}\nCategory: {category}{loc}\nPriority: {priority}\n"
           f"Forwarded to: {DEPARTMENTS[category]}\n\n{_EN[priority]}\n")
    if duplicate_of:
        msg += (f"\nA complaint about the same issue ({duplicate_of}) is already open; yours has been linked to it "
                f"({dup_count} citizens have now reported this problem).\n")
    return msg + f"\nExpected resolution: within {_eta(hours)} (by about {eta_dt})."
