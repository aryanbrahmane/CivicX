"""CivicX - Streamlit dashboard.     streamlit run app.py"""
import json
import re
from html import escape as _esc

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from civicx import analytics as an
from civicx import db
from civicx.config import (CATEGORIES, DEPARTMENTS, MODEL_DIR, PRIORITIES, SLA_HOURS)
from civicx.lexicon import EN_STOPWORDS, MR_STOPWORDS
from civicx.models import find_duplicate, load_artifacts
from civicx.preprocess import detect_language, normalize, preprocess, stem_en, stem_mr
from civicx.signals import explain, extract_signals

# ------------------------------------------------------------------ constants
NAVY, BLUE, SKY, GREY = "#1d3a8a", "#3b5bdb", "#748ffc", "#98a2b3"
PCOL = {"Critical": "#b42318", "High": "#dc6803", "Medium": "#b98900", "Low": "#027a48"}
PBG = {"Critical": "#fef3f2", "High": "#fffaeb", "Medium": "#fefdf0", "Low": "#ecfdf3"}
LANG_NAME = {"en": "English", "mr": "Marathi"}
PRIORITY_DEF = {
    "Critical": "Immediate danger to life, health or property, or essential services failing at a hospital.",
    "High": "Widespread or prolonged disruption, or a vulnerable group is affected.",
    "Medium": "Service disruption with limited impact; routine action needed.",
    "Low": "Minor inconvenience, billing queries or requests.",
}
EXAMPLES = {
    "Select an example": "",
    "English - water supply (5 days)": "Our street in Andheri hasn't had water supply for the last 5 days. Children and elderly people are suffering.",
    "English - live electric wire": "A live electric wire has fallen on the road in Dadar. Please act urgently!",
    "English - bus service": "The BEST bus on the route near Colaba is always very late.",
    "Marathi - water supply (5 days)": "अंधेरीमध्ये गेल्या ५ दिवसांपासून पाणीपुरवठा बंद आहे. लहान मुले आणि वृद्ध नागरिकांना खूप त्रास होत आहे.",
    "Marathi - potholes": "कुर्ल्यात रस्त्यावर मोठे खड्डे पडले असून वाहतूक कोंडी होत आहे.",
    "Marathi - garbage": "गोवंडीमध्ये कचरा उचलला गेला नाही, प्रचंड दुर्गंधी आहे.",
    "Marathi - hospital power cut": "रुग्णालयातील आयसीयूमध्ये वीज नाही, कृपया तातडीने कार्यवाही करा. वांद्रे",
}
_TOKEN = re.compile(r"[\u0900-\u097F]+|[a-z0-9]+")
_DEV = re.compile(r"[\u0900-\u097F]")
_STRETCH = tuple(int(x) for x in st.__version__.split(".")[:2] if x.isdigit()) >= (1, 50)


# ------------------------------------------------------------------ small helpers
def esc(s):
    """HTML-escape user/model text; also neutralise '$' so Streamlit does not treat it as LaTeX."""
    return _esc(str(s)).replace("$", "&#36;")


def br(s):
    """Escaped text with line breaks preserved (html() strips raw newlines)."""
    return esc(s).replace("\n", "<br>")


def html(s):
    """Render HTML without markdown treating indented lines as code blocks."""
    st.markdown("\n".join(l.strip() for l in s.splitlines() if l.strip()), unsafe_allow_html=True)


def show_chart(fig):
    if _STRETCH:
        st.plotly_chart(fig, width="stretch")
    else:
        st.plotly_chart(fig, use_container_width=True)


def show_df(df, **kw):
    if _STRETCH:
        st.dataframe(df, width="stretch", hide_index=True, **kw)
    else:
        st.dataframe(df, use_container_width=True, hide_index=True, **kw)


def style_fig(fig, h=330, legend=False):
    fig.update_layout(template="simple_white", height=h, margin=dict(t=48, l=10, r=10, b=10), showlegend=legend,
                      title_font=dict(size=15, color="#101828"), font=dict(family="Segoe UI, Arial, sans-serif", size=12, color="#344054"),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    fig.update_xaxes(gridcolor="#eaecf0"); fig.update_yaxes(gridcolor="#eaecf0")
    return fig


def pct(v, d=1):
    return "n/a" if v is None else f"{v * 100:.{d}f}%"


def badge(p):
    return f'<span class="badge" style="background:{PCOL[p]}">{esc(p)}</span>'


def chips(items):
    return "".join(f'<span class="chip">{esc(str(i))}</span>' for i in items) or '<span class="muted">None</span>'


def section(title, sub=None):
    html(f'<div class="sec"><div class="sec-t">{esc(title)}</div>' + (f'<div class="sec-s">{esc(sub)}</div>' if sub else "") + "</div>")


def kpi(label, value, sub=""):
    return (f'<div class="kpi"><div class="kpi-l">{esc(label)}</div><div class="kpi-v">{esc(str(value))}</div>'
            f'<div class="kpi-s">{esc(sub)}</div></div>')


def kpi_row(items):
    cols = st.columns(len(items))
    for c, (l, v, s) in zip(cols, items):
        with c:
            html(kpi(l, v, s))


def bars(rows, color=BLUE):
    """rows: [(label, 0..1, optional colour)] -> horizontal HTML bars"""
    out = ""
    for r in rows:
        lab, v = r[0], r[1]
        col = r[2] if len(r) > 2 else color
        out += (f'<div class="brow"><div class="blab">{esc(lab)}</div><div class="btrack"><div class="bfill" '
                f'style="width:{max(v, 0) * 100:.1f}%;background:{col}"></div></div><div class="bval">{v * 100:.1f}%</div></div>')
    return out


def kv_table(rows):
    return '<table class="kv">' + "".join(f"<tr><th>{esc(k)}</th><td>{v}</td></tr>" for k, v in rows) + "</table>"


def inject_css():
    html("""<style>
    #MainMenu {visibility:hidden;} footer {visibility:hidden;}
    .block-container {padding-top:1.4rem; padding-bottom:3rem; max-width:1280px;}
    html, body, [class*="css"] {font-family:"Segoe UI","Inter","Noto Sans Devanagari","Mangal",Arial,sans-serif;}
    .hero {background:linear-gradient(115deg,#12245e 0%,#1d3a8a 55%,#3454c5 100%); color:#fff; border-radius:14px; padding:26px 32px; margin-bottom:18px;}
    .hero h1 {color:#fff; font-size:30px; margin:0; letter-spacing:.2px; font-weight:700;}
    .hero p {color:#d0d9f5; margin:6px 0 0; font-size:15px;}
    .hero .tags {margin-top:14px;}
    .hero .tag {display:inline-block; border:1px solid rgba(255,255,255,.35); color:#e6ecff; border-radius:99px; padding:3px 12px; font-size:12px; margin-right:8px;}
    .sec {margin:22px 0 10px;} .sec-t {font-size:18px; font-weight:700; color:#101828;} .sec-s {font-size:13px; color:#667085; margin-top:2px;}
    .kpi {background:#fff; border:1px solid #e4e7ec; border-radius:12px; padding:14px 16px; min-height:96px; box-shadow:0 1px 2px rgba(16,24,40,.04);}
    .kpi-l {font-size:12px; color:#667085; text-transform:uppercase; letter-spacing:.6px; font-weight:600;}
    .kpi-v {font-size:28px; font-weight:700; color:#101828; line-height:1.25; margin-top:2px;}
    .kpi-s {font-size:12px; color:#667085; min-height:16px;}
    .badge {display:inline-block; color:#fff; font-weight:600; font-size:12px; padding:3px 11px; border-radius:99px; letter-spacing:.3px;}
    .chip {display:inline-block; background:#eef2ff; color:#1d3a8a; border:1px solid #dbe3ff; border-radius:6px; padding:2px 9px; margin:2px 4px 2px 0; font-size:13px;}
    .muted {color:#98a2b3; font-size:13px;}
    .card {background:#fff; border:1px solid #e4e7ec; border-radius:12px; padding:16px 18px; margin-bottom:12px; box-shadow:0 1px 2px rgba(16,24,40,.04);}
    .card h4 {margin:0 0 10px; font-size:15px; color:#101828;}
    .step {background:#fff; border:1px solid #e4e7ec; border-top:3px solid #3b5bdb; border-radius:10px; padding:14px 16px; min-height:150px;}
    .step b {color:#1d3a8a; font-size:13px; letter-spacing:.4px;} .step div {font-size:13.5px; color:#475467; margin-top:4px;}
    .brow {display:flex; align-items:center; gap:10px; margin:7px 0;} .blab {width:34%; font-size:13px; color:#344054;}
    .btrack {flex:1; background:#eaecf0; height:9px; border-radius:6px; overflow:hidden;} .bfill {height:100%; border-radius:6px;}
    .bval {width:56px; text-align:right; font-size:12.5px; color:#475467; font-variant-numeric:tabular-nums;}
    table.kv {width:100%; border-collapse:collapse; font-size:13.5px;} table.kv th {text-align:left; color:#667085; font-weight:600; width:46%; padding:7px 4px; border-bottom:1px solid #f2f4f7;}
    table.kv td {padding:7px 4px; border-bottom:1px solid #f2f4f7; color:#101828;}
    .reply {background:#f8f9fc; border:1px solid #e4e7ec; border-left:4px solid #3b5bdb; border-radius:8px; padding:14px 18px; font-size:14.5px; line-height:1.6; color:#1d2939;}
    .note {background:#f8f9fc; border:1px solid #e4e7ec; border-radius:10px; padding:12px 16px; font-size:13.5px; color:#475467;}
    .alert {border-radius:10px; padding:12px 16px; font-size:14px; margin:8px 0; border:1px solid;}
    .a-warn {background:#fffaeb; border-color:#fedf89; color:#93370d;} .a-ok {background:#ecfdf3; border-color:#abefc6; color:#067647;}
    .a-bad {background:#fef3f2; border-color:#fecdca; color:#b42318;}
    .scorebar {height:14px; background:#eaecf0; border-radius:8px; overflow:hidden; margin:6px 0 2px;} .scorebar div {height:100%; border-radius:8px;}
    .stTabs [data-baseweb="tab-list"] {gap:4px; border-bottom:1px solid #e4e7ec;}
    .stTabs [data-baseweb="tab"] {padding:10px 16px; font-weight:600; color:#667085;}
    .stTabs [aria-selected="true"] {color:#1d3a8a;}
    [data-testid="stSidebar"] {background:#fff; border-right:1px solid #e4e7ec;}
    .sb-brand {font-size:22px; font-weight:800; color:#1d3a8a; letter-spacing:.3px;} .sb-sub {font-size:12.5px; color:#667085; margin-bottom:10px;}
    </style>""")


# ------------------------------------------------------------------ cached resources
@st.cache_resource(show_spinner="Loading language models...")
def get_app():
    from civicx.pipeline import CivicX
    art = load_artifacts()
    return CivicX(), art.get("meta", {})


def alert(kind, text):
    html(f'<div class="alert a-{kind}">{text}</div>')


# ------------------------------------------------------------------ result renderer
def render_result(r, saved, cx):
    p = r["processed"]
    sig = r["signals"]
    pri, score = r["priority"], r["urgency_score"]
    section("Analysis result", "Generated by the CivicX pipeline" + (" and saved as a ticket" if saved else " (preview - nothing has been saved)"))
    top = st.columns(4)
    with top[0]: html(kpi("Language", "Marathi" if r["lang"] == "mr" else "English", "detected from script" ))
    with top[1]: html(kpi("Category", r["category"], f"{r['cat_conf'] * 100:.0f}% confidence"))
    with top[2]:
        html(f'<div class="kpi"><div class="kpi-l">Priority</div><div class="kpi-v" style="color:{PCOL[pri]}">{esc(pri)}</div>'
             f'<div class="kpi-s">urgency score {score * 100:.0f}%</div></div>')
    with top[3]: html(kpi("Location", r["location"] or "Not detected", "gazetteer match"))

    left, right = st.columns(2)
    with left:
        reasons = "".join(f"<li>{esc(x)}</li>" for x in explain(sig, "en")) or "<li>No special urgency cues were found in the text.</li>"
        html(f'<div class="card"><h4>Urgency assessment</h4><div class="muted">Urgency score</div>'
             f'<div class="scorebar"><div style="width:{score * 100:.0f}%;background:{PCOL[pri]}"></div></div>'
             f'<div style="font-size:13px;color:#475467;margin-bottom:10px">{score * 100:.0f}% - {badge(pri)}</div>'
             f'<div class="muted" style="margin-top:8px">Factors that influenced the priority</div><ul style="margin:4px 0 10px 18px;font-size:14px">{reasons}</ul>'
             f'<div class="muted">Class probabilities</div>' + bars([(PRIORITIES[i], float(r["priority_probs"][i]), PCOL[PRIORITIES[i]]) for i in range(4)]) + "</div>")
        yn = lambda v: "Yes" if v else "No"
        dur = f"{sig['duration_days']:g} day(s)" if sig["duration_days"] > 0 else "Not mentioned"
        html('<div class="card"><h4>Extracted signals</h4>' + kv_table([
            ("Duration mentioned", esc(dur)), ("Urgent wording", yn(sig["urgent_kw"])),
            ("Hospital or patients affected", yn(sig["hospital"])), ("Children or elderly affected", yn(sig["children_elderly"])),
            ("Students or school affected", yn(sig["school"])), ("Public-safety hazard", yn(sig["public_safety"])),
            ("Severity words found", esc(", ".join(sig["severe_words"]) or "None")), ("Negation present", yn(sig["negation"]))]) + "</div>")
    with right:
        dept = DEPARTMENTS[r["category"]]
        html('<div class="card"><h4>Category confidence (top 3)</h4>' + bars([(c, float(pr)) for c, pr in r["top3"]]) + "</div>")
        html('<div class="card"><h4>Routing</h4>' + kv_table([
            ("Responsible department", esc(dept)), ("Target resolution time", f"{SLA_HOURS[pri]} hours"),
            ("Complaint language", LANG_NAME[r["lang"]])]) + "</div>")
        if saved:
            if r["is_duplicate"]:
                msg = (f'<b>Duplicate detected.</b> This complaint was linked to the existing issue <b>{esc(r["duplicate_of"])}</b> '
                       f'(similarity {r["dup_sim"]:.2f}). {r["dup_count"]} citizens have now reported this issue.')
                if r.get("parent_escalated_to"):
                    msg += f' The original ticket was escalated to <b>{esc(r["parent_escalated_to"])}</b> priority because the issue is widespread.'
                alert("warn", msg)
            else:
                alert("ok", f'<b>New issue registered.</b> Ticket number <b>{esc(r["ticket_no"])}</b>. No similar open complaint was found.')
        else:
            pool = db.open_pool(cx.con)
            m, sim = find_duplicate(r["embedding"], r["category"], r["location"], pool, cx.thr)
            if m:
                alert("warn", f'<b>Duplicate check (preview).</b> If submitted, this complaint would be linked to <b>{esc(m["ticket_no"])}</b> (similarity {sim:.2f}).')
            else:
                alert("ok", "<b>Duplicate check (preview).</b> No similar open complaint exists; submitting would create a new issue.")

    if saved:
        section("Auto-generated reply", "Written in the same language as the complaint")
        html(f'<div class="reply">{br(r["response"])}</div>')
    with st.expander("NLP pipeline details (preprocessing output)"):
        html(kv_table([("Normalised text", esc(p.normalized)), ("Tokens after stop-word removal and stemming", chips(p.tokens)),
                       ("Concept tokens (language independent)", chips(p.concepts)), ("Location (gazetteer NER)", esc(p.location or "None"))]))


# ------------------------------------------------------------------ tabs
def tab_overview(cx, meta, n_tickets):
    section("About CivicX", "Automated complaint triage and clustering for Mumbai civic issues, in English and Marathi")
    st.write("Municipal bodies receive thousands of citizen complaints every day. Most are sorted by hand or with keyword rules, which causes slow "
             "routing, duplicate tickets, no prioritisation and generic replies. CivicX applies Natural Language Processing to read each complaint "
             "(English, Marathi or mixed), decide what it is about and how urgent it is, merge duplicates into a single issue, draft a reply in the "
             "citizen's own language and give managers a live picture of civic problems.")
    section("How a complaint is processed")
    steps = [("1. PREPROCESS", "Language-specific cleaning: normalisation, tokenisation, stop-words, stemming and lemmatisation. Marathi has its own stemmer."),
             ("2. CATEGORISE", "TF-IDF word and character features with Logistic Regression assign one of 8 civic categories."),
             ("3. PRIORITISE", "Duration, impact and urgency cues are extracted and combined with the text model to give Low to Critical."),
             ("4. DEDUPLICATE", "Sentence embeddings and cosine similarity link repeat reports of the same problem, in either language."),
             ("5. RESPOND", "A bilingual reply with ticket number, department and target time is generated and stored.")]
    for c, (t, d) in zip(st.columns(5), steps):
        with c:
            html(f'<div class="step"><b>{t}</b><div>{esc(d)}</div></div>')
    section("Categories and responsible departments")
    show_df(pd.DataFrame({"Category": CATEGORIES, "Responsible department": [DEPARTMENTS[c] for c in CATEGORIES]}))
    section("Priority levels")
    show_df(pd.DataFrame({"Priority": PRIORITIES[::-1], "Meaning": [PRIORITY_DEF[p] for p in PRIORITIES[::-1]],
                          "Target resolution (hours)": [SLA_HOURS[p] for p in PRIORITIES[::-1]]}))
    section("Language handling")
    c1, c2 = st.columns(2)
    with c1:
        html('<div class="card"><h4>English</h4><div style="font-size:14px;color:#475467">Lower-casing, contraction expansion (n\'t to not), regex tokenisation, '
             'stop-word removal that keeps negations, irregular-form lemma table and a rule-based suffix stemmer.</div></div>')
    with c2:
        html('<div class="card"><h4>Marathi</h4><div style="font-size:14px;color:#475467">Unicode normalisation, Devanagari digit conversion, script-aware tokenisation, '
             'Marathi stop-words, a lemma dictionary and a longest-suffix stemmer for case markers such as मध्ये, पासून and च्या. '
             'Mumbai localities are recognised in both scripts.</div></div>')
    section("Data note")
    html('<div class="note">The models are trained on a bilingual corpus of about 4,200 complaints that was generated from templates covering 25 Mumbai localities, '
         'because no public labelled Marathi civic-complaint dataset was available. Performance on real-world text should be validated before operational use. '
         'See the Model Performance tab for results on unseen and hand-written complaints.</div>')


def tab_submit(cx):
    section("Submit a complaint", "Type in English or Marathi. Use Analyse to preview, or Submit to create a ticket.")
    with st.container(border=True):
        c1, c2 = st.columns([2, 1])
        with c1:
            st.selectbox("Load a sample complaint", list(EXAMPLES), key="ex_pick",
                         on_change=lambda: st.session_state.update(complaint_text=EXAMPLES[st.session_state.ex_pick]))
        with c2:
            lang_choice = st.selectbox("Language", ["Auto-detect", "English", "Marathi"], key="lang_pick")
        text = st.text_area("Complaint text", key="complaint_text", height=130, max_chars=1500,
                            placeholder="Describe the problem and its location. Example: No water supply in Andheri for 3 days.")
        b1, b2, _ = st.columns([1, 1, 3])
        analyse = b1.button("Analyse", key="btn_an")
        submit = b2.button("Submit complaint", type="primary", key="btn_sub")
    lang = {"Auto-detect": None, "English": "en", "Marathi": "mr"}[lang_choice]
    if analyse or submit:
        if not text.strip():
            st.session_state.pop("last", None)
            alert("bad", "Please enter a complaint before continuing.")
        else:
            res = cx.submit(text, lang) if submit else cx.analyze(text, lang)
            st.session_state["last"] = (res, bool(submit))
    if "last" in st.session_state:
        render_result(*st.session_state["last"], cx)


def filter_block(df):
    c1, c2, c3 = st.columns([1, 2, 1])
    period = c1.selectbox("Period", ["All time", "Last 7 days", "Last 14 days", "Last 30 days"], key="d_per")
    cats = c2.multiselect("Category", CATEGORIES, key="d_cat")
    langs = c3.multiselect("Language", ["English", "Marathi"], key="d_lang")
    d = df.copy()
    if cats: d = d[d.category.isin(cats)]
    if langs: d = d[d.lang.isin([k for k, v in LANG_NAME.items() if v in langs])]
    base = d
    n = {"Last 7 days": 7, "Last 14 days": 14, "Last 30 days": 30}.get(period)
    if n and not d.empty:
        d = d[d.created_at > d.created_at.max() - pd.Timedelta(days=n)]
    return d, base


def tab_dashboard(cx, df):
    section("Operations dashboard", "Live view of all complaints stored in the system")
    if df.empty:
        alert("warn", "No complaints are stored yet. Submit a complaint or run <code>python seed_demo.py</code> to load demonstration data.")
        return
    d, base = filter_block(df)
    if d.empty:
        alert("warn", "No complaints match the selected filters.")
        return
    k = an.kpis(d)
    avg = "n/a" if pd.isna(k["avg_response_hours"]) else f"{k['avg_response_hours']:.1f} h"
    dup_rate = k["duplicates_merged"] / k["total"] * 100 if k["total"] else 0
    kpi_row([("Total complaints", k["total"], "received"), ("Unique issues", k["unique_issues"], f"{k['duplicates_merged']} duplicates merged"),
             ("Open issues", k["open_issues"], "awaiting resolution"), ("Critical (open)", k["critical_open"], "need immediate action"),
             ("Avg. resolution time", avg, "resolved issues"), ("Duplicate rate", f"{dup_rate:.0f}%", "of all complaints")])
    st.write("")
    c1, c2 = st.columns(2)
    cc = d.category.value_counts().rename_axis("category").reset_index(name="n").sort_values("n")
    f1 = px.bar(cc, x="n", y="category", orientation="h", title="Complaints by category", text="n")
    f1.update_traces(marker_color=BLUE, textposition="outside"); f1.update_xaxes(title=""); f1.update_yaxes(title="")
    with c1: show_chart(style_fig(f1))
    pc = pd.DataFrame({"priority": PRIORITIES[::-1], "n": [k["priority_counts"][p] for p in PRIORITIES[::-1]]})
    f2 = px.bar(pc, x="priority", y="n", title="Open issues by priority", text="n", color="priority", color_discrete_map=PCOL)
    f2.update_traces(textposition="outside"); f2.update_xaxes(title=""); f2.update_yaxes(title="")
    with c2: show_chart(style_fig(f2))

    c3, c4 = st.columns([2, 1])
    daily = d.assign(day=d.created_at.dt.date).groupby(["day", "lang"]).size().reset_index(name="n")
    daily["Language"] = daily.lang.map(LANG_NAME)
    f3 = px.bar(daily, x="day", y="n", color="Language", title="Complaints per day", color_discrete_map={"English": BLUE, "Marathi": "#f79009"})
    f3.update_xaxes(title=""); f3.update_yaxes(title="")
    with c3: show_chart(style_fig(f3, legend=True))
    lg = d.lang.map(LANG_NAME).value_counts().reset_index(); lg.columns = ["Language", "n"]
    f4 = px.pie(lg, names="Language", values="n", hole=.55, title="Language split", color="Language", color_discrete_map={"English": BLUE, "Marathi": "#f79009"})
    f4.update_traces(textinfo="percent+label")
    with c4: show_chart(style_fig(f4))

    c5, c6 = st.columns(2)
    tr = an.trending(base)
    if not tr.empty:
        tr = tr.sort_values("change_pct")
        f5 = go.Figure(go.Bar(x=tr.change_pct, y=tr.category, orientation="h", text=[f"{v:+.0f}%" for v in tr.change_pct], textposition="outside",
                              marker_color=["#b42318" if v > 10 else ("#027a48" if v < -10 else GREY) for v in tr.change_pct]))
        f5.update_layout(title="Trend: last 7 days vs previous 7 days (% change)"); f5.update_xaxes(title="")
        with c5: show_chart(style_fig(f5))
    hs = an.hotspots(d).sort_values("complaints")
    f6 = px.bar(hs, x="complaints", y="location", orientation="h", title="Localities with most complaints", text="complaints")
    f6.update_traces(marker_color="#6941c6", textposition="outside"); f6.update_xaxes(title=""); f6.update_yaxes(title="")
    with c6: show_chart(style_fig(f6))

    section("Resolution performance against target", "Resolved issues only (duplicates excluded)")
    parents = d[d.is_duplicate == 0]
    res = parents[parents.status == "Resolved"].copy()
    if res.empty:
        st.caption("No issues have been resolved yet.")
    else:
        res["hours"] = (res.resolved_at - res.created_at).dt.total_seconds() / 3600
        rows = []
        for p in PRIORITIES[::-1]:
            g = res[res.priority == p]
            if len(g):
                rows.append({"Priority": p, "Resolved issues": len(g), "Avg. hours": round(g.hours.mean(), 1), "Target (hours)": SLA_HOURS[p],
                             "Resolved within target": f"{(g.hours <= SLA_HOURS[p]).mean() * 100:.0f}%"})
        show_df(pd.DataFrame(rows))


def tab_clusters(df):
    section("Issue clusters", "Complaints describing the same problem in the same locality, in English or Marathi, are merged into a single issue")
    if df.empty:
        alert("warn", "No complaints stored yet.")
        return
    n = st.slider("Number of clusters to show", 5, 50, 15, key="cl_n")
    ct = an.cluster_table(df, n)
    allc = df.groupby("cluster_id").size()
    kpi_row([("Issues (clusters)", len(allc), "unique problems"), ("Issues with repeat reports", int((allc > 1).sum()), "two or more complaints"),
             ("Largest cluster", int(allc.max()), "complaints in one issue"), ("Complaints merged", int(df.is_duplicate.sum()), "avoided as separate tickets")])
    st.write("")
    ct["Language"] = ct.languages.map(lambda s: " + ".join(LANG_NAME.get(x, x) for x in s.split("+")))
    view = ct[["ticket_no", "reports", "category", "location", "priority", "Language", "first_reported", "last_reported", "sample_complaint"]]
    show_df(view, column_config={
        "ticket_no": "Ticket", "reports": st.column_config.ProgressColumn("Reports", min_value=0, max_value=int(ct.reports.max()), format="%d"),
        "category": "Category", "location": "Locality", "priority": "Priority",
        "first_reported": st.column_config.DatetimeColumn("First reported", format="DD MMM YYYY, HH:mm"),
        "last_reported": st.column_config.DatetimeColumn("Last reported", format="DD MMM YYYY, HH:mm"),
        "sample_complaint": "Representative complaint"})
    section("Inspect a cluster")
    meta = ct.set_index("cluster_id")
    pick = st.selectbox("Cluster", list(meta.index), key="cl_pick",
                        format_func=lambda i: f"{meta.loc[i, 'ticket_no']}  |  {meta.loc[i, 'category']}  |  {meta.loc[i, 'location'] or 'Unknown'}  |  {meta.loc[i, 'reports']} report(s)")
    m = df[df.cluster_id == pick].sort_values("created_at").copy()
    m["Language"] = m.lang.map(LANG_NAME)
    show_df(m[["ticket_no", "created_at", "Language", "dup_sim", "text"]], column_config={
        "ticket_no": "Ticket", "created_at": st.column_config.DatetimeColumn("Received", format="DD MMM YYYY, HH:mm"),
        "dup_sim": st.column_config.NumberColumn("Similarity to original", format="%.2f"), "text": "Complaint"})


def tab_tickets(cx, df):
    section("Ticket register", "Search, filter, export and resolve tickets")
    if df.empty:
        alert("warn", "No tickets stored yet.")
        return
    if st.session_state.get("flash"):
        alert("ok", esc(st.session_state.pop("flash")))
    c1, c2, c3, c4 = st.columns(4)
    cat = c1.multiselect("Category", CATEGORIES, key="t_cat")
    pri = c2.multiselect("Priority", PRIORITIES[::-1], key="t_pri")
    stt = c3.selectbox("Status", ["All", "Open", "Resolved"], key="t_st")
    lng = c4.selectbox("Language", ["All", "English", "Marathi"], key="t_lang")
    c5, c6 = st.columns([2, 2])
    q = c5.text_input("Search in complaint text or ticket number", key="t_q")
    kind = c6.radio("Show", ["All tickets", "Primary issues only", "Linked duplicates only"], horizontal=True, key="t_kind")
    v = df.copy()
    if cat: v = v[v.category.isin(cat)]
    if pri: v = v[v.priority.isin(pri)]
    if stt != "All": v = v[v.status == stt]
    if lng != "All": v = v[v.lang == ("en" if lng == "English" else "mr")]
    if kind == "Primary issues only": v = v[v.is_duplicate == 0]
    if kind == "Linked duplicates only": v = v[v.is_duplicate == 1]
    if q.strip():
        v = v[v.text.str.contains(q.strip(), case=False, regex=False) | v.ticket_no.str.contains(q.strip(), case=False, regex=False)]
    v = v.sort_values("created_at", ascending=False)
    st.caption(f"{len(v)} of {len(df)} tickets")
    t = v.assign(Language=v.lang.map(LANG_NAME), Type=v.is_duplicate.map({0: "Primary", 1: "Linked duplicate"}), urgency=(v.urgency_score * 100).round(0))
    show_df(t[["ticket_no", "created_at", "Language", "category", "priority", "urgency", "location", "status", "Type", "text"]], column_config={
        "ticket_no": "Ticket", "created_at": st.column_config.DatetimeColumn("Received", format="DD MMM YYYY, HH:mm"), "category": "Category",
        "priority": "Priority", "urgency": st.column_config.ProgressColumn("Urgency (%)", min_value=0, max_value=100, format="%d"),
        "location": "Locality", "status": "Status", "text": "Complaint"})
    st.download_button("Download filtered tickets (CSV)", v.drop(columns=["response", "id", "parent_id"]).to_csv(index=False).encode("utf-8-sig"),
                       file_name="civicx_tickets.csv", mime="text/csv", key="dl")
    section("Ticket details")
    if v.empty:
        st.caption("No ticket selected.")
        return
    tn = st.selectbox("Ticket", v.ticket_no.head(500).tolist(), key="t_pick")
    row = v[v.ticket_no == tn].iloc[0]
    left, right = st.columns(2)
    with left:
        html('<div class="card"><h4>Ticket summary</h4>' + kv_table([
            ("Ticket number", esc(row.ticket_no)), ("Received", esc(row.created_at.strftime("%d %b %Y, %H:%M"))), ("Language", LANG_NAME[row.lang]),
            ("Category", f"{esc(row.category)} ({row.cat_conf * 100:.0f}% confidence)"), ("Priority", badge(row.priority) + f" &nbsp; urgency {row.urgency_score * 100:.0f}%"),
            ("Locality", esc(row.location or "Not detected")), ("Department", esc(row.department)), ("Status", esc(row.status)),
            ("Linked to", "Original issue" if row.is_duplicate == 0 else f"Existing issue (similarity {row.dup_sim:.2f})")]) + "</div>")
    with right:
        html(f'<div class="card"><h4>Complaint</h4><div style="font-size:14.5px;line-height:1.6">{esc(row.text)}</div>'
             f'<div class="muted" style="margin-top:10px">Priority factors</div><div style="font-size:13.5px">{esc(row.reasons or "None")}</div></div>')
        html(f'<div class="card"><h4>Reply sent to citizen</h4><div class="reply" style="font-size:13.5px">{br(row.response or "")}</div></div>')
    if row.status != "Resolved" and st.button("Mark as resolved", key="resolve"):
        db.update(cx.con, int(row.id), status="Resolved", resolved_at=pd.Timestamp.now().isoformat(timespec="seconds"))
        st.session_state["flash"] = f"Ticket {row.ticket_no} was marked as resolved."
        st.rerun()


def tab_lab():
    section("NLP lab", "See exactly how a complaint is transformed before it reaches the models")
    if "lab_text" not in st.session_state:
        st.session_state["lab_text"] = "कुर्ल्यात रस्त्यावर मोठे खड्डे पडले असून वाहतूक कोंडी होत आहे"
    with st.container(border=True):
        pick = st.selectbox("Load a sample", list(EXAMPLES), key="lab_pick", on_change=lambda: st.session_state.update(lab_text=EXAMPLES[st.session_state.lab_pick] or st.session_state.lab_text))
        text = st.text_area("Text", key="lab_text", height=100, max_chars=1500)
    if not text.strip():
        return
    p = preprocess(text)
    steps = st.columns(3)
    with steps[0]: html('<div class="card"><h4>1. Language identification</h4>' + kv_table([("Detected language", LANG_NAME[p.lang]), ("Method", "Devanagari vs Latin character ratio")]) + "</div>")
    with steps[1]: html('<div class="card"><h4>2. Normalisation</h4><div style="font-size:14px">' + esc(p.normalized) + "</div></div>")
    with steps[2]: html('<div class="card"><h4>3. Entities and concepts</h4>' + kv_table([("Location (NER)", esc(p.location or "None")), ("Concept tokens", chips(p.concepts))]) + "</div>")
    section("Token-level processing", "Each token, whether it is removed as a stop-word, and its stem or lemma")
    rows = []
    for tok in _TOKEN.findall(normalize(text, p.lang)):
        if p.lang == "mr" and _DEV.search(tok):
            stop, stem = tok in MR_STOPWORDS, stem_mr(tok)
        else:
            stop, stem = tok in EN_STOPWORDS, (stem_en(tok) if not tok.isdigit() else tok)
        rows.append({"Token": tok, "Stop-word": "Removed" if stop else "Kept", "Stem / lemma": "-" if stop else stem})
    show_df(pd.DataFrame(rows))
    html('<div class="card"><h4>Final tokens passed to the models</h4>' + chips(p.tokens) + "</div>")
    sig = extract_signals(p.normalized, p.lang)
    section("Extracted urgency signals")
    yn = lambda v: "Yes" if v else "No"
    html('<div class="card">' + kv_table([("Duration (days)", f"{sig['duration_days']:g}" if sig["duration_days"] else "Not mentioned"), ("Urgent wording", yn(sig["urgent_kw"])),
         ("Hospital / patients", yn(sig["hospital"])), ("Children / elderly", yn(sig["children_elderly"])), ("School / students", yn(sig["school"])),
         ("Public-safety hazard", yn(sig["public_safety"])), ("Severity words", esc(", ".join(sig["severe_words"]) or "None")), ("Negation", yn(sig["negation"]))]) + "</div>")


def tab_perf():
    section("Model performance", "Evaluation results saved by train.py")
    mp = MODEL_DIR / "metrics.json"
    if not mp.exists():
        alert("warn", "metrics.json not found. Run <code>python train.py</code> first.")
        return
    M = json.load(open(mp, encoding="utf-8"))
    g = M.get
    dup = g("dup_cosine_+_category_location_gate", {})
    hw = g("urgency_handwritten_model_only", {})
    kpi_row([("Category accuracy", pct(g("category_handwritten_acc")), "hand-written test set"),
             ("Category, unseen wording", pct(g("category_unseen_paraphrase_acc")), "paraphrase test"),
             ("Urgency within one level", pct(hw.get("within_one")), "hand-written test set"),
             ("Duplicate detection F1", f"{dup.get('f1', 0):.2f}", "same category and locality pairs")])
    html('<div class="note" style="margin-top:12px"><b>How to read these numbers.</b> The training corpus is template based, so scores on a random split are '
         'optimistic. The unseen-paraphrase and hand-written results are the realistic estimates of how the system behaves on new wording.</div>')

    section("Module 1: Category classification")
    comp = g("category_model_comparison_macroF1", {})
    c1, c2 = st.columns(2)
    with c1:
        st.caption("Algorithm comparison (macro-F1, random 80/20 split)")
        show_df(pd.DataFrame({"Model": list(comp), "Macro-F1": [f"{v:.3f}" for v in comp.values()]}))
        st.caption("All four models tie on this split, so the Logistic Regression model with word and character features was chosen for its probability outputs and robustness to Marathi word forms.")
    with c2:
        st.caption("Accuracy under different evaluation settings")
        show_df(pd.DataFrame([
            ("Random split, all", g("category_random_split_acc")), ("Random split, English", g("category_random_split_acc_en")), ("Random split, Marathi", g("category_random_split_acc_mr")),
            ("Unseen paraphrase", g("category_unseen_paraphrase_acc")), ("Train English, test Marathi", g("category_en_to_mr_transfer_acc")),
            ("Hand-written, all", g("category_handwritten_acc")), ("Hand-written, English", g("category_handwritten_acc_en")), ("Hand-written, Marathi", g("category_handwritten_acc_mr"))],
            columns=["Evaluation", "Accuracy"]).assign(Accuracy=lambda x: x.Accuracy.map(pct)))
    img = MODEL_DIR / "category_confusion.png"
    if img.exists():
        st.image(str(img), caption="Category confusion matrix (random split)", width=520)

    section("Module 2: Urgency detection")
    show_df(pd.DataFrame([
        ("Keyword-rule baseline", pct(g("urgency_keyword_baseline_acc")), "Exact match"),
        ("CivicX, random split", pct(g("urgency_random_split_acc")), "Exact match"), ("CivicX, random split", pct(g("urgency_within_one_level")), "Within one level"),
        ("CivicX, unseen paraphrase", pct(g("urgency_unseen_paraphrase_acc")), "Exact match"),
        ("CivicX, hand-written set", pct(hw.get("acc")), "Exact match"), ("CivicX, hand-written set", pct(hw.get("within_one")), "Within one level")],
        columns=["System / test", "Accuracy", "Measure"]))
    st.caption(f"Random-split macro-F1: {g('urgency_random_split_macroF1', 0):.3f}. Urgency labels follow rules written for this project, so exact-match accuracy on hand-written complaints is the most conservative figure.")

    section("Module 3: Duplicate detection and clustering")
    d0 = g("dup_cosine_only", {})
    show_df(pd.DataFrame([("Cosine similarity only", d0.get("precision"), d0.get("recall"), d0.get("f1")),
                          ("Cosine + category and locality gate", dup.get("precision"), dup.get("recall"), dup.get("f1"))],
                         columns=["Method", "Precision", "Recall", "F1"]).round(3))
    kpi_row([("Recall, same language", pct(g("dup_same_language_recall")), "duplicate pairs"), ("Recall, English-Marathi", pct(g("dup_cross_lingual_recall")), "cross-language pairs"),
             ("Clustering ARI", f"{g('cluster_ARI', 0):.2f}", "Adjusted Rand Index")])


# ------------------------------------------------------------------ main
def main():
    st.set_page_config(page_title="CivicX | Complaint Triage and Clustering", layout="wide", initial_sidebar_state="expanded")
    inject_css()
    if not (MODEL_DIR / "civicx_models.joblib").exists():
        alert("bad", "Models are not trained yet. Run <code>python generate_dataset.py</code> and then <code>python train.py</code>, then refresh this page.")
        st.stop()
    cx, meta = get_app()
    df = db.tickets_df(cx.con)
    with st.sidebar:
        html('<div class="sb-brand">CivicX</div><div class="sb-sub">Complaint triage and clustering</div>')
        html('<div class="card"><h4>System status</h4>' + kv_table([
            ("Models", "Loaded"), ("Training complaints", f"{meta.get('n_train', 'n/a'):,}" if isinstance(meta.get("n_train"), int) else "n/a"),
            ("Languages", "English, Marathi"), ("Embedding", "Sentence LSA" if cx.emb.backend == "lsa" else "Multilingual SBERT"),
            ("Duplicate threshold", f"{cx.thr:.2f}"), ("Tickets in database", f"{len(df):,}")]) + "</div>")
        html('<div class="card"><h4>Quick guide</h4><div style="font-size:13px;color:#475467">1. Open <b>Submit Complaint</b> and enter text in English or Marathi.<br>'
             '2. Use <b>Analyse</b> to preview or <b>Submit</b> to create a ticket.<br>3. Monitor results in <b>Dashboard</b>, <b>Issue Clusters</b> and <b>Ticket Register</b>.</div></div>')
        st.caption("NLP Mini Project - St. Francis Institute of Technology")
    html('<div class="hero"><h1>CivicX</h1><p>End-to-end NLP framework for automated complaint triage and clustering</p>'
         '<div class="tags"><span class="tag">English and Marathi</span><span class="tag">Mumbai civic services</span>'
         '<span class="tag">Category</span><span class="tag">Urgency</span><span class="tag">Duplicate clustering</span><span class="tag">Auto-reply</span></div></div>')
    tabs = st.tabs(["Overview", "Submit Complaint", "Dashboard", "Issue Clusters", "Ticket Register", "NLP Lab", "Model Performance"])
    with tabs[0]: tab_overview(cx, meta, len(df))
    with tabs[1]: tab_submit(cx)
    with tabs[2]: tab_dashboard(cx, df)
    with tabs[3]: tab_clusters(df)
    with tabs[4]: tab_tickets(cx, df)
    with tabs[5]: tab_lab()
    with tabs[6]: tab_perf()


if __name__ == "__main__":
    main()
