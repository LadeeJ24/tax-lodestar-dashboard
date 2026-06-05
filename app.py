"""
Tax Lodestar — IRS Ruling Analytics Prototype
==============================================
Streamlit dashboard for needle-in-haystack analysis across Subchapter C rulings.

Run locally:
    pip install -r requirements.txt
    streamlit run app.py

Deployed on Streamlit Community Cloud — see README for live URL.
"""

import re
from pathlib import Path

import pandas as pd
import streamlit as st

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Tax Lodestar — §382 Analytics",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_DIR = Path(__file__).parent / "data"
RULINGS_CSV = DATA_DIR / "Rulings.csv"
UILC_CSV = DATA_DIR / "UILC_Codes.csv"
VOCAB_CSV = DATA_DIR / "Transaction_Vocabulary.csv"

# Fields that contain narrative/searchable text (used across the app)
# Metadata fields — paraphrased summaries the human extractor wrote
TEXT_SEARCH_FIELDS = [
    "subject_summary",
    "issues_presented",
    "conclusions",
    "key_authorities_cited",
    "primary_code_sections",
    "uilc_dictionary_mapping",
    "uilc_on_document",
    "transaction_type",
    "timeline_vector_raw",
    "timeline_vector",
    "favorable_logic_pivot",
    "compelled_representations",
]

# Full-text field — raw pdftotext output of the actual IRS document.
# Search this whenever you need precise regulatory phrases or subsection citations
# that the metadata fields may paraphrase away.
FULL_TEXT_FIELD = "full_text"

# -----------------------------------------------------------------------------
# Doctrinal Queries — the "why" canon. Each doctrine maps to:
#   - uilc_codes: IRS's own classification (most reliable)
#   - citations:  regulation / statute strings to look for in key_authorities_cited
#                 or primary_code_sections (catches rulings without UILC tagging)
#   - phrases:    fallback doctrinal language to scan across all text fields
# A ruling matches if ANY of these hit (OR logic across all three signal types).
# -----------------------------------------------------------------------------
DOCTRINAL_QUERIES = {
    "presumption_no_cross_ownership": {
        "label": "Presumption of (j)(2)(iii)(B)(1) — segregation-rule rebuttal",
        "description": (
            "Rulings that rebut the segregation-rule presumption in Treas. Reg. § 1.382-2T(j)(2)(iii)(B)(1) "
            "(that two public groups created on an equity structure shift do not overlap) by establishing "
            "Actual Knowledge under § 1.382-2T(k)(2) — typically via Written Inquiries or SEC-filing review. "
            "Doctrinal hit requires the literal regulatory phrase 'the presumption described in "
            "§ 1.382-2T(j)(2)(iii)(B)(1) will not apply' (or its 'Treas. Reg.' variant) — surface mechanics "
            "alone do not match."
        ),
        # No UILC codes — the broad codes (0382.07-00, 0382.07-05) sweep in general
        # owner-shift/segregation rulings that don't actually rebut this presumption.
        # The doctrine is identified ONLY by the precise subsection citation OR the
        # literal regulatory phrase.
        "uilc_codes": [],
        "citations": [
            "1.382-2T(j)(2)(iii)(B)(1)",
            "§ 1.382-2T(j)(2)(iii)(B)(1)",
        ],
        "phrases": [
            "presumption described in § 1.382-2T(j)(2)(iii)(B)(1) will not apply",
            "presumption described in Treas. Reg. § 1.382-2T(j)(2)(iii)(B)(1) will not apply",
            "presumption described in § 1.382-2T(j)(2)(iii)(B)(1)",
            "presumption described in Treas. Reg. § 1.382-2T(j)(2)(iii)(B)(1)",
            "1.382-2T(j)(2)(iii)(B)(1) will not apply",
        ],
        "year_range": None,
    },
    "section_382_l_5": {
        "label": "§382(l)(5) bankruptcy exception",
        "description": (
            "Rulings invoking the §382(l)(5) bankruptcy-emergence exception — no §382 limitation "
            "applies if (i) the loss corporation is under court jurisdiction in a title 11 or similar "
            "case, and (ii) qualified creditors and old shareholders own ≥50% of the new stock. "
            "Matching prefers literal '§ 382(l)(5)' or '§ 1.382-9' references in the actual ruling text "
            "(strongest), or the doctrinal phrase 'qualified creditor' (which is the regulatory "
            "term of art for the (l)(5) creditor-continuity test)."
        ),
        # Drop UILC — the broad 0382.07-00 sweeps general owner-shift rulings
        "uilc_codes": [],
        "citations": [
            "§ 382(l)(5)", "382(l)(5)",
            "section 382(l)(5)",
            "§ 1.382-9", "1.382-9",
        ],
        "phrases": [
            "qualified creditor",
            "§ 382(l)(5)",
            "section 382(l)(5)",
            "382(l)(5) applies",
            "382(l)(5) bankruptcy",
            "l)(5) exception",
        ],
        "year_range": None,
    },
    "actual_knowledge_2008_2020": {
        "label": "Actual knowledge under § 1.382-2T(k)(2) (2008–2020)",
        "description": (
            "Rulings during 2008–2020 establishing Actual Knowledge under § 1.382-2T(k)(2) — "
            "the regulatory mechanism for overcoming the segregation/presumption rules by "
            "showing the loss corporation actually knows the relevant ownership facts."
        ),
        "uilc_codes": [],
        "citations": [
            "§ 1.382-2T(k)(2)", "1.382-2T(k)(2)",
            "section 1.382-2T(k)(2)",
        ],
        "phrases": [
            "actual knowledge within the meaning of",
            "actual knowledge of the stock ownership",
            "acceptable method of determining actual knowledge",
            "written inquiries",
            "written questionnaires",
        ],
        "year_range": (2008, 2020),
    },
    "standard_382_representations": {
        "label": "Standard §382 representations (5-rep boilerplate)",
        "description": (
            "Rulings carrying the recurring 5-representation package taxpayers must make to "
            "obtain §382 segregation-rule / actual-knowledge relief: (1) loss corporation as "
            "defined in §382(k)(1); (2) single class of outstanding stock; (3) no other "
            "outstanding interests or obligations that would be treated as stock; (4) no "
            "actual knowledge of other 5%-owners; (5) the relevant transaction qualifies as a "
            "tax-free reorganization under §368(a). Matching is OR across the five rep stems "
            "in the full text — a single hit surfaces the ruling; the §382 rep package "
            "recurs across most segregation-rule and (k)(2)-actual-knowledge rulings, so "
            "expect broad recall (this is the point — it is the controlled-vocabulary index "
            "of which rulings carry which reps)."
        ),
        # No UILC codes and no broad citations: §382(k)(1) and §368(a) appear
        # in nearly every §382 ruling for unrelated reasons (loss-corp definition,
        # generic reorg references). The doctrinal signal is the REP LANGUAGE itself,
        # so match precisely on the 5 phrases below.
        "uilc_codes": [],
        "citations": [],
        "phrases": [
            "loss corporation as defined in section 382(k)(1)",
            "class of outstanding stock",
            "outstanding interests or obligations that would be",
            "no actual knowledge",
            "tax-free reorganization",
        ],
        # Per-rep tagging: each phrase above maps 1:1 to a short label. Order is
        # preserved so the rendered checkmark row reads as a stable 5-column matrix
        # across all rulings. This converts "OR match over 5 phrases" into structured
        # per-rep attributes the user can scan at a glance.
        "rep_labels": [
            "loss corp",          # §382(k)(1) definition
            "single class",       # only one class of outstanding stock
            "no other interests", # no other stock-like obligations
            "no actual knowledge",# of other 5%-owners
            "tax-free reorg",     # §368(a) qualification
        ],
        "year_range": None,
    },
}


# -----------------------------------------------------------------------------
# Doctrinal Neighborhood — the cross-section interaction graph
# -----------------------------------------------------------------------------
# This is the product’s strategic claim: §382 does not live alone. A real
# M&A tax partner working a §382 question is simultaneously running attribution
# under §318, watching for plain-vanilla preferred under §1504(a)(4), tracking
# carryover under §381, paralleling the §383/§384 limitations, and worrying
# about §165(g) worthless-stock and §163(j) interest carryforward interactions.
# The neighborhood map below is the controlled-vocabulary picture of that.
#
# Schema:
#   NEIGHBORHOOD["sections"][section_id] = {
#       label, statute_label, role ("hub"|"satellite"), one_liner
#   }
#   NEIGHBORHOOD["interactions"] = list of dicts with from_, to, label, explanation
#
# Adding a new section or interaction is a one-line edit; the UI rebuilds itself.
NEIGHBORHOOD = {
    "sections": {
        "382": {
            "label": "§382",
            "statute_label": "§382 — NOL limitation after ownership change",
            "role": "hub",
            "one_liner": (
                "Caps a loss corporation's ability to use pre-change NOLs after a\n"
                "“ownership change” (more-than-50%-point shift in 5%-shareholder\n"
                "ownership over a 3-year testing period). The whole neighborhood\n"
                "orbits this section."
            ),
        },
        "318": {
            "label": "§318",
            "statute_label": "§318 — Constructive ownership (attribution)",
            "role": "satellite",
            "one_liner": (
                "Attribution rules that determine who constructively owns what.\n"
                "Every §382 ownership-change calculation runs through §318\n"
                "(modified) to identify the 5%-shareholders being tested."
            ),
        },
        "1504": {
            "label": "§1504(a)(4)",
            "statute_label": "§1504(a)(4) — Plain-vanilla preferred stock",
            "role": "satellite",
            "one_liner": (
                "Defines plain-vanilla preferred stock (non-voting, limited and\n"
                "preferred as to dividends, non-convertible, non-participating).\n"
                "§382 excludes this class from the ownership-change calculation."
            ),
        },
        "381": {
            "label": "§381",
            "statute_label": "§381 — Carryover of tax attributes",
            "role": "satellite",
            "one_liner": (
                "Governs which tax attributes (NOLs, credits, E&P, methods)\n"
                "carry over in a tax-free acquisition or liquidation. §381 says\n"
                "the attributes carry; §382 caps how much you can use."
            ),
        },
        "383": {
            "label": "§383",
            "statute_label": "§383 — Limitation on credits and capital losses",
            "role": "satellite",
            "one_liner": (
                "Parallel limitation to §382, applied to general business credits,\n"
                "foreign tax credits, minimum tax credits, and capital loss\n"
                "carryforwards. Same ownership-change trigger as §382."
            ),
        },
        "384": {
            "label": "§384",
            "statute_label": "§384 — Built-in loss limitation (SRLY-style)",
            "role": "satellite",
            "one_liner": (
                "Limits the use of acquired built-in losses against pre-acquisition\n"
                "gains of an acquiring group. Conceptually parallel to §382(h)\n"
                "NUBIL/NUBIG — same underlying loss, different limitation regime."
            ),
        },
        "165g": {
            "label": "§165(g)",
            "statute_label": "§165(g) — Worthless securities (subsidiary stock)",
            "role": "satellite",
            "one_liner": (
                "Ordinary-loss treatment when stock of an affiliated subsidiary\n"
                "becomes worthless. Worthlessness may itself trigger a §382\n"
                "ownership change in the parent (deemed disposition)."
            ),
        },
        "163j": {
            "label": "§163(j)",
            "statute_label": "§163(j) — Business interest expense limitation",
            "role": "satellite",
            "one_liner": (
                "Caps deductibility of net business interest. Disallowed interest\n"
                "carryforwards are loss-corporation attributes that get caught\n"
                "by the §382 limitation post-ownership-change."
            ),
        },
    },
    # Interactions are intentionally directional: from_ is the section whose
    # rule operates ON to_. "382 -> 318" means "§382 uses §318 attribution."
    "interactions": [
        {
            "from_": "382", "to": "318",
            "label": "uses attribution",
            "explanation": (
                "§382's ownership-change calculation borrows §318(a) attribution\n"
                "(with modifications under §382(l)(3) and the regulations) to\n"
                "determine 5%-shareholder ownership across the testing period."
            ),
        },
        {
            "from_": "382", "to": "1504",
            "label": "excludes plain-vanilla preferred",
            "explanation": (
                "§1504(a)(4) preferred stock is excluded from the §382 ownership-\n"
                "change calculation. A misclassification flips the ownership math."
            ),
        },
        {
            "from_": "381", "to": "382",
            "label": "attributes carry; §382 caps usage",
            "explanation": (
                "§381 governs WHICH attributes survive a tax-free acquisition;\n"
                "§382 then determines HOW MUCH of those carryforwards can be used\n"
                "each year. The two sections work as a pair on every carryover."
            ),
        },
        {
            "from_": "383", "to": "382",
            "label": "parallel cap on credits / cap losses",
            "explanation": (
                "§383 mirrors §382 for general business credits, foreign tax credits,\n"
                "minimum tax credits, and capital loss carryforwards. Same ownership-\n"
                "change trigger; computed off the same §382 limitation base."
            ),
        },
        {
            "from_": "384", "to": "382",
            "label": "parallel BIL limitation (§382(h) sibling)",
            "explanation": (
                "§384 limits built-in losses of one corporation from offsetting\n"
                "pre-acquisition gains of another in an affiliated/SRLY context.\n"
                "Often analyzed alongside §382(h) NUBIL on the same transaction."
            ),
        },
        {
            "from_": "165g", "to": "382",
            "label": "worthlessness may trigger ownership change",
            "explanation": (
                "A §165(g)(3) worthless-subsidiary deduction in the parent can be\n"
                "treated as a deemed disposition. If significant, that disposition\n"
                "contributes to a §382 ownership change at the parent level."
            ),
        },
        {
            "from_": "163j", "to": "382",
            "label": "interest carryforwards caught by §382",
            "explanation": (
                "Disallowed business interest carryforwards under §163(j) are\n"
                "“loss-corporation” attributes for §382 purposes (Notice 2018-28,\n"
                "§1.382-2(a)(8)). After an ownership change they are subject to\n"
                "the §382 annual limitation."
            ),
        },
    ],
}


DISPLAY_ORDER = [
    "ruling_number",
    "document_type",
    "control_number",
    "date_released",
    "date_issued",
    "issuing_office",
    "signatory",
    "subject_summary",
    "transaction_type",
    "issues_presented",
    "conclusions",
    "uilc_on_document",
    "uilc_dictionary_mapping",
    "primary_code_sections",
    "key_authorities_cited",
    "timeline_vector_raw",
    "timeline_vector",
    "favorable_logic_pivot",
    "compelled_representations",
    "precedential_value",
    "migration_status",
    "migration_review",
]

# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------
# Use file mtime as part of the cache key so that updates to CSVs invalidate
# the cache automatically on Streamlit Cloud (where the container is long-lived).
def _data_version() -> str:
    import os
    parts = []
    for p in (RULINGS_CSV, UILC_CSV, VOCAB_CSV):
        try:
            parts.append(f"{p}:{os.path.getmtime(p):.0f}")
        except OSError:
            parts.append(f"{p}:missing")
    return "|".join(parts)


@st.cache_data
def load_data(_version: str):
    rulings = pd.read_csv(RULINGS_CSV, dtype=str).fillna("")
    uilc = pd.read_csv(UILC_CSV, dtype=str).fillna("")
    vocab = pd.read_csv(VOCAB_CSV, dtype=str).fillna("")

    # Parse date_released into proper datetime where possible.
    # Some rows are non-ISO (e.g., "Sept 19, 2013") — try multiple formats.
    parsed = pd.to_datetime(
        rulings["date_released"], errors="coerce", format="%Y-%m-%d"
    )
    # Fallback: try pandas' flexible parser on the rows that failed
    mask = parsed.isna() & (rulings["date_released"].astype(str).str.len() > 0)
    if mask.any():
        flexible = pd.to_datetime(
            rulings.loc[mask, "date_released"], errors="coerce"
        )
        parsed.loc[mask] = flexible
    rulings["date_parsed"] = parsed
    rulings["year"] = rulings["date_parsed"].dt.year

    # Two haystacks per ruling:
    #   _meta_haystack: paraphrased metadata fields (subject_summary, etc.)
    #   _full_text:     raw pdftotext output of the actual ruling
    #   _haystack:      union of both — the catch-all that user text search hits
    def build_meta(row):
        return " | ".join(str(row.get(f, "")) for f in TEXT_SEARCH_FIELDS).lower()
    rulings["_meta_haystack"] = rulings.apply(build_meta, axis=1)
    if FULL_TEXT_FIELD in rulings.columns:
        rulings["_full_text"] = rulings[FULL_TEXT_FIELD].astype(str).str.lower()
    else:
        rulings["_full_text"] = ""
    rulings["_haystack"] = rulings["_meta_haystack"] + " || FULL_TEXT || " + rulings["_full_text"]

    return rulings, uilc, vocab

rulings, uilc, vocab = load_data(_data_version())

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------
def highlight_terms(text: str, terms: list[str]) -> str:
    """Return text with each term wrapped in markdown bold + colored span."""
    if not text or not terms:
        return text
    result = str(text)
    for term in terms:
        if not term:
            continue
        pattern = re.compile(re.escape(term), re.IGNORECASE)
        result = pattern.sub(
            lambda m: f"<mark style='background-color:#fff3a3;padding:0 2px;'>{m.group(0)}</mark>",
            result,
        )
    return result


def parse_uilc_codes(uilc_text: str) -> list[str]:
    """Extract the list of UILC codes from the multiline text in uilc_on_document."""
    out = []
    for line in str(uilc_text).split("\n"):
        for chunk in line.split():
            chunk = chunk.strip()
            if chunk and re.match(r"^\d{3,4}\.", chunk):
                out.append(chunk)
    return out


def count_matches(haystack: str, terms: list[str]) -> int:
    return sum(1 for t in terms if t and t.lower() in haystack)


# -----------------------------------------------------------------------------
# Sidebar — filters
# -----------------------------------------------------------------------------
st.sidebar.title("🧭 Tax Lodestar")
st.sidebar.caption("Subchapter C ruling analytics prototype")
st.sidebar.markdown("---")

st.sidebar.subheader("Filters")

# Initialize session state
if "active_doctrine" not in st.session_state:
    st.session_state["active_doctrine"] = None

# Search box — supports comma-separated terms (OR) and "AND" for intersection
search_query = st.sidebar.text_input(
    "Search text (any field)",
    value="",
    placeholder='e.g.  actual knowledge  •  382(l)(5)  •  preferred stock',
    help="Type one or more search terms. Use `AND` between terms to require all of them. Use commas for OR. Case-insensitive.",
)

# Document type filter
doc_types = sorted([d for d in rulings["document_type"].unique() if d])
selected_doc_types = st.sidebar.multiselect(
    "Document type",
    doc_types,
    default=doc_types,
)

# Date range
min_year = int(rulings["year"].dropna().min()) if rulings["year"].notna().any() else 1998
max_year = int(rulings["year"].dropna().max()) if rulings["year"].notna().any() else 2026
year_range = st.sidebar.slider(
    "Year range (date_released)",
    min_value=min_year,
    max_value=max_year,
    value=(min_year, max_year),
)

# UILC code filter
all_uilc_codes = sorted(set(uilc["uilc_code"].dropna().unique()))
selected_uilc = st.sidebar.multiselect(
    "UILC codes (rulings citing ANY selected)",
    all_uilc_codes,
    default=[],
    help="Leave empty to include all rulings",
)

st.sidebar.markdown("---")
st.sidebar.caption(
    f"**Corpus:** {len(rulings)} rulings · "
    f"{len(uilc)} UILC codes · "
    f"{len(vocab)} vocab terms"
)

# -----------------------------------------------------------------------------
# Doctrinal Query matcher — runs ONLY if active_doctrine is set
# -----------------------------------------------------------------------------
def normalize_uilc(code: str) -> str:
    code = str(code).strip()
    if "." not in code:
        return code
    left, right = code.split(".", 1)
    if left.isdigit() and len(left) < 4:
        left = left.zfill(4)
    if "-" not in right:
        right = right + "-00"
    return f"{left}.{right}"


def match_doctrine(row, doctrine):
    """Returns (matched: bool, signals: list[str]) explaining WHY it matched.

    Signal types (in order of evidentiary strength):
      "full-text phrase: <p>" — the literal phrase appears in the actual ruling PDF text
      "full-text cite: <c>"   — the regulation citation appears in the actual ruling text
      "cite: <c>"             — the citation appears in the metadata (primary_code_sections, etc.)
      "meta phrase: <p>"      — the phrase appears in human-written metadata only
      "UILC <code>"           — the ruling carries that UILC code on its face
    """
    signals = []

    # UILC match
    target_codes = {normalize_uilc(c) for c in doctrine["uilc_codes"]}
    row_codes = {normalize_uilc(c) for c in parse_uilc_codes(row.get("uilc_on_document", ""))}
    uilc_hits = target_codes & row_codes
    if uilc_hits:
        signals.extend(f"UILC {c}" for c in sorted(uilc_hits))

    # Citation match — check metadata blob AND the full ruling text
    cite_blob = " ".join([
        str(row.get("key_authorities_cited", "")),
        str(row.get("primary_code_sections", "")),
        str(row.get("uilc_dictionary_mapping", "")),
    ]).lower()
    full_text = str(row.get("_full_text", "")).lower()
    meta_text = str(row.get("_meta_haystack", "")).lower()
    for cite in doctrine["citations"]:
        cite_lc = cite.lower()
        if cite_lc in full_text:
            signals.append(f"full-text cite: {cite}")
        elif cite_lc in cite_blob:
            signals.append(f"cite: {cite}")

    # Phrase match — prefer full-text hits over metadata hits
    for phrase in doctrine["phrases"]:
        p_lc = phrase.lower()
        if p_lc in full_text:
            signals.append(f'full-text phrase: “{phrase}”')
        elif p_lc in meta_text:
            signals.append(f'meta phrase: “{phrase}”')

    return (len(signals) > 0, signals)


# -----------------------------------------------------------------------------
# Apply filters
# -----------------------------------------------------------------------------
filtered = rulings.copy()
active_doctrine_key = st.session_state.get("active_doctrine")
active_doctrine = DOCTRINAL_QUERIES.get(active_doctrine_key) if active_doctrine_key else None

# When a doctrinal query is active, it overrides the text search but still respects
# doc-type, year, and UILC sidebar filters
doctrine_match_signals = {}  # ruling_number -> list of signals
doctrine_per_rep_hits = {}   # ruling_number -> {rep_label: bool} (only for rep-package doctrines)
if active_doctrine:
    matches = []
    for _, row in filtered.iterrows():
        ok, signals = match_doctrine(row, active_doctrine)
        # If this doctrine declares rep_labels, also compute per-rep hits so
        # the UI can render a structured checkmark row instead of just a count.
        if active_doctrine.get("rep_labels"):
            full_text_lc = str(row.get("_full_text", "")).lower()
            meta_text_lc = str(row.get("_meta_haystack", "")).lower()
            phrases = active_doctrine["phrases"]
            labels = active_doctrine["rep_labels"]
            per_rep = {}
            for label, phrase in zip(labels, phrases):
                p_lc = phrase.lower()
                # Prefer full-text hit; fall back to metadata so paraphrase-only
                # rulings still register (will be flagged in the weaker tier).
                per_rep[label] = (p_lc in full_text_lc) or (p_lc in meta_text_lc)
            doctrine_per_rep_hits[row["ruling_number"]] = per_rep
        if ok:
            matches.append(row["ruling_number"])
            doctrine_match_signals[row["ruling_number"]] = signals
    filtered = filtered[filtered["ruling_number"].isin(matches)]

    # Apply doctrine's year_range if specified (and user hasn't narrowed further)
    if active_doctrine["year_range"]:
        dy_lo, dy_hi = active_doctrine["year_range"]
        filtered = filtered[
            (filtered["year"].isna()) |
            ((filtered["year"] >= dy_lo) & (filtered["year"] <= dy_hi))
        ]

# Text search — supports AND and OR (commas) — still runs even when a doctrine is active
search_terms_or = []
search_terms_and = []
if search_query.strip():
    raw = search_query.strip()
    if " AND " in raw.upper():
        # split on AND (case-insensitive)
        parts = re.split(r"\s+AND\s+", raw, flags=re.IGNORECASE)
        search_terms_and = [p.strip().lower() for p in parts if p.strip()]
        search_terms_or = []
    else:
        # commas = OR, single term = OR with one element
        parts = [p.strip().lower() for p in raw.split(",") if p.strip()]
        search_terms_or = parts
        search_terms_and = []

if search_terms_and:
    mask = filtered["_haystack"].apply(
        lambda h: all(t in h for t in search_terms_and)
    )
    filtered = filtered[mask]
elif search_terms_or:
    mask = filtered["_haystack"].apply(
        lambda h: any(t in h for t in search_terms_or)
    )
    filtered = filtered[mask]

# Doc type filter
if selected_doc_types:
    filtered = filtered[filtered["document_type"].isin(selected_doc_types)]

# Year range
filtered = filtered[
    (filtered["year"].isna()) |
    ((filtered["year"] >= year_range[0]) & (filtered["year"] <= year_range[1]))
]

# UILC filter
if selected_uilc:
    def has_any_selected_uilc(row_uilc_text):
        codes_in_ruling = parse_uilc_codes(row_uilc_text)
        # normalize each for comparison
        normalized = set()
        for code in codes_in_ruling:
            # zero-pad to dictionary form
            if "." in code:
                left, right = code.split(".", 1)
                if left.isdigit() and len(left) < 4:
                    left = left.zfill(4)
                if "-" not in right:
                    right = right + "-00"
                normalized.add(f"{left}.{right}")
            else:
                normalized.add(code)
        return any(c in normalized for c in selected_uilc)
    mask = filtered["uilc_on_document"].apply(has_any_selected_uilc)
    filtered = filtered[mask]

# Highlight terms = whatever the user typed + any phrases from the active doctrine
highlight_list = search_terms_or + search_terms_and
if active_doctrine:
    highlight_list = highlight_list + active_doctrine["phrases"]

# -----------------------------------------------------------------------------
# Main pane
# -----------------------------------------------------------------------------
st.title("Tax Lodestar")
if "full_text_char_count" in rulings.columns:
    _ft_chars = int(pd.to_numeric(rulings["full_text_char_count"], errors="coerce").fillna(0).sum())
else:
    _ft_chars = int(rulings.get(FULL_TEXT_FIELD, pd.Series([""]*len(rulings))).astype(str).str.len().sum())
st.markdown(
    "**Subchapter C ruling analytics prototype.** "
    f"Search across {len(rulings)} §382 IRS rulings (PLRs, FSAs, CCAs, TAMs) "
    f"— metadata **plus** ~{_ft_chars/1000:.0f}K characters of full ruling text — "
    "to find regulatory language the raw IRS database can't surface."
)

# Metric cards
col1, col2, col3, col4 = st.columns(4)
col1.metric("Matching rulings", len(filtered), delta=f"of {len(rulings)} total")
col2.metric(
    "PLRs", int((filtered["document_type"] == "PLR").sum())
)
col3.metric(
    "FSA/TAM/CCA",
    int(filtered["document_type"].isin(["FSA", "TAM", "CCA"]).sum())
)
years_in_filter = filtered["year"].dropna()
if len(years_in_filter):
    col4.metric("Year span", f"{int(years_in_filter.min())}–{int(years_in_filter.max())}")
else:
    col4.metric("Year span", "—")

# -----------------------------------------------------------------------------
# Doctrinal Neighborhood Map — the cross-section interaction picture
# -----------------------------------------------------------------------------
st.markdown("---")
st.subheader("📍 Doctrinal neighborhood")
st.caption(
    "§382 does not live alone. An M&A tax partner working a §382 question is "
    "simultaneously running attribution under §318, watching for plain-vanilla "
    "preferred under §1504(a)(4), tracking carryover under §381, paralleling the "
    "§383 and §384 limitations, and worrying about §165(g) worthless-stock and "
    "§163(j) interest-carryforward interactions. This map shows the controlled-"
    "vocabulary picture of that neighborhood."
)

# Render the neighborhood as a directed graph via Streamlit's native DOT support.
# The hub (§382) sits center; satellites radiate around it. Edge labels carry
# the doctrinal interaction (not just "uses" but WHAT it uses).
def _build_neighborhood_dot(nh: dict) -> str:
    lines = [
        "digraph neighborhood {",
        '  rankdir=LR;',
        '  bgcolor="transparent";',
        '  node [fontname="Helvetica", fontsize=14, style="filled,rounded", shape=box, margin="0.25,0.15"];',
        '  edge [fontname="Helvetica", fontsize=10, color="#555555", fontcolor="#333333"];',
        "",
        "  // Sections (nodes)",
    ]
    for sid, s in nh["sections"].items():
        if s["role"] == "hub":
            fill, border, fontcolor, penwidth = "#01696F", "#01696F", "white", "2"
        else:
            fill, border, fontcolor, penwidth = "#F9F8F5", "#D4D1CA", "#28251D", "1"
        # Use the statute_label so the box tells the user what the section IS,
        # not just its number.
        node_label = s["label"]
        lines.append(
            f'  "{sid}" [label="{node_label}", '
            f'fillcolor="{fill}", color="{border}", fontcolor="{fontcolor}", penwidth={penwidth}];'
        )
    lines.append("")
    lines.append("  // Interactions (edges)")
    for ix in nh["interactions"]:
        # Escape any double quotes in labels defensively.
        edge_label = ix["label"].replace('"', '\\"')
        lines.append(
            f'  "{ix["from_"]}" -> "{ix["to"]}" [label="  {edge_label}  "];'
        )
    lines.append("}")
    return "\n".join(lines)

with st.expander("🗺️ View neighborhood map (graph)", expanded=True):
    st.graphviz_chart(_build_neighborhood_dot(NEIGHBORHOOD), use_container_width=True)
    st.caption(
        "Edges are directional: an arrow from A → B means A's rule operates on B "
        "(e.g., “§381 → §382” means §381 hands attributes to §382, which caps them)."
    )

# Tabular view: the same interactions as a sortable, scannable data table.
# This is the view a partner uses to actually READ the neighborhood; the graph
# above is the at-a-glance picture.
with st.expander("📋 View interactions as a table", expanded=False):
    _interactions_df = pd.DataFrame([
        {
            "From": NEIGHBORHOOD["sections"][ix["from_"]]["label"],
            "→": "→",
            "To": NEIGHBORHOOD["sections"][ix["to"]]["label"],
            "Interaction": ix["label"],
            "Explanation": ix["explanation"].replace("\n", " "),
        }
        for ix in NEIGHBORHOOD["interactions"]
    ])
    st.dataframe(_interactions_df, use_container_width=True, hide_index=True)

# Section legend: lets a partner click each satellite section and read what it is
# and why it matters to §382. This is the part that turns the map from “pretty "
# picture” into a teaching surface.
with st.expander("ℹ️ Section legend (what each statute does)", expanded=False):
    for sid, s in NEIGHBORHOOD["sections"].items():
        role_marker = "🎯" if s["role"] == "hub" else "·"
        st.markdown(f"**{role_marker} {s['statute_label']}**")
        st.markdown(s["one_liner"].replace("\n", " "))
        st.markdown("")

# -----------------------------------------------------------------------------
# Doctrinal Queries — first-class, evidence-based
# -----------------------------------------------------------------------------
st.markdown("---")
st.subheader("Doctrinal lens")
st.caption(
    "Pick a doctrine to filter the corpus to rulings that actually carry it. "
    "Matching uses **UILC code + regulation citation + doctrinal phrase** — not just summary text — "
    "so rulings surface even when the summary uses surface descriptions "
    "(e.g., \"Schedule 13D/13G filings\") rather than naming the doctrine. "
    "Once narrowed, use the search box in the sidebar to drill into facts "
    "(e.g., `SEC filings`, `written inquiries`, `shareholder X`)."
)

# Build picker options from the canonical DOCTRINAL_QUERIES dict so adding a new
# doctrine requires no UI changes — the picker scales automatically.
NO_DOCTRINE_LABEL = "— No doctrinal lens (show all rulings) —"
_doctrine_keys = list(DOCTRINAL_QUERIES.keys())
_picker_options = [NO_DOCTRINE_LABEL] + [
    DOCTRINAL_QUERIES[k]["label"] for k in _doctrine_keys
]
_label_to_key = {DOCTRINAL_QUERIES[k]["label"]: k for k in _doctrine_keys}

# Default the picker to whatever is currently active in session_state, if any.
_current_key = st.session_state.get("active_doctrine")
_current_label = (
    DOCTRINAL_QUERIES[_current_key]["label"]
    if _current_key and _current_key in DOCTRINAL_QUERIES
    else NO_DOCTRINE_LABEL
)
_picker_index = _picker_options.index(_current_label)

# Give the picker a stable widget key so Streamlit manages its state directly.
# We then read the widget's value out of session_state and write it to the
# canonical 'active_doctrine' slot, triggering a rerun via st.rerun() when the
# value actually changes. This is the correct pattern for widget-driven state
# in Streamlit: setting session_state inside the script does NOT auto-rerun,
# so the previous implementation silently lagged by one rerun.
_picked_label = st.selectbox(
    "Doctrine",
    options=_picker_options,
    index=_picker_index,
    help=(
        "Type to filter (e.g., `presumption`, `bankruptcy`, `(l)(5)`, `representations`). "
        "This is the controlled-vocabulary index of §382 doctrines — not a free-text search."
    ),
    label_visibility="collapsed",
    key="doctrine_picker",
)

# Sync picker selection back to the canonical key. If the value differs from
# what's already in session_state, force a rerun so downstream filter logic
# (which runs ABOVE this widget in the script) sees the new value on the
# next pass.
_new_active = None if _picked_label == NO_DOCTRINE_LABEL else _label_to_key[_picked_label]
if st.session_state.get("active_doctrine") != _new_active:
    st.session_state["active_doctrine"] = _new_active
    st.rerun()

# Show the selected doctrine's description inline — previously hidden in tooltip.
_active_key_for_caption = st.session_state.get("active_doctrine")
if _active_key_for_caption:
    _doc = DOCTRINAL_QUERIES[_active_key_for_caption]
    with st.expander("ℹ️ About this doctrine", expanded=False):
        st.markdown(_doc["description"])
        # Show the controlled vocabulary that drives matching for transparency.
        if _doc.get("citations"):
            st.markdown("**Citations matched:** " + ", ".join(f"`{c}`" for c in _doc["citations"]))
        if _doc.get("phrases"):
            st.markdown("**Doctrinal phrases matched:** " + ", ".join(f"`{p}`" for p in _doc["phrases"]))
        if _doc.get("uilc_codes"):
            st.markdown("**UILC codes matched:** " + ", ".join(f"`{c}`" for c in _doc["uilc_codes"]))
        if _doc.get("rep_labels"):
            st.markdown(
                "**Rep tags:** " + ", ".join(f"`{lab}`" for lab in _doc["rep_labels"])
                + "  \n*Each matched ruling is scored on which of these reps it carries.*"
            )

# (No separate "clear" button needed — selecting the first option in the picker
# above clears the doctrinal lens.)

st.markdown("---")

# -----------------------------------------------------------------------------
# Results
# -----------------------------------------------------------------------------
if len(filtered) == 0:
    st.warning("No rulings match the current filters. Try widening the search.")
else:
    sort_options = {
        "Date released (newest first)": ("date_parsed", False),
        "Date released (oldest first)": ("date_parsed", True),
        "Ruling number (ascending)": ("ruling_number", True),
    }
    sort_choice = st.selectbox("Sort results by", list(sort_options.keys()), index=0)
    sort_col, ascending = sort_options[sort_choice]
    filtered = filtered.sort_values(sort_col, ascending=ascending, na_position="last")

    st.subheader(f"Results — {len(filtered)} ruling(s)")

    # Active doctrine banner with match-quality breakdown
    if active_doctrine and doctrine_match_signals:
        # Tally each ruling by its STRONGEST signal type.
        # Strength order (high → low):
        #   full-text phrase > full-text cite > meta cite > meta phrase > UILC only
        ft_phrase = ft_cite = meta_cite = meta_phrase = uilc_only = 0
        for rn in filtered["ruling_number"]:
            sigs = doctrine_match_signals.get(rn, [])
            if any(s.startswith("full-text phrase") for s in sigs):
                ft_phrase += 1
            elif any(s.startswith("full-text cite") for s in sigs):
                ft_cite += 1
            elif any(s.startswith("cite:") for s in sigs):
                meta_cite += 1
            elif any(s.startswith("meta phrase") for s in sigs):
                meta_phrase += 1
            elif any(s.startswith("UILC") for s in sigs):
                uilc_only += 1
        total = ft_phrase + ft_cite + meta_cite + meta_phrase + uilc_only
        st.info(
            f"🎯 **Doctrine:** {active_doctrine['label']} — {total} ruling(s) matched  \n"
            f"• **{ft_phrase}** with the literal doctrinal phrase in the actual ruling text (strongest)  \n"
            f"• **{ft_cite}** citing the precise regulation in the actual ruling text  \n"
            f"• **{meta_cite}** with the citation in metadata only (weaker — review carefully)  \n"
            f"• **{meta_phrase}** with the phrase only in human metadata (weakest — likely paraphrase noise)  \n"
            f"• **{uilc_only}** via UILC code only (broadest — review carefully)"
        )

    # -------------------------------------------------------------------------
    # Rep matrix: only renders for doctrines that declare rep_labels (e.g., the
    # standard §382 representations doctrine). Shows, at a glance, which of the
    # N controlled-vocabulary reps each ruling carries. This is the scannable
    # view of the rep package — 5/5 vs 1/5 visible without expanding any card.
    # -------------------------------------------------------------------------
    if active_doctrine and active_doctrine.get("rep_labels") and doctrine_per_rep_hits:
        rep_labels = active_doctrine["rep_labels"]
        matrix_rows = []
        for _, row in filtered.iterrows():
            rn = row["ruling_number"]
            per_rep = doctrine_per_rep_hits.get(rn, {})
            n_hits = sum(1 for label in rep_labels if per_rep.get(label))
            entry = {
                "Ruling": rn,
                "Score": f"{n_hits}/{len(rep_labels)}",
            }
            for label in rep_labels:
                entry[label] = "✅" if per_rep.get(label) else "—"
            matrix_rows.append((n_hits, entry))
        # Sort by score descending so full-package rulings rise to the top
        matrix_rows.sort(key=lambda x: -x[0])
        matrix_df = pd.DataFrame([e for _, e in matrix_rows])
        with st.expander(
            f"📊 Rep matrix — which of the {len(rep_labels)} reps each ruling carries "
            f"(click to expand)",
            expanded=True,
        ):
            st.caption(
                "Each column is one of the standard §382 reps. ✅ means the rep "
                "phrase appears verbatim in the ruling text (or metadata fallback). "
                "Rulings sorted by completeness of the rep package."
            )
            st.dataframe(
                matrix_df,
                use_container_width=True,
                hide_index=True,
            )

    # Render each ruling as a card
    for _, row in filtered.iterrows():
        # Build a header badge from doctrine signals if available
        header_badge = ""
        if active_doctrine:
            sigs = doctrine_match_signals.get(row["ruling_number"], [])
            # If this is a rep-package doctrine, the header badge becomes an
            # N/M score — more informative than a generic signal-type label.
            if active_doctrine.get("rep_labels"):
                per_rep = doctrine_per_rep_hits.get(row["ruling_number"], {})
                rep_labels = active_doctrine["rep_labels"]
                n_hits = sum(1 for label in rep_labels if per_rep.get(label))
                if n_hits:
                    header_badge = f"  —  ✅ {n_hits}/{len(rep_labels)} reps"
            elif sigs:
                # Show concise signal types in the header, strongest first
                types = []
                if any(s.startswith("full-text phrase") for s in sigs):
                    types.append("📝 full-text phrase")
                if any(s.startswith("full-text cite") for s in sigs):
                    types.append("📜 full-text cite")
                if any(s.startswith("cite:") for s in sigs):
                    types.append("📄 meta cite")
                if any(s.startswith("meta phrase") for s in sigs):
                    types.append("💬 meta phrase")
                if any(s.startswith("UILC") for s in sigs):
                    types.append("🏷️ UILC")
                if types:
                    header_badge = "  —  " + " + ".join(types)

        with st.expander(
            f"**{row['ruling_number']}** · {row['document_type']} · "
            f"{row['date_released']} · {row['subject_summary'][:120]}{header_badge}",
            expanded=False,
        ):
            # Show "why it matched" first if a doctrine is active
            if active_doctrine:
                sigs = doctrine_match_signals.get(row["ruling_number"], [])
                # For rep-package doctrines, lead with a per-rep checkmark row;
                # this is the structured view that turns retrieval into a tag table.
                if active_doctrine.get("rep_labels"):
                    per_rep = doctrine_per_rep_hits.get(row["ruling_number"], {})
                    rep_labels = active_doctrine["rep_labels"]
                    parts = [
                        f"{'✅' if per_rep.get(label) else '❌'} {label}"
                        for label in rep_labels
                    ]
                    st.markdown("**🔖 Reps carried:** " + "  ·  ".join(parts))
                if sigs:
                    st.markdown(
                        f"**🎯 Why this matched:** " + " · ".join(sigs)
                    )
                    st.markdown("")
            # Top row of metadata
            meta_col1, meta_col2, meta_col3 = st.columns(3)
            meta_col1.markdown(f"**Document type:** {row['document_type']}")
            meta_col1.markdown(f"**Control number:** {row['control_number'] or '—'}")
            meta_col2.markdown(f"**Date released:** {row['date_released'] or '—'}")
            meta_col2.markdown(f"**Date issued:** {row['date_issued'] or '—'}")
            meta_col3.markdown(f"**Issuing office:** {row['issuing_office'] or '—'}")
            meta_col3.markdown(f"**Signatory:** {row['signatory'] or '—'}")

            st.markdown("---")

            # Subject
            if row["subject_summary"]:
                st.markdown("**Subject**")
                st.markdown(
                    highlight_terms(row["subject_summary"], highlight_list),
                    unsafe_allow_html=True,
                )

            # Issues + conclusions if present
            for label, fname in [
                ("Issues presented", "issues_presented"),
                ("Conclusions", "conclusions"),
                ("Favorable logic pivot", "favorable_logic_pivot"),
                ("Compelled representations", "compelled_representations"),
            ]:
                if row.get(fname, "").strip():
                    st.markdown(f"**{label}**")
                    st.markdown(
                        highlight_terms(row[fname], highlight_list),
                        unsafe_allow_html=True,
                    )

            # UILC codes
            tab1, tab2, tab3, tab_ft, tab4 = st.tabs([
                "UILC Codes",
                "Code Sections & Authorities",
                "Timeline Vector",
                "📝 Full text",
                "Raw record",
            ])

            with tab1:
                if row["uilc_dictionary_mapping"]:
                    st.markdown(
                        highlight_terms(row["uilc_dictionary_mapping"], highlight_list),
                        unsafe_allow_html=True,
                    )
                else:
                    st.caption("No UILC dictionary mapping available.")

            with tab2:
                if row["primary_code_sections"]:
                    st.markdown("**Primary Code Sections & Regulations**")
                    st.markdown(
                        highlight_terms(row["primary_code_sections"], highlight_list),
                        unsafe_allow_html=True,
                    )
                if row["key_authorities_cited"]:
                    st.markdown("**Key Authorities Cited**")
                    st.markdown(
                        highlight_terms(row["key_authorities_cited"], highlight_list),
                        unsafe_allow_html=True,
                    )

            with tab3:
                if row["timeline_vector_raw"]:
                    st.markdown(
                        highlight_terms(row["timeline_vector_raw"], highlight_list),
                        unsafe_allow_html=True,
                    )
                else:
                    st.caption("No timeline vector data.")

            with tab_ft:
                full_text = str(row.get(FULL_TEXT_FIELD, ""))
                if not full_text.strip():
                    st.caption("No full text available for this ruling.")
                else:
                    char_count = int(row.get("full_text_char_count") or len(full_text))
                    st.caption(
                        f"Source: pdftotext -layout extraction of IRS PDF — {char_count:,} characters. "
                        "Highlighted excerpts shown first; full text below."
                    )

                    # Build excerpt list: every match of every active term + doctrinal phrase/cite
                    excerpt_terms = []
                    if active_doctrine:
                        excerpt_terms += [p for p in active_doctrine["phrases"]]
                        excerpt_terms += [c for c in active_doctrine["citations"]]
                    excerpt_terms += [t for t in highlight_list if t]
                    # de-dupe while preserving order, case-insensitive
                    seen = set()
                    excerpt_terms = [
                        t for t in excerpt_terms
                        if (t.lower() not in seen and not seen.add(t.lower()))
                    ]

                    if excerpt_terms:
                        ft_lower = full_text.lower()
                        # collect all match offsets across all terms
                        offsets = []
                        for term in excerpt_terms:
                            tl = term.lower()
                            if not tl:
                                continue
                            start = 0
                            while True:
                                i = ft_lower.find(tl, start)
                                if i < 0:
                                    break
                                offsets.append((i, i + len(term), term))
                                start = i + max(1, len(term))
                        offsets.sort()
                        # merge nearby offsets into excerpt windows
                        windows = []
                        WIN = 250
                        for start_i, end_i, term in offsets:
                            ws = max(0, start_i - WIN)
                            we = min(len(full_text), end_i + WIN)
                            if windows and ws <= windows[-1][1]:
                                windows[-1] = (windows[-1][0], max(we, windows[-1][1]), windows[-1][2] + [term])
                            else:
                                windows.append((ws, we, [term]))
                        if windows:
                            st.markdown(f"**{len(offsets)} match(es) across {len(windows)} excerpt(s):**")
                            for ws, we, terms in windows[:10]:
                                excerpt = full_text[ws:we]
                                prefix = "…" if ws > 0 else ""
                                suffix = "…" if we < len(full_text) else ""
                                rendered = highlight_terms(prefix + excerpt + suffix, terms)
                                st.markdown(rendered, unsafe_allow_html=True)
                                st.markdown("")
                            if len(windows) > 10:
                                st.caption(f"…and {len(windows) - 10} more excerpt(s) below the cutoff. "
                                           "Use Ctrl-F on the full text to find them.")
                        else:
                            st.caption("No matches found in full text for the active terms.")
                    else:
                        st.caption("No active search terms or doctrinal query. Showing full text below.")

                    with st.expander("📄 Show entire ruling text", expanded=False):
                        st.text_area(
                            "Full text",
                            value=full_text,
                            height=400,
                            label_visibility="collapsed",
                            key=f"ft_{row['ruling_number']}",
                        )

            with tab4:
                st.json({
                    k: row[k]
                    for k in DISPLAY_ORDER
                    if k in row and row[k]
                })

# -----------------------------------------------------------------------------
# Analytics panel (bottom)
# -----------------------------------------------------------------------------
st.markdown("---")
with st.expander("📊 Landscape — distribution of filtered rulings", expanded=False):
    if len(filtered) > 0:
        # By document type
        dt_col, yr_col = st.columns(2)
        with dt_col:
            st.markdown("**By document type**")
            dt_counts = filtered["document_type"].value_counts()
            st.bar_chart(dt_counts)
        with yr_col:
            st.markdown("**By year released**")
            yr_counts = filtered["year"].dropna().astype(int).value_counts().sort_index()
            st.bar_chart(yr_counts)

        # UILC distribution
        st.markdown("**Top UILC codes in filtered set**")
        code_counts = {}
        for _, row in filtered.iterrows():
            for code in parse_uilc_codes(row["uilc_on_document"]):
                # normalize
                if "." in code:
                    left, right = code.split(".", 1)
                    if left.isdigit() and len(left) < 4:
                        left = left.zfill(4)
                    if "-" not in right:
                        right = right + "-00"
                    norm = f"{left}.{right}"
                else:
                    norm = code
                code_counts[norm] = code_counts.get(norm, 0) + 1

        code_df = pd.DataFrame(
            sorted(code_counts.items(), key=lambda x: -x[1])[:15],
            columns=["UILC code", "Rulings"]
        )
        # join with UILC titles
        code_df = code_df.merge(
            uilc[["uilc_code", "title"]],
            left_on="UILC code",
            right_on="uilc_code",
            how="left",
        )
        code_df["title"] = code_df["title"].fillna("(not in dictionary)")
        code_df = code_df[["UILC code", "title", "Rulings"]]
        st.dataframe(code_df, use_container_width=True, hide_index=True)
    else:
        st.caption("No data to chart with current filters.")

# Footer
st.markdown("---")
st.caption(
    "Tax Lodestar prototype · Built on the unified 19-field Subchapter C schema · "
    f"Data: {len(rulings)} §382 rulings (Dec 1998 – Jul 2023)"
)
