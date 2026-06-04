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
        "year_range": None,
    },
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
if active_doctrine:
    matches = []
    for _, row in filtered.iterrows():
        ok, signals = match_doctrine(row, active_doctrine)
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
# Doctrinal Queries — first-class, evidence-based
# -----------------------------------------------------------------------------
st.markdown("---")
st.subheader("Doctrinal queries")
st.caption(
    "These search by **UILC code + regulation citation + doctrinal phrase** — not just summary text. "
    "They surface rulings even when the summary uses surface descriptions (e.g., \"Schedule 13D/13G filings\") "
    "rather than naming the doctrine."
)

# Row 1
canned_col1, canned_col2 = st.columns(2)
if canned_col1.button(
    "1️⃣ Presumption rebuttal — § 1.382-2T(j)(2)(iii)(B)(1)",
    use_container_width=True,
    help=DOCTRINAL_QUERIES["presumption_no_cross_ownership"]["description"],
):
    st.session_state["active_doctrine"] = "presumption_no_cross_ownership"

if canned_col2.button(
    "2️⃣ §382(l)(5) bankruptcy exception",
    use_container_width=True,
    help=DOCTRINAL_QUERIES["section_382_l_5"]["description"],
):
    st.session_state["active_doctrine"] = "section_382_l_5"

# Row 2
canned_col3, canned_col4 = st.columns(2)
if canned_col3.button(
    "3️⃣ Actual knowledge, 2008–2020",
    use_container_width=True,
    help=DOCTRINAL_QUERIES["actual_knowledge_2008_2020"]["description"],
):
    st.session_state["active_doctrine"] = "actual_knowledge_2008_2020"

if canned_col4.button(
    "4️⃣ Standard §382 representations (5-rep package)",
    use_container_width=True,
    help=DOCTRINAL_QUERIES["standard_382_representations"]["description"],
):
    st.session_state["active_doctrine"] = "standard_382_representations"

if st.session_state.get("active_doctrine"):
    if st.button("✕ Clear doctrinal query", type="secondary"):
        st.session_state["active_doctrine"] = None
        st.rerun()

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

    # Render each ruling as a card
    for _, row in filtered.iterrows():
        # Build a header badge from doctrine signals if available
        header_badge = ""
        if active_doctrine:
            sigs = doctrine_match_signals.get(row["ruling_number"], [])
            if sigs:
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
