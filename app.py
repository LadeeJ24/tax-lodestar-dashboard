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
@st.cache_data
def load_data():
    rulings = pd.read_csv(RULINGS_CSV, dtype=str).fillna("")
    uilc = pd.read_csv(UILC_CSV, dtype=str).fillna("")
    vocab = pd.read_csv(VOCAB_CSV, dtype=str).fillna("")

    # Parse date_released into proper datetime where possible
    rulings["date_parsed"] = pd.to_datetime(
        rulings["date_released"], errors="coerce", format="%Y-%m-%d"
    )
    rulings["year"] = rulings["date_parsed"].dt.year

    # Build a single haystack column per ruling for fast text search
    def build_haystack(row):
        return " | ".join(str(row.get(f, "")) for f in TEXT_SEARCH_FIELDS).lower()
    rulings["_haystack"] = rulings.apply(build_haystack, axis=1)

    return rulings, uilc, vocab

rulings, uilc, vocab = load_data()

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
# Apply filters
# -----------------------------------------------------------------------------
filtered = rulings.copy()

# Text search — supports AND and OR (commas)
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

# Highlight terms = whatever the user typed
highlight_list = search_terms_or + search_terms_and

# -----------------------------------------------------------------------------
# Main pane
# -----------------------------------------------------------------------------
st.title("Tax Lodestar")
st.markdown(
    "**Subchapter C ruling analytics prototype.** "
    "Search across 68 §382 IRS rulings (PLRs, FSAs, CCAs, TAMs) "
    "to find patterns the raw IRS database can't surface."
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
# Quick canned queries
# -----------------------------------------------------------------------------
st.markdown("---")
st.subheader("Try a needle-in-haystack query")

canned_col1, canned_col2, canned_col3 = st.columns(3)
if canned_col1.button(
    "1️⃣ Presumption of no cross-ownership",
    use_container_width=True,
    help="Rulings discussing the §382 actual-knowledge presumption (Treas. Reg. § 1.382-2T(k))",
):
    st.session_state["preset_search"] = "actual knowledge, no ownership, presumption, cross-ownership"

if canned_col2.button(
    "2️⃣ §382(l)(5) rulings",
    use_container_width=True,
    help="Rulings invoking the §382(l)(5) bankruptcy exception",
):
    st.session_state["preset_search"] = "382(l)(5)"

if canned_col3.button(
    "3️⃣ Actual knowledge, 2008–2020",
    use_container_width=True,
    help="Rulings addressing actual knowledge during 2008-2020",
):
    st.session_state["preset_search"] = "actual knowledge"
    st.session_state["preset_years"] = (2008, 2020)

if st.session_state.get("preset_search"):
    st.info(
        f"💡 Canned query loaded: **{st.session_state['preset_search']}**. "
        "Copy this into the sidebar **Search text** box and adjust the year filter if needed. "
        "(Streamlit doesn't let me write back into sidebar inputs from a button — click and paste.)"
    )

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

    # Render each ruling as a card
    for _, row in filtered.iterrows():
        with st.expander(
            f"**{row['ruling_number']}** · {row['document_type']} · "
            f"{row['date_released']} · {row['subject_summary'][:120]}",
            expanded=False,
        ):
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
            tab1, tab2, tab3, tab4 = st.tabs([
                "UILC Codes",
                "Code Sections & Authorities",
                "Timeline Vector",
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
