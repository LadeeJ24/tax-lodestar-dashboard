# Tax Lodestar — IRS Ruling Analytics Prototype

A Streamlit dashboard for needle-in-haystack analysis across Subchapter C IRS rulings.

## What it does

- Full-text search across all narrative fields in 68 §382 rulings
- Filter by document type (PLR/FSA/TAM/CCA), date range, and UILC code
- Click into any ruling to see its full record with search terms highlighted
- Landscape view: distribution of rulings by document type, year, and UILC code

## Sample queries

- "Presumption of no cross-ownership" — actual knowledge doctrine under §382
- "§382(l)(5) rulings" — bankruptcy exception cases
- "Actual knowledge, 2008–2020" — temporal slice of a doctrinal question

## Tech stack

- Streamlit + pandas
- Data loaded from CSVs (committed in `data/`)
- No backend, no database — runs entirely in-memory

## Schema

19 unified fields mapping Master Index-Enforcer prompt output:
- Document identity: ruling_number, document_type, control_number, dates, signatory, precedential_value
- Indexing: uilc_on_document, uilc_dictionary_mapping, primary_code_sections, key_authorities_cited
- Substance: transaction_type, issues_presented, conclusions, subject_summary
- Differentiators: timeline_vector, favorable_logic_pivot, compelled_representations

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open http://localhost:8501

## Deploy

Pushed to Streamlit Community Cloud — see live URL in repo description.
