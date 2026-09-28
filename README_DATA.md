# Seed data for the hospital resource allocation system

Everything in this folder is synthetic — generated for development and testing, not sourced from
any real hospital, patient, or staff record.

## What each piece is for

**policies/** — the source text the RAG pipeline chunks, embeds, and retrieves from. Original
documents written for this project, structured like real hospital policy (numbered sections,
defined thresholds, escalation rules).

**seed/beds.csv and seed/staff.csv** — reference data for the capacity lookups the bed and staff
agents call as tools. Occupancy is pre-set to roughly ICU 78%, general ward 65%, isolation 40%.

**seed/synthetic_patient_requests.csv** — 68 requests spanning every severity level, several
diagnosis categories, all three monitoring levels, and all four arrival modes, plus one explicit
8-patient mass-casualty batch at the end.

**fixtures/eval_fixtures.json** — 13 requests paired with an expected router output and expected
agent decision, hand-derived from the rules in policies/. Use as assertion targets for unit and
integration tests, not as clinical ground truth.

## The constraint this data honors

No file here contains a name, a national identifier, an insurance or payer ID, or any other direct
identifier — only the operational fields the system needs (patient_ref, age, diagnosis category,
severity, isolation flag, monitoring level, arrival mode). Keep any data added later to the same
shape.
