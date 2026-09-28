# Compliance Rules

Effective date: 2026-01-01

## 1.0 Mass-casualty activation

Multi-patient incidents trigger parallel triage processing with priority ordering by acuity, not
arrival order. When a mass-casualty request is detected (multiple patients submitted under a
single incident, or `arrival_mode` of `mass_casualty`), the workflow must process each patient
concurrently and record a complete, independent audit trail for every patient in the batch — no
patient's processing may be skipped or deferred because of batch size.

## 2.0 Audit and grounding requirements

Every agent recommendation must carry at least one policy citation grounding its decision. A
recommendation issued without a supporting citation above the similarity floor must be marked
`policy_grounded: false` and have its confidence score lowered accordingly, rather than being
presented as policy-backed. This rule exists to keep the system's compliance guarantees
enforceable in code rather than aspirational.
