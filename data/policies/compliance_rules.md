# Compliance Rules

Policy ID: COMP-RULE-01
Effective date: 2026-01-15
Owner: Compliance and Risk Management

## 1.0 Mass-casualty activation

When an incident presents more patients than can be processed sequentially within normal
turnaround targets, mass-casualty activation may be declared by the attending or the on-call
administrator. Activation triggers parallel triage processing with priority ordering by acuity,
not arrival order, coordinated with Section 4 of the Emergency Protocols. Activation status,
the declaring individual, and the timestamp are recorded as the first entry in the incident's
audit trail.

## 2. De-identification and data handling

### 2.1 Minimum necessary data

Automated placement and staffing decisions operate on de-identified operational fields only —
synthetic patient reference, age, diagnosis category, severity, isolation flag, and monitoring
level. Direct identifiers (name, national identifier, insurance or payer identifiers, contact
information) are out of scope for this system and must not be introduced into its data model.

### 2.2 Access control

Access to placement, staffing, and audit records is role-based. Reviewers acting on the human
approval gate see only the fields required to make an approval decision.

## 3. Audit and record retention

### 3.1 Append-only audit trail

Every automated or human decision affecting placement, staffing, or routing is logged in an
audit trail that permits new entries but not modification or deletion of prior entries. This
applies at the system access-control level, not only as an application-level convention.

### 3.2 Retention

Audit records are retained for the period required by applicable recordkeeping regulation and are
not purged as part of routine data cleanup.

## 4. Human review requirements

A decision must be routed to human review when any of the following hold: the system's confidence
score falls below the defined threshold, two or more agents produce conflicting recommendations,
the case is flagged critical, or a policy conflict is detected during consolidation. This
requirement cannot be bypassed by a downstream process; it is enforced at the point confidence is
evaluated.

## 5. Documentation standards

Every automated recommendation must cite the specific policy section that grounds it. A
recommendation with no supporting citation is treated as ungrounded and its confidence score is
lowered accordingly rather than presented as a confident decision.
