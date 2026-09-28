# Bed Allocation Policy

Effective date: 2026-01-01

## 2.4 General ward defaults

Stable patients without elevated monitoring needs default to general ward placement. A general
ward bed is the correct assignment whenever severity is moderate or minor, monitoring level is
"none" or "standard", and isolation is not required. Ward assignment should prefer the patient's
diagnosis-appropriate ward (e.g. cardiac complaints to the telemetry-capable ward) when capacity
allows, but general ward placement itself does not require specialist bed availability.

## 3.2 Critical acuity placement

Patients presenting with critical acuity indicators are placed in monitored ICU beds ahead of
ward assignment. Critical acuity includes severity classified as "critical" or any requirement
for ICU-level monitoring, regardless of arrival mode. ICU placement takes priority over ward
capacity constraints — if no ICU bed is immediately available, the request must be escalated to
human review rather than defaulted to a lower level of care.

## 4.1 Isolation criteria

Suspected or confirmed infectious status requires isolation-room placement regardless of ward
capacity. Isolation placement applies whenever `requires_isolation` is true, independent of
severity level. Isolation rooms take precedence over general ward assignment but do not override
a simultaneous critical-acuity ICU requirement — in that case, an ICU bed with isolation
capability should be sought first, and the conflict flagged for staffing confirmation if one is
not available.
