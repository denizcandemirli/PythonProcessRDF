### Issues and recommendations

1) Fusion column robustness
- Path: compute_similarity.py
- Evidence: requires specific old column names; failed earlier with new canonical columns.
- Impact: pipeline break when upstream emits S_content/S_typed/S_edge/S_struct.
- Fix: extended pick() candidates to include new names.

2) Windows console Unicode
- Path: visualize_total_similarity.py
- Evidence: UnicodeEncodeError for '\u2192' arrow.
- Impact: visualization step aborts on Windows console encodings.
- Fix: replaced arrow with ASCII '->'.

3) Parameter surface visibility
- Path: repository-wide
- Evidence: many scripts expose CLI parameters without central inventory.
- Impact: reproducibility and tuning friction.
- Fix: added tools/gen_parameters.py to synthesize parameters.csv.

4) SHACL not executed
- Path: N/A
- Evidence: no SHACL engine wired; only ontology heuristics.
- Impact: domain/range violations undetected automatically.
- Mitigation: provided shapes_min.ttl and instructions to run pyshacl; recommend integrating pyshacl.

### Patches applied
- compute_similarity.py: accept new canonical columns.
- visualize_total_similarity.py: ASCII-only logging.

See patches/ for PR-ready diffs (to be organized if needed).
