# Verdict Graph: Specification

> PRIVATE AND PROPRIETARY. Owned by Kanjani AI Research. See NOTICE.md.

**Status:** Specification only — not implemented
**Layer:** Domain Graph (governance evidence domain)
**Depends on:** codeproperty-graph, document-graph, lexical-graph
**Related:** AIGP RFC-032 (Post-Hoc Evaluation Loop), RFC-035 (Mediation Vector Profile), D-DNA

---

## 1. What Is the Verdict Graph?

The Verdict Graph is a Domain Graph whose domain is **software governance evidence** — the accumulation of observations, verdicts, quality signals, and provenance about software artifacts over their lifecycle.

```
Code Property Graph  =  "What IS the code?"       (structure, flow, calls)
Verdict Graph        =  "What do we KNOW about    (verdicts, evidence, posture,
                         the code?"                provenance, compliance)
```

It is the **Master Evidence Graph** described in the research papers — made executable as a queryable property graph in Neptune.

---

## 2. Why It Exists

Software governance today is stateless:
- A scan runs, produces a report, the report is filed
- Next scan runs, produces another report — no connection to the previous one
- Nobody can ask: "Show me the *trajectory* of this component's quality over time"

The Verdict Graph makes governance **stateful**:
- Every verdict is a node with timestamp, evidence, observer, and result
- Verdicts connect to the artifact they evaluate (CPG nodes)
- History accumulates — you can query trajectories, not just snapshots
- Provenance is first-class: who observed, what evidence, under what authority

---

## 3. Node Types

| Node Type | Description | Key Properties |
|-----------|-------------|----------------|
| `Artifact` | The thing being evaluated (code unit, deployment, config) | artifact_id, type, version, source_repo |
| `Verdict` | A judgment rendered on an artifact | verdict_id, result (MATCH/MISMATCH/VIOLATION), timestamp, confidence |
| `Evidence` | A signed, provenance-bound piece of evidence | evidence_id, type, source, hash, admissibility_level |
| `Observer` | Who/what rendered the verdict | observer_id, type (human/machine/hybrid), calibration_status |
| `Concern` | The evaluative question being asked | concern_id, domain_of_concern, concern_class |
| `Policy` | The governance rule that triggered evaluation | policy_id, version, authority, scope |
| `MediationVector` | A point-in-time measurement snapshot (RFC-035) | vector_id, timestamp, stage, variable_values |
| `CouplingState` | Trajectory state for process-level concerns (ADD-035-001) | coupling_id, session_count, current_stage, progression_rate |

---

## 4. Edge Types

| Edge Type | From → To | Meaning |
|-----------|-----------|---------|
| `EVALUATES` | Verdict → Artifact | This verdict evaluates this artifact |
| `EVIDENCED_BY` | Verdict → Evidence | This verdict is supported by this evidence |
| `RENDERED_BY` | Verdict → Observer | This observer rendered this verdict |
| `CONCERNS` | Verdict → Concern | This verdict addresses this concern |
| `AUTHORIZED_BY` | Verdict → Policy | This policy authorized the evaluation |
| `SUPERSEDES` | Verdict → Verdict | This verdict replaces the previous one |
| `DERIVED_FROM` | Evidence → Evidence | Provenance chain |
| `OBSERVED_IN` | MediationVector → Artifact | This measurement observed this artifact |
| `PRODUCED_BY` | MediationVector → Observer | Who made this measurement |
| `PART_OF` | MediationVector → CouplingState | This measurement belongs to this trajectory |
| `REFERENCES` | Artifact → CPGNode | Links verdict subject to code structure |

---

## 5. Relationship to Code Property Graph

```
Verdict Graph                          Code Property Graph
─────────────                          ────────────────────
Artifact(id="parser-v2.1")  ──REFERENCES──→  METHOD(full_name="com.example.Parser.parse")
    │
    ├── EVALUATES ←── Verdict(result=MATCH, t=2026-06-30)
    │                     │
    │                     ├── EVIDENCED_BY → Evidence(type=D-DNA, hash=...)
    │                     ├── RENDERED_BY  → Observer(type=machine, id=sast-scanner)
    │                     └── CONCERNS     → Concern(class=code_quality)
    │
    └── EVALUATES ←── Verdict(result=MISMATCH, t=2026-06-15)
                          │
                          └── SUPERSEDES → Verdict(result=MATCH, t=2026-06-01)
```

The REFERENCES edge is the **bridge** between the Verdict Graph and the Code Property Graph. It links governance observations to the code structure they observe.

---

## 6. Relationship to AIGP Protocol

| AIGP Concept | Verdict Graph Node/Edge |
|--------------|------------------------|
| RFC-032 VERIFY verdict | `Verdict` node with result, timestamp, evidence |
| RFC-035 Mediation Vector | `MediationVector` node with variable_values |
| D-DNA signed evidence | `Evidence` node with hash, provenance, admissibility_level |
| RFC-034 Domain of Concern | `Concern` node with domain, class |
| RFC-037 Observer | `Observer` node with type, calibration, accreditation |
| ADD-035-001 Coupling | `CouplingState` node with stage, progression_rate |
| ADD-035-004 Progressive Degradation | Trajectory of `MediationVector` nodes over time |

The Verdict Graph is **AIGP made queryable** — the protocol's evidence and verdicts stored as graph state that can be traversed, aggregated, and analyzed.

---

## 7. DRY Architecture (Projected)

```
verdict-graph (domain layer — ~200 lines)
├── schema.py          — Verdict/Evidence/Observer/Concern types
├── models.py          — VerdictNode, EvidenceNode, etc. with from_aigp()
├── verdict_tracker.py — Accumulate verdicts over time, link to artifacts
├── posture_query.py   — Query concern posture trajectories
└── provenance.py      — Evidence chain traversal

document-graph (infrastructure — REUSED)
├── Node, Edge, batch_nodes_unwind
├── CypherBuilder, tenant-scoped labels
└── Multi-tenancy

lexical-graph (foundation — REUSED)
├── GraphStoreFactory, Neptune connection
└── TenantId
```

---

## 8. Key Queries the Verdict Graph Enables

```cypher
-- "What is the current quality posture of this artifact?"
MATCH (a:Artifact {artifact_id: $id})<-[:EVALUATES]-(v:Verdict)
WHERE NOT (v)<-[:SUPERSEDES]-()
RETURN v.result, v.timestamp, v.confidence

-- "Show me the verdict trajectory for this component over 6 months"
MATCH (a:Artifact {artifact_id: $id})<-[:EVALUATES]-(v:Verdict)
WHERE v.timestamp > datetime() - duration('P6M')
RETURN v.result, v.timestamp ORDER BY v.timestamp

-- "Which artifacts have unresolved MISMATCH verdicts?"
MATCH (v:Verdict {result: 'MISMATCH'})-[:EVALUATES]->(a:Artifact)
WHERE NOT (v)<-[:SUPERSEDES]-()
RETURN a.artifact_id, v.timestamp, v.concern_class

-- "What evidence supports this verdict?"
MATCH (v:Verdict {verdict_id: $id})-[:EVIDENCED_BY]->(e:Evidence)
RETURN e.type, e.source, e.admissibility_level, e.hash

-- "Show the progressive degradation trajectory for this coupling"
MATCH (cs:CouplingState {coupling_id: $id})<-[:PART_OF]-(mv:MediationVector)
RETURN mv.timestamp, mv.stage, mv.variable_values
ORDER BY mv.timestamp

-- "Which code methods have never been evaluated?"
MATCH (m:METHOD)
WHERE NOT (:Artifact)-[:REFERENCES]->(m)
RETURN m.full_name, m.filename
```

---

## 9. Delta Logic

Like codeproperty-graph, the Verdict Graph has delta semantics — but **additive**, not replacement:

| codeproperty-graph | verdict-graph |
|--------------------|---------------|
| Full replacement on change | Append-only (verdicts accumulate) |
| Old tenant purged | Old verdicts stay (linked via SUPERSEDES) |
| Manifest = method signatures | No manifest needed (timestamps are natural ordering) |
| Skip if no code change | Always write new verdicts (observation is always new) |

The Verdict Graph never deletes — it only supersedes. History is preserved for audit trail and trajectory analysis.

---

## 10. Temporal Model

```
t=0: First scan
     Verdict(MATCH) → Artifact(parser-v2.0)

t=1: Code changes, new scan
     Verdict(MISMATCH) → Artifact(parser-v2.1)
         └── SUPERSEDES → Verdict(MATCH) at t=0

t=2: Fix applied, re-scan
     Verdict(MATCH) → Artifact(parser-v2.2)
         └── SUPERSEDES → Verdict(MISMATCH) at t=1

Query at any time: "What's the CURRENT posture?"
→ Find verdicts with no incoming SUPERSEDES edge = latest state
```

---

## 11. Integration Points

| System | Integration |
|--------|-------------|
| CI/CD pipeline | Writes Verdict nodes after each scan/test |
| SAST scanner (Joern, Semgrep) | Observer; Evidence is the scan output |
| AIGP Governance Server | Writes CHECK/VERIFY results as Verdicts |
| SBOM (CycloneDX) | Evidence node linked to dependency verdicts |
| D-DNA signer | Signs Evidence nodes; hash + signature stored |
| Human reviewer | Observer(type=human); manual Verdict |
| Quality Moderator (RFC-032) | Reads verdict trajectory; adapts posture |

---

## 12. When to Build

The Verdict Graph should be built when:
1. codeproperty-graph is stable in production (code structure is queryable)
2. AIGP RFC-036 (calculation semantics) is drafted (defines how vectors combine)
3. A governance pipeline exists that produces verdicts to store

It should NOT be built prematurely — without real verdicts flowing, the graph is empty and adds no value. The spec exists so the architecture is ready when the pipeline is.

---

## 13. Summary

| Aspect | Value |
|--------|-------|
| GraphRAG type | Domain Graph (governance evidence domain) |
| Domain | Software verdicts, evidence, provenance, quality posture |
| Enables | Stateful governance — trajectories, not just snapshots |
| Architecture | ~200 lines domain code + document-graph + lexical-graph (DRY) |
| Key innovation | SUPERSEDES edge makes history queryable without deletion |
| AIGP alignment | Makes RFC-032/035/037 concepts queryable in Neptune |
| Prerequisite | Real governance pipeline producing verdicts |

---

© 2024-2026 Kanjani AI Research. All rights reserved.
