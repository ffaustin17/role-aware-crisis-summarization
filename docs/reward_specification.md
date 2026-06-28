# Role-Aware Reward Specification

This document defines the first transparent reward function for role-aware
crisis tweet summarization. It is designed for generated summaries where each
record has a source tweet/context, one or more responder roles, and a candidate
summary.

## Composite Reward

The baseline reward is a weighted average of four normalized component scores:

```text
reward = 0.35 relevance
       + 0.25 factuality
       + 0.20 role_coverage
       + 0.20 urgency
```

Each component is scored on a 0.0 to 1.0 scale. Higher is better.

## Role Criteria

Multi-role rows use the union of the criteria for all assigned roles. Rows are
not expanded by role.

### EMS

EMS summaries should prioritize people needing medical attention and transport.

Tracked criteria:
- injury and casualty awareness
- medical urgency
- triage and transport readiness
- vulnerable people, trapped people, or rescue needs

Grounded keywords and phrases:
- `injured`, `injuries`, `injury`, `wounded`, `casualties`, `fatalities`,
  `dead`, `death toll`, `killed`, `hurt`, `people hurt`
- `medical`, `medical aid`, `medical assistance`, `medical support`,
  `emergency medical`, `medical emergencies`, `patient`, `patient care`,
  `treatment`, `care`, `triage`, `transport`, `patient transport`
- `urgent`, `urgency`, `critical`, `vulnerable`, `rescue`, `trapped`,
  `mass casualty`
- `ambulance`, `ambulance response`, `hospital`, `MMU`, `CERT`

### Firefighter

Fire summaries should prioritize fire behavior, containment, exposure hazards,
and rescue conditions.

Tracked criteria:
- fire spread
- containment
- hazardous materials or exposure
- structural danger, collapse, and search/rescue conditions
- smoke, haze, and air-quality danger

Grounded keywords and phrases:
- `fire`, `fires`, `wildfire`, `flames`, `smoke`, `haze`
- `spread`, `fire spread`, `containment`, `contain`, `uncontained`, `growing`
- `fire control`, `FC`, `firefighting`, `fire crews`
- `hazardous`, `hazardous material`, `hazmat`, `material exposure`,
  `chemical`, `gas`, `toxic`, `fumes`, `smoke inhalation`
- `structural`, `structural safety`, `collapse`, `trapped`,
  `search and rescue`, `search rescue`, `rescue operations`
- `air quality`, `PSI`, `smoke exposure`, `USAR`

### Police

Police summaries should prioritize public safety, scene control, evacuation,
traffic/access, and security threats.

Tracked criteria:
- threat and public safety
- crowd control
- evacuation, access, and traffic control
- scene security
- criminal activity, unrest, or protest monitoring

Grounded keywords and phrases:
- `threat`, `threats`, `safety threat`, `public safety`, `security`
- `law enforcement`, `public order`
- `crowd`, `crowd control`, `scene security`, `secure the area`, `perimeter`
- `evacuation`, `evacuate`, `evacuations`, `access`, `access control`,
  `traffic`, `traffic control`, `road closure`, `road closures`,
  `blocked roads`
- `criminal activity`, `criminal investigation`, `crime prevention`, `arrest`,
  `arrested`, `investigate`, `probe`, `unrest`, `protest`
- `DCC`, `dispatch`, `TEU`, `CPU`

## Component Definitions

### Relevance

Relevance measures whether the candidate summary is semantically related to the
source tweet first, while still giving some credit for matching role/disaster
context. The preferred implementation uses SentenceTransformer cosine
similarity to compute:

```text
tweet_relevance   = similarity(candidate summary, source tweet)
context_relevance = similarity(candidate summary, role/disaster/context fields)
relevance         = 0.70 tweet_relevance + 0.30 context_relevance
```

If context fields are unavailable, relevance falls back to tweet-only scoring.
If the source tweet is unavailable, relevance falls back to the existing combined
source context.

Cosine similarity is normalized to the 0.0 to 1.0 range. A lexical fallback may
be used for smoke tests or environments without model access.

### Factuality

The local default implementation uses a transparent source-grounding proxy. It
rewards candidate summaries whose content words, numbers, and compact
identifiers are supported by the source context. This proxy is useful for fast
local scoring and debugging because it has no large model dependency.

The scoring script also supports an optional MiniCheck backend. MiniCheck scores
sentence-level claims from the generated summary against the source context, then
averages the support probabilities into the factuality component. This should be
the preferred backend for final reward experiments once the dependency and model
checkpoint are available in the Kaggle environment.

### Role Coverage

Role coverage measures whether the candidate summary mentions role-specific
criteria that are also present in the source context. For example, an EMS row
with source evidence of injuries should receive more credit if the candidate
mentions injuries, casualties, triage, medical support, or patient transport.

If the source has no configured evidence for a role category, that category is
not counted against the candidate. The scorer also records diagnostic counts for
the number of applicable source-backed categories and the number covered by the
candidate.

### Urgency

Urgency measures whether urgent source evidence is reflected in the candidate.
Urgency evidence includes casualties, injuries, evacuation, trapped people,
active fire spread, hazardous materials, threat language, and other immediate
response cues.

The current implementation scores urgency with concept categories rather than
raw keyword overlap. A candidate receives credit for covering the same urgency
concepts found in the source, such as casualty/injury, rescue/evacuation, active
hazard, or severity/threat evidence. This reduces brittle mismatches like
`dead` versus `fatalities` or `evacuated` versus `evacuation`.

If no urgency evidence appears in the source context, the urgency score is
neutral rather than punitive.

## Known Limitations

- The current factuality component is a proxy and may over-penalize valid
  paraphrases.
- MiniCheck improves factual support checking, but it does not fully solve
  omission or generic-summary problems by itself.
- Keyword coverage can still miss semantically correct wording that uses
  unexpected vocabulary, although role coverage and urgency now use broader
  phrase lists and category-level diagnostics.
- The reward is designed for analysis and ranking, not as a final human-quality
  judgment.
- The target summaries are synthetic, so reward outputs should be interpreted as
  task-specific diagnostics rather than evidence of real-world operational
  readiness.
