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
  `dead`, `death toll`, `killed`
- `medical`, `medical support`, `medical emergencies`, `patient`, `triage`,
  `transport`, `patient transport`
- `urgent`, `urgency`, `critical`, `vulnerable`, `rescue`, `trapped`
- `ambulance`, `hospital`, `MMU`, `CERT`

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
- `hazardous`, `hazardous material`, `hazmat`, `material exposure`,
  `chemical`, `gas`
- `structural`, `collapse`, `trapped`, `search rescue`, `rescue operations`
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
- `crowd`, `crowd control`, `scene security`
- `evacuation`, `evacuate`, `evacuations`, `access`, `traffic`, `road closure`,
  `blocked roads`
- `criminal activity`, `arrest`, `arrested`, `investigate`, `probe`, `unrest`,
  `protest`
- `DCC`, `dispatch`, `TEU`

## Component Definitions

### Relevance

Relevance measures whether the candidate summary is semantically related to the
source tweet and available context. The preferred implementation uses
SentenceTransformer cosine similarity between:

```text
candidate summary
source tweet/context
```

Cosine similarity is normalized to the 0.0 to 1.0 range. A lexical fallback may
be used for smoke tests or environments without model access.

### Factuality

The first implementation uses a transparent source-grounding proxy. It rewards
candidate summaries whose content words, numbers, and compact identifiers are
supported by the source context.

This is not a full MiniFactScore/MiniCheck replacement. A learned factuality
checker should replace or augment this proxy once the model dependency is stable
in the Kaggle environment.

### Role Coverage

Role coverage measures whether the candidate summary mentions role-specific
criteria that are also present in the source context. For example, an EMS row
with source evidence of injuries should receive more credit if the candidate
mentions injuries, casualties, triage, medical support, or patient transport.

If the source has no configured evidence for a role category, that category is
not counted against the candidate.

### Urgency

Urgency measures whether urgent source evidence is reflected in the candidate.
Urgency evidence includes casualties, injuries, evacuation, trapped people,
active fire spread, hazardous materials, threat language, and other immediate
response cues.

If no urgency evidence appears in the source context, the urgency score is
neutral rather than punitive.

## Known Limitations

- The current factuality component is a proxy and may over-penalize valid
  paraphrases.
- Keyword coverage can miss semantically correct wording that uses unexpected
  vocabulary.
- The reward is designed for analysis and ranking, not as a final human-quality
  judgment.
- The target summaries are synthetic, so reward outputs should be interpreted as
  task-specific diagnostics rather than evidence of real-world operational
  readiness.
