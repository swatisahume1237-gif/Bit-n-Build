# SymbioLoop — Autonomous Industrial Symbiosis Network

Multi-agent system that discovers, evaluates, and negotiates B2B deals that route
one factory's manufacturing byproduct into another factory's production pipeline
as raw material — reducing landfill waste, virgin material demand, and CO2e emissions.

## Problem

Manufacturers generate byproducts (fly ash, slag, sawdust, spent grain, foundry sand,
chemical sludge, etc.) that are frequently landfilled or incinerated, while nearby
factories buy virgin equivalents of the same material. Industrial symbiosis (the
Kalundborg model) shows this waste-to-input routing is highly profitable and
sustainable at scale — but discovering compatible partners, checking safety/quality
compatibility, negotiating terms, and estimating logistics is a slow, manual,
relationship-driven process today. SymbioLoop automates the full pipeline with
autonomous agents.

## Architecture

```
data/companies.json
        |
        v
[1] Ingestion Agent      -> loads & validates company byproduct/need profiles
        |
        v
[2] Matchmaking Agent    -> rule-based matching: category + hazard safety + quality
        |
        v
[3] Logistics Agent      -> haversine distance, transport cost/emissions, viability score, ranking
        |
        v
[4] Negotiation Agent    -> two LLM personas (supplier/buyer) autonomously negotiate
        |                    within each side's real commercial constraints (price floor/
        |                    ceiling, volume, contract length, transport-cost share),
        |                    over multiple rounds, and only reach a deal if a proposed
        |                    AGREEMENT is (a) valid structured JSON, (b) re-validated
        |                    against BOTH sides' hard constraints, and (c) explicitly
        |                    CONFIRMED by the counterparty — not just declared by one side.
        |                    Checks long-term memory first for a still-active prior deal
        |                    between the same pair before negotiating from scratch.
        v
[5] Reporting Agent      -> aggregates into a sustainability impact report
        |
        v
dashboard/index.html     -> map + deal list + negotiation transcripts + impact stats
```

`orchestrator.py` ties the five agents together and makes one explicit planning
decision: only the top-N highest-viability matches are sent to the (costly) LLM
negotiation step, rather than negotiating every candidate pair.

## Why this is "agentic" (not just an LLM call)

- **Multi-step planning**: orchestrator selects which matches are worth negotiating
  based on the logistics agent's ranking, before spending any LLM budget.
- **Tool use**: agents call real computation (haversine distance, cost/emissions
  formulas, transport-split optimization) that the LLM does not do itself — the LLM
  only reasons and bargains over agent-supplied constraints.
- **Multi-turn autonomous negotiation within real constraints**: two independent LLM
  personas, each seeing only its own side's price floor/ceiling, volume, and contract
  preferences (never the other side's), negotiate to a proposal — with no human in
  the loop.
- **Deals are earned, not assumed**: an `AGREEMENT:` from one side is only a proposal.
  It's re-validated against both sides' hard constraints and requires an explicit
  `CONFIRMED` from the counterparty before it's recorded as a real deal. If the two
  sides' ranges genuinely don't overlap, the negotiation is allowed to fail —
  see `output/negotiation_log.json` for a real example (fly_ash: seller minimum
  ₹1,600/ton vs buyer maximum ₹1,472/ton — correctly negotiated to `NO_DEAL` by both
  the scripted fallback and the live LLM negotiation, independently, over 4 full
  rounds).
- **Long-term memory**: every completed negotiation is persisted to
  `output/negotiation_log.json`. Before negotiating a pair again, the agent checks
  for a still-active prior agreement (tracked via `contract_months` -> `expires_at`)
  and reuses it instead of renegotiating from scratch — and also passes the most
  recent historical price for either party as an anchor hint on a fresh negotiation,
  so the system genuinely uses its own history rather than just archiving it.
- **Explainability / safety**: hazard compatibility (matchmaker.py) is enforced as a
  hard rule before any LLM step — the system never reasons its way into an unsafe
  match.
- **Reliability under real API conditions**: the negotiation agent retries on
  transient LLM provider errors and degrades to a clearly-labeled failed record
  (rather than crashing the whole pipeline run) if a provider error persists — so one
  bad API call doesn't take down every other match's negotiation.

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file in the project root (this file is gitignored — never commit it):
```
GROQ_API_KEY=your_groq_api_key_here
```

Then run:
```bash
python orchestrator.py
```

Open the dashboard via a local server (needed for the JSON fetch):
```bash
cd dashboard
python -m http.server 8000
# visit http://localhost:8000
```

### Demo mode (no API key required)

If `GROQ_API_KEY` is not set, `negotiator.py` automatically falls back to a
deterministic negotiation that still respects each side's real price/volume/duration
constraints (not a single hard-coded price) — so the pipeline and dashboard still run
end-to-end, and can still genuinely fail if a pair's constraints don't overlap. This
is a deliberate reliability choice: don't rely on live network/API calls working
flawlessly during a live demo.

### Live LLM negotiation

Get a key from https://console.groq.com. The negotiation agent uses Groq's hosted
`openai/gpt-oss-20b` (OpenAI's open-weight reasoning model) via the official `groq`
Python SDK. Note: **Groq** (the inference API company, groq.com) is a different
company from xAI's **Grok** model — same-sounding name, unrelated API.

`gpt-oss-20b` is a reasoning model — it spends part of its token budget on hidden
internal reasoning before producing its visible reply. We explicitly set
`reasoning_effort="low"` and a generous `max_tokens` to avoid truncating the model's
response mid-JSON, which we hit during development (see commit history).

**Known provider-side issue**: Groq's `gpt-oss` models intermittently throw
`"Tool choice is none, but model called a tool"` even when no tools are configured —
a documented, acknowledged bug on Groq's side, not something in this codebase. The
negotiation agent retries transient LLM errors automatically and records a graceful
failure (rather than crashing) if the error persists past the retry budget.

## Data

`data/companies.json` is a synthetic but realistic dataset modeled on real Karnataka/
Tamil Nadu industrial clusters and known real-world symbiosis pairs (fly ash → cement,
bagasse → biomass power, foundry sand → road construction, spent grain → animal feed,
FGD gypsum → plasterboard, chrome sludge → specialized hazardous processing). No public
real-time API exists for industrial byproduct data, so this is hand-built for the
prototype — flagged transparently rather than presented as live data.

## Assumptions to state clearly in the demo

- Transport cost (₹3.5/ton-km) and emissions factors (0.062 kg CO2/ton-km truck,
  0.45 tCO2e/ton landfill diversion saving) are illustrative industry-average
  estimates, not verified logistics data — swap in a freight-rate/emissions API
  for production use.
- Matching is rule-based (category + hazard + quality) rather than semantic/embedding
  based, by design: safety-critical routing needs to be auditable, not a black box.
- Negotiation outcomes are genuinely earned within each side's stated commercial
  constraints, not guaranteed — some matches will correctly fail if price ranges
  don't overlap, which the demo should show as a feature, not hide as a failure.

## Next steps if time allows

- Swap static dataset for a CSV upload flow so any company can add its own profile.
- Real freight-cost API integration.
- Surface memory-reuse events (existing active contracts) in the dashboard.
