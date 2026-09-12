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
        |                    price, volume, delivery terms over multiple turns;
        |                    persists every negotiation to output/negotiation_log.json (memory)
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
  formulas) that the LLM does not do itself — the LLM only reasons over agent output.
- **Multi-turn autonomous negotiation**: two independent LLM personas with different,
  partially conflicting goals and no visibility into each other's system prompt
  negotiate to a structured agreement, without a human in the loop.
- **Memory**: every negotiation is appended to `output/negotiation_log.json`,
  giving the system a persistent record it could query in future runs
  (e.g. "don't re-negotiate a route we already have a live 12-month deal on").
- **Explainability / safety**: hazard compatibility is enforced as a hard rule
  before any LLM step — the system never reasons its way into an unsafe match.

## Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your_key_here   # optional — see below
python orchestrator.py
```

Then open `dashboard/index.html` via a local server (needed for the JSON fetch):

```bash
cd dashboard
python -m http.server 8000
# visit http://localhost:8000
```

### Demo mode (no API key required)

If `ANTHROPIC_API_KEY` is not set, `negotiator.py` automatically falls back to a
deterministic scripted negotiation so the full pipeline and dashboard still run
end-to-end. This is a deliberate reliability choice — do not rely on live network/API
calls working flawlessly during a live demo.

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

## Next steps if time allows

- Swap static dataset for a CSV upload flow so any company can add its own profile.
- Add a "renegotiate" trigger reading from `negotiation_log.json` memory.
- Real freight-cost API integration.
