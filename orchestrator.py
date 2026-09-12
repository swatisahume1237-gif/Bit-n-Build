"""
SymbioLoop Orchestrator
------------------------
Runs the full agentic pipeline:

  Ingestion -> Matchmaking -> Logistics ranking -> Negotiation (top N) -> Reporting

This is the "planning" layer of the system: it decides how many top matches
are worth spending negotiation effort on (an LLM call per side per turn is not
free), which is itself a simple but real agentic planning decision.

Run with:  python orchestrator.py
Output written to output/results.json (used by the dashboard).
"""

import sys
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR / "agents"))

from ingestion import IngestionAgent
from matchmaker import MatchmakingAgent
from logistics import LogisticsAgent
from negotiator import NegotiationAgent
from reporter import ReportingAgent

TOP_N_TO_NEGOTIATE = 6
# Planning decision, not an arbitrary constant: an LLM call per side per turn
# costs money and latency, so the orchestrator only spends negotiation effort
# on the matches Logistics has already ranked as most viable. This is itself
# a simple but real agentic planning step -
#   11 possible matches -> Logistics filters/ranks -> top 6 economically
#   viable matches -> LLM negotiation
# rather than negotiating all candidate matches indiscriminately.


def main():
    print("=== SymbioLoop: Industrial Symbiosis Agent Network ===\n")

    ingestion = IngestionAgent(str(BASE_DIR / "data" / "companies.json"))
    companies = ingestion.load()
    companies_summary = ingestion.summarize(companies)

    raw_matches = MatchmakingAgent().find_matches(companies)
    ranked_matches = LogisticsAgent().enrich_and_rank(raw_matches)

    top_matches = ranked_matches[:TOP_N_TO_NEGOTIATE]
    print(f"\n[Orchestrator] Selecting top {TOP_N_TO_NEGOTIATE} matches for negotiation "
          f"out of {len(ranked_matches)} candidates.\n")

    negotiator = NegotiationAgent()
    negotiations = []
    for match in top_matches:
        print(f"[Orchestrator] Negotiating: {match['supplier_name']} -> {match['buyer_name']} "
              f"({match['category']})")
        negotiations.append(negotiator.negotiate(match))

    report = ReportingAgent().build_report(ranked_matches, negotiations)

    # Pipeline stage summary - drives the "Ingestion -> Matchmaking ->
    # Logistics -> Negotiation -> Reporting" progress bar on the dashboard so
    # the five-agent architecture is visible at a glance, not just implied.
    pipeline = [
        {"stage": "Ingestion", "detail": f"{companies_summary['num_companies']} companies"},
        {"stage": "Matchmaking", "detail": f"{len(raw_matches)} compatible"},
        {"stage": "Logistics", "detail": f"Top {len(top_matches)} ranked"},
        {"stage": "Negotiation", "detail": f"{report['deals_agreed']} agreements"},
        {"stage": "Reporting", "detail": f"{report['total_waste_diverted_tons_month']} t diverted"},
    ]

    output = {
        "companies_summary": companies_summary,
        "companies": companies,  # full profiles, used by the dashboard's "Run Simulation" panel
        "matches": ranked_matches,
        "negotiations": negotiations,
        "report": report,
        "pipeline": pipeline,
        "top_n_to_negotiate": TOP_N_TO_NEGOTIATE,
    }

    (BASE_DIR / "output").mkdir(exist_ok=True)
    (BASE_DIR / "output" / "results.json").write_text(json.dumps(output, indent=2))
    (BASE_DIR / "dashboard" / "data.json").write_text(json.dumps(output, indent=2))

    print("\n[Orchestrator] Done. Results written to output/results.json and dashboard/data.json")


if __name__ == "__main__":
    main()
