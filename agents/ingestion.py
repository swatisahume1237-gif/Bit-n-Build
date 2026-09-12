"""
Ingestion Agent
----------------
Responsible for loading raw industrial company data (byproducts + material needs)
and normalizing it into a consistent in-memory structure for downstream agents.

In a real deployment this agent would pull from ERP systems, waste manifests,
or IoT sensor feeds. For this prototype it reads a structured JSON file that
mimics a municipal/industrial-association data dump.
"""

import json
from pathlib import Path

# Resolve the default path relative to the project root (this file's
# grandparent directory), not the current working directory, so this agent
# behaves the same whether invoked as `python orchestrator.py` from the
# project root or `python agents/ingestion.py` from anywhere else.
_BASE_DIR = Path(__file__).resolve().parent.parent
_DEFAULT_DATA_PATH = _BASE_DIR / "data" / "companies.json"


class IngestionAgent:
    def __init__(self, data_path: str = None):
        self.data_path = Path(data_path) if data_path else _DEFAULT_DATA_PATH

    def load(self) -> list[dict]:
        """Load and validate company records. Raises on malformed data
        rather than silently dropping records, since bad data here would
        corrupt every downstream match."""
        with open(self.data_path, "r", encoding="utf-8") as f:
            companies = json.load(f)

        for c in companies:
            required = {"id", "name", "industry", "city", "lat", "lon", "byproducts", "needs"}
            missing = required - c.keys()
            if missing:
                raise ValueError(f"Company {c.get('id', '?')} missing fields: {missing}")

        print(f"[IngestionAgent] Loaded {len(companies)} company profiles.")
        return companies

    def summarize(self, companies: list[dict]) -> dict:
        """Quick stats used for the impact report and sanity-checking the run."""
        total_byproduct_streams = sum(len(c["byproducts"]) for c in companies)
        total_need_streams = sum(len(c["needs"]) for c in companies)
        return {
            "num_companies": len(companies),
            "byproduct_streams": total_byproduct_streams,
            "need_streams": total_need_streams,
        }


if __name__ == "__main__":
    agent = IngestionAgent()  # uses _DEFAULT_DATA_PATH, resolved from project root
    data = agent.load()
    print(agent.summarize(data))
