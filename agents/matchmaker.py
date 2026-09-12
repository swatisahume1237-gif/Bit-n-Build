"""
Safety note: the hazard check below is a compatibility gate based on source-data flags only — it is not a substitute for real regulatory, transport, or handling-permit verification in an actual deployment.
Matchmaking Agent
------------------
Given normalized company profiles, finds candidate matches between a
company's waste/byproduct stream and another company's raw-material need.

Matching reasoning (in order of priority):
  1. Category match       - byproduct.category == need.category
  2. Hazard compatibility - a hazardous byproduct can only match a need
                             that explicitly accepts hazardous input
  3. Quality compatibility- need's quality_spec must be satisfied by the
                             byproduct's quality_grade (or spec is "any")
  4. Volume compatibility - flagged as partial/full fulfilment, not a hard filter,
                             since partial matches are still useful in the real world

This is intentionally rule-based rather than a black-box embedding match:
in industrial symbiosis, category and hazard compatibility are safety-critical
and must be auditable/explainable, which is why we score-and-explain rather than
opaquely rank.

Scope note (important for the pipeline story): this agent only answers
"CAN these two companies technically/safely work together?" - it does not
judge whether a match is economically or environmentally attractive. That
question ("IS this match worth pursuing?") belongs to the Logistics Agent,
which enriches these candidates with distance/cost/CO2e and produces the
viability ranking. Keeping that split explicit is what makes the two agents
separately testable and auditable.

Hazard-safety note: the hazard gate below is a hard filter based on the
`hazard_flag` / `hazard_accepted` metadata supplied in the source data. It
is NOT a claim that the system performs real regulatory, transport, or
handling-permit compliance checks - a production version would need to
verify regulatory classification, transport restrictions, facility permits,
and material safety specifications against authoritative sources.
"""

QUALITY_ORDER = {"grade_A": 3, "grade_B": 2, "grade_C": 1}


def _quality_satisfies(spec: str, grade: str) -> bool:
    if spec == "any":
        return True
    if spec == "grade_A_or_B":
        return grade in ("grade_A", "grade_B")
    return spec == grade


class MatchmakingAgent:
    def find_matches(self, companies: list[dict]) -> list[dict]:
        matches = []

        for supplier in companies:
            for byproduct in supplier["byproducts"]:
                for buyer in companies:
                    if buyer["id"] == supplier["id"]:
                        continue
                    for need in buyer["needs"]:
                        if need["category"] != byproduct["category"]:
                            continue

                        if byproduct["hazard_flag"] and not need.get("hazard_accepted", False):
                            continue  # hard gate: hazard_flag metadata vs hazard_accepted metadata only

                        if not _quality_satisfies(need["quality_spec"], byproduct["quality_grade"]):
                            continue

                        fulfilment_ratio = min(
                            byproduct["volume_tons_month"] / need["volume_tons_month"], 1.0
                        )

                        matches.append({
                            "supplier_id": supplier["id"],
                            "supplier_name": supplier["name"],
                            "supplier_city": supplier["city"],
                            "supplier_lat": supplier["lat"],
                            "supplier_lon": supplier["lon"],
                            "buyer_id": buyer["id"],
                            "buyer_name": buyer["name"],
                            "buyer_city": buyer["city"],
                            "buyer_lat": buyer["lat"],
                            "buyer_lon": buyer["lon"],
                            "category": byproduct["category"],
                            "byproduct_description": byproduct["description"],
                            "need_description": need["description"],
                            "supply_volume_tons_month": byproduct["volume_tons_month"],
                            "demand_volume_tons_month": need["volume_tons_month"],
                            "fulfilment_ratio": round(fulfilment_ratio, 2),
                            "quality_grade": byproduct["quality_grade"],
                            "hazardous": byproduct["hazard_flag"],
                            # Real negotiation constraints, carried through from source data,
                            # so the Negotiation Agent has an actual feasible bargaining space
                            # instead of improvising numbers.
                            "seller_terms": byproduct.get("commercial_terms", {}),
                            "buyer_terms": need.get("commercial_terms", {}),
                            "disposal_cost_inr_per_ton": byproduct.get("disposal_cost_inr_per_ton"),
                            "virgin_material_cost_inr_per_ton": need.get("virgin_material_cost_inr_per_ton"),
                        })

        print(f"[MatchmakingAgent] Found {len(matches)} candidate matches.")
        return matches


if __name__ == "__main__":
    import json
    from ingestion import IngestionAgent  # uses its own project-root-relative default path

    companies = IngestionAgent().load()
    matches = MatchmakingAgent().find_matches(companies)
    print(json.dumps(matches, indent=2))
