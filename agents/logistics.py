"""
Logistics Agent
----------------
Enriches candidate matches with distance, estimated transport cost, and
estimated CO2e impact, then ranks matches so the negotiation agent only
spends effort on the most viable pairs.

Scope note: this agent answers "IS this technically-valid match economically
and environmentally attractive?" - separate from the Matchmaking Agent's
"CAN these two companies work together at all?" question.

Cost/emission constants below are illustrative industry-average estimates
for a hackathon prototype, NOT verified logistics data. Documented here for
transparency - swap in real freight/emissions APIs for a production version.

Distance note: `distance_km` is straight-line (Haversine) geographic
distance, NOT an actual road/routing distance - a real deployment would
call a road-routing API. We surface it in the UI as "estimated distance"
to avoid implying it's a driving route.
"""

import math

TRUCK_COST_PER_TON_KM_INR = 3.5          # approx domestic road freight rate
TRUCK_EMISSION_KG_CO2_PER_TON_KM = 0.062  # approx diesel truck freight factor
LANDFILL_DIVERSION_SAVING_TON_CO2E = 0.45  # avoided landfill/incineration emissions per ton diverted

# Weights for the normalized 0-100 viability score. Tuned for demo
# interpretability over precision - documented so a judge can see exactly
# how the score is composed rather than a single opaque number.
VIABILITY_WEIGHTS = {
    "fulfilment": 0.40,   # can the buyer's need actually be met
    "logistics": 0.20,    # closer routes score higher (proxy for cost/feasibility)
    "co2_benefit": 0.15,  # net emissions benefit after transport
    "match_quality": 0.25,  # quality-grade headroom above what the buyer required
}


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


QUALITY_HEADROOM = {"grade_A": 1.0, "grade_B": 0.66, "grade_C": 0.33}
MAX_REASONABLE_DISTANCE_KM = 500  # beyond this, logistics score bottoms out at 0


class LogisticsAgent:
    def enrich_and_rank(self, matches: list[dict]) -> list[dict]:
        for m in matches:
            distance_km = haversine_km(
                m["supplier_lat"], m["supplier_lon"], m["buyer_lat"], m["buyer_lon"]
            )
            volume = min(m["supply_volume_tons_month"], m["demand_volume_tons_month"])

            transport_cost_inr = distance_km * volume * TRUCK_COST_PER_TON_KM_INR
            transport_emissions_ton_co2e = (
                distance_km * volume * TRUCK_EMISSION_KG_CO2_PER_TON_KM
            ) / 1000

            diversion_saving_ton_co2e = volume * LANDFILL_DIVERSION_SAVING_TON_CO2E
            net_co2e_saved = diversion_saving_ton_co2e - transport_emissions_ton_co2e

            m["distance_km"] = round(distance_km, 1)
            m["transport_cost_inr_per_month"] = round(transport_cost_inr, 0)
            m["transport_emissions_ton_co2e_month"] = round(transport_emissions_ton_co2e, 2)
            m["net_co2e_saved_ton_month"] = round(net_co2e_saved, 2)

            # --- Normalized 0-100 viability score ---
            # Each component is scaled to 0-1 first, so the final number reads
            # like a percentage/grade rather than an unbounded raw figure.
            fulfilment_component = m["fulfilment_ratio"]  # already 0-1

            logistics_component = max(0.0, 1 - (distance_km / MAX_REASONABLE_DISTANCE_KM))

            gross_possible = diversion_saving_ton_co2e or 1  # avoid div-by-zero
            co2_component = max(0.0, min(1.0, net_co2e_saved / gross_possible))

            match_quality_component = QUALITY_HEADROOM.get(m["quality_grade"], 0.5)

            score_0_1 = (
                fulfilment_component * VIABILITY_WEIGHTS["fulfilment"]
                + logistics_component * VIABILITY_WEIGHTS["logistics"]
                + co2_component * VIABILITY_WEIGHTS["co2_benefit"]
                + match_quality_component * VIABILITY_WEIGHTS["match_quality"]
            )

            m["viability_score"] = round(score_0_1 * 100, 1)
            m["viability_breakdown"] = {
                "fulfilment_pct": round(fulfilment_component * 100, 1),
                "logistics_pct": round(logistics_component * 100, 1),
                "co2_benefit_pct": round(co2_component * 100, 1),
                "match_quality_pct": round(match_quality_component * 100, 1),
            }

        ranked = sorted(matches, key=lambda x: x["viability_score"], reverse=True)
        print(f"[LogisticsAgent] Ranked {len(ranked)} matches by viability score (0-100).")
        return ranked


if __name__ == "__main__":
    import json
    from ingestion import IngestionAgent  # uses its own project-root-relative default path
    from matchmaker import MatchmakingAgent

    companies = IngestionAgent().load()
    matches = MatchmakingAgent().find_matches(companies)
    ranked = LogisticsAgent().enrich_and_rank(matches)
    print(json.dumps(ranked[:3], indent=2))
