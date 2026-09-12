"""
Reporting Agent
----------------
Aggregates matches + negotiation outcomes into a single impact summary:
tons of waste diverted from landfill/disposal, net CO2e saved, and estimated
monthly economic impact across the network. This is what you show on the
final dashboard screen / end of the demo video.

Correctness rule (important): a negotiation only counts as a deal if its
status is exactly "agreed". Negotiations that failed (no overlapping
price range, turn limit hit, or malformed AGREEMENT payload) are excluded
from every downstream total - "we ran a negotiation" is not the same claim
as "we closed a deal".
"""


class ReportingAgent:
    def build_report(self, ranked_matches: list[dict], negotiations: list[dict]) -> dict:
        agreed = [n for n in negotiations if n.get("status") == "agreed"]
        failed = [n for n in negotiations if n.get("status") != "agreed"]

        matches_by_id = {
            f"{m['supplier_id']}->{m['buyer_id']}:{m['category']}": m for m in ranked_matches
        }

        total_volume = 0.0
        total_net_co2e = 0.0
        total_transport_cost = 0.0
        total_buyer_savings = 0.0
        total_supplier_savings = 0.0
        top_deals = []

        for n in agreed:
            m = matches_by_id.get(n["match_id"])
            if not m:
                continue

            volume = n.get("volume_tons_month") or min(
                m["supply_volume_tons_month"], m["demand_volume_tons_month"]
            )
            price = n.get("price_inr_per_ton")
            transport_cost = m["transport_cost_inr_per_month"]

            # --- Real "cost saved" calculation, not just transport cost ---
            virgin_cost = m.get("virgin_material_cost_inr_per_ton")
            disposal_cost = m.get("disposal_cost_inr_per_ton")

            buyer_savings = (virgin_cost - price) * volume if (virgin_cost and price is not None) else None
            supplier_savings = disposal_cost * volume if disposal_cost else None

            total_volume += volume
            total_net_co2e += m["net_co2e_saved_ton_month"]
            total_transport_cost += transport_cost
            if buyer_savings is not None:
                total_buyer_savings += buyer_savings
            if supplier_savings is not None:
                total_supplier_savings += supplier_savings

            top_deals.append({
                "route": f"{m['supplier_name']} -> {m['buyer_name']}",
                "material": m["category"],
                "status": n["status"],
                "volume_tons_month": volume,
                "price_inr_per_ton": price,
                "contract_months": n.get("contract_months"),
                "net_co2e_saved_ton_month": m["net_co2e_saved_ton_month"],
                "estimated_distance_km": m["distance_km"],
                "viability_score": m["viability_score"],
                "buyer_savings_inr_per_month": round(buyer_savings, 0) if buyer_savings is not None else None,
                "supplier_savings_inr_per_month": round(supplier_savings, 0) if supplier_savings is not None else None,
                "mode": n.get("mode"),
            })

        total_economic_benefit = total_buyer_savings + total_supplier_savings - total_transport_cost

        report = {
            "total_matches_found": len(ranked_matches),
            "negotiations_attempted": len(negotiations),
            "deals_agreed": len(agreed),
            "deals_failed": len(failed),
            "total_waste_diverted_tons_month": round(total_volume, 1),
            "total_net_co2e_saved_ton_month": round(total_net_co2e, 1),
            "total_monthly_transport_cost_inr": round(total_transport_cost, 0),
            "total_buyer_savings_inr_per_month": round(total_buyer_savings, 0),
            "total_supplier_savings_inr_per_month": round(total_supplier_savings, 0),
            "total_economic_benefit_inr_per_month": round(total_economic_benefit, 0),
            "top_deals": top_deals,
            "failed_negotiations": [
                {
                    "route": f"{n['supplier']} -> {n['buyer']}",
                    "material": n["category"],
                    "reason": n.get("failure_reason", "Unknown"),
                }
                for n in failed
            ],
            # Disclaimers surfaced verbatim on the dashboard so a judge never
            # has to ask "where did this number come from?"
            "notes": {
                "co2e": "Estimated net avoided emissions/month, based on prototype emission "
                        "assumptions (see agents/logistics.py) - not a certified carbon accounting figure.",
                "distance": "Distances are straight-line (estimated), not actual road-routing distances.",
                "data": "Synthetic demonstration data modeled on realistic industrial clusters and "
                        "common symbiosis relationships - not real current market listings.",
                "hazard": "Hazard compatibility is a hard gate based on the supplied material metadata, "
                          "not a regulatory or permit compliance check.",
                "economics": "Buyer/supplier savings are estimated against illustrative virgin-material "
                             "and disposal-cost baselines documented in data/companies.json.",
            },
        }
        print(f"[ReportingAgent] {report['deals_agreed']} deals agreed "
              f"({report['deals_failed']} failed) out of {report['negotiations_attempted']} negotiated, "
              f"{report['total_waste_diverted_tons_month']} tons/month diverted, "
              f"{report['total_net_co2e_saved_ton_month']} tCO2e/month net saved.")
        return report
