"""
Negotiation Agent
------------------
This is the core "agentic" showcase: two LLM-driven personas (Supplier agent
and Buyer agent) negotiate the terms of a byproduct-to-raw-material deal
autonomously, over multiple rounds, within a real bargaining space defined by
each side's commercial constraints, and converge on a STRUCTURED agreement
(not a sentence of prose).

Design notes:
- Each persona is given ONLY its own company's context (its constraints,
  costs, minimum/maximum acceptable terms) - neither sees the other's system
  prompt. This mirrors real bilateral negotiation and avoids the LLM
  "negotiating with itself" in a way that trivializes the task.
- Both sides negotiate inside an actual feasible space: seller minimum price,
  buyer maximum price, minimum/required volume, preferred contract length,
  and max transport-cost share, all sourced from the source company data
  (see agents/matchmaker.py). If the ranges genuinely don't overlap, the
  negotiation is allowed to fail (status: "failed") rather than always
  "succeeding" - a scripted-always-agrees negotiation isn't a negotiation.
- A proposed "AGREEMENT:" is only a PROPOSAL. It only becomes a deal once
  (a) it is parsed as valid structured JSON, (b) it is re-validated against
  BOTH sides' hard constraints (price/volume/duration/transport-split), and
  (c) the counterparty explicitly confirms those exact terms with a
  "CONFIRMED" reply. One side declaring "AGREEMENT" is never sufficient by
  itself - see `validate_agreement` and `_seek_confirmation` below.
- Conversation state is passed explicitly turn-by-turn (working memory for
  THIS negotiation).
- LONG-TERM MEMORY: before negotiating, this agent searches
  output/negotiation_log.json for a previous agreed deal between the same
  supplier/buyer/category that has not yet expired (see `contract_months`
  -> `expires_at`). If one is still active, it reuses those terms without
  creating a new fake negotiation record in the log (reuse is reported back
  to the caller/dashboard, but is not itself persisted as a new history
  entry - otherwise the log would grow forever on repeated runs). If no
  active agreement exists, it still passes the most recent past price for
  this supplier or buyer as an anchor hint, so returning parties don't start
  from zero. This is what actually makes it "memory" rather than just an
  append-only log.
- Every NEW negotiation (live or scripted) is persisted to
  output/negotiation_log.json. Memory-reuse events are not.
- If no API key is available, DEMO_MODE produces a deterministic negotiation
  that still respects each side's real constraints (not a fixed price per
  hazard flag) so the pipeline (and your demo) never breaks mid-presentation,
  and still visibly varies deal-to-deal.
"""

import os
import re
import json
from pathlib import Path
from datetime import datetime, timezone, timedelta

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

DEMO_MODE = os.environ.get("GROQ_API_KEY") is None
MODEL = os.environ.get("SYMBIOLOOP_MODEL", "openai/gpt-oss-20b")
# Each "round" is one supplier turn + one buyer turn (up to 2 LLM calls), so
# MAX_ROUNDS=4 means up to 8 LLM calls total, not 4. Named MAX_ROUNDS (rather
# than MAX_TURNS) precisely to avoid that ambiguity.
MAX_ROUNDS = 4

# Resolve paths relative to the project root (this file's grandparent
# directory), not the current working directory, so `python orchestrator.py`
# and `python agents/negotiator.py` both work regardless of where they're
# invoked from.
BASE_DIR = Path(__file__).resolve().parent.parent
LOG_PATH = BASE_DIR / "output" / "negotiation_log.json"

DEFAULT_SELLER_TERMS = {
    "minimum_price_inr_per_ton": 800,
    "preferred_price_inr_per_ton": 1400,
    "minimum_volume_tons_month": 0,
    "preferred_contract_months": 12,
    "max_transport_share": 0.5,
}
DEFAULT_BUYER_TERMS = {
    "maximum_price_inr_per_ton": 1600,
    "required_volume_tons_month": 0,
    "preferred_contract_months": 12,
    "max_transport_share": 0.5,
}

# Approximate a month as 30 days for contract-expiry math. This is a
# documented simplification for the prototype, not calendar-accurate billing.
_DAYS_PER_MONTH = 30


def _client():
    return Groq(api_key=os.environ["GROQ_API_KEY"])


def _match_id(match: dict) -> str:
    return f"{match['supplier_id']}->{match['buyer_id']}:{match['category']}"


def _load_memory() -> list[dict]:
    if LOG_PATH.exists():
        try:
            return json.loads(LOG_PATH.read_text())
        except json.JSONDecodeError:
            return []
    return []


class NegotiationMemory:
    """Thin retrieval layer over past negotiation outcomes. This is what turns
    the append-only log into something the agent actually *uses*, rather than
    just archives."""

    def __init__(self, history: list[dict]):
        self.history = history

    def existing_agreement(self, match: dict):
        """The most recent agreed deal for this exact supplier/buyer/category
        pair, IF it hasn't expired yet (based on the contract's own
        `contract_months` duration, stored as `expires_at`). An expired
        agreement is treated as lapsed, not permanently active - the
        relationship must be renegotiated. Legacy records with no tracked
        expiry (from before this fix) are also treated as lapsed rather than
        assumed to still be active forever."""
        mid = _match_id(match)
        record = next(
            (r for r in reversed(self.history)
             if r.get("match_id") == mid and r.get("status") == "agreed"),
            None,
        )
        if not record:
            return None

        expires_at = record.get("expires_at")
        if not expires_at:
            return None  # no tracked expiry -> don't assume forever-active

        try:
            expiry = datetime.fromisoformat(expires_at)
        except ValueError:
            return None

        if datetime.now(timezone.utc) >= expiry:
            return None  # contract term has lapsed - must renegotiate

        return record

    def anchor_price_hint(self, match: dict):
        """Most recent agreed price involving either party for this category,
        used to anchor a NEW negotiation (different counterparties) instead of
        starting from zero context."""
        candidates = [
            r for r in reversed(self.history)
            if r.get("status") == "agreed"
            and r.get("category") == match["category"]
            and (r.get("supplier") == match["supplier_name"] or r.get("buyer") == match["buyer_name"])
        ]
        if candidates:
            return candidates[0].get("price_inr_per_ton")
        return None


def compute_transport_split(seller_terms: dict, buyer_terms: dict) -> str:
    """Pick an explicit, defensible transport-cost split, rather than
    mechanically taking min(seller_max, buyer_max) as the seller's share
    (which produces results like 80/20 with no justification other than
    "that's the bigger of the two caps").

    Preference order:
      1. A standard 50/50 split, if both sides' maximum transport-cost share
         can accommodate it.
      2. Otherwise, the midpoint of the band of splits that satisfy BOTH
         sides' caps simultaneously (seller_share <= seller_max AND
         buyer_share <= buyer_max, i.e. seller_share >= 100 - buyer_max).
      3. If the caps don't leave any feasible band at all (rare edge case:
         seller_max + buyer_max < 100%), fall back to a best-effort midpoint -
         this should be treated as a negotiation blocker by
         validate_agreement's transport-split check.
    """
    seller_max_pct = seller_terms["max_transport_share"] * 100
    buyer_max_pct = buyer_terms["max_transport_share"] * 100

    lower_bound = max(0.0, 100 - buyer_max_pct)  # min seller share buyer's cap allows
    upper_bound = min(100.0, seller_max_pct)     # max seller share seller's cap allows

    if lower_bound <= 50.0 <= upper_bound:
        seller_share = 50.0
    elif lower_bound <= upper_bound:
        seller_share = (lower_bound + upper_bound) / 2.0
    else:
        seller_share = (seller_max_pct + (100 - buyer_max_pct)) / 2.0

    seller_share_int = max(0, min(100, int(round(seller_share))))
    return f"{seller_share_int}/{100 - seller_share_int}"


def validate_agreement(data: dict, match: dict, seller_terms: dict, buyer_terms: dict):
    """Re-validate a parsed AGREEMENT payload against BOTH sides' hard
    constraints. Presence of the required keys is not enough - an LLM (or a
    malformed scripted computation) could produce a structurally valid JSON
    object with values that violate price, volume, duration, or transport
    limits. Returns (True, None) if valid, else (False, reason)."""
    try:
        price = float(data["price_inr_per_ton"])
        volume = float(data["volume_tons_month"])
        months = int(data["contract_months"])
    except (KeyError, TypeError, ValueError):
        return False, "Agreement fields are missing or not numeric."

    delivery_terms = data.get("delivery_terms")
    if not isinstance(delivery_terms, str) or not delivery_terms.strip():
        return False, "Delivery terms must be a non-empty string."

    if price <= 0:
        return False, "Price must be positive."
    if price < seller_terms["minimum_price_inr_per_ton"]:
        return False, "Price is below the seller's minimum acceptable price."
    if price > buyer_terms["maximum_price_inr_per_ton"]:
        return False, "Price is above the buyer's maximum acceptable price."

    max_volume = min(match["supply_volume_tons_month"], match["demand_volume_tons_month"])
    if volume <= 0 or volume > max_volume:
        return False, "Volume is outside the physically available supply/demand range."
    if volume < seller_terms["minimum_volume_tons_month"]:
        return False, "Volume is below the seller's minimum committed volume."
    if volume < buyer_terms["required_volume_tons_month"]:
        return False, "Volume is below the buyer's required volume."

    if months <= 0:
        return False, "Contract duration must be a positive number of months."

    transport_split = data.get("transport_split")
    if not isinstance(transport_split, str) or "/" not in transport_split:
        return False, "Transport split must be a 'seller/buyer' percentage string."
    try:
        seller_pct_str, buyer_pct_str = transport_split.split("/", 1)
        seller_pct, buyer_pct = float(seller_pct_str), float(buyer_pct_str)
    except ValueError:
        return False, "Transport split is not a valid 'seller/buyer' percentage pair."
    if abs((seller_pct + buyer_pct) - 100) > 0.5:
        return False, "Transport split percentages must sum to 100."
    if seller_pct > seller_terms["max_transport_share"] * 100 + 0.5:
        return False, "Transport split exceeds the seller's maximum transport-cost share."
    if buyer_pct > buyer_terms["max_transport_share"] * 100 + 0.5:
        return False, "Transport split exceeds the buyer's maximum transport-cost share."

    return True, None


def _supplier_system_prompt(match: dict, seller_terms: dict, anchor_hint, preferred_split: str) -> str:
    hint_line = (
        f"\nYou previously agreed a similar deal around INR {anchor_hint}/ton - use that as context, "
        f"but you may adjust it." if anchor_hint else ""
    )
    return f"""You are the negotiation agent representing {match['supplier_name']}, a supplier of
{match['byproduct_description']} ({match['category']}).
You have {match['supply_volume_tons_month']} tons/month available.
Your hard constraints (do not violate these):
- Minimum acceptable price: INR {seller_terms['minimum_price_inr_per_ton']}/ton
- Preferred price: INR {seller_terms['preferred_price_inr_per_ton']}/ton
- Minimum volume you will commit to: {seller_terms['minimum_volume_tons_month']} tons/month
- Preferred contract length: {seller_terms['preferred_contract_months']} months
- Maximum share of transport cost you'll accept: {int(seller_terms['max_transport_share']*100)}%
- Standard transport-cost split for this route: {preferred_split} (seller/buyer) - you may deviate
  only within your own maximum share limit above.{hint_line}
Your goals, in priority order:
1. Secure a recurring monthly offtake agreement (avoid one-off deals).
2. Get as close to your preferred price as possible - never accept below your minimum.
3. Minimise your logistics burden - prefer buyer picks up or shares transport cost.
Negotiate directly and concisely (2-4 sentences per turn).
If you and the buyer reach agreement, end your final message with a line starting exactly with
"AGREEMENT:" followed by ONLY a compact JSON object with keys: price_inr_per_ton (number),
volume_tons_month (number), contract_months (number), delivery_terms (string), transport_split (string).
This is a PROPOSAL, not a done deal - the counterparty must separately confirm these exact terms
before it is final, so only propose terms you are genuinely willing to have accepted as-is.
If after several turns no deal within your constraints is possible, end your final message with a line
starting exactly with "NO_DEAL:" followed by a one-sentence reason."""


def _buyer_system_prompt(match: dict, buyer_terms: dict, anchor_hint, preferred_split: str) -> str:
    hint_line = (
        f"\nA similar deal was previously agreed around INR {anchor_hint}/ton - use that as context, "
        f"but you may push for better." if anchor_hint else ""
    )
    return f"""You are the negotiation agent representing {match['buyer_name']}, which needs
{match['need_description']} ({match['category']}) as an input material.
Your hard constraints (do not violate these):
- Maximum acceptable price: INR {buyer_terms['maximum_price_inr_per_ton']}/ton
- Required volume: {buyer_terms['required_volume_tons_month']} tons/month
- Preferred contract length: {buyer_terms['preferred_contract_months']} months
- Maximum share of transport cost you'll accept: {int(buyer_terms['max_transport_share']*100)}%
- Standard transport-cost split for this route: {preferred_split} (seller/buyer) - you may deviate
  only within your own maximum share limit above.{hint_line}
Your goals, in priority order:
1. Lock in a reliable recurring supply to replace virgin raw material purchases.
2. Negotiate as far below your maximum price as possible.
3. Keep transport distance/cost manageable (this route is {match['distance_km']} km).
Negotiate directly and concisely (2-4 sentences per turn).
If you and the supplier reach agreement, end your final message with a line starting exactly with
"AGREEMENT:" followed by ONLY a compact JSON object with keys: price_inr_per_ton (number),
volume_tons_month (number), contract_months (number), delivery_terms (string), transport_split (string).
This is a PROPOSAL, not a done deal - the counterparty must separately confirm these exact terms
before it is final, so only propose terms you are genuinely willing to have accepted as-is.
If after several turns no deal within your constraints is possible, end your final message with a line
starting exactly with "NO_DEAL:" followed by a one-sentence reason."""


_CONFIRMATION_PROMPT_TEMPLATE = """The other party has proposed the following agreement:
{proposal_json}
If you accept these EXACT terms, respond with only the single word: CONFIRMED
If you do not accept them, respond with a line starting exactly with "NO_DEAL:" followed by a
one-sentence reason. Do not propose new counter-terms here - a plain rejection ends this
negotiation attempt."""


def _call_llm(client, system_prompt: str, history: list[dict]) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        *history,
    ]

    resp = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        max_tokens=300,
        temperature=0.3,
    )

    return resp.choices[0].message.content.strip()


def _parse_agreement_json(raw: str):
    """Extract and validate the structural shape of the JSON payload that
    follows 'AGREEMENT:'. Returns None (never raises) if it doesn't parse -
    callers must treat that as a failed negotiation rather than guessing at
    terms.

    Uses a real JSON decoder anchored at the first '{', rather than a greedy
    `\\{.*\\}` regex, so trailing prose after a well-formed object (or a
    second JSON-looking block later in the message) can never get folded
    into the parsed result."""
    raw = raw.strip()
    start = raw.find("{")
    if start == -1:
        return None
    decoder = json.JSONDecoder()
    try:
        data, _ = decoder.raw_decode(raw[start:])
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    required = {"price_inr_per_ton", "volume_tons_month", "contract_months",
                "delivery_terms", "transport_split"}
    if not required.issubset(data.keys()):
        return None
    return data


def _seek_confirmation(client, counterpart_sys: str, counterpart_history: list[dict],
                        proposal: dict, transcript: list[dict], proposer_speaker: str):
    """Ask the counterpart to explicitly confirm a proposed AGREEMENT. Returns
    (confirmed: bool, failure_reason: str | None). A proposal is never
    self-confirming - only an explicit "CONFIRMED" from the OTHER side (not
    the one that proposed it) finalizes a deal."""
    counterpart_speaker = "buyer" if proposer_speaker == "supplier" else "supplier"
    confirmation_prompt = _CONFIRMATION_PROMPT_TEMPLATE.format(
        proposal_json=json.dumps(proposal)
    )
    counterpart_history.append({"role": "user", "content": confirmation_prompt})
    reply = _call_llm(client, counterpart_sys, counterpart_history)
    counterpart_history.append({"role": "assistant", "content": reply})
    transcript.append({"speaker": counterpart_speaker, "text": reply})

    if re.match(r"^\s*CONFIRMED\b", reply, re.IGNORECASE):
        return True, None
    if "NO_DEAL:" in reply:
        return False, reply.split("NO_DEAL:", 1)[1].strip()
    return False, "Counterparty did not explicitly confirm the proposed terms."


def _scripted_negotiation(match: dict, seller_terms: dict, buyer_terms: dict, anchor_hint) -> dict:
    """Deterministic fallback so the demo works with zero API dependency -
    but it still negotiates within each side's REAL constraints instead of a
    single hard-coded price, and can genuinely fail if ranges don't overlap
    OR if the constructed terms don't survive the same validation the live
    LLM path is held to."""
    seller_min = seller_terms["minimum_price_inr_per_ton"]
    seller_pref = seller_terms["preferred_price_inr_per_ton"]
    buyer_max = buyer_terms["maximum_price_inr_per_ton"]
    volume = min(match["supply_volume_tons_month"], match["demand_volume_tons_month"])
    contract_months = min(seller_terms["preferred_contract_months"], buyer_terms["preferred_contract_months"])
    transport_split = compute_transport_split(seller_terms, buyer_terms)

    if seller_min > buyer_max:
        # No feasible overlap - an honest failure, not a forced deal.
        transcript = [
            {"speaker": "supplier", "text": f"Our minimum for {match['category']} is INR {seller_min}/ton "
                                             f"given our disposal-avoidance economics."},
            {"speaker": "buyer", "text": f"We can't go above INR {buyer_max}/ton against our virgin-material "
                                          f"benchmark. That gap doesn't close for us."},
            {"speaker": "supplier", "text": f"Understood - INR {buyer_max}/ton is below what we can accept. "
                                             f"NO_DEAL: price ranges do not overlap."},
        ]
        return {
            "transcript": transcript,
            "status": "failed",
            "failure_reason": f"Seller minimum (INR {seller_min}/ton) exceeds buyer maximum (INR {buyer_max}/ton).",
            "mode": "scripted_fallback",
        }

    # Opening offers bracket the feasible range; settle near the midpoint,
    # nudged toward the seller's preferred price when the range comfortably
    # allows it, and pulled toward any historical anchor price.
    midpoint = (seller_min + buyer_max) / 2
    settle = min(buyer_max, max(seller_min, (midpoint + seller_pref) / 2))
    if anchor_hint:
        settle = (settle + anchor_hint) / 2
    settle = int(round(settle / 10.0)) * 10  # round to nearest 10 for readability
    settle = max(seller_min, min(buyer_max, settle))

    opening_supplier = int(round(min(seller_pref, buyer_max * 1.1)))
    opening_buyer = int(round(max(seller_min * 0.85, buyer_max * 0.8)))

    proposal = {
        "price_inr_per_ton": settle,
        "volume_tons_month": volume,
        "contract_months": contract_months,
        "delivery_terms": "supplier site",
        "transport_split": transport_split,
    }
    valid, reason = validate_agreement(proposal, match, seller_terms, buyer_terms)

    transcript = [
        {"speaker": "supplier", "text": f"We can offer {volume} tons/month of {match['category']} "
                                         f"at INR {opening_supplier}/ton, recurring monthly, delivery from our site."},
        {"speaker": "buyer", "text": f"That's above our target. We can do INR {opening_buyer}/ton if you commit "
                                      f"to a {contract_months}-month recurring contract and we split transport cost "
                                      f"{transport_split}."},
        {"speaker": "supplier", "text": f"Acceptable at INR {settle}/ton flat, {contract_months}-month recurring, "
                                         f"transport split {transport_split}. "
                                         f"AGREEMENT: {json.dumps(proposal)}"},
    ]

    if not valid:
        transcript.append({"speaker": "buyer", "text": f"NO_DEAL: {reason}"})
        return {
            "transcript": transcript,
            "status": "failed",
            "failure_reason": reason,
            "mode": "scripted_fallback",
        }

    transcript.append({"speaker": "buyer", "text": "CONFIRMED: agreed to the terms above."})
    return {
        "transcript": transcript,
        "status": "agreed",
        "mode": "scripted_fallback",
        **proposal,
    }


class NegotiationAgent:
    def __init__(self):
        self.memory = NegotiationMemory(_load_memory())

    def negotiate(self, match: dict) -> dict:
        seller_terms = {**DEFAULT_SELLER_TERMS, **(match.get("seller_terms") or {})}
        buyer_terms = {**DEFAULT_BUYER_TERMS, **(match.get("buyer_terms") or {})}

        # --- Long-term memory check: is there already an active (unexpired)
        # agreement? If so, reuse it WITHOUT writing a new "memory_reuse"
        # entry into the persistent log - otherwise the log (and thus every
        # future existing_agreement() scan) grows by one record per run even
        # though no new negotiation happened.
        existing = self.memory.existing_agreement(match)
        if existing:
            record = {
                "match_id": _match_id(match),
                "supplier": match["supplier_name"],
                "buyer": match["buyer_name"],
                "category": match["category"],
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "transcript": [],
                "status": "agreed",
                "price_inr_per_ton": existing["price_inr_per_ton"],
                "volume_tons_month": existing["volume_tons_month"],
                "contract_months": existing["contract_months"],
                "delivery_terms": existing["delivery_terms"],
                "transport_split": existing["transport_split"],
                "mode": "memory_reuse",
                "memory_note": f"Existing active agreement found (originally agreed "
                               f"{existing['timestamp']}, valid until {existing.get('expires_at', 'n/a')}) "
                               f"- reused without renegotiation.",
            }
            return record  # NOT persisted to the log - see docstring above

        anchor_hint = self.memory.anchor_price_hint(match)
        preferred_split = compute_transport_split(seller_terms, buyer_terms)

        if DEMO_MODE:
            result = _scripted_negotiation(match, seller_terms, buyer_terms, anchor_hint)
        else:
            client = _client()
            supplier_sys = _supplier_system_prompt(match, seller_terms, anchor_hint, preferred_split)
            buyer_sys = _buyer_system_prompt(match, buyer_terms, anchor_hint, preferred_split)

            supplier_history, buyer_history = [], []
            transcript = []
            structured = None
            status = "failed"
            failure_reason = "No agreement reached within round limit."
            opening = "Please open the negotiation with your opening offer."
            supplier_history.append({"role": "user", "content": opening})

            for _round in range(MAX_ROUNDS):
                supplier_msg = _call_llm(client, supplier_sys, supplier_history)
                transcript.append({"speaker": "supplier", "text": supplier_msg})
                supplier_history.append({"role": "assistant", "content": supplier_msg})
                buyer_history.append({"role": "user", "content": supplier_msg})

                if "AGREEMENT:" in supplier_msg:
                    proposal = _parse_agreement_json(supplier_msg.split("AGREEMENT:", 1)[1])
                    if not proposal:
                        status, failure_reason = "failed", "Agreement declared but terms were not valid structured data."
                        break
                    valid, reason = validate_agreement(proposal, match, seller_terms, buyer_terms)
                    if not valid:
                        status, failure_reason = "failed", f"Proposed agreement violated constraints: {reason}"
                        break
                    confirmed, reason = _seek_confirmation(
                        client, buyer_sys, buyer_history, proposal, transcript, proposer_speaker="supplier"
                    )
                    supplier_history.append({"role": "user", "content": transcript[-1]["text"]})
                    if confirmed:
                        structured, status = proposal, "agreed"
                    else:
                        status, failure_reason = "failed", reason
                    break
                if "NO_DEAL:" in supplier_msg:
                    failure_reason = supplier_msg.split("NO_DEAL:", 1)[1].strip()
                    status = "failed"
                    break

                buyer_msg = _call_llm(client, buyer_sys, buyer_history)
                transcript.append({"speaker": "buyer", "text": buyer_msg})
                buyer_history.append({"role": "assistant", "content": buyer_msg})
                supplier_history.append({"role": "user", "content": buyer_msg})

                if "AGREEMENT:" in buyer_msg:
                    proposal = _parse_agreement_json(buyer_msg.split("AGREEMENT:", 1)[1])
                    if not proposal:
                        status, failure_reason = "failed", "Agreement declared but terms were not valid structured data."
                        break
                    valid, reason = validate_agreement(proposal, match, seller_terms, buyer_terms)
                    if not valid:
                        status, failure_reason = "failed", f"Proposed agreement violated constraints: {reason}"
                        break
                    confirmed, reason = _seek_confirmation(
                        client, supplier_sys, supplier_history, proposal, transcript, proposer_speaker="buyer"
                    )
                    buyer_history.append({"role": "user", "content": transcript[-1]["text"]})
                    if confirmed:
                        structured, status = proposal, "agreed"
                    else:
                        status, failure_reason = "failed", reason
                    break
                if "NO_DEAL:" in buyer_msg:
                    failure_reason = buyer_msg.split("NO_DEAL:", 1)[1].strip()
                    status = "failed"
                    break

            result = {
                "transcript": transcript,
                "status": status,
                "mode": "live_llm",
            }
            if status == "agreed" and structured:
                result.update(structured)
            else:
                result["failure_reason"] = failure_reason

        record = {
            "match_id": _match_id(match),
            "supplier": match["supplier_name"],
            "buyer": match["buyer_name"],
            "category": match["category"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **result,
        }

        if record["status"] == "agreed":
            started_at = datetime.now(timezone.utc)
            expires_at = started_at + timedelta(days=_DAYS_PER_MONTH * int(record.get("contract_months") or 0))
            record["started_at"] = started_at.isoformat()
            record["expires_at"] = expires_at.isoformat()

        self._append_to_memory(record)
        return record

    def _append_to_memory(self, record: dict):
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        history = _load_memory()
        history.append(record)
        LOG_PATH.write_text(json.dumps(history, indent=2))
        self.memory = NegotiationMemory(history)  # keep in-run memory fresh


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(BASE_DIR / "agents"))
    from ingestion import IngestionAgent
    from matchmaker import MatchmakingAgent
    from logistics import LogisticsAgent

    companies = IngestionAgent(str(BASE_DIR / "data" / "companies.json")).load()
    matches = LogisticsAgent().enrich_and_rank(MatchmakingAgent().find_matches(companies))
    top_match = matches[0]
    result = NegotiationAgent().negotiate(top_match)
    print(json.dumps(result, indent=2))
