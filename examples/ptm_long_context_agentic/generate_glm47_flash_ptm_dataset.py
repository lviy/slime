from __future__ import annotations

import json
from pathlib import Path


OUTPUT_PATH = Path(__file__).with_name("glm47_flash_ptm_rollout_prompts.jsonl")


TOOL_SPECS = [
    (
        "calendar_lookup",
        "Query the enterprise calendar index for meetings, owners, participants, and change history.",
        '{"employee_id":"string","date_range":"string","timezone":"string"}',
        '{"meetings":[{"id":"string","owner":"string","start":"string","end":"string","confidence":"float"}]}',
    ),
    (
        "calendar_reschedule_simulator",
        "Simulate moving one or more meetings while preserving hard constraints and approval chains.",
        '{"meeting_ids":["string"],"candidate_windows":["string"],"priority_rule":"string"}',
        '{"accepted_plan":[{"meeting_id":"string","new_time":"string"}],"rejected_reasons":["string"]}',
    ),
    (
        "flight_inventory_snapshot",
        "Return a simulated airline inventory snapshot with seats, cabin rules, and rebooking penalties.",
        '{"origin":"string","destination":"string","date":"string","cabin":"string"}',
        '{"options":[{"flight_no":"string","fare":"number","seats_left":"int","risk":"string"}]}',
    ),
    (
        "hotel_policy_checker",
        "Check simulated hotel options against company travel policy, loyalty tier, and safety constraints.",
        '{"city":"string","date_range":"string","budget_cap":"number","traveler_tier":"string"}',
        '{"hotels":[{"name":"string","nightly_rate":"number","policy":"string","notes":"string"}]}',
    ),
    (
        "ground_transport_router",
        "Plan airport transfer, rail, or car service with estimated arrival confidence and disruption notes.",
        '{"city":"string","pickup":"string","dropoff":"string","priority":"string"}',
        '{"routes":[{"mode":"string","eta_minutes":"int","risk":"string","cost":"number"}]}',
    ),
    (
        "incident_dashboard",
        "Read a synthetic SRE dashboard including error rate, p95, p99, saturation, and alert states.",
        '{"service":"string","window":"string","region":"string"}',
        '{"metrics":{"error_rate":"float","p95_ms":"int","p99_ms":"int"},"alerts":["string"]}',
    ),
    (
        "logs_query",
        "Search simulated structured logs with filters on service, deployment, region, and correlation id.",
        '{"service":"string","query":"string","limit":"int"}',
        '{"matches":[{"timestamp":"string","level":"string","message":"string","correlation_id":"string"}]}',
    ),
    (
        "trace_summarizer",
        "Summarize synthetic distributed traces and identify the most suspicious span or downstream dependency.",
        '{"service":"string","trace_window":"string","hint":"string"}',
        '{"suspect_spans":[{"span":"string","latency_ms":"int","reason":"string"}]}',
    ),
    (
        "feature_flag_audit",
        "Inspect rollout state for simulated feature flags, targeting rules, and blast radius.",
        '{"flag_prefix":"string","environment":"string"}',
        '{"flags":[{"name":"string","state":"string","rule":"string","blast_radius":"string"}]}',
    ),
    (
        "deploy_history_reader",
        "Return a synthetic deploy history with timestamps, owners, canary notes, and rollback metadata.",
        '{"service":"string","lookback_hours":"int"}',
        '{"deploys":[{"id":"string","time":"string","owner":"string","change_summary":"string"}]}',
    ),
    (
        "supplier_catalog_reader",
        "Read a simulated supplier catalog with inventory, GPU SKU mapping, price, and qualification notes.",
        '{"commodity":"string","region":"string","urgency":"string"}',
        '{"suppliers":[{"name":"string","inventory":"int","unit_price":"number","qualified":"bool"}]}',
    ),
    (
        "contract_clause_analyzer",
        "Summarize simulated contract clauses related to penalties, warranties, cancellation, and payment terms.",
        '{"supplier":"string","contract_version":"string"}',
        '{"clauses":[{"topic":"string","summary":"string","risk":"string"}]}',
    ),
    (
        "compliance_risk_checker",
        "Return synthetic compliance flags for export control, data handling, audit trail, and geographic restrictions.",
        '{"supplier":"string","shipment_region":"string","item_type":"string"}',
        '{"risk_level":"string","findings":["string"]}',
    ),
    (
        "warehouse_eta_model",
        "Estimate simulated lead time from warehouses, customs handoff, and last-mile schedules.",
        '{"origin_region":"string","destination_region":"string","shipment_mode":"string"}',
        '{"eta_days":"int","confidence":"float","blocking_issues":["string"]}',
    ),
    (
        "order_record_reader",
        "Read a synthetic commerce order record with payment state, fulfillment state, and exception tags.",
        '{"order_id":"string"}',
        '{"order":{"payment_state":"string","fulfillment_state":"string","tags":["string"]}}',
    ),
    (
        "refund_policy_lookup",
        "Summarize the simulated refund and replacement policy for a specific category and customer tier.",
        '{"category":"string","customer_tier":"string","reason":"string"}',
        '{"policy":"string","approval_threshold":"number","manual_review":"bool"}',
    ),
    (
        "fraud_signal_scan",
        "Return synthetic fraud signals including device risk, velocity, chargeback history, and address mismatch.",
        '{"customer_id":"string","order_id":"string"}',
        '{"signals":[{"name":"string","severity":"string","evidence":"string"}]}',
    ),
    (
        "shipment_tracker",
        "Track a simulated parcel timeline and detect suspicious scans, delivery gaps, or route anomalies.",
        '{"tracking_id":"string"}',
        '{"events":[{"time":"string","status":"string","location":"string","note":"string"}]}',
    ),
    (
        "crm_case_reader",
        "Read a synthetic CRM timeline with prior tickets, escalation notes, and sentiment markers.",
        '{"customer_id":"string","lookback_days":"int"}',
        '{"cases":[{"id":"string","topic":"string","outcome":"string","sentiment":"string"}]}',
    ),
    (
        "policy_memory_search",
        "Search internal synthetic policy memory for procedures, exceptions, escalation paths, and templates.",
        '{"topic":"string","audience":"string","strictness":"string"}',
        '{"documents":[{"title":"string","excerpt":"string","confidence":"float"}]}',
    ),
    (
        "budget_guardrail_checker",
        "Apply a simulated cost guardrail and explain where the plan exceeds budget or reserve limits.",
        '{"scenario":"string","proposed_cost":"number","budget_cap":"number"}',
        '{"status":"string","delta":"number","notes":["string"]}',
    ),
    (
        "stakeholder_directory_lookup",
        "Resolve simulated stakeholders, approvers, backups, and escalation delegates.",
        '{"team":"string","role_needed":"string"}',
        '{"contacts":[{"name":"string","role":"string","timezone":"string","backup":"string"}]}',
    ),
    (
        "knowledge_base_diff",
        "Compare two synthetic policy or runbook versions and summarize operational differences.",
        '{"doc_a":"string","doc_b":"string","focus":"string"}',
        '{"differences":[{"topic":"string","before":"string","after":"string"}]}',
    ),
    (
        "simulation_notepad_write",
        "Store an internal synthetic working note. This tool is for transcript realism only.",
        '{"section":"string","content":"string"}',
        '{"stored":"bool","reference_id":"string"}',
    ),
]


def build_tool_catalog() -> str:
    blocks = []
    for index, (name, desc, args_schema, return_schema) in enumerate(TOOL_SPECS, start=1):
        blocks.append(
            "\n".join(
                [
                    f"Tool {index}: {name}",
                    f"Description: {desc}",
                    f"Input schema: {args_schema}",
                    f"Return schema: {return_schema}",
                    "Usage rules:",
                    "- Never claim the tool was truly executed.",
                    "- Treat all results as simulations used to construct a coherent transcript.",
                    "- Prefer concise arguments with realistic enterprise field names.",
                    "- If uncertain, add a short caveat in the simulated tool result instead of refusing.",
                    "Failure patterns to mention when helpful:",
                    "- stale cache",
                    "- incomplete inventory snapshot",
                    "- regional mismatch",
                    "- policy ambiguity",
                    "- downstream timeout",
                ]
            )
        )
    return "\n\n".join(blocks)


def build_system_message() -> str:
    return (
        "You are Enterprise Simulation Agent, a transcript generator used for PrefixTreeMerging stress tests.\n"
        "Your job is to produce a realistic but fully simulated multi-turn agentic trace.\n"
        "Do not claim any external tool was actually executed. Every tool call and tool result is synthetic.\n"
        "Output requirements:\n"
        "1. Produce exactly 6 turns in the assistant completion.\n"
        "2. Every turn must be explicit and labeled Turn 1 through Turn 6.\n"
        "3. At least 3 turns must include a <tool_call> JSON block.\n"
        "4. At least 3 turns must include a <tool_result> block.\n"
        "5. Preserve a highly structured format so multiple samples share long common prefixes.\n"
        "6. End the final turn with the exact line FINAL_STATUS: TASK_DONE\n"
        "7. Keep the answer self-contained and plausible for enterprise operations, incident response, or logistics.\n"
        "8. Do not apologize. Do not mention that this is a benchmark.\n"
        "Canonical completion skeleton:\n"
        "Turn 1 - Situation framing and initial plan.\n"
        "Turn 2 - First simulated tool call and result summary.\n"
        "Turn 3 - Intermediate reasoning plus second simulated tool call.\n"
        "Turn 4 - Second tool result and reconciliation of conflicting signals.\n"
        "Turn 5 - Final validation tool call or policy lookup.\n"
        "Turn 6 - Final decision memo, risk summary, and FINAL_STATUS: TASK_DONE.\n"
        "The tool catalog below is intentionally long. Reuse its wording faithfully when appropriate.\n\n"
        + build_tool_catalog()
    )


def build_history_messages() -> list[dict[str, str]]:
    history = []

    history.append(
        {
            "role": "user",
            "content": (
                "Historical case A. A regional travel coordinator needs to re-sequence a leadership roadshow after a storm."
                " Preserve executive preferences, meeting priority, and policy constraints."
            ),
        }
    )
    history.append(
        {
            "role": "assistant",
            "content": (
                "Turn 1 - I will inspect calendar conflicts, candidate flights, and hotel policy before proposing a revised route.\n"
                "Turn 2 - <tool_call>{\"name\":\"calendar_lookup\",\"arguments\":{\"employee_id\":\"exec-204\",\"date_range\":\"2026-06-02 to 2026-06-05\",\"timezone\":\"Asia/Singapore\"}}</tool_call>\n"
                "<tool_result>Simulated result: two board meetings are fixed, one customer dinner can move within plus or minus 90 minutes, and the owner delegate is available for asynchronous approval.</tool_result>\n"
                "Turn 3 - <tool_call>{\"name\":\"flight_inventory_snapshot\",\"arguments\":{\"origin\":\"SIN\",\"destination\":\"HND\",\"date\":\"2026-06-03\",\"cabin\":\"business\"}}</tool_call>\n"
                "<tool_result>Simulated result: three viable flights remain, but the cheapest option has moderate disruption risk and limited reaccommodation support.</tool_result>\n"
                "Turn 4 - I cross-checked hotel policy and identified one compliant hotel near the backup venue with flexible cancellation.\n"
                "Turn 5 - <tool_call>{\"name\":\"ground_transport_router\",\"arguments\":{\"city\":\"Tokyo\",\"pickup\":\"Haneda Airport\",\"dropoff\":\"Marunouchi conference center\",\"priority\":\"arrival certainty\"}}</tool_call>\n"
                "<tool_result>Simulated result: rail is cheapest, chauffeured car is most reliable under rain disruption, shared shuttle has schedule risk.</tool_result>\n"
                "Turn 6 - Recommendation: move the customer dinner, select the reliable mid-price flight, and book the flexible hotel. FINAL_STATUS: TASK_DONE"
            ),
        }
    )

    history.append(
        {
            "role": "user",
            "content": (
                "Historical case B. A payments team sees rising p99 latency during a staged release and needs a containment plan."
            ),
        }
    )
    history.append(
        {
            "role": "assistant",
            "content": (
                "Turn 1 - I will correlate deploy history, feature flags, traces, and the incident dashboard.\n"
                "Turn 2 - <tool_call>{\"name\":\"incident_dashboard\",\"arguments\":{\"service\":\"checkout-api\",\"window\":\"last_30m\",\"region\":\"global\"}}</tool_call>\n"
                "<tool_result>Simulated result: error rate is only mildly elevated, but p99 doubled in two regions and saturation alerts fired on a shared dependency.</tool_result>\n"
                "Turn 3 - <tool_call>{\"name\":\"deploy_history_reader\",\"arguments\":{\"service\":\"checkout-api\",\"lookback_hours\":\"8\"}}</tool_call>\n"
                "<tool_result>Simulated result: the latest canary introduced a routing heuristic that increased calls to a slow entitlement service.</tool_result>\n"
                "Turn 4 - <tool_call>{\"name\":\"feature_flag_audit\",\"arguments\":{\"flag_prefix\":\"checkout_\",\"environment\":\"prod\"}}</tool_call>\n"
                "<tool_result>Simulated result: one pricing flag reached 35 percent of traffic without region pinning, increasing cross-region chatter.</tool_result>\n"
                "Turn 5 - I note that rollback is low risk because the schema is unchanged and the canary owner is online.\n"
                "Turn 6 - Containment plan: disable the routing heuristic, pin the pricing flag to one region, and monitor traces for 20 minutes. FINAL_STATUS: TASK_DONE"
            ),
        }
    )

    history.append(
        {
            "role": "user",
            "content": (
                "Historical case C. Procurement is comparing GPU suppliers under budget, compliance, and delivery pressure."
            ),
        }
    )
    history.append(
        {
            "role": "assistant",
            "content": (
                "Turn 1 - I will compare inventory, contract risk, compliance, and ETA confidence before ranking suppliers.\n"
                "Turn 2 - <tool_call>{\"name\":\"supplier_catalog_reader\",\"arguments\":{\"commodity\":\"H100-class accelerator\",\"region\":\"APAC\",\"urgency\":\"critical\"}}</tool_call>\n"
                "<tool_result>Simulated result: Supplier North has the best inventory, Supplier Harbor has the best price, Supplier Delta has the strongest service credits.</tool_result>\n"
                "Turn 3 - <tool_call>{\"name\":\"contract_clause_analyzer\",\"arguments\":{\"supplier\":\"Supplier Harbor\",\"contract_version\":\"v7\"}}</tool_call>\n"
                "<tool_result>Simulated result: Harbor includes a steep cancellation penalty and weak warranty replacement language.</tool_result>\n"
                "Turn 4 - <tool_call>{\"name\":\"compliance_risk_checker\",\"arguments\":{\"supplier\":\"Supplier North\",\"shipment_region\":\"Singapore\",\"item_type\":\"accelerator\"}}</tool_call>\n"
                "<tool_result>Simulated result: no hard stop, but export screening adds one review gate and raises execution risk slightly.</tool_result>\n"
                "Turn 5 - I balanced the price advantage against delivery certainty and support obligations.\n"
                "Turn 6 - Recommended ranking: North first, Delta second, Harbor third. FINAL_STATUS: TASK_DONE"
            ),
        }
    )
    return history


def build_current_user_message(task_name: str, task_body: str) -> str:
    return (
        f"Active task: {task_name}\n"
        "Generate a fresh simulated transcript for the active task.\n"
        "Use the same formatting style as the historical examples, but do not copy their decisions.\n"
        "Hard requirements:\n"
        "- Exactly 6 labeled turns.\n"
        "- The final line must be FINAL_STATUS: TASK_DONE\n"
        "- Mention at least 3 tool calls and 3 tool results.\n"
        "- The transcript must look like a realistic multi-turn agent workflow, even though all tools are simulated.\n"
        "- Reuse the enterprise terminology from the tool catalog and policy history so that many trajectories share long prefixes.\n"
        "- Introduce enough variation that 16 sampled trajectories are not identical.\n"
        "- Keep each turn dense and information-rich.\n\n"
        "Detailed task brief:\n"
        f"{task_body}"
    )


TASKS = [
    (
        "travel_replan",
        (
            "A chief strategy officer is traveling from Shanghai to Singapore and then to Tokyo for a board prep, "
            "customer roundtable, and press interview. A weather event may disrupt the Tokyo leg, one meeting can move, "
            "and the traveler strongly prefers a same-day arrival buffer. The transcript should simulate comparing flights, "
            "calendar constraints, hotel policy, and ground transport, then produce a concrete replanning memo."
        ),
    ),
    (
        "payments_incident",
        (
            "A payment authorization service has rising p99 in two regions after a staged release. The transcript should "
            "simulate checking incident metrics, deploy history, logs, traces, and feature flags, then produce a containment "
            "plan that balances rollback speed, customer impact, and blast radius."
        ),
    ),
    (
        "gpu_procurement",
        (
            "Infra leadership needs to secure an emergency order of accelerators for a new cluster. The transcript should "
            "simulate comparing multiple suppliers on inventory, contract clauses, compliance, ETA, and budget guardrails, "
            "then produce a ranked recommendation with explicit tradeoffs."
        ),
    ),
    (
        "commerce_risk_resolution",
        (
            "A high-value ecommerce order has conflicting signals: payment succeeded, delivery appears delayed, the customer "
            "has prior VIP treatment, and several fraud indicators are elevated. The transcript should simulate checking the "
            "order record, shipment tracker, CRM history, refund policy, and fraud signals, then produce a decision memo on "
            "refund, replacement, or manual review."
        ),
    ),
]


def build_records() -> list[dict]:
    records = []
    system_message = {"role": "system", "content": build_system_message()}
    history_messages = build_history_messages()

    for task_name, task_body in TASKS:
        records.append(
            {
                "case_id": f"ptm_agentic_{task_name}",
                "answer": "TASK_DONE",
                "messages": [
                    system_message,
                    *history_messages,
                    {"role": "user", "content": build_current_user_message(task_name, task_body)},
                ],
            }
        )
    return records


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    records = build_records()
    with OUTPUT_PATH.open("w", encoding="utf-8") as fout:
        for record in records:
            fout.write(json.dumps(record, ensure_ascii=True) + "\n")
    print(f"Wrote {len(records)} records to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
