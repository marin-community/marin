"""Wire JSON schemas for GLM constrained output; runtime/semantic checks still apply."""

from .prompts import ENVIRONMENTS, VERIFIERS
from .validation import AXES


def text():
    return {"type": "string", "minLength": 1}


def enum(values):
    return {"type": "string", "enum": list(values)}


def array(item, minimum=0, maximum=None):
    result = {"type": "array", "items": item, "minItems": minimum}
    if maximum is not None:
        result["maxItems"] = maximum
    return result


def obj(properties):
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def strings(minimum=0):
    return array(text(), minimum)


def plan_schema(capability_id):
    return obj(
        {
            "capability_id": {"type": "string", "const": capability_id},
            "coverage_rationale": text(),
            "research_priorities": strings(),
            "excluded_combinations": array(
                obj(
                    {
                        "environment": enum(ENVIRONMENTS),
                        "verification": enum(VERIFIERS),
                        "reason": text(),
                    }
                )
            ),
            "slots": array(
                obj(
                    {
                        "slot": {"type": "integer", "minimum": 1, "maximum": 10},
                        "title": text(),
                        "workflow": text(),
                        "environment": enum(ENVIRONMENTS),
                        "verification": enum(VERIFIERS),
                        "distinctive_challenge": text(),
                        "status": enum(["propose", "null"]),
                        "reason": {"type": ["string", "null"]},
                    }
                ),
                10,
                10,
            ),
        }
    )


def proposal_schema(capability_id, slot):
    fields = {
        "capability_id": {"type": "string", "const": capability_id},
        "slot": {"type": "integer", "const": slot},
        "status": enum(["proposed"]),
        "null_reason": {"type": "null"},
        "environment": enum(ENVIRONMENTS),
        "verification": enum(VERIFIERS),
    }
    fields.update(
        {
            key: text()
            for key in (
                "title",
                "task_family",
                "capability_alignment",
                "environment_rationale",
                "workflow",
                "task_brief",
            )
        }
    )
    fields.update(
        {
            key: strings(1)
            for key in (
                "inputs",
                "deliverables",
                "constraints",
                "difficulty_drivers",
                "validation_plan",
            )
        }
    )
    fields["grounding"] = obj(
        {
            "known_facts": strings(),
            "research_needed": strings(),
            "sources": array(
                obj(
                    {
                        "query_or_url": text(),
                        "purpose": text(),
                        "verification_status": enum(["unverified"]),
                        "license_check": text(),
                        "fallback": text(),
                    }
                )
            ),
        }
    )
    fields["environment_spec"] = obj(
        {
            "initial_state": text(),
            "tools": strings(),
            "reset": text(),
            "dependency_strategy": text(),
            "resource_estimate": text(),
        }
    )
    fields["verification_spec"] = obj(
        {
            "observable_success": text(),
            "grader_design": text(),
            "positive_controls": strings(1),
            "negative_controls": strings(1),
            "anti_shortcuts": strings(1),
            "rubric": strings(1),
        }
    )
    fields["builder_plan"] = array(
        obj(
            {
                "session": text(),
                "depends_on": strings(),
                "goal": text(),
                "handoff_artifacts": strings(1),
                "acceptance_checks": strings(1),
            }
        ),
        1,
    )
    fields["risks"] = array(
        obj({"risk": text(), "mitigation": text(), "abandon_if": text()}), 1
    )
    fields["data_policy"] = obj(
        {
            "provenance": text(),
            "license": text(),
            "split_group": text(),
            "contamination_check": text(),
            "private_evaluator_data": strings(1),
        }
    )
    null = obj(
        {
            "capability_id": fields["capability_id"],
            "slot": fields["slot"],
            "status": enum(["null"]),
            "null_reason": text(),
        }
    )
    return {"anyOf": [obj(fields), null]}


def review_schema(capability_id, slots):
    slots = list(slots)
    return obj(
        {
            "capability_id": {"type": "string", "const": capability_id},
            "portfolio_verdict": enum(["accept", "repair", "reject"]),
            "portfolio_issues": strings(),
            "missing_slots": {
                "type": "array",
                "items": {"type": "integer"},
                "const": sorted(set(range(1, 11)) - set(slots)),
            },
            "reviews": array(
                obj(
                    {
                        "slot": {"type": "integer", "enum": slots},
                        "verdict": enum(["accept", "repair", "reject", "null"]),
                        "scores": obj(
                            {
                                axis: {"type": "integer", "minimum": 1, "maximum": 5}
                                for axis in sorted(AXES)
                            }
                        ),
                        "critical_failures": strings(),
                        "issues": strings(),
                        "required_changes": strings(),
                    }
                ),
                len(slots),
                len(slots),
            ),
        }
    )


def response_format(name, schema):
    return {
        "type": "json_schema",
        "json_schema": {"name": name, "strict": True, "schema": schema},
    }
