import json
import os
import re
import time
import logging

from flask import Flask, jsonify, render_template, request
from google import genai
from google.genai import types


# ============================================================
# 1. FLASK CONFIGURATION
# ============================================================

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 1 * 1024 * 1024

logging.basicConfig(level=logging.INFO)
app.logger.setLevel(logging.INFO)

# Accept either environment variable name.
# Set your actual key in Render Environment settings.
API_KEY_ENV_NAMES = ("GEMINI_API_KEY", "GeminiAPIKey")

MODEL_NAME = os.getenv(
    "GEMINI_MODEL",
    "gemini-2.5-flash"
).strip()

MAX_RETRIES = 3


# ============================================================
# 2. SYSTEM INSTRUCTIONS
# ============================================================

SYSTEM_INSTRUCTIONS = """
You are RepairMate, a cautious repair-diagnosis assistant for
household appliances, consumer electronics, bicycles, and common
non-industrial equipment.

You are not a certified technician.

Return valid JSON only, with these keys:
status, summary, follow_up_question, likely_causes, safe_checks,
recommended_next_steps, when_to_get_help, safety_note.

Rules:
- status must be "needs_info" or "diagnosis".
- Ask one concise follow-up question if important information is missing.
- Recommend only low-risk external checks.
- Never recommend opening sealed devices.
- Never recommend working on live electrical circuits.
- Never recommend bypassing safety devices.
- Never recommend handling gas lines or refrigerants.
- Never recommend handling swollen or damaged lithium batteries.
- For smoke, sparks, burning smells, gas smells, exposed wiring,
  electric shock, fire, or swollen batteries, prioritize safety.
- Tell the user to move away from hazards and contact qualified
  professionals or emergency services when appropriate.
- Do not invent prices, parts, warranties, or certainty.
- Do not claim a diagnosis is confirmed when it is only a possibility.
- Use clear, simple language.
- Return a JSON object, not Markdown or code fences.
"""


# ============================================================
# 3. EMERGENCY SAFETY TERMS
# ============================================================

DANGER_TERMS = [
    "gas leak",
    "smell gas",
    "gas smell",
    "leaking gas",
    "smoke",
    "sparks",
    "burning smell",
    "burning odor",
    "burnt smell",
    "exposed wire",
    "exposed wiring",
    "electric shock",
    "electrical shock",
    "swollen battery",
    "battery swelling",
    "battery is swollen",
    "lithium battery swelling",
    "fire",
    "flames",
]


# ============================================================
# 4. LOCAL REPAIR KNOWLEDGE BASE
# ============================================================

KNOWLEDGE = [
    {
        "topic": "Laptop black screen",
        "keywords": [
            "laptop", "black screen", "display", "monitor"
        ],
        "checks": [
            "Check that the device and display have power.",
            "Check the brightness setting.",
            "Check the selected display input if applicable.",
            "Disconnect external peripherals and restart only if safe."
        ],
        "warning": (
            "Do not open the power supply or work inside a device "
            "connected to mains power."
        )
    },
    {
        "topic": "Laptop overheating",
        "keywords": [
            "laptop", "overheating", "hot", "fan", "shut down"
        ],
        "checks": [
            "Place the laptop on a hard, flat surface.",
            "Ensure the ventilation openings are not blocked.",
            "Close demanding applications if the system responds.",
            "Use manufacturer diagnostics if available."
        ],
        "warning": (
            "Stop using the laptop if it becomes extremely hot, "
            "smokes, smells burnt, or has a swollen battery."
        )
    },
    {
        "topic": "Phone charging",
        "keywords": [
            "phone", "charging", "charger", "battery", "usb"
        ],
        "checks": [
            "Try a known-good compatible cable and adapter.",
            "Try a different outlet if it is safe.",
            "Inspect the charging port visually without inserting "
            "metal objects."
        ],
        "warning": (
            "Do not charge a wet, unusually hot, damaged, or "
            "swollen device."
        )
    },
    {
        "topic": "Printer paper jam",
        "keywords": [
            "printer", "paper", "jam", "feed"
        ],
        "checks": [
            "Turn the printer off.",
            "Follow the manufacturer's manual to access the paper path.",
            "Remove accessible paper gently in the indicated direction."
        ],
        "warning": (
            "Allow hot components to cool and do not force stuck parts."
        )
    },
    {
        "topic": "Wi-Fi connection",
        "keywords": [
            "wifi", "wi-fi", "router", "internet", "network"
        ],
        "checks": [
            "Check whether other devices can connect.",
            "Restart the device and router if safe.",
            "Check signal strength and service status.",
            "Record any displayed error message."
        ],
        "warning": (
            "Avoid factory-resetting a router before saving "
            "its configuration details."
        )
    },
    {
        "topic": "Washing machine drainage",
        "keywords": [
            "washing machine", "washer", "drain", "water", "draining"
        ],
        "checks": [
            "Disconnect power before inspecting accessible parts, "
            "provided it is safe to approach.",
            "Check for a visibly kinked drain hose.",
            "Follow the manufacturer's filter-cleaning procedure."
        ],
        "warning": (
            "Water may be hot or heavy. Stop if leakage or electrical "
            "hazards are present."
        )
    },
    {
        "topic": "Air conditioner cooling",
        "keywords": [
            "air conditioner", "cooling", "warm air", "ice"
        ],
        "checks": [
            "Check the thermostat mode and setting.",
            "Clean only the user-accessible filter.",
            "Ensure vents are open and the outdoor unit is unobstructed."
        ],
        "warning": (
            "Do not handle refrigerants or electrical components."
        )
    },
    {
        "topic": "Vehicle no-start",
        "keywords": [
            "car", "vehicle", "engine", "start", "crank", "battery"
        ],
        "checks": [
            "Note whether the engine clicks, cranks slowly, or does not crank.",
            "Check dashboard warnings and fuel level.",
            "Only visually inspect accessible battery terminals if safe."
        ],
        "warning": (
            "Avoid hot or moving parts, fuel leaks, and high-voltage "
            "hybrid or electric vehicle systems."
        )
    }
]


# ============================================================
# 5. LOCAL KNOWLEDGE RETRIEVAL
# ============================================================

def retrieve_notes(query, limit=3):
    """
    Retrieve relevant local repair notes using keyword matching.
    This works locally and does not consume Gemini API quota.
    """

    query = (query or "").lower()
    ranked = []

    for note in KNOWLEDGE:
        score = sum(
            1 for keyword in note["keywords"]
            if keyword in query
        )

        if score > 0:
            ranked.append((score, note))

    ranked.sort(key=lambda item: item[0], reverse=True)

    return [
        note for _, note in ranked[:limit]
    ]


# ============================================================
# 6. ENVIRONMENT VARIABLE HELPERS
# ============================================================

def get_api_key():
    """Read the Gemini API key from supported environment variables."""

    for variable_name in API_KEY_ENV_NAMES:
        value = os.getenv(variable_name, "").strip()

        if value:
            return value

    return ""


# ============================================================
# 7. LOCAL SAFETY RESPONSE
# ============================================================

def urgent_safety_report():
    return {
        "status": "diagnosis",
        "summary": (
            "This may involve an immediate safety hazard. "
            "Stop troubleshooting and move away from the hazard."
        ),
        "follow_up_question": "",
        "likely_causes": [
            {
                "cause": "Potentially hazardous fault",
                "likelihood": "Unknown",
                "reason": (
                    "The reported symptoms may indicate a safety risk. "
                    "The cause cannot be confirmed remotely."
                )
            }
        ],
        "safe_checks": [
            "Keep away from the hazard.",
            "Stop using the item if it is safe to do so."
        ],
        "recommended_next_steps": [
            (
                "Do not touch exposed wires, leaking gas, hot or "
                "swollen batteries, or damaged electrical equipment."
            ),
            (
                "For suspected gas, leave the area and contact "
                "local emergency or gas services from a safe location."
            ),
            (
                "For smoke or fire, move away and contact local "
                "emergency services."
            )
        ],
        "when_to_get_help": [
            "Contact an appropriate qualified professional before "
            "using the item again."
        ],
        "safety_note": (
            "Do not approach or unplug equipment if doing so would "
            "bring you near a hazard."
        )
    }


# ============================================================
# 8. JSON RESPONSE PARSER
# ============================================================

def parse_json_response(raw):
    """Parse JSON, including responses accidentally wrapped in fences."""

    if not raw:
        return None

    raw = raw.strip()

    raw = re.sub(
        r"^```(?:json)?\s*",
        "",
        raw,
        flags=re.IGNORECASE
    )

    raw = re.sub(r"\s*```$", "", raw)

    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        pass

    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)

    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return None


# ============================================================
# 9. GEMINI DIAGNOSIS REQUEST WITH RETRIES
# ============================================================

def generate_gemini_response(client, prompt_data):
    """
    Call Gemini with limited retries for temporary failures.

    Authentication and invalid-request errors are not retried.
    """

    for attempt in range(MAX_RETRIES):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=json.dumps(
                    prompt_data,
                    ensure_ascii=False
                ),
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTIONS,
                    temperature=0.2,
                    response_mime_type="application/json",
                    max_output_tokens=1200
                )
            )

            response_text = getattr(response, "text", None)

            if not response_text:
                raise RuntimeError(
                    "Gemini returned an empty response."
                )

            return response_text

        except Exception as exc:
            error_text = str(exc).lower()

            # Retry only errors that may be temporary.
            retryable_terms = [
                "429",
                "500",
                "502",
                "503",
                "504",
                "resource_exhausted",
                "unavailable",
                "deadline",
                "timed out",
                "timeout",
                "temporarily",
                "connection reset"
            ]

            retryable = any(
                term in error_text
                for term in retryable_terms
            )

            app.logger.exception(
                "Gemini request failed on attempt %s/%s",
                attempt + 1,
                MAX_RETRIES
            )

            if not retryable or attempt == MAX_RETRIES - 1:
                raise

            # Exponential backoff: 1 second, then 2 seconds.
            time.sleep(2 ** attempt)

    raise RuntimeError(
        "Gemini could not complete the request after retries."
    )


# ============================================================
# 10. MAIN DIAGNOSIS FUNCTION
# ============================================================

def diagnose(data):
    if not isinstance(data, dict):
        raise ValueError("Invalid request data.")

    message = data.get("symptoms", "")

    if not isinstance(message, str):
        raise ValueError("The problem description must be text.")

    message = message.strip()

    if not message:
        raise ValueError(
            "Please describe the device and the problem."
        )

    if len(message) > 5000:
        raise ValueError(
            "Please keep the problem description under 5,000 characters."
        )

    # Safety check before making an external API request.
    lower_message = message.lower()

    if any(term in lower_message for term in DANGER_TERMS):
        report = urgent_safety_report()
        report["matched_notes"] = []
        report["model"] = "local-safety-check"
        return report

    api_key = get_api_key()

    if not api_key:
        raise RuntimeError(
            "Gemini API key is missing. Add GEMINI_API_KEY "
            "or GeminiAPIKey in Render Environment settings."
        )

    device_type = data.get("device_type", "Not specified")
    model_number = data.get("model_number", "Not provided")
    history = data.get("history", "")

    if not isinstance(device_type, str):
        device_type = "Not specified"

    if not isinstance(model_number, str):
        model_number = "Not provided"

    if not isinstance(history, str):
        history = ""

    device_type = device_type[:100]
    model_number = model_number[:150]
    history = history[:1500]

    # Retrieve relevant local knowledge.
    notes = retrieve_notes(
        f"{device_type} {model_number} {message} {history}"
    )

    prompt_data = {
        "device_type": device_type,
        "model_number": model_number,
        "symptoms": message,
        "additional_history": history,
        "retrieved_local_repair_notes": notes
    }

    client = genai.Client(api_key=api_key)

    raw_response = generate_gemini_response(
        client,
        prompt_data
    )

    result = parse_json_response(raw_response)

    if not isinstance(result, dict):
        app.logger.error(
            "Gemini returned invalid JSON. Response length: %s",
            len(raw_response)
        )

        raise RuntimeError(
            "Gemini returned an unreadable response. Please retry."
        )

    # Validate the expected output structure.
    defaults = {
        "status": "diagnosis",
        "summary": "Preliminary diagnosis only.",
        "follow_up_question": "",
        "likely_causes": [],
        "safe_checks": [],
        "recommended_next_steps": [],
        "when_to_get_help": [],
        "safety_note": (
            "Stop if anything feels unsafe and contact "
            "a qualified professional."
        )
    }

    for key, default in defaults.items():
        if not isinstance(result.get(key), type(default)):
            result[key] = default

    if result["status"] not in ("needs_info", "diagnosis"):
        result["status"] = "diagnosis"

    # Attach local retrieval information.
    result["matched_notes"] = [
        {"topic": note["topic"]}
        for note in notes
    ]

    result["model"] = MODEL_NAME

    return result


# ============================================================
# 11. HOMEPAGE
# ============================================================

@app.get("/")
def index():
    return render_template("index.html")


# ============================================================
# 12. HEALTH CHECK
# ============================================================

@app.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "model": MODEL_NAME,
        "api_key_configured": bool(get_api_key())
    })


# ============================================================
# 13. DIAGNOSIS API ENDPOINT
# ============================================================

@app.post("/api/diagnose")
def api_diagnose():
    try:
        data = request.get_json(silent=True)

        if not isinstance(data, dict):
            return jsonify({
                "error": (
                    "Invalid request. Please submit the diagnosis "
                    "form again."
                )
            }), 400

        result = diagnose(data)

        return jsonify(result), 200

    except ValueError as exc:
        return jsonify({
            "error": str(exc)
        }), 400

    except RuntimeError as exc:
        app.logger.error(
            "Diagnosis service error: %s",
            str(exc)
        )

        return jsonify({
            "error": str(exc)
        }), 503

    except Exception:
        app.logger.exception(
            "Unexpected diagnosis request failure"
        )

        return jsonify({
            "error": (
                "Diagnosis failed. Please check the Render logs "
                "for the underlying Gemini or application error."
            )
        }), 502


# ============================================================
# 14. APPLICATION STARTUP
# ============================================================

if __name__ == "__main__":
    port = int(os.getenv("PORT", "10000"))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
