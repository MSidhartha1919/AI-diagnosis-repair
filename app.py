import json
import os
import re
from flask import Flask, jsonify, render_template, request
from google import genai
from google.genai import types

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 1 * 1024 * 1024

API_KEY_ENV = "GeminiAPIKey"
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

SYSTEM_INSTRUCTIONS = """You are RepairMate, a cautious repair-diagnosis assistant for household appliances, consumer electronics, bicycles, and common non-industrial equipment. You are not a certified technician. Return valid JSON with keys status, summary, follow_up_question, likely_causes, safe_checks, recommended_next_steps, when_to_get_help, safety_note. status is needs_info or diagnosis. Ask one concise follow-up question if key information is missing. Recommend only low-risk external checks. Never recommend opening sealed devices, working on live circuits, bypassing safety devices, handling gas lines/refrigerants, or handling swollen/damaged lithium batteries. For smoke, sparks, burning smell, gas smell, exposed wiring, electric shock, fire, or swollen battery, prioritize stopping use, moving away if needed, and contacting qualified help or emergency services. Do not invent prices, parts, warranties, or certainty. JSON only."""

DANGER_TERMS = [
    "gas leak", "smell gas", "gas smell", "smoke", "sparks", "burning smell",
    "burning odor", "exposed wire", "exposed wiring", "electric shock",
    "swollen battery", "battery swelling", "fire", "flames"
]

KNOWLEDGE = [
    {"topic": "Laptop black screen", "keywords": ["laptop", "black screen", "display", "monitor"],
     "checks": ["Check that the device and display have power.", "Check brightness and the selected display input if applicable.", "Disconnect external peripherals and restart only if there are no safety warning signs."],
     "warning": "Do not open the power supply or work inside a device connected to mains power."},
    {"topic": "Laptop overheating", "keywords": ["laptop", "overheating", "hot", "fan", "shut down"],
     "checks": ["Place the laptop on a hard, flat surface.", "Ensure vents are not blocked.", "Close demanding applications if the system remains responsive.", "Use manufacturer diagnostics if available."],
     "warning": "Power down and seek service if it is extremely hot, smells burnt, smokes, or the battery is swollen."},
    {"topic": "Phone charging", "keywords": ["phone", "charging", "charger", "battery", "usb"],
     "checks": ["Try a known-good, compatible cable and adapter.", "Try a different outlet.", "Inspect the port visually without inserting metal objects."],
     "warning": "Do not charge a wet, unusually hot, damaged, or swollen device."},
    {"topic": "Printer paper jam", "keywords": ["printer", "paper", "jam", "feed"],
     "checks": ["Turn the printer off.", "Follow the model's manual to access the paper path.", "Remove accessible paper gently in the indicated direction."],
     "warning": "Let hot components cool and do not force stuck parts."},
    {"topic": "Wi-Fi connection", "keywords": ["wifi", "wi-fi", "router", "internet", "network"],
     "checks": ["Check whether other devices connect.", "Restart the device and router if safe.", "Check signal strength and service status.", "Record any error message."],
     "warning": "Avoid factory-resetting a router before saving configuration details."},
    {"topic": "Washing machine drainage", "keywords": ["washing machine", "washer", "drain", "water", "draining"],
     "checks": ["Unplug the machine before inspecting accessible parts.", "Check for a visibly kinked drain hose.", "Use the manual's filter-cleaning procedure."],
     "warning": "Water can be hot/heavy. Stop if there is leakage or an electrical smell."},
    {"topic": "Air conditioner cooling", "keywords": ["air conditioner", "ac", "cooling", "warm air", "ice"],
     "checks": ["Check thermostat mode and setting.", "Clean only the user-accessible filter.", "Make sure vents are open and the outdoor unit is unobstructed."],
     "warning": "Do not handle refrigerant or electrical components."},
    {"topic": "Vehicle no-start", "keywords": ["car", "vehicle", "engine", "start", "crank", "battery"],
     "checks": ["Note whether the engine clicks, cranks slowly, or does not crank.", "Check dashboard warnings and fuel level.", "Visually inspect accessible battery terminals only if safe."],
     "warning": "Avoid hot/moving parts, fuel leaks, and high-voltage hybrid/EV systems; seek qualified assistance if unsure."}
]

def retrieve_notes(query, limit=3):
    query = (query or "").lower()
    ranked = []
    for note in KNOWLEDGE:
        score = sum(1 for kw in note["keywords"] if kw in query)
        if score:
            ranked.append((score, note))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return [note for _, note in ranked[:limit]]

def urgent_safety_report():
    return {
        "status": "diagnosis",
        "summary": "This may involve an immediate safety hazard. Stop troubleshooting.",
        "follow_up_question": "",
        "likely_causes": [{"cause": "Potentially hazardous fault", "likelihood": "Unknown",
                           "reason": "The symptoms can indicate a safety risk; the cause cannot be confirmed remotely."}],
        "safe_checks": ["Keep away from the hazard. Stop using the item if it is safe to do so."],
        "recommended_next_steps": [
            "Do not touch exposed wires, leaking gas, hot or swollen batteries, or damaged electrical equipment.",
            "For suspected gas, leave the area and contact local emergency/gas services from a safe location.",
            "For smoke or fire, move away and contact local emergency services."
        ],
        "when_to_get_help": ["Contact an appropriate qualified professional before using the item again."],
        "safety_note": "Do not approach or unplug equipment if doing so would bring you near a hazard."
    }

def parse_json_response(raw):
    raw = (raw or "").strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.I)
    raw = re.sub(r"\s*```$", "", raw)
    try:
        return json.loads(raw)
    except Exception:
        match = re.search(r"\{.*\}", raw, flags=re.S)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                return None
    return None

def diagnose(data):
    message = (data.get("symptoms") or "").strip()
    if not message:
        raise ValueError("Please describe the device and the problem.")
    if len(message) > 5000:
        raise ValueError("Please keep the problem description under 5,000 characters.")

    if any(term in message.lower() for term in DANGER_TERMS):
        report = urgent_safety_report()
        report["matched_notes"] = []
        report["model"] = "local-safety-check"
        return report

    api_key = os.getenv(API_KEY_ENV, "").strip()
    if not api_key:
        raise RuntimeError(f"Missing {API_KEY_ENV}. Add it in Render Environment settings.")

    device_type = (data.get("device_type") or "Not specified")[:100]
    model_number = (data.get("model_number") or "Not provided")[:150]
    history = (data.get("history") or "")[:1500]
    notes = retrieve_notes(f"{device_type} {model_number} {message} {history}")

    prompt_data = {
        "device_type": device_type,
        "model_number": model_number,
        "symptoms": message,
        "additional_history": history,
        "retrieved_local_repair_notes": notes
    }
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=json.dumps(prompt_data, ensure_ascii=False),
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTIONS,
            temperature=0.2,
            response_mime_type="application/json",
            max_output_tokens=1200
        )
    )
    result = parse_json_response(getattr(response, "text", ""))
    if not isinstance(result, dict):
        raise RuntimeError("Gemini returned an unreadable response. Please retry.")
    defaults = {
        "status": "diagnosis",
        "summary": "Preliminary diagnosis only.",
        "follow_up_question": "",
        "likely_causes": [],
        "safe_checks": [],
        "recommended_next_steps": [],
        "when_to_get_help": [],
        "safety_note": "Stop if anything feels unsafe and contact a qualified professional."
    }
    for key, default in defaults.items():
        if not isinstance(result.get(key), type(default)):
            result[key] = default
    if result["status"] not in ("needs_info", "diagnosis"):
        result["status"] = "diagnosis"
    result["matched_notes"] = [{"topic": n["topic"]} for n in notes]
    result["model"] = MODEL_NAME
    return result

@app.get("/")
def index():
    return render_template("index.html")

@app.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "model": MODEL_NAME,
        "api_key_configured": bool(os.getenv(API_KEY_ENV, "").strip())
    })

@app.post("/api/diagnose")
def api_diagnose():
    try:
        data = request.get_json(silent=True) or {}
        return jsonify(diagnose(data))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except RuntimeError as exc:
        app.logger.warning("Diagnosis service issue: %s", exc)
        return jsonify({"error": str(exc)}), 503
    except Exception:
        app.logger.exception("Diagnosis request failed")
        return jsonify({"error": "Diagnosis failed. Check Render logs, Gemini key, quota, and model availability."}), 502

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "10000")), debug=False)
