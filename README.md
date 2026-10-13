# RepairMate — Agentic AI Repair Diagnosis Agent

RepairMate is a Flask website for preliminary repair troubleshooting. It runs a local safety screen, retrieves matching local repair notes, and uses Google's Gemini API to produce a structured diagnosis report.

## Files
- `app.py` — Flask backend, Gemini integration, local retrieval, safety screening, and health endpoint.
- `templates/index.html` — Website interface.
- `requirements.txt` — Python dependencies.
- `render.yaml` — Optional Render Blueprint configuration.
- `.gitignore` — Helps prevent secrets and temporary files from being committed.

## Get a Gemini API key
Create a key in Google AI Studio: https://aistudio.google.com/app/apikey
Free quota/model access can change by account, region, and date. Check the applicable usage limits. This project does not require an OpenAI API key.

## Deploy to Render
1. Upload the project files to a GitHub repository. Never commit an API key or `.env` file.
2. In Render, choose **New + → Web Service**, then connect the repository.
3. Use build command: `pip install -r requirements.txt`
4. Use start command: `gunicorn app:app`
5. Add environment variables:
   - `GeminiAPIKey` = your secret Gemini API key
   - `GEMINI_MODEL` = `gemini-2.5-flash`
6. Deploy and open the Render URL.
7. Check `/health`. It verifies the web process is running and whether the key variable is present; it does not validate the key against Gemini.

## Run locally
Install dependencies using `pip install -r requirements.txt`, set `GeminiAPIKey` in your shell environment, and run `python app.py`.

## Safety
This is preliminary guidance only and cannot inspect equipment physically. It does not replace a qualified repair professional. For smoke, sparks, gas smell, exposed wiring, fire, electric shock, or a swollen battery, stop troubleshooting and seek appropriate emergency or professional assistance.
