import os
import json
import re
import time
import google.generativeai as genai
from app.services.interfaces import BaseAIService
from app.services.observability import metrics


def _extract_json(text: str) -> dict:
    """Parse a JSON object from an LLM reply, tolerating code fences, <think>
    reasoning blocks, and trailing prose after the object (some models, incl.
    the Catalyst GLM, append an explanation after the JSON — which plain
    json.loads rejects as 'Extra data')."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    # Prefer a fenced ```json block (greedy, so nested braces survive).
    if "```" in text:
        m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.DOTALL)
        if m:
            text = m.group(1)
    # Try every '{' as a start and raw_decode from there — a reasoning model may
    # emit prose (with stray braces) before the real object, so return the first
    # brace position that yields a dict with our expected keys, else the first
    # that parses at all.
    decoder = json.JSONDecoder()
    first_ok = None
    for i, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text[i:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            if first_ok is None:
                first_ok = obj
            if obj.keys() & {"summary", "detected_patterns", "translated_query", "ok"}:
                return obj
    if first_ok is not None:
        return first_ok
    return json.loads(text)


def _strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


# Appended to prompts when the investigator asked in Kannada, so the model
# responds natively in Kannada (JSON keys stay English for parsing).
_KANNADA_JSON = (
    " IMPORTANT: Write ALL string VALUES (summary, detected_patterns, "
    "recommended_actions, audit_explanation) in Kannada (ಕನ್ನಡ script). "
    "Keep the JSON keys in English."
)
_KANNADA_TEXT = " IMPORTANT: Write the entire response in Kannada (ಕನ್ನಡ script)."

class GeminiAIService(BaseAIService):
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)

    def analyze_crime_pattern(self, query_text: str, historical_records: list, language: str = "en") -> dict:
        try:
            model = genai.GenerativeModel("gemini-1.5-flash")
            records_summary = json.dumps(historical_records[:30], default=str)
            lang_note = _KANNADA_JSON if language == "kn" else ""
            prompt = f"""
            You are a lead crime analyst for the Karnataka State Police.
            Analyze the following investigator query and relevant database crime records.
            Identify key crime patterns, hotspot indicators, anomalies, and recommended actions.
            Return the output STRICTLY as a JSON object with these keys:
            - 'summary' (str): Short overview of the analysis.
            - 'detected_patterns' (list of str): List of key crime patterns identified.
            - 'confidence_score' (float): Confidence value between 0.0 and 1.0.
            - 'recommended_actions' (list of str): Strategic police recommendations.
            - 'audit_explanation' (str): Technical explanation of your reasoning (Chain-of-Thought).

            Investigator Query: {query_text}
            Crime Records: {records_summary}
            {lang_note}
            """
            response = model.generate_content(prompt)
            text = response.text.strip()
            if text.startswith("```json"):
                text = text.split("```json")[1].split("```")[0].strip()
            elif text.startswith("```"):
                text = text.split("```")[1].split("```")[0].strip()
            return json.loads(text)
        except Exception as e:
            return {
                "summary": f"Analyzed query: '{query_text}' (Fallback mode)",
                "detected_patterns": ["Pattern detection completed via fallback heuristics. Category density matches standard averages."],
                "confidence_score": 0.5,
                "recommended_actions": ["Verify regional records manually. Monitor recidivism parameters."],
                "audit_explanation": f"API Error Fallback. Error details: {str(e)}"
            }

    def calculate_risk_score_explanation(self, criminal_name: str, priors_count: int, crime_types: list) -> str:
        try:
            model = genai.GenerativeModel("gemini-1.5-flash")
            prompt = f"""
            Generate an explainable AI risk assessment for the following criminal suspect.
            Name: {criminal_name}
            Number of Priors: {priors_count}
            Types of Crimes Committed: {', '.join(crime_types)}

            Provide a 3-4 sentence explanation detailing why this person carries a specific recidivism risk.
            Keep the tone professional and clinical, suitable for a police intelligence dashboard.
            """
            response = model.generate_content(prompt)
            return response.text.strip()
        except Exception:
            return f"Standard risk assessment: The suspect {criminal_name} has {priors_count} priors involving severe charges. Co-offender associations suggest persistent criminal patterns."

    def translate_kannada_query(self, query_text: str) -> dict:
        query_cleaned = query_text.strip()
        try:
            model = genai.GenerativeModel("gemini-1.5-flash")
            prompt = f"""
            You are a translation assistant for the Karnataka State Police.
            Detect the language of the following input. If it is in Kannada, translate it to English. If it is in English, keep it as is.
            Provide output strictly as a JSON object with these keys:
            - 'original_query' (str)
            - 'translated_query' (str)
            - 'detected_language' (str) ('kn' or 'en')
            - 'confidence' (float)

            Input Query: {query_cleaned}
            """
            response = model.generate_content(prompt)
            text = response.text.strip()
            if text.startswith("```json"):
                text = text.split("```json")[1].split("```")[0].strip()
            elif text.startswith("```"):
                text = text.split("```")[1].split("```")[0].strip()
            return json.loads(text)
        except Exception:
            is_kannada = any(ord(char) > 127 for char in query_cleaned)
            return {
                "original_query": query_cleaned,
                "translated_query": query_cleaned,
                "detected_language": "kn" if is_kannada else "en",
                "confidence": 0.7
            }

    def transcribe_kannada_audio(self, audio_bytes: bytes) -> str:
        return "ಬೆಂಗಳೂರಿನಲ್ಲಿ ನಡೆದ ಇತ್ತೀಚಿನ ಕಳ್ಳತನ ಪ್ರಕರಣಗಳ ವಿವರ ಕೊಡಿ"


class MockAIService(BaseAIService):
    def analyze_crime_pattern(self, query_text: str, historical_records: list, language: str = "en") -> dict:
        if language == "kn":
            return {
                "summary": "ಇತ್ತೀಚಿನ ಎಫ್‌ಐಆರ್ ದಾಖಲೆಗಳ ವಿಶ್ಲೇಷಣೆ (ಡೆಮೊ ಪ್ರತಿಕ್ರಿಯೆ).",
                "detected_patterns": [
                    "ಬೆಂಗಳೂರಿನಲ್ಲಿ ರಾತ್ರಿ ವೇಳೆ ಸೈಬರ್ ಅಪರಾಧ ಪ್ರಕರಣಗಳಲ್ಲಿ ಸ್ವಲ್ಪ ಏರಿಕೆ.",
                    "ಹುಬ್ಬಳ್ಳಿ ಮತ್ತು ಬೆಳಗಾವಿಯಲ್ಲಿ ಸಂಭಾವ್ಯ ಮಾದಕವಸ್ತು ಜಾಲ.",
                    "ಕಲಬುರಗಿಯಲ್ಲಿ ನಿರುದ್ಯೋಗದೊಂದಿಗೆ ಕಳ್ಳತನದ ಸಂಬಂಧ.",
                ],
                "confidence_score": 0.85,
                "recommended_actions": [
                    "ಬೆಂಗಳೂರು ನಗರ ವಲಯಗಳಲ್ಲಿ ರಾತ್ರಿ 10ರಿಂದ 3ರವರೆಗೆ ಗಸ್ತು ಹೆಚ್ಚಿಸಿ.",
                    "ಪುನರಪರಾಧಿಗಳ ಜಾಲವನ್ನು ಪರಿಶೀಲಿಸಿ.",
                    "ಹೆಚ್ಚಿನ ನಿರುದ್ಯೋಗ ವಲಯಗಳಲ್ಲಿ ಕೌಶಲ್ಯ ತರಬೇತಿ ಕಾರ್ಯಕ್ರಮ.",
                ],
                "audit_explanation": "90 ದಾಖಲೆಗಳ ವಿಶ್ಲೇಷಣೆ; 3 ಪ್ರಮುಖ ಅಸಂಗತತೆಗಳು ಪತ್ತೆ.",
            }
        return {
            "summary": f"Mock AI Analysis for query: '{query_text}'",
            "detected_patterns": [
                "Slight rise in Cybercrime incidents during night hours in Bengaluru.",
                "Potential narcotics distribution ring spanning Hubballi and Belagavi.",
                "Socio-economic factors suggest high unemployment correlation with minor thefts in Kalaburagi."
            ],
            "confidence_score": 0.85,
            "recommended_actions": [
                "Increase police patrol frequency in Bengaluru Urban zones between 10 PM and 3 AM.",
                "Query joint networks of repeat offenders 'Blade Ramesh' and 'Double Anand'.",
                "Conduct localized skill training initiatives in high-unemployment crime clusters."
            ],
            "audit_explanation": "Chain-of-thought: Analyzed 90 historical crime reports. Cross-referenced category densities and location-time clusters. Found 3 key anomaly peaks."
        }

    def calculate_risk_score_explanation(self, criminal_name: str, priors_count: int, crime_types: list) -> str:
        return f"Reasoning for {criminal_name}: Priors: {priors_count}. Crime types: {', '.join(crime_types)}. Recidivism probability is elevated due to repeating offenses of high severity. Social network shows links to active crime ring bosses."

    def translate_kannada_query(self, query_text: str) -> dict:
        query_cleaned = query_text.strip()
        local_kannada_map = {
            "ಬೆಂಗಳೂರಿನಲ್ಲಿ ಕಳ್ಳತನ": "Theft in Bengaluru",
            "ಕಳ್ಳತನ": "Theft",
            "ಕೊಲೆ ಪ್ರಕರಣ": "Homicide case",
            "ನಾರ್ಕೋಟಿಕ್ಸ್": "Narcotics",
            "ಸೈಬರ್ ಕ್ರೈಮ್": "Cybercrime",
            "ಹೊದಿಕೆ": "Summary",
            "ವರದಿ": "Report",
        }
        translated = local_kannada_map.get(query_cleaned, query_cleaned)
        is_kannada = any(ord(char) > 127 for char in query_cleaned)
        return {
            "original_query": query_cleaned,
            "translated_query": translated,
            "detected_language": "kn" if is_kannada else "en",
            "confidence": 0.95
        }

    def transcribe_kannada_audio(self, audio_bytes: bytes) -> str:
        return "ಬೆಂಗಳೂರಿನಲ್ಲಿ ನಡೆದ ಇತ್ತೀಚಿನ ಕಳ್ಳತನ ಪ್ರಕರಣಗಳ ವಿವರ ಕೊಡಿ"


class LLMAIService(BaseAIService):
    """
    Real LLM analysis via an OpenAI-compatible API. Defaults to Groq's free
    endpoint (open models such as Llama 3.x). Configured entirely by env:

      FALLBACK_AI_BASE_URL  (default https://api.groq.com/openai/v1)
      FALLBACK_AI_MODEL     (default llama-3.3-70b-versatile)
      FALLBACK_AI_API_KEYS  (comma-separated; tried in order for rate-limit
                             failover on the free tier)

    Every method degrades to the heuristic MockAIService on any error, so a
    missing/invalid key or a network blip never breaks the request.
    """

    def __init__(self):
        self.base_url = (os.getenv("FALLBACK_AI_BASE_URL") or "https://api.groq.com/openai/v1").rstrip("/")
        self.model = (
            os.getenv("FALLBACK_AI_MODEL") or os.getenv("GROQ_MODEL") or "llama-3.3-70b-versatile"
        )
        raw_keys = os.getenv("FALLBACK_AI_API_KEYS") or os.getenv("GROQ_API_KEY") or ""
        self.keys = [k.strip() for k in raw_keys.split(",") if k.strip()]
        self.provider = "groq" if "groq" in self.base_url else self.base_url

    @staticmethod
    def is_configured() -> bool:
        return bool(os.getenv("FALLBACK_AI_API_KEYS") or os.getenv("GROQ_API_KEY"))

    def _chat(self, system: str, user: str, json_mode: bool = False, max_tokens: int = 1024) -> str:
        import requests

        url = f"{self.base_url}/chat/completions"
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.4,
            "max_tokens": max_tokens,
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}

        last_err: Exception = RuntimeError("No FALLBACK_AI_API_KEYS configured")
        for key in self.keys:  # rotate keys on rate-limit / auth failure
            try:
                resp = requests.post(
                    url,
                    headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                    json=body,
                    timeout=30,
                )
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"].strip()
            except Exception as exc:  # noqa: BLE001 — try the next key
                last_err = exc
                continue
        raise last_err

    def analyze_crime_pattern(self, query_text: str, historical_records: list, language: str = "en") -> dict:
        try:
            records = json.dumps(historical_records[:30], default=str)
            system = (
                "You are a lead crime analyst for the Karnataka State Police. "
                "Reply ONLY with a single JSON object, no prose."
            )
            user = (
                "Analyse the investigator query against the recent FIR records and return JSON with keys: "
                "summary (string), detected_patterns (array of strings), confidence_score (number 0-1), "
                "recommended_actions (array of strings), audit_explanation (string). "
                f"Investigator query: {query_text}\nFIR records: {records}"
                + (_KANNADA_JSON if language == "kn" else "")
            )
            start = time.perf_counter()
            result = json.loads(self._chat(system, user, json_mode=True, max_tokens=1100))
            metrics.record_ai(True, self.provider, self.model, (time.perf_counter() - start) * 1000)
            return result
        except Exception as exc:  # noqa: BLE001 — graceful degrade
            metrics.record_ai(False, error=exc)
            print(f"[LLMAIService] analyze_crime_pattern fell back: {exc}")
            return MockAIService().analyze_crime_pattern(query_text, historical_records, language)

    def calculate_risk_score_explanation(self, criminal_name: str, priors_count: int, crime_types: list) -> str:
        try:
            system = (
                "You are a police intelligence analyst. Write a clinical, professional 3-4 sentence "
                "recidivism risk assessment suitable for a dashboard. No preamble."
            )
            user = (
                f"Suspect: {criminal_name}. Prior cases: {priors_count}. "
                f"Offence types: {', '.join(crime_types) or 'unknown'}."
            )
            start = time.perf_counter()
            result = self._chat(system, user, max_tokens=320)
            metrics.record_ai(True, self.provider, self.model, (time.perf_counter() - start) * 1000)
            return result
        except Exception as exc:  # noqa: BLE001
            metrics.record_ai(False, error=exc)
            print(f"[LLMAIService] risk_explanation fell back: {exc}")
            return MockAIService().calculate_risk_score_explanation(criminal_name, priors_count, crime_types)

    def translate_kannada_query(self, query_text: str) -> dict:
        cleaned = query_text.strip()
        try:
            system = "You are a translation assistant for the Karnataka State Police. Reply ONLY with JSON."
            user = (
                "Detect the language of the input. If Kannada, translate to English; if English, keep it. "
                "Return JSON with keys: original_query (string), translated_query (string), "
                "detected_language ('kn' or 'en'), confidence (number 0-1). "
                f"Input: {cleaned}"
            )
            start = time.perf_counter()
            result = json.loads(self._chat(system, user, json_mode=True, max_tokens=400))
            metrics.record_ai(True, self.provider, self.model, (time.perf_counter() - start) * 1000)
            return result
        except Exception as exc:  # noqa: BLE001
            metrics.record_ai(False, error=exc)
            print(f"[LLMAIService] translate fell back: {exc}")
            return MockAIService().translate_kannada_query(query_text)

    def transcribe_kannada_audio(self, audio_bytes: bytes) -> str:
        # Groq chat models don't transcribe audio; keep the existing stub.
        return MockAIService().transcribe_kannada_audio(audio_bytes)


class _CatalystToken:
    """
    Process-wide cache for the Catalyst OAuth access token.

    Catalyst access tokens expire in ~1 hour, so a static token in env is only
    good for one hour. When the long-lived refresh-token credentials are set,
    this mints an access token from them, caches it, and auto-renews ~2 min
    before expiry — the durable way to run Catalyst as a deployed primary AI.
    A static GLM_AI_TOKEN (quick testing / manual override) always wins.

    Note: keys use a GLM_ prefix (not CATALYST_) because Catalyst reserves the
    CATALYST_ env-var prefix and its console rejects user-defined CATALYST_* vars.
    """

    _access_token: str = ""
    _expires_at: float = 0.0

    @classmethod
    def get(cls) -> str:
        static = os.getenv("GLM_AI_TOKEN")
        if static:
            return static
        if cls._access_token and time.time() < cls._expires_at:
            return cls._access_token
        cls._refresh()
        return cls._access_token

    @classmethod
    def invalidate(cls) -> None:
        cls._access_token = ""
        cls._expires_at = 0.0

    @classmethod
    def _refresh(cls) -> None:
        import requests

        accounts = os.getenv("GLM_ACCOUNTS_URL", "https://accounts.zoho.in").rstrip("/")
        resp = requests.post(
            f"{accounts}/oauth/v2/token",
            params={
                "grant_type": "refresh_token",
                "client_id": os.getenv("GLM_CLIENT_ID", ""),
                "client_secret": os.getenv("GLM_CLIENT_SECRET", ""),
                "refresh_token": os.getenv("GLM_REFRESH_TOKEN", ""),
            },
            timeout=20,
        )
        resp.raise_for_status()
        data = resp.json()
        token = data.get("access_token")
        if not token:
            # Zoho returns {"error": "..."} with HTTP 200 on bad creds.
            raise RuntimeError(f"Catalyst token refresh failed: {data}")
        cls._access_token = token
        cls._expires_at = time.time() + max(60, int(data.get("expires_in", 3600)) - 120)


class CatalystGLMService(BaseAIService):
    """
    Catalyst QuickML GLM chat — the PRIMARY AI. OpenAI-compatible; on any
    failure each method delegates to `fallback` (Groq, then heuristic mock), so
    the chain is Catalyst → Groq → mock.

    Config (env) — either a static token OR the refresh-token credentials.
    Keys use a GLM_ prefix because Catalyst reserves CATALYST_* (its console
    rejects user-defined CATALYST_* env vars):
      Static (expires in ~1h, for quick tests):
        GLM_AI_TOKEN     Bearer access token
      Durable (auto-renewed, for deployment):
        GLM_REFRESH_TOKEN  long-lived refresh token (never expires)
        GLM_CLIENT_ID      Self Client id from api-console.zoho.in
        GLM_CLIENT_SECRET  Self Client secret
        GLM_ACCOUNTS_URL   default https://accounts.zoho.in
      Endpoint (safe to bake):
        GLM_AI_URL    default the project's GLM chat endpoint
        GLM_AI_ORG    default 60080167463
        GLM_AI_MODEL  default crm-di-glm47b_30b_it
    """

    DEFAULT_URL = "https://api.catalyst.zoho.in/quickml/v1/project/46808000000019001/glm/chat"

    def __init__(self, fallback: BaseAIService):
        self.url = os.getenv("GLM_AI_URL") or self.DEFAULT_URL
        self.org = os.getenv("GLM_AI_ORG", "60080167463")
        self.model = os.getenv("GLM_AI_MODEL", "crm-di-glm47b_30b_it")
        self.fallback = fallback

    @staticmethod
    def is_configured() -> bool:
        if os.getenv("GLM_AI_TOKEN"):
            return True
        return bool(
            os.getenv("GLM_REFRESH_TOKEN")
            and os.getenv("GLM_CLIENT_ID")
            and os.getenv("GLM_CLIENT_SECRET")
        )

    def _chat(self, system: str, user: str, max_tokens: int = 1024, json_mode: bool = False,
              temperature: float = 0.4) -> str:
        import requests

        # JSON tasks want determinism (less reasoning drift); prose can breathe.
        if json_mode:
            temperature = 0.1
        # Two endpoint shapes: GLM chat is OpenAI-style (`messages`); the VLM
        # (e.g. VL-Qwen) uses `prompt` + `system_prompt` (+ optional images).
        if "/vlm/" in self.url:
            body = {
                "model": self.model,
                "prompt": user,
                "system_prompt": system,
                "images": [],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": 0.9,
                "top_k": 50,
            }
        else:
            body = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": False,
            }
            # Reasoning models (this GLM included) otherwise spend the whole
            # token budget "thinking" and never reach the JSON — slow AND
            # unparseable. Ask the server to disable thinking for JSON tasks.
            if json_mode:
                body["chat_template_kwargs"] = {"enable_thinking": False}
        # NB: `json_mode` intentionally does NOT send OpenAI's response_format —
        # this QuickML GLM endpoint rejects it with HTTP 400. JSON is coaxed via
        # the prompt and parsed leniently (_extract_json digs the object out).
        # Retry once on 401: the cached access token may have just expired, so
        # invalidate and re-mint from the refresh token (skip when a static
        # token is pinned — re-minting can't help there).
        using_refresh = not os.getenv("GLM_AI_TOKEN")
        for attempt in range(2):
            resp = requests.post(
                self.url,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {_CatalystToken.get()}",
                    "CATALYST-ORG": self.org,
                },
                json=body,
                timeout=45,
            )
            if resp.status_code == 401 and attempt == 0 and using_refresh:
                _CatalystToken.invalidate()
                continue
            if resp.status_code >= 400:
                # Surface the body — a bare "400 Client Error" hides the reason
                # (unsupported param, bad model name, quota, etc.).
                raise RuntimeError(f"GLM HTTP {resp.status_code}: {resp.text[:300]}")
            if os.getenv("GLM_DEBUG", "").lower() == "true":
                print(f"[CatalystGLM] raw response: {resp.text[:700]}")
            return self._extract_content(resp.json()).strip()

    @staticmethod
    def _extract_content(data) -> str:
        """
        Pull the generated text from a QuickML GLM response. The endpoint is
        *mostly* OpenAI-shaped but not guaranteed, so probe the known layouts
        and, if none match, raise with the actual keys so the fallback log
        surfaces the real structure instead of a bare KeyError.
        """
        if isinstance(data, str):
            return data
        if not isinstance(data, dict):
            raise RuntimeError(f"GLM: unexpected response type {type(data).__name__}")
        # OpenAI-style: choices[0].message.content | choices[0].text
        choices = data.get("choices")
        if isinstance(choices, list) and choices:
            c0 = choices[0]
            if isinstance(c0, dict):
                msg = c0.get("message") or c0.get("delta") or {}
                if isinstance(msg, dict) and msg.get("content"):
                    return msg["content"]
                if c0.get("text"):
                    return c0["text"]
        # Flat alternatives seen across Zoho/QuickML serving responses
        for key in ("output", "text", "response", "content", "result",
                    "answer", "generated_text", "prediction", "completion"):
            v = data.get(key)
            if isinstance(v, str) and v.strip():
                return v
        # Nested envelopes: {"data": {...|"..."}} or {"message": {"content": "..."}}
        nested = data.get("data")
        if isinstance(nested, (dict, str)):
            try:
                return CatalystGLMService._extract_content(nested)
            except RuntimeError:
                pass
        msg = data.get("message")
        if isinstance(msg, dict) and msg.get("content"):
            return msg["content"]
        raise RuntimeError(
            f"GLM: no content field; keys={list(data.keys())} "
            f"sample={json.dumps(data, default=str)[:400]}"
        )

    def analyze_crime_pattern(self, query_text: str, historical_records: list, language: str = "en") -> dict:
        try:
            # Fewer, leaner records → less for the reasoning model to chew on.
            # A terse prompt matters: verbose "no markdown / start with {"
            # instructions get echoed back as chain-of-thought and burn the
            # token budget before the JSON is produced.
            records = json.dumps(historical_records[:12], default=str)
            system = (
                "You are a crime analyst for the Karnataka State Police. "
                "Respond with a single JSON object only."
            )
            user = (
                "Keys: summary (string), detected_patterns (array, <=4), "
                "confidence_score (0-1), recommended_actions (array, <=4), "
                "audit_explanation (string).\n"
                f"Query: {query_text}\nFIR records: {records}"
                + (_KANNADA_JSON if language == "kn" else "")
            )
            start = time.perf_counter()
            # Cap tokens so a slow reasoning pass still returns before the
            # AppSail gateway's ~30s request timeout (which shows as the UI
            # hanging on "…"). Enough for reasoning + a compact JSON object.
            raw = self._chat(system, user, 1400, json_mode=True)
            # Use Catalyst only when it yields a real analysis object; if it just
            # reasoned out loud (no parseable JSON), fall back for a clean answer
            # instead of surfacing raw chain-of-thought.
            result = _extract_json(raw)
            if not (isinstance(result, dict) and result.get("summary")):
                raise ValueError("no analysis JSON in GLM reply")
            metrics.record_ai(True, "catalyst", self.model, (time.perf_counter() - start) * 1000)
            return result
        except Exception as exc:  # noqa: BLE001
            print(f"[CatalystGLM] analyze_crime_pattern fell back: {exc}")
            return self.fallback.analyze_crime_pattern(query_text, historical_records, language)

    def calculate_risk_score_explanation(self, criminal_name: str, priors_count: int, crime_types: list) -> str:
        try:
            system = (
                "You are a police intelligence analyst. Write a clinical, professional 3-4 sentence "
                "recidivism risk assessment suitable for a dashboard. No preamble."
            )
            user = (
                f"Suspect: {criminal_name}. Prior cases: {priors_count}. "
                f"Offence types: {', '.join(crime_types) or 'unknown'}."
            )
            start = time.perf_counter()
            result = _strip_think(self._chat(system, user, 320))
            metrics.record_ai(True, "catalyst", self.model, (time.perf_counter() - start) * 1000)
            return result
        except Exception as exc:  # noqa: BLE001
            print(f"[CatalystGLM] risk_explanation fell back: {exc}")
            return self.fallback.calculate_risk_score_explanation(criminal_name, priors_count, crime_types)

    def translate_kannada_query(self, query_text: str) -> dict:
        try:
            system = (
                "You are a translation assistant for the Karnataka State Police. "
                "Output ONLY a single JSON object, nothing else — no reasoning, no "
                "markdown, no text before or after. Start your reply with '{'."
            )
            user = (
                "Detect the language of the input. If Kannada, translate to English; if English, keep it. "
                "Return JSON with keys: original_query (string), translated_query (string), "
                "detected_language ('kn' or 'en'), confidence (number 0-1). "
                f"Input: {query_text.strip()}"
            )
            start = time.perf_counter()
            result = _extract_json(self._chat(system, user, 400, json_mode=True))
            metrics.record_ai(True, "catalyst", self.model, (time.perf_counter() - start) * 1000)
            return result
        except Exception as exc:  # noqa: BLE001
            print(f"[CatalystGLM] translate fell back: {exc}")
            return self.fallback.translate_kannada_query(query_text)

    def transcribe_kannada_audio(self, audio_bytes: bytes) -> str:
        return self.fallback.transcribe_kannada_audio(audio_bytes)


class AIServiceFactory:
    @staticmethod
    def get_ai_service() -> BaseAIService:
        """
        AI selection (Open/Closed). Chain: Catalyst GLM (primary, if
        CATALYST_AI_TOKEN) → Groq (FALLBACK_AI_*) → Gemini → heuristic mock.
        """
        if LLMAIService.is_configured():
            base: BaseAIService = LLMAIService()
        else:
            api_key = os.getenv("GEMINI_API_KEY")
            mock_pipeline = os.getenv("MOCK_AI_PIPELINE", "true").lower() == "true"
            base = MockAIService() if (not api_key or mock_pipeline) else GeminiAIService(api_key=api_key)

        if CatalystGLMService.is_configured():
            return CatalystGLMService(fallback=base)
        return base
