# Vyuha Network API — Catalyst AppSail (Docker image runtime).
# Docker is used instead of the managed Python runtime because the backend
# pulls native wheels (bcrypt, cryptography, reportlab, pydantic-core) that
# cannot be reliably vendored from a macOS dev machine to Catalyst's Linux
# runtime. The image builds those on Linux at deploy time.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install deps first for layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Runtime config for the Catalyst deployment. The client (Web Client Hosting)
# is a separate origin, and the API is stateless Bearer-token auth, so wildcard
# CORS (credentials off) is safe.
# Catalyst's AppSail gateway handles CORS; the app must not add its own or the
# browser sees duplicate Access-Control-Allow-Origin headers.
#
# Only NON-SECRET plumbing is baked here so it survives container recycles
# (runtime values set via /admin live only in one process and are lost on
# restart). Real AI is on (MOCK_AI_PIPELINE=false) with Groq as the provider
# and the Zia OCR endpoint pre-wired. SECRETS — FALLBACK_AI_API_KEYS (Groq key),
# ZIA_CODELIB_SECRET, CATALYST_AI_TOKEN, DATABASE_URL — are NOT baked in; set
# them via /admin or the Catalyst console so they never enter the image or git.
ENV DISABLE_APP_CORS=true \
    AUTO_SEED=true \
    MOCK_AI_PIPELINE=false \
    CATALYST_AI_ENABLED=false \
    FALLBACK_AI_BASE_URL=https://api.groq.com/openai/v1 \
    FALLBACK_AI_MODEL=llama-3.3-70b-versatile \
    ZIA_OCR_URL=https://vyuha-network-60080167463.development.catalystserverless.in/server/catalyst-zia-services/ocr

# Catalyst injects the port via X_ZOHO_CATALYST_LISTEN_PORT (default 9000).
EXPOSE 9000
CMD ["python3", "-u", "catalyst_server.py"]
