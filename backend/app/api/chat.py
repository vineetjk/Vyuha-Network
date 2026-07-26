import base64
import hashlib
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import Response
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.db import models
from app.schemas import ChatQuery, ChatReply
from app.services.interfaces import BaseAIService, BasePDFService
from app.services import catalyst_ai
from app.api.deps import get_current_user, get_ai_service, get_pdf_service

router = APIRouter(prefix="/chat", tags=["Intelligent Conversational AI"])

@router.post("", response_model=ChatReply)
def post_chat_query(
    payload: ChatQuery, 
    db: Session = Depends(get_db), 
    ai_service: BaseAIService = Depends(get_ai_service),
    current_user: models.User = Depends(get_current_user)
):
    """Submit a text or voice query to the KSP crime records database assistant."""
    query_text = payload.query_text
    
    # 1. Handle base64 audio decoding if present
    if payload.voice_audio_base64:
        try:
            audio_bytes = base64.b64decode(payload.voice_audio_base64)
            # Transcribe audio using STT service fallback
            query_text = ai_service.transcribe_kannada_audio(audio_bytes)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid voice audio payload: {str(e)}"
            )

    if not query_text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query text or voice audio is required"
        )

    # 2. Language detection is a local Unicode check — no AI round-trip. The
    #    analyzer model reads Kannada natively, so we pass the query as-is and
    #    tell the model to answer in Kannada (via language=). This removes a
    #    slow, failure-prone second LLM call that was doubling chat latency.
    # Kannada Unicode block is U+0C80–U+0CFF.
    detected_lang = "kn" if any(0x0C80 <= ord(ch) <= 0x0CFF for ch in query_text) else "en"
    translated_text = query_text

    # 3. Fetch recent FIR cases as LLM context. Keep the payload compact — the
    #    analyzer is a reasoning model, so long BriefFacts blobs balloon its
    #    thinking and push latency past the gateway timeout. District/category/
    #    gravity/date are enough to spot patterns; description is trimmed short.
    cases = (
        db.query(models.CaseMaster)
        .order_by(models.CaseMaster.CrimeRegisteredDate.desc())
        .limit(20)
        .all()
    )
    historical_data = []
    for c in cases:
        historical_data.append({
            "district": c.unit.district.DistrictName if c.unit and c.unit.district else None,
            "category": c.minor_head.CrimeHeadName if c.minor_head else None,
            "gravity": c.gravity.LookupValue if c.gravity else None,
            "date": c.CrimeRegisteredDate.strftime("%Y-%m-%d") if c.CrimeRegisteredDate else None,
            "brief": (c.BriefFacts or "")[:90],
        })

    # 4. Invoke LLM pattern analyzer. Pass the detected language so the model
    #    responds natively in Kannada when the query was Kannada (no fragile
    #    English->Kannada back-translation of the assembled reply).
    ai_response = ai_service.analyze_crime_pattern(translated_text, historical_data, language=detected_lang)
    summary = ai_response.get("summary", "No summary returned")
    patterns = ai_response.get("detected_patterns", []) or []
    recommendations = ai_response.get("recommended_actions", []) or []
    confidence = ai_response.get("confidence_score")

    # Flat text kept for the audit ledger, PDF export and history restore.
    heading_patterns = "ಪತ್ತೆಯಾದ ಮಾದರಿಗಳು" if detected_lang == "kn" else "Detected Patterns"
    heading_actions = "ಶಿಫಾರಸು ಕ್ರಮಗಳು" if detected_lang == "kn" else "Recommended Actions"
    formatted_reply = f"{summary}\n\n"
    if patterns:
        formatted_reply += f"<b>{heading_patterns}:</b>\n" + "\n".join(f"- {p}" for p in patterns) + "\n\n"
    if recommendations:
        formatted_reply += f"<b>{heading_actions}:</b>\n" + "\n".join(f"- {r}" for r in recommendations)

    # 5. Calculate audit verification hash
    timestamp = datetime.now(timezone.utc)
    raw_hash_data = f"{query_text}{formatted_reply}{timestamp}"
    ver_hash = hashlib.sha256(raw_hash_data.encode("utf-8")).hexdigest()

    # 6. Save audit record (store naive UTC in the column; the API returns the
    #    aware value so clients get an ISO string with a UTC offset and render
    #    it in local time instead of misreading UTC as local).
    audit_entry = models.ChatAudit(
        user_id=current_user.id,
        query_text=query_text,
        reply_text=formatted_reply,
        audio_url=None, # In case we add synthesis audio link later
        timestamp=timestamp.replace(tzinfo=None)
    )
    db.add(audit_entry)
    db.commit()

    # 7. Optional Catalyst Zia enrichment (sentiment + keywords of the query).
    #    Runs only when Catalyst AI is enabled; None otherwise.
    nlp = catalyst_ai.analyze_text(translated_text)

    return {
        "original_query": query_text,
        "translated_query": translated_text,
        "reply_text": formatted_reply,
        "language": detected_lang,
        "timestamp": timestamp,
        "verification_hash": ver_hash,
        "sentiment": (nlp or {}).get("sentiment"),
        "sentiment_score": (nlp or {}).get("score"),
        "keywords": (nlp or {}).get("keywords"),
        "summary": summary,
        "detected_patterns": patterns,
        "recommended_actions": recommendations,
        "confidence": confidence,
    }

@router.get("/history")
def get_chat_history(db: Session = Depends(get_db), current_user: models.User = Depends(get_current_user)):
    """Fetch previous chat audit records for auditing and ledger check."""
    audits = db.query(models.ChatAudit).filter(models.ChatAudit.user_id == current_user.id).order_by(models.ChatAudit.timestamp.desc()).all()
    result = []
    for a in audits:
        result.append({
            "id": a.id,
            "query_text": a.query_text,
            "reply_text": a.reply_text,
            # Stored naive UTC → emit with a UTC offset so clients localise it.
            "timestamp": a.timestamp.replace(tzinfo=timezone.utc).isoformat() if a.timestamp else None,
        })
    return result

@router.get("/export-pdf")
def get_pdf_report(
    db: Session = Depends(get_db), 
    pdf_service: BasePDFService = Depends(get_pdf_service),
    current_user: models.User = Depends(get_current_user)
):
    """Generate and return confidential investigation query audit logs PDF."""
    audits = db.query(models.ChatAudit).filter(models.ChatAudit.user_id == current_user.id).order_by(models.ChatAudit.timestamp.asc()).all()
    
    query_history = []
    for a in audits:
        query_history.append({
            "query_text": a.query_text,
            "reply_text": a.reply_text.replace("<b>", "").replace("</b>", ""), # Strip HTML bold tags for PDF
            "timestamp": a.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        })

    if not query_history:
        # Provide placeholder if empty
        query_history.append({
            "query_text": "No previous query logs found.",
            "reply_text": "Ledger is currently empty.",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

    pdf_data = pdf_service.generate_chat_report(query_history)
    
    # Return as PDF application attachment stream
    return Response(
        content=pdf_data,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f"attachment; filename=ksp_audit_report_{int(datetime.now().timestamp())}.pdf"
        }
    )
