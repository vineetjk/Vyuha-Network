import {
  FileDown,
  Fingerprint,
  Languages,
  ScanText,
  Send,
  ShieldCheck,
  Smile,
  Sparkles,
} from 'lucide-react';
import { useEffect, useMemo, useRef, useState } from 'react';
import type { KeyboardEvent } from 'react';
import { ChatAPI, IntelAPI } from '../../api/endpoints';
import { CATALYST_AI, extractErrorMessage } from '../../api/client';
import { Badge } from '../../components/ui/Badge';
import { AiAnswer, hasStructuredAnswer } from '../../components/ui/AiAnswer';
import { Card } from '../../components/ui/Card';
import { EmptyState } from '../../components/ui/states';
import { formatDateTime } from '../../lib/format';
import type { ChatMessage } from '../../types';

const SUGGESTIONS = [
  'Which districts show rising theft this month?',
  'Summarize narcotics patterns near Bengaluru',
  'ಬೆಂಗಳೂರಿನಲ್ಲಿ ಸೈಬರ್ ಅಪರಾಧ ಪ್ರವೃತ್ತಿ ಏನು?',
  'Recommend patrol priorities for high-risk zones',
];

let messageSeq = 0;
const nextId = () => `msg-${Date.now()}-${messageSeq++}`;

/** Strip the backend's inline <b> markup; render as plain text. */
const cleanReply = (text: string) => text.replace(/<\/?b>/g, '');

// Reconstruct the structured answer from a stored flat reply so restored history
// renders as the styled AiAnswer instead of raw text. The reply_text format is
// "{summary}\n\n<b>heading:</b>\n- item…" with patterns first, actions second —
// we key off order (not the heading text) so it works for English and Kannada.
const parseStructured = (
  raw: string,
): { summary: string; detected_patterns: string[]; recommended_actions: string[] } | null => {
  if (!raw || !raw.includes('<b>')) return null;
  const parts = raw.split('<b>');
  const summary = parts[0].replace(/<\/?b>/g, '').trim();
  const blocks = parts.slice(1).map((seg) =>
    (seg.split('</b>')[1] ?? '')
      .split('\n')
      .map((l) => l.trim())
      .filter((l) => l.startsWith('- '))
      .map((l) => l.slice(2).trim())
      .filter(Boolean),
  );
  return { summary, detected_patterns: blocks[0] ?? [], recommended_actions: blocks[1] ?? [] };
};

export function AssistantPage() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [draft, setDraft] = useState('');
  const [sending, setSending] = useState(false);
  const [exporting, setExporting] = useState(false);
  const [scanning, setScanning] = useState(false);
  const [historyCount, setHistoryCount] = useState<number | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const ocrInputRef = useRef<HTMLInputElement>(null);

  // Zia OCR: extract text from an uploaded FIR/case document into the composer.
  const scanDocument = async (file: File) => {
    setScanning(true);
    try {
      const res = await IntelAPI.ocr(file);
      if (res.available && res.text) {
        setDraft((prev) => (prev ? `${prev}\n${res.text}` : (res.text ?? '')));
        textareaRef.current?.focus();
      } else {
        setMessages((prev) => [
          ...prev,
          {
            id: nextId(),
            sender: 'assistant',
            text: res.message ?? 'No text could be extracted from that document.',
            timestamp: new Date().toISOString(),
            error: true,
          },
        ]);
      }
    } catch (error) {
      setMessages((prev) => [
        ...prev,
        {
          id: nextId(),
          sender: 'assistant',
          text: extractErrorMessage(error, 'Document scan failed.'),
          timestamp: new Date().toISOString(),
          error: true,
        },
      ]);
    } finally {
      setScanning(false);
    }
  };

  // Seed the transcript from the audit ledger.
  useEffect(() => {
    let cancelled = false;
    ChatAPI.history()
      .then((entries) => {
        if (cancelled) return;
        setHistoryCount(entries.length);
        const restored: ChatMessage[] = entries
          .slice(0, 10)
          .reverse()
          .flatMap((entry) => [
            {
              id: `${entry.id}-q`,
              sender: 'user' as const,
              text: entry.query_text,
              timestamp: entry.timestamp,
            },
            {
              id: `${entry.id}-a`,
              sender: 'assistant' as const,
              text: cleanReply(entry.reply_text),
              timestamp: entry.timestamp,
              ...(parseStructured(entry.reply_text) ?? {}),
            },
          ]);
        setMessages(restored);
      })
      .catch(() => {
        if (!cancelled) setHistoryCount(0);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: 'smooth' });
  }, [messages, sending]);

  const send = async (text: string) => {
    const query = text.trim();
    if (!query || sending) return;
    setDraft('');
    setSending(true);
    setMessages((prev) => [
      ...prev,
      { id: nextId(), sender: 'user', text: query, timestamp: new Date().toISOString() },
    ]);
    try {
      const reply = await ChatAPI.send(query);
      setMessages((prev) => [
        ...prev,
        {
          id: nextId(),
          sender: 'assistant',
          text: cleanReply(reply.reply_text),
          timestamp: reply.timestamp,
          language: reply.language,
          translatedQuery:
            reply.translated_query !== reply.original_query ? reply.translated_query : undefined,
          hash: reply.verification_hash,
          sentiment: reply.sentiment,
          keywords: reply.keywords,
          summary: reply.summary,
          detected_patterns: reply.detected_patterns,
          recommended_actions: reply.recommended_actions,
          confidence: reply.confidence,
        },
      ]);
      setHistoryCount((count) => (count === null ? count : count + 1));
    } catch (error) {
      setMessages((prev) => [
        ...prev,
        {
          id: nextId(),
          sender: 'assistant',
          text: extractErrorMessage(error, 'The assistant could not process this query. Please retry.'),
          timestamp: new Date().toISOString(),
          error: true,
        },
      ]);
    } finally {
      setSending(false);
      textareaRef.current?.focus();
    }
  };

  const onComposerKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      void send(draft);
    }
  };

  const exportPdf = async () => {
    setExporting(true);
    try {
      const blob = await ChatAPI.exportPdf();
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `vyuha_audit_report_${new Date().toISOString().slice(0, 10)}.pdf`;
      document.body.appendChild(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(url);
    } catch {
      // Non-fatal; surface inline.
      setMessages((prev) => [
        ...prev,
        {
          id: nextId(),
          sender: 'assistant',
          text: 'PDF export failed. Please try again.',
          timestamp: new Date().toISOString(),
          error: true,
        },
      ]);
    } finally {
      setExporting(false);
    }
  };

  // Hero empty-state: only when there's truly no conversation yet.
  const showEmptyState = useMemo(() => messages.length === 0 && !sending, [messages, sending]);
  // Suggestion chips: always available above the composer while the input is
  // empty — regardless of prior history (which is restored from the ledger).
  const showSuggestionChips = !sending && draft.trim() === '';

  return (
    <main className="page page--flush assistant-page">
      <div className="chat-column">
        <div className="chat-scroll" ref={scrollRef}>
          {showEmptyState ? (
            <EmptyState
              icon={<Sparkles size={30} />}
              title="Ask the investigation assistant"
              message="Query the crime records database in English or Kannada — patterns, hotspots, advisories and case summaries. Every exchange is hash-verified and audited."
            />
          ) : (
            messages.map((message) => {
              const structured =
                message.sender === 'assistant' && !message.error && hasStructuredAnswer(message);
              return (
              <div
                key={message.id}
                className={`chat-msg chat-msg--${message.sender} fade-in`}
              >
                <div
                  className="chat-msg__bubble"
                  style={message.error ? { borderColor: 'var(--status-critical)', color: 'var(--status-critical)' } : undefined}
                >
                  {structured ? <AiAnswer data={message} /> : message.text}
                </div>
                <div className="chat-msg__meta">
                  <span>{formatDateTime(message.timestamp)}</span>
                  {message.language === 'kn' && (
                    <span style={{ display: 'inline-flex', alignItems: 'center', gap: 4 }}>
                      <Languages size={11} />
                      Kannada
                    </span>
                  )}
                  {message.hash && (
                    <span
                      className="chat-msg__hash"
                      title={`SHA-256 verification hash: ${message.hash}`}
                      style={{ display: 'inline-flex', alignItems: 'center', gap: 4 }}
                    >
                      <Fingerprint size={11} />
                      {message.hash.slice(0, 12)}…
                    </span>
                  )}
                </div>
                {!structured && (message.sentiment || (message.keywords && message.keywords.length > 0)) && (
                  <div
                    style={{ display: 'flex', flexWrap: 'wrap', gap: 6, padding: '2px 4px 0' }}
                    title="Catalyst Zia analysis of your query"
                  >
                    {message.sentiment && (
                      <Badge
                        tone={
                          /pos/i.test(message.sentiment)
                            ? 'good'
                            : /neg/i.test(message.sentiment)
                              ? 'critical'
                              : 'neutral'
                        }
                      >
                        <Smile size={11} /> {message.sentiment}
                      </Badge>
                    )}
                    {message.keywords?.slice(0, 5).map((kw) => (
                      <Badge key={kw} tone="accent">
                        {kw}
                      </Badge>
                    ))}
                  </div>
                )}
              </div>
              );
            })
          )}

          {sending && (
            <div className="chat-msg chat-msg--assistant">
              <div className="chat-msg__bubble">
                <span className="typing-dots" aria-label="Assistant is thinking">
                  <span />
                  <span />
                  <span />
                </span>
              </div>
            </div>
          )}
        </div>

        <div className="chat-composer">
          {showSuggestionChips && (
            <div className="suggestion-row">
              {SUGGESTIONS.map((suggestion) => (
                <button
                  key={suggestion}
                  className="suggestion-chip"
                  onClick={() => void send(suggestion)}
                >
                  {suggestion}
                </button>
              ))}
            </div>
          )}
          <div className="chat-composer__row">
            {CATALYST_AI && (
              <>
                <input
                  ref={ocrInputRef}
                  type="file"
                  accept="image/*,.pdf"
                  hidden
                  onChange={(e) => {
                    const file = e.target.files?.[0];
                    if (file) void scanDocument(file);
                    e.target.value = '';
                  }}
                />
                <button
                  className="btn btn--secondary"
                  onClick={() => ocrInputRef.current?.click()}
                  disabled={scanning}
                  aria-label="Scan a document with OCR"
                  title="Extract text from an FIR / case document (Zia OCR)"
                  style={{ height: 42, width: 46, flex: 'none' }}
                >
                  <ScanText size={16} />
                </button>
              </>
            )}
            <textarea
              ref={textareaRef}
              rows={1}
              value={draft}
              placeholder="Ask about patterns, hotspots, suspects… (English or ಕನ್ನಡ)"
              onChange={(e) => setDraft(e.target.value)}
              onKeyDown={onComposerKeyDown}
              aria-label="Message the investigation assistant"
            />
            <button
              className="btn btn--primary"
              onClick={() => void send(draft)}
              disabled={sending || !draft.trim()}
              aria-label="Send query"
              style={{ height: 42, width: 46 }}
            >
              <Send size={16} />
            </button>
          </div>
          <div className="chat-composer__hint">
            {scanning
              ? 'Scanning document with Zia OCR…'
              : 'Enter to send · Shift+Enter for a new line · Replies are grounded in the live FIR database'}
          </div>
        </div>
      </div>

      <aside className="assistant-aside">
        <Card title="Audit ledger" subtitle="Cryptographically registered queries">
          <p style={{ fontSize: 12.5, color: 'var(--text-2)', lineHeight: 1.6 }}>
            {historyCount === null
              ? 'Loading ledger…'
              : `${historyCount} audited ${historyCount === 1 ? 'exchange' : 'exchanges'} on record for your account.`}
          </p>
          <button
            className="btn btn--secondary btn--sm"
            onClick={() => void exportPdf()}
            disabled={exporting}
            style={{ marginTop: 10 }}
          >
            <FileDown size={13} />
            {exporting ? 'Preparing report…' : 'Export PDF report'}
          </button>
        </Card>

        <Card title="Capabilities">
          <ul style={{ margin: 0, paddingLeft: 0, listStyle: 'none', display: 'grid', gap: 10, fontSize: 12.5, color: 'var(--text-2)' }}>
            <li style={{ display: 'flex', gap: 8 }}>
              <Sparkles size={14} style={{ flex: 'none', marginTop: 2, color: 'var(--accent)' }} />
              Pattern detection and advisory actions grounded in the latest 50 FIRs
            </li>
            <li style={{ display: 'flex', gap: 8 }}>
              <Languages size={14} style={{ flex: 'none', marginTop: 2, color: 'var(--accent)' }} />
              Bilingual — ask in Kannada, get answers in Kannada
            </li>
            <li style={{ display: 'flex', gap: 8 }}>
              <ShieldCheck size={14} style={{ flex: 'none', marginTop: 2, color: 'var(--accent)' }} />
              Every reply carries a SHA-256 verification hash for the audit trail
            </li>
          </ul>
        </Card>
      </aside>
    </main>
  );
}
