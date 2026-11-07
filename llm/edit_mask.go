package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"sync"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"
)

// EditMaskConfig travels only over the private Go-to-runner connection. In
// particular, it is never passed to apply-template or a tokenizer.
type EditMaskConfig struct {
	RequestID string   `json:"request_id"`
	Version   int      `json:"version"`
	Root      string   `json:"root"`
	Files     []string `json:"files"`
	PathField string   `json:"path_field"`
}

type editMaskTelemetry struct {
	mu      sync.Mutex
	pending []byte
	spans   map[string]trace.Span
}

func (s *llamaServerRunner) registerEditMask(ctx context.Context, id string) func() {
	t := &s.editMaskTelemetry
	t.mu.Lock()
	if t.spans == nil {
		t.spans = make(map[string]trace.Span)
	}
	t.spans[id] = trace.SpanFromContext(ctx)
	t.mu.Unlock()
	return func() { t.mu.Lock(); delete(t.spans, id); t.mu.Unlock() }
}

// Native stderr writes may split an event, or contain several events. Only the
// explicit numeric/status fields enter telemetry, never arguments or contents.
func (t *editMaskTelemetry) consume(chunk []byte) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.pending = append(t.pending, chunk...)
	for {
		pos := bytes.IndexByte(t.pending, '\n')
		if pos < 0 {
			break
		}
		line := t.pending[:pos]
		t.pending = t.pending[pos+1:]
		_, payload, found := bytes.Cut(line, []byte("OLLAMA_EDIT_MASK "))
		if !found {
			continue
		}
		var event struct {
			RequestID    string `json:"request_id"`
			Status       string `json:"status"`
			Reason       string `json:"reason"`
			CallIndex    int    `json:"call_index"`
			Files        int    `json:"files"`
			ContentBytes int64  `json:"content_bytes"`
			IndexBytes   int64  `json:"index_bytes"`
			MaskedTokens int64  `json:"masked_tokens"`
		}
		if json.Unmarshal(payload, &event) != nil {
			continue
		}
		if span := t.spans[event.RequestID]; span != nil {
			span.AddEvent("ollama.edit_mask."+event.Status, trace.WithAttributes(
				attribute.String("reason", event.Reason), attribute.Int("call_index", event.CallIndex),
				attribute.Int("files", event.Files), attribute.Int64("content_bytes", event.ContentBytes),
				attribute.Int64("index_bytes", event.IndexBytes), attribute.Int64("masked_tokens", event.MaskedTokens)))
		}
	}
	if len(t.pending) > 64*1024 {
		t.pending = nil
	}
}

func (s *llamaServerRunner) requireEditMaskCapability(ctx context.Context) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, fmt.Sprintf("http://127.0.0.1:%d/health", s.port), nil)
	if err != nil {
		return err
	}
	res, err := s.httpClient().Do(req)
	if err != nil {
		return fmt.Errorf("oldString mask capability check: %w", err)
	}
	defer res.Body.Close()
	var health struct {
		EditMask int `json:"ollama_oldstring_mask"`
	}
	if res.StatusCode != http.StatusOK || json.NewDecoder(res.Body).Decode(&health) != nil || health.EditMask != 1 {
		return fmt.Errorf("OLLAMA_OPENCODE_EDIT_CONSTRAINTS requires a patched llama.cpp payload advertising oldString-mask-v1; rebuild the native payload")
	}
	trace.SpanFromContext(ctx).AddEvent("ollama.edit_mask.capability", trace.WithAttributes(attribute.Int("version", health.EditMask)))
	return nil
}
