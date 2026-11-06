package server

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/ollama/ollama/api"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/propagation"
	"go.opentelemetry.io/otel/trace"
)

var inferenceTracer = otel.Tracer("github.com/ollama/ollama/server")

func inferenceTelemetryMiddleware(route string) gin.HandlerFunc {
	return func(c *gin.Context) {
		if c.Request == nil {
			c.Next()
			return
		}

		parent := otel.GetTextMapPropagator().Extract(c.Request.Context(), propagation.HeaderCarrier(c.Request.Header))
		ctx, span := inferenceTracer.Start(parent, c.Request.Method+" "+route,
			trace.WithSpanKind(trace.SpanKindServer),
			trace.WithAttributes(
				attribute.String("http.request.method", c.Request.Method),
				attribute.String("http.route", route),
			),
		)
		c.Request = c.Request.WithContext(ctx)

		defer span.End()
		c.Next()

		status := c.Writer.Status()
		span.SetAttributes(attribute.Int("http.response.status_code", status))
		if status >= http.StatusInternalServerError {
			span.SetStatus(codes.Error, http.StatusText(status))
		}
	}
}

type generationProgress struct {
	chunks    int64
	lastEvent time.Time
}

func (p *generationProgress) record(span trace.Span, done bool, inputTokens int, cachedInputTokens *int, outputTokens int, inputDuration, outputDuration time.Duration, doneReason string) {
	p.chunks++
	if !span.IsRecording() {
		return
	}

	now := time.Now()
	if !done && p.chunks%32 != 0 && now.Sub(p.lastEvent) < time.Second {
		return
	}
	p.lastEvent = now

	attrs := []attribute.KeyValue{
		attribute.Int64("ollama.generation.chunks", p.chunks),
	}
	if done {
		attrs = append(attrs,
			attribute.Int("gen_ai.usage.input_tokens", inputTokens),
			attribute.Int("gen_ai.usage.output_tokens", outputTokens),
			attribute.Int64("ollama.prompt_eval_duration_ms", inputDuration.Milliseconds()),
			attribute.Int64("ollama.eval_duration_ms", outputDuration.Milliseconds()),
			attribute.String("ollama.generation.done_reason", doneReason),
		)
		if cachedInputTokens != nil {
			attrs = append(attrs, attribute.Int("ollama.prompt_eval_cached_tokens", *cachedInputTokens))
		}
		span.SetAttributes(attrs...)
	}
	span.AddEvent("ollama.generation.progress", trace.WithAttributes(attrs...))
}

func recordTraceError(span trace.Span, stage string, err error) {
	if err == nil {
		return
	}
	span.SetAttributes(
		attribute.String("ollama.error.stage", stage),
		attribute.String("error.type", fmt.Sprintf("%T", err)),
	)
	span.SetStatus(codes.Error, stage+" failed")
}

func startChatStage(ctx context.Context, name string) (context.Context, trace.Span) {
	return inferenceTracer.Start(ctx, "ollama.chat."+name)
}

func traceChatMessages(messages []api.Message) (string, error) {
	// Image payloads are binary and can make spans enormous. Preserve the text
	// and tool context while leaving those payloads out of the trace.
	traceMessages := append([]api.Message(nil), messages...)
	for i := range traceMessages {
		traceMessages[i].Images = nil
	}

	data, err := json.Marshal(traceMessages)
	if err != nil {
		return "", err
	}
	return string(data), nil
}
