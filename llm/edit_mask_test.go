package llm

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"os/exec"
	"reflect"
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
	"golang.org/x/sync/semaphore"
)

type maskTransport func(*http.Request) (*http.Response, error)

func (f maskTransport) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }
func maskHTTPResponse(body string) *http.Response {
	return &http.Response{StatusCode: 200, Body: io.NopCloser(strings.NewReader(body)), Header: make(http.Header)}
}

func TestEditMaskCapabilityAndPrivateMetadata(t *testing.T) {
	for _, capability := range []bool{false, true} {
		t.Run(map[bool]string{false: "old_payload", true: "patched_payload"}[capability], func(t *testing.T) {
			sent := false
			runner := &llamaServerRunner{sem: semaphore.NewWeighted(1), cmd: &exec.Cmd{}}
			runner.client = &http.Client{Transport: maskTransport(func(req *http.Request) (*http.Response, error) {
				if req.URL.Path == "/health" {
					if capability {
						return maskHTTPResponse(`{"status":"ok","ollama_oldstring_mask":1}`), nil
					}
					return maskHTTPResponse(`{"status":"ok"}`), nil
				}
				if req.URL.Path != "/v1/chat/completions" {
					t.Fatalf("unexpected endpoint %s", req.URL.Path)
				}
				var body map[string]json.RawMessage
				if err := json.NewDecoder(req.Body).Decode(&body); err != nil {
					t.Fatal(err)
				}
				if _, ok := body["_ollama_oldstring_mask"]; !ok {
					t.Fatal("private metadata not sent")
				}
				if string(body["backend_sampling"]) != "false" {
					t.Fatal("backend sampling not disabled")
				}
				sent = true
				return maskHTTPResponse("data: [DONE]\n\n"), nil
			})}
			opts := api.DefaultOptions()
			req := ChatRequest{Messages: []api.Message{{Role: "user", Content: "edit it"}}, Options: &opts,
				EditMask: &EditMaskConfig{Version: 1, RequestID: "test", Root: "/tmp", Files: []string{}, PathField: "filePath"}}
			err := runner.Chat(context.Background(), req, func(ChatResponse) {})
			if capability && err != nil {
				t.Fatal(err)
			}
			if !capability && (err == nil || !strings.Contains(err.Error(), "oldString-mask-v1")) {
				t.Fatalf("missing capability error: %v", err)
			}
			if sent != capability {
				t.Fatalf("completion sent = %v", sent)
			}
		})
	}
}

func TestEditMaskPromptInputsEqual(t *testing.T) {
	opts := api.DefaultOptions()
	props := api.NewToolPropertiesMap()
	for _, name := range []string{"newString", "oldString", "filePath"} {
		props.Set(name, api.ToolProperty{Type: api.PropertyType{"string"}})
	}
	req := ChatRequest{Messages: []api.Message{{Role: "system", Content: "<env>\nWorking directory: /tmp\n</env>"}}, Options: &opts,
		Tools: api.Tools{{Type: "function", Function: api.ToolFunction{Name: "edit", Parameters: api.ToolFunctionParameters{Type: "object", Required: []string{"newString", "oldString", "filePath"}, Properties: props}}}}}
	runner := &llamaServerRunner{}
	for _, afterRead := range []bool{false, true} {
		if afterRead {
			req.Messages = append(req.Messages, api.Message{Role: "tool", ToolName: "read", Content: "file contents\n"})
		}
		control, err := runner.llamaServerChatRequest(req, true)
		if err != nil {
			t.Fatal(err)
		}
		req.EditMask = &EditMaskConfig{Version: 1, Root: "/tmp", PathField: "filePath"}
		constrained, err := runner.llamaServerChatRequest(req, true)
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(control, constrained) {
			t.Fatal("private mask leaked into template request")
		}
		encoded, _ := json.Marshal(req)
		if strings.Contains(string(encoded), "path_field") || strings.Contains(string(encoded), "EditMask") {
			t.Fatal("private metadata is publicly serialized")
		}
		req.EditMask = nil
	}
}

func TestEditMaskTelemetrySplitLines(t *testing.T) {
	recorder := tracetest.NewSpanRecorder()
	provider := sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(recorder))
	ctx, span := provider.Tracer("test").Start(context.Background(), "generate")
	runner := &llamaServerRunner{}
	unregister := runner.registerEditMask(ctx, "request")
	payload := `prefix OLLAMA_EDIT_MASK {"request_id":"request","status":"active","reason":"selected_file","files":2,"content_bytes":42,"index_bytes":168,"masked_tokens":30,"contents":"NEVER_TELEMETRY"}` + "\n"
	runner.editMaskTelemetry.consume([]byte(payload[:25]))
	runner.editMaskTelemetry.consume([]byte(payload[25:]))
	unregister()
	span.End()
	spans := recorder.Ended()
	if len(spans) != 1 || len(spans[0].Events()) != 1 {
		t.Fatal("missing mask telemetry")
	}
	event := spans[0].Events()[0]
	if event.Name != "ollama.edit_mask.active" {
		t.Fatalf("event = %+v", event)
	}
	for _, value := range event.Attributes {
		if strings.Contains(value.Value.Emit(), "NEVER_TELEMETRY") {
			t.Fatal("file contents in telemetry")
		}
	}
}

func TestEditMaskExperimentDisablesDraft(t *testing.T) {
	t.Setenv("OLLAMA_OPENCODE_EDIT_EXPERIMENT", "1")
	t.Setenv("OLLAMA_OPENCODE_EDIT_CONSTRAINTS", "0")
	opts := api.DefaultOptions()
	opts.DraftNumPredict = 5
	args := appendDraftArgs(nil, draftTypeMTP, "model", opts)
	if !reflect.DeepEqual(args, []string{"--spec-type", "none"}) {
		t.Fatalf("draft args = %q", args)
	}
	body, err := (&llamaServerRunner{}).llamaServerChatRequest(ChatRequest{Options: &opts}, true)
	if err != nil || body["backend_sampling"] != false {
		t.Fatalf("backend setting: %v %v", body, err)
	}
}
