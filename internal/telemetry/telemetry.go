package telemetry

import (
	"context"
	"fmt"
	"os"
	"strings"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracehttp"
	"go.opentelemetry.io/otel/propagation"
	"go.opentelemetry.io/otel/sdk/resource"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"

	"github.com/ollama/ollama/version"
)

// Init configures OTLP tracing from the standard OpenTelemetry environment
// variables. Tracing is enabled by default and can be disabled with
// OTEL_TRACES_EXPORTER=none.
func Init(ctx context.Context) (shutdown func(context.Context) error, enabled bool, err error) {
	exporter := strings.TrimSpace(os.Getenv("OTEL_TRACES_EXPORTER"))

	switch exporter {
	case "none":
		return nil, false, nil
	case "", "otlp":
	default:
		return nil, false, fmt.Errorf("unsupported OTEL_TRACES_EXPORTER %q (supported: otlp, none)", exporter)
	}

	protocol := strings.TrimSpace(os.Getenv("OTEL_EXPORTER_OTLP_TRACES_PROTOCOL"))
	if protocol == "" {
		protocol = strings.TrimSpace(os.Getenv("OTEL_EXPORTER_OTLP_PROTOCOL"))
	}
	if protocol != "" && protocol != "http/protobuf" && protocol != "http/json" {
		return nil, false, fmt.Errorf("unsupported OTLP trace protocol %q (supported: http/protobuf, http/json)", protocol)
	}

	spanExporter, err := otlptracehttp.New(ctx)
	if err != nil {
		return nil, false, fmt.Errorf("create OTLP span exporter: %w", err)
	}

	serviceName := strings.TrimSpace(os.Getenv("OTEL_SERVICE_NAME"))
	if serviceName == "" {
		serviceName = "ollama"
	}
	res, err := resource.New(ctx,
		resource.WithFromEnv(),
		resource.WithTelemetrySDK(),
		resource.WithAttributes(
			attribute.String("service.name", serviceName),
			attribute.String("service.version", version.Version),
		),
	)
	if err != nil {
		return nil, false, fmt.Errorf("create OpenTelemetry resource: %w", err)
	}

	provider := sdktrace.NewTracerProvider(
		sdktrace.WithBatcher(spanExporter),
		sdktrace.WithResource(res),
	)
	otel.SetTracerProvider(provider)
	otel.SetTextMapPropagator(propagation.NewCompositeTextMapPropagator(
		propagation.TraceContext{},
		propagation.Baggage{},
	))

	return provider.Shutdown, true, nil
}
