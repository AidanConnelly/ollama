package server

import (
	"encoding/json"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
)

func TestConstrainOpenCodeEditTools(t *testing.T) {
	root := t.TempDir()
	content := "const message = \"the quick brown cat\";\nconsole.log(message);\n"
	if err := os.WriteFile(filepath.Join(root, "example.ts"), []byte(content), 0o600); err != nil {
		t.Fatal(err)
	}

	args := api.NewToolCallFunctionArguments()
	args.Set("path", "example.ts")
	messages := []api.Message{
		{Role: "system", Content: "<env>\nWorking directory: " + root + "\n</env>"},
		{Role: "assistant", ToolCalls: []api.ToolCall{{Function: api.ToolCallFunction{
			Name: "read", Arguments: args,
		}}}},
	}

	got := constrainOpenCodeEditTools(api.Tools{openCodeEditTestTool()}, messages)
	if len(got) != 1 {
		t.Fatalf("got %d tools, want 1", len(got))
	}
	parameters := got[0].Function.Parameters
	if len(parameters.AnyOf) != 0 {
		t.Fatalf("parameters contain unsupported root anyOf: %v", parameters.AnyOf)
	}

	if !slices.Equal(parameters.Required, []string{"path", "oldString", "newString"}) {
		t.Fatalf("required order = %v", parameters.Required)
	}
	if parameters.AdditionalProperties == nil || *parameters.AdditionalProperties {
		t.Fatalf("additionalProperties = %v, want false", parameters.AdditionalProperties)
	}

	var propertyOrder []string
	for name := range parameters.Properties.All() {
		propertyOrder = append(propertyOrder, name)
	}
	if !slices.Equal(propertyOrder, []string{"path", "oldString", "newString", "replaceAll"}) {
		t.Fatalf("property order = %v", propertyOrder)
	}

	pathProperty, _ := parameters.Properties.Get("path")
	if !slices.Equal(pathProperty.Enum, []any{"example.ts"}) {
		t.Fatalf("path enum = %v", pathProperty.Enum)
	}
	oldStringProperty, _ := parameters.Properties.Get("oldString")
	if len(oldStringProperty.Enum) == 0 {
		t.Fatal("oldString enum is empty")
	}
	seen := make(map[string]bool, len(oldStringProperty.Enum))
	for _, value := range oldStringProperty.Enum {
		candidate, ok := value.(string)
		if !ok {
			t.Fatalf("oldString enum value has type %T", value)
		}
		if !strings.Contains(content, candidate) {
			t.Errorf("oldString candidate %q is absent from the file", candidate)
		}
		if seen[candidate] {
			t.Errorf("duplicate oldString candidate %q", candidate)
		}
		seen[candidate] = true
	}
	if !seen["the quick brown cat"] {
		t.Fatal("oldString enum does not contain the existing phrase")
	}

	newStringProperty, _ := parameters.Properties.Get("newString")
	if len(newStringProperty.Enum) != 0 {
		t.Fatalf("newString enum = %v, want unconstrained", newStringProperty.Enum)
	}

	encoded, err := json.Marshal(parameters)
	if err != nil {
		t.Fatal(err)
	}
	jsonText := string(encoded)
	pathIndex := strings.Index(jsonText, `"path"`)
	oldIndex := strings.Index(jsonText, `"oldString"`)
	newIndex := strings.Index(jsonText, `"newString"`)
	if pathIndex < 0 || oldIndex <= pathIndex || newIndex <= oldIndex {
		t.Fatalf("serialized property order is not path, oldString, newString: %s", jsonText)
	}
}

func TestConstrainOpenCodeEditToolsUsesMostRecentReadableFile(t *testing.T) {
	root := t.TempDir()
	for name, content := range map[string]string{
		"older.txt": "older phrase\n",
		"newer.txt": "newer phrase\n",
	} {
		if err := os.WriteFile(filepath.Join(root, name), []byte(content), 0o600); err != nil {
			t.Fatal(err)
		}
	}

	messages := []api.Message{
		{Role: "system", Content: "<env>\nWorking directory: " + root + "\n</env>"},
		{Role: "assistant", ToolCalls: []api.ToolCall{
			openCodeReadTestCall("older.txt"),
			openCodeReadTestCall("newer.txt"),
		}},
	}

	got := constrainOpenCodeEditTools(api.Tools{openCodeEditTestTool()}, messages)
	pathProperty, _ := got[0].Function.Parameters.Properties.Get("path")
	if !slices.Equal(pathProperty.Enum, []any{"newer.txt"}) {
		t.Fatalf("path enum = %v, want only the most recent read", pathProperty.Enum)
	}
	oldStringProperty, _ := got[0].Function.Parameters.Properties.Get("oldString")
	seen := make(map[string]bool, len(oldStringProperty.Enum))
	for _, value := range oldStringProperty.Enum {
		seen[value.(string)] = true
	}
	if !seen["newer phrase"] || seen["older phrase"] {
		t.Fatalf("oldString enum is not tied to the most recent read: %v", oldStringProperty.Enum)
	}
}

func TestConstrainOpenCodeEditToolsSupportsOpenCodeFilePath(t *testing.T) {
	root := t.TempDir()
	content := "export const message = \"the quick brown cat\";\n"
	path := filepath.Join(root, "example.ts")
	if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
		t.Fatal(err)
	}

	args := api.NewToolCallFunctionArguments()
	args.Set("filePath", path)
	messages := []api.Message{
		{Role: "system", Content: "<env>\nWorking directory: " + root + "\n</env>"},
		{Role: "assistant", ToolCalls: []api.ToolCall{{Function: api.ToolCallFunction{
			Name: "read", Arguments: args,
		}}}},
	}

	got := constrainOpenCodeEditTools(api.Tools{openCodeEditTestToolWithPathField("filePath")}, messages)
	parameters := got[0].Function.Parameters
	if len(parameters.AnyOf) != 0 {
		t.Fatalf("parameters contain unsupported root anyOf: %v", parameters.AnyOf)
	}
	if !slices.Equal(parameters.Required, []string{"filePath", "oldString", "newString"}) {
		t.Fatalf("required order = %v", parameters.Required)
	}
	pathProperty, _ := parameters.Properties.Get("filePath")
	if !slices.Equal(pathProperty.Enum, []any{path}) {
		t.Fatalf("filePath enum = %v, want %q", pathProperty.Enum, path)
	}
	oldStringProperty, _ := parameters.Properties.Get("oldString")
	seen := make(map[string]bool, len(oldStringProperty.Enum))
	for _, value := range oldStringProperty.Enum {
		seen[value.(string)] = true
	}
	if !seen["the quick brown cat"] {
		t.Fatal("oldString enum does not contain the existing phrase")
	}
}

func TestConstrainOpenCodeEditToolsIgnoresReadOutsideWorkingDirectory(t *testing.T) {
	root := t.TempDir()
	outside := filepath.Join(t.TempDir(), "outside.txt")
	if err := os.WriteFile(outside, []byte("must not be offered"), 0o600); err != nil {
		t.Fatal(err)
	}

	args := api.NewToolCallFunctionArguments()
	args.Set("path", outside)
	messages := []api.Message{
		{Role: "system", Content: "<env>\nWorking directory: " + root + "\n</env>"},
		{Role: "assistant", ToolCalls: []api.ToolCall{{Function: api.ToolCallFunction{
			Name: "read", Arguments: args,
		}}}},
	}

	got := constrainOpenCodeEditTools(api.Tools{openCodeEditTestTool()}, messages)
	if len(got) != 1 || len(got[0].Function.Parameters.AnyOf) != 0 {
		t.Fatal("edit schema was constrained using a file outside the working directory")
	}
	pathProperty, _ := got[0].Function.Parameters.Properties.Get("path")
	if len(pathProperty.Enum) != 0 {
		t.Fatalf("path enum = %v, want unchanged schema", pathProperty.Enum)
	}
}

func openCodeReadTestCall(path string) api.ToolCall {
	arguments := api.NewToolCallFunctionArguments()
	arguments.Set("path", path)
	return api.ToolCall{Function: api.ToolCallFunction{
		Name: "read", Arguments: arguments,
	}}
}

func openCodeEditTestTool() api.Tool {
	return openCodeEditTestToolWithPathField("path")
}

func openCodeEditTestToolWithPathField(pathField string) api.Tool {
	properties := api.NewToolPropertiesMap()
	properties.Set(pathField, api.ToolProperty{Type: api.PropertyType{"string"}})
	properties.Set("oldString", api.ToolProperty{Type: api.PropertyType{"string"}})
	properties.Set("newString", api.ToolProperty{Type: api.PropertyType{"string"}})
	properties.Set("replaceAll", api.ToolProperty{Type: api.PropertyType{"boolean"}})

	return api.Tool{
		Type: "function",
		Function: api.ToolFunction{
			Name: "edit",
			Parameters: api.ToolFunctionParameters{
				Type:       "object",
				Properties: properties,
			},
		},
	}
}
