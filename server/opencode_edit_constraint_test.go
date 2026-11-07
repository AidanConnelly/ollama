package server

import (
	"context"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"slices"
	"testing"

	"github.com/ollama/ollama/api"
)

func openCodeEditTestTool() api.Tool {
	props := api.NewToolPropertiesMap()
	for _, name := range []string{"newString", "oldString", "filePath"} {
		props.Set(name, api.ToolProperty{Type: api.PropertyType{"string"}})
	}
	return api.Tool{Type: "function", Function: api.ToolFunction{Name: "edit", Parameters: api.ToolFunctionParameters{
		Type: "object", Required: []string{"newString", "oldString", "filePath"}, Properties: props,
	}}}
}

func TestOpenCodeEditMetadataPreservesInputsAfterRead(t *testing.T) {
	t.Setenv("OLLAMA_OPENCODE_EDIT_CONSTRAINTS", "1")
	root := t.TempDir()
	if err := os.WriteFile(filepath.Join(root, "source.txt"), []byte("same same\n"), 0600); err != nil {
		t.Fatal(err)
	}
	tools := api.Tools{openCodeEditTestTool()}
	msgs := []api.Message{{Role: "system", Content: "<env>\nWorking directory: " + root + "\n</env>"}}
	args := api.NewToolCallFunctionArguments()
	args.Set("filePath", "source.txt")
	for _, afterRead := range []bool{false, true} {
		if afterRead {
			msgs = append(msgs, api.Message{Role: "assistant", ToolCalls: []api.ToolCall{{Function: api.ToolCallFunction{Name: "read", Arguments: args}}}},
				api.Message{Role: "tool", ToolName: "read", Content: "same same\n"})
		}
		before, _ := json.Marshal([]any{tools, msgs})
		config, err := prepareOpenCodeEditMask(context.Background(), tools, msgs)
		if err != nil {
			t.Fatal(err)
		}
		if config == nil || config.Root != root || !slices.Contains(config.Files, "source.txt") {
			t.Fatalf("metadata = %+v", config)
		}
		after, _ := json.Marshal([]any{tools, msgs})
		if string(before) != string(after) {
			t.Fatal("mask changed prompt inputs")
		}
	}
	if err := os.WriteFile(filepath.Join(root, ".gitignore"), []byte("source.txt\n"), 0600); err != nil {
		t.Fatal(err)
	}
	refreshed, err := prepareOpenCodeEditMask(context.Background(), tools, msgs)
	if err != nil || slices.Contains(refreshed.Files, "source.txt") {
		t.Fatalf("manifest did not refresh: %+v %v", refreshed, err)
	}
}

func TestOpenCodeEligibleManifest(t *testing.T) {
	for _, repository := range []bool{false, true} {
		t.Run(map[bool]string{false: "without_git", true: "with_git"}[repository], func(t *testing.T) {
			root, outside := t.TempDir(), t.TempDir()
			files := map[string]string{
				".gitignore": "*.log\n!keep.log\nblocked/\n",
				"keep.log":   "keep", "skip.log": "skip", "text": "text", "blocked/keep": "skip",
				"nested/.gitignore": "*.tmp\n!keep.tmp\n!allowed.log\n",
				"nested/keep.tmp":   "keep", "nested/skip.tmp": "skip", "nested/allowed.log": "keep",
				"nested/skip.log": "skip", ".git/metadata": "skip",
			}
			if repository {
				if b, err := exec.Command("git", "init", "-q", root).CombinedOutput(); err != nil {
					t.Fatalf("git init: %v %s", err, b)
				}
			}
			for path, text := range files {
				full := filepath.Join(root, path)
				if err := os.MkdirAll(filepath.Dir(full), 0700); err != nil {
					t.Fatal(err)
				}
				if err := os.WriteFile(full, []byte(text), 0600); err != nil {
					t.Fatal(err)
				}
			}
			if err := os.WriteFile(filepath.Join(outside, "secret"), []byte("outside"), 0600); err != nil {
				t.Fatal(err)
			}
			for link, target := range map[string]string{"inside-link": "text", "outside-link": filepath.Join(outside, "secret"), "dir-link": "nested"} {
				if err := os.Symlink(target, filepath.Join(root, link)); err != nil {
					t.Fatal(err)
				}
			}
			got, err := openCodeEligibleManifest(context.Background(), root)
			if err != nil {
				t.Fatal(err)
			}
			want := []string{".gitignore", "inside-link", "keep.log", "nested/.gitignore", "nested/allowed.log", "nested/keep.tmp", "text"}
			if !reflect.DeepEqual(got, want) {
				t.Fatalf("manifest = %q, want %q", got, want)
			}
		})
	}
}

func TestOpenCodeMaskOptIn(t *testing.T) {
	t.Setenv("OLLAMA_OPENCODE_EDIT_CONSTRAINTS", "0")
	config, err := prepareOpenCodeEditMask(context.Background(), api.Tools{openCodeEditTestTool()}, nil)
	if config != nil || err != nil {
		t.Fatalf("disabled: %+v %v", config, err)
	}
}
