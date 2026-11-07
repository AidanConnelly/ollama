package server

import (
	"bytes"
	"context"
	"fmt"
	"log/slog"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"

	"github.com/google/uuid"
	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/llm"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"
)

const openCodeEditToolName = "edit"

// No messages or schemas are modified. Files are read and indexed lazily by the
// native sampler; the manifest is rebuilt for every request, including follow-ups.
func prepareOpenCodeEditMask(ctx context.Context, tools api.Tools, messages []api.Message) (*llm.EditMaskConfig, error) {
	if !envconfig.OpenCodeEditConstraints() {
		return nil, nil
	}
	field := ""
	for _, tool := range tools {
		if f, ok := openCodeEditPathField(tool); ok {
			field = f
			break
		}
	}
	if field == "" {
		return nil, nil
	}
	root := openCodeWorkingDirectory(messages)
	if root == "" || !filepath.IsAbs(root) {
		slog.Info("oldString mask skipped", "reason", "missing_working_directory")
		trace.SpanFromContext(ctx).AddEvent("ollama.edit_mask.skipped", trace.WithAttributes(attribute.String("reason", "missing_working_directory")))
		return nil, nil
	}
	root, err := filepath.EvalSymlinks(root)
	if err != nil {
		return nil, fmt.Errorf("oldString mask working directory: %w", err)
	}
	paths, err := openCodeEligibleManifest(ctx, root)
	if err != nil {
		return nil, err
	}
	slog.Info("oldString mask configured", "files", len(paths), "implementation", "oldString-mask-v1")
	trace.SpanFromContext(ctx).AddEvent("ollama.edit_mask.configured", trace.WithAttributes(attribute.Int("files", len(paths))))
	return &llm.EditMaskConfig{RequestID: uuid.NewString(), Version: 1, Root: root, Files: paths, PathField: field}, nil
}

// An empty private Git index applies Git's own ignore semantics even when root
// is not a repository. It also treats tracked-but-ignored files consistently.
// Neither the user's index nor their repository configuration is read/written.
func openCodeEligibleManifest(ctx context.Context, root string) ([]string, error) {
	gitDir, err := os.MkdirTemp("", "ollama-edit-manifest-")
	if err != nil {
		return nil, err
	}
	defer os.RemoveAll(gitDir)
	run := func(args ...string) ([]byte, error) {
		cmd := exec.CommandContext(ctx, "git", args...)
		// Discard inherited Git overrides, global excludes and repository config.
		for _, entry := range os.Environ() {
			if !strings.HasPrefix(entry, "GIT_") {
				cmd.Env = append(cmd.Env, entry)
			}
		}
		cmd.Env = append(cmd.Env, "GIT_CONFIG_NOSYSTEM=1", "GIT_CONFIG_GLOBAL="+os.DevNull, "GIT_CONFIG_COUNT=0")
		out, err := cmd.Output()
		if err != nil {
			return nil, fmt.Errorf("oldString mask manifest requires Git: %w", err)
		}
		return out, nil
	}
	if _, err := run("init", "--bare", "--quiet", gitDir); err != nil {
		return nil, err
	}
	out, err := run("--git-dir="+gitDir, "--work-tree="+root, "-c", "core.bare=false", "-c", "core.excludesFile="+os.DevNull,
		"ls-files", "--others", "--exclude-standard", "-z")
	if err != nil {
		return nil, err
	}
	paths := []string{}
	for _, raw := range bytes.Split(out, []byte{0}) {
		if len(raw) == 0 {
			continue
		}
		rel := string(raw)
		if strings.Contains("/"+filepath.ToSlash(rel)+"/", "/.git/") {
			continue
		}
		full := filepath.Join(root, rel)
		resolved, err := filepath.EvalSymlinks(full)
		if err != nil {
			continue
		}
		inside, err := filepath.Rel(root, resolved)
		if err != nil || inside == ".." || strings.HasPrefix(inside, ".."+string(filepath.Separator)) {
			continue
		}
		info, err := os.Stat(full)
		if err != nil || !info.Mode().IsRegular() {
			continue
		}
		paths = append(paths, filepath.ToSlash(rel))
	}
	sort.Strings(paths)
	return paths, nil
}

func isOpenCodeEditTool(tool api.Tool) bool {
	_, ok := openCodeEditPathField(tool)
	return ok
}

func openCodeEditPathField(tool api.Tool) (string, bool) {
	if tool.Function.Name != openCodeEditToolName || tool.Function.Parameters.Properties == nil {
		return "", false
	}
	pathField := ""
	for _, name := range []string{"filePath", "path"} {
		property, ok := tool.Function.Parameters.Properties.Get(name)
		if ok && len(property.Type) == 1 && property.Type[0] == "string" {
			pathField = name
			break
		}
	}
	if pathField == "" {
		return "", false
	}
	for _, name := range []string{"oldString", "newString"} {
		property, ok := tool.Function.Parameters.Properties.Get(name)
		if !ok || len(property.Type) != 1 || property.Type[0] != "string" {
			return "", false
		}
	}
	return pathField, true
}

func openCodeWorkingDirectory(messages []api.Message) string {
	var directory string
	for _, message := range messages {
		if message.Role != "system" {
			continue
		}
		inEnvironment := false
		for _, line := range strings.Split(message.Content, "\n") {
			line = strings.TrimSpace(line)
			switch line {
			case "<env>":
				inEnvironment = true
				continue
			case "</env>":
				inEnvironment = false
				continue
			}
			if inEnvironment {
				if value, ok := strings.CutPrefix(line, "Working directory:"); ok {
					directory = strings.TrimSpace(value)
				}
			}
		}
	}
	return directory
}
