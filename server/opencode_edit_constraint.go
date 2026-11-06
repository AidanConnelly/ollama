package server

import (
	"encoding/json"
	"index/suffixarray"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"unicode"
	"unicode/utf8"

	"github.com/ollama/ollama/api"
)

const (
	openCodeEditToolName       = "edit"
	openCodeReadToolName       = "read"
	openCodeEditMaxFiles       = 4
	openCodeEditMaxCandidates  = 2048
	openCodeEditMaxFileBytes   = 256 * 1024
	openCodeEditMaxSchemaBytes = 256 * 1024
)

type openCodeRead struct {
	path   string
	offset int
	limit  int
}

// constrainOpenCodeEditTools augments the OpenCode edit schema with the most
// recent readable file from OpenCode read calls. The flat schema's ordered
// required fields make llama.cpp's native tool grammar emit path, oldString,
// then newString. oldString choices are exact, unique spans between lexical
// boundaries within a line, plus complete spans of up to eight lines. This is
// intentionally a bounded prototype rather than a general substring automaton.
func constrainOpenCodeEditTools(tools api.Tools, messages []api.Message) api.Tools {
	root := openCodeWorkingDirectory(messages)
	if root == "" || !filepath.IsAbs(root) {
		return tools
	}

	reads := recentOpenCodeReads(messages)
	if len(reads) == 0 {
		return tools
	}

	editIndex := -1
	editPathField := ""
	for i := range tools {
		if field, ok := openCodeEditPathField(tools[i]); ok {
			editIndex = i
			editPathField = field
			break
		}
	}
	if editIndex < 0 {
		return tools
	}

	root, err := filepath.EvalSymlinks(root)
	if err != nil {
		return tools
	}
	root, err = filepath.Abs(root)
	if err != nil {
		return tools
	}

	editTool := tools[editIndex]
	var parameters *api.ToolFunctionParameters
	for _, read := range reads {
		rel, text, err := readOpenCodeFile(root, read.path)
		if err != nil {
			continue
		}
		candidates := uniqueEditSnippets(text, read.offset, read.limit)
		if len(candidates) == 0 {
			continue
		}

		properties := api.NewToolPropertiesMap()
		pathProperty, _ := editTool.Function.Parameters.Properties.Get(editPathField)
		pathProperty.Type = api.PropertyType{"string"}
		pathValue := rel
		if editPathField == "filePath" {
			pathValue = filepath.Join(root, filepath.FromSlash(rel))
		}
		pathProperty.Enum = []any{pathValue}
		properties.Set(editPathField, pathProperty)

		oldStringProperty, _ := editTool.Function.Parameters.Properties.Get("oldString")
		oldStringProperty.Type = api.PropertyType{"string"}
		oldStringProperty.Enum = make([]any, len(candidates))
		for i, candidate := range candidates {
			oldStringProperty.Enum[i] = candidate
		}
		properties.Set("oldString", oldStringProperty)

		newStringProperty, _ := editTool.Function.Parameters.Properties.Get("newString")
		newStringProperty.Type = api.PropertyType{"string"}
		newStringProperty.Enum = nil
		properties.Set("newString", newStringProperty)

		required := []string{editPathField, "oldString", "newString"}
		if replaceAll, ok := editTool.Function.Parameters.Properties.Get("replaceAll"); ok {
			replaceAll.Type = api.PropertyType{"boolean"}
			properties.Set("replaceAll", replaceAll)
		}

		noAdditionalProperties := false
		constrained := api.ToolFunctionParameters{
			Type:                 "object",
			Required:             required,
			AdditionalProperties: &noAdditionalProperties,
			Properties:           properties,
		}
		parameters = &constrained
		break
	}
	if parameters == nil {
		return tools
	}

	constrained := append(api.Tools(nil), tools...)
	function := constrained[editIndex].Function
	function.Parameters = *parameters
	constrained[editIndex].Function = function

	encoded, err := json.Marshal(constrained)
	if err != nil || len(encoded) > openCodeEditMaxSchemaBytes {
		return tools
	}
	return constrained
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

func recentOpenCodeReads(messages []api.Message) []openCodeRead {
	var reads []openCodeRead
	seen := make(map[string]struct{})
	for i := len(messages) - 1; i >= 0 && len(reads) < openCodeEditMaxFiles; i-- {
		for j := len(messages[i].ToolCalls) - 1; j >= 0 && len(reads) < openCodeEditMaxFiles; j-- {
			call := messages[i].ToolCalls[j]
			if call.Function.Name != openCodeReadToolName {
				continue
			}
			var path string
			for _, name := range []string{"filePath", "path"} {
				if value, ok := call.Function.Arguments.Get(name); ok {
					path, _ = value.(string)
				}
				if path != "" {
					break
				}
			}
			if path == "" {
				continue
			}
			if _, ok := seen[path]; ok {
				continue
			}
			seen[path] = struct{}{}

			read := openCodeRead{path: path, offset: 1, limit: 2000}
			if value, ok := call.Function.Arguments.Get("offset"); ok {
				if n := numericInt(value); n > 0 {
					read.offset = n
				}
			}
			if value, ok := call.Function.Arguments.Get("limit"); ok {
				if n := numericInt(value); n > 0 {
					read.limit = min(n, 2000)
				}
			}
			reads = append(reads, read)
		}
	}
	return reads
}

func numericInt(value any) int {
	switch value := value.(type) {
	case int:
		return value
	case int64:
		return int(value)
	case float64:
		return int(value)
	case json.Number:
		n, _ := value.Int64()
		return int(n)
	default:
		return 0
	}
}

func readOpenCodeFile(root, requested string) (string, string, error) {
	path := requested
	if !filepath.IsAbs(path) {
		path = filepath.Join(root, filepath.FromSlash(path))
	}
	path, err := filepath.EvalSymlinks(path)
	if err != nil {
		return "", "", err
	}
	path, err = filepath.Abs(path)
	if err != nil {
		return "", "", err
	}
	rel, err := filepath.Rel(root, path)
	if err != nil || rel == "." || rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) {
		return "", "", fs.ErrPermission
	}

	info, err := os.Stat(path)
	if err != nil {
		return "", "", err
	}
	if !info.Mode().IsRegular() || info.Size() <= 0 || info.Size() > openCodeEditMaxFileBytes {
		return "", "", fs.ErrInvalid
	}
	content, err := os.ReadFile(path)
	if err != nil {
		return "", "", err
	}
	if !utf8.Valid(content) || strings.IndexByte(string(content), 0) >= 0 {
		return "", "", fs.ErrInvalid
	}
	return filepath.ToSlash(rel), string(content), nil
}

func uniqueEditSnippets(content string, offset, limit int) []string {
	lines := strings.SplitAfter(content, "\n")
	start := max(offset-1, 0)
	if start >= len(lines) {
		return nil
	}
	end := min(start+limit, len(lines))
	lines = lines[start:end]
	if len(lines) == 0 {
		return nil
	}

	var snippets []string
	seen := make(map[string]struct{})
	index := suffixarray.New([]byte(content))
	add := func(snippet string) bool {
		if snippet == "" || len(snippet) > 4096 || strings.TrimSpace(snippet) == "" {
			return false
		}
		if _, ok := seen[snippet]; ok {
			return false
		}
		if len(index.Lookup([]byte(snippet), 2)) != 1 {
			return false
		}
		seen[snippet] = struct{}{}
		snippets = append(snippets, snippet)
		return len(snippets) >= openCodeEditMaxCandidates
	}

	for _, line := range lines {
		boundaries := lexicalBoundaries(line)
		for i := 0; i+1 < len(boundaries); i++ {
			for j := i + 1; j < len(boundaries); j++ {
				if add(line[boundaries[i]:boundaries[j]]) {
					return snippets
				}
			}
		}
	}

	for width := 2; width <= 8; width++ {
		for i := 0; i+width <= len(lines); i++ {
			snippet := strings.Join(lines[i:i+width], "")
			if add(snippet) {
				return snippets
			}
		}
	}
	return snippets
}

func lexicalBoundaries(line string) []int {
	boundaries := []int{0}
	previousClass := byte(0)
	for offset, r := range line {
		currentClass := byte(3)
		switch {
		case r == '_' || (r >= '0' && r <= '9') || (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || unicode.IsLetter(r):
			currentClass = 1
		case unicode.IsSpace(r):
			currentClass = 2
		}
		if offset > 0 && currentClass != previousClass {
			boundaries = append(boundaries, offset)
		}
		previousClass = currentClass
	}
	if boundaries[len(boundaries)-1] != len(line) {
		boundaries = append(boundaries, len(line))
	}
	return boundaries
}
