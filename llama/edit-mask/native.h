#pragma once
#include "matcher.h"
#include "chat.h"
#include "chat-peg-parser.h"
#include "log.h"

namespace ollama_edit {
// Escape XML string bytes into an unfinished JSON string without repairing
// partial UTF-8. The mask's incremental decoder owns Unicode validation.
inline std::string json_prefix(std::string_view value) {
    std::string result;
    for (unsigned char c : value) {
        if (c == '"' || c == '\\') { result += '\\'; result += char(c); }
        else if (c < 0x20) {
            static const char * hex = "0123456789abcdef";
            result += "\\u00"; result += hex[c >> 4]; result += hex[c & 15];
        } else result += char(c);
    }
    return result;
}
inline bool incomplete_utf8(std::string_view tail) {
    if (tail.empty()) return false;
    auto c = uint8_t(tail[0]);
    size_t width = c >= 0xc2 && c <= 0xdf ? 2 : c >= 0xe0 && c <= 0xef ? 3 : c >= 0xf0 && c <= 0xf4 ? 4 : 0;
    if (!width || tail.size() >= width) return false;
    for (size_t i = 1; i < tail.size(); ++i) if ((uint8_t(tail[i]) & 0xc0) != 0x80) return false;
    return true;
}
struct native_parser final : parser {
    std::shared_ptr<const common_chat_parser_params> params;
    common_chat_input input;
    std::vector<call> last;
    std::vector<std::string> triggers;
    size_t trigger_tail = 0;
    bool xml = false, fast_string = false, triggered = false;
    explicit native_parser(const common_chat_session & session)
        : params(std::make_shared<common_chat_parser_params>(session.edit_mask_parser_params())),
          xml(session.parser().has_rule("xml-arg-string")) {
        common_params_sampling sampling;
        session.apply_sampling(sampling);
        if (sampling.grammar_lazy) {
            for (const auto & trigger : sampling.grammar_triggers) {
                if (trigger.type != COMMON_GRAMMAR_TRIGGER_TYPE_WORD || trigger.value.empty()) { triggers.clear(); break; }
                triggers.push_back(trigger.value);
                trigger_tail = std::max(trigger_tail, trigger.value.size() - 1);
            }
        }
    }
    std::unique_ptr<parser> clone() const override { return std::make_unique<native_parser>(*this); }
    std::vector<call> feed(const std::string & bytes, int32_t token) override {
        // Use the native lazy grammar's own triggers to prove that a candidate
        // cannot enter a tool call. Include the accepted suffix for split tags.
        if (!triggered && last.empty() && !triggers.empty()) {
            std::string tail = input.text.substr(input.size() - std::min(input.size(), trigger_tail)) + bytes;
            bool hit = false;
            for (const auto & trigger : triggers) if (tail.find(trigger) != std::string::npos) { hit = true; break; }
            if (!hit) { input.append(bytes, token); return {}; }
            triggered = true;
        }
        input.append(bytes, token);
        // The AST established an open string extending exactly to the current
        // input boundary. These bytes cannot close it or cross an argument.
        if (fast_string && !last.empty() && bytes.find_first_of(xml ? "\n<" : "\"\\") == std::string::npos) {
            last.back().arguments += xml ? json_prefix(bytes) : bytes;
            return last;
        }
        common_chat_input effective = input;
        effective.prepend(params->generation_prompt);
        common_peg_parse_context ctx(std::move(effective.text), std::move(effective.tokens), COMMON_PEG_PARSE_FLAG_LENIENT);
        auto parsed = params->parser.parse(ctx);
        std::vector<call> out;
        call * current = nullptr;
        bool string_open = false;
        size_t fields = 0, value_end = 0;
        ctx.ast.visit(parsed, [&](const common_peg_ast_node & node) {
            using tags = common_chat_peg_builder;
            if (node.tag == tags::TOOL_OPEN) {
                out.push_back({"", "{"}); current = &out.back(); fields = 0; string_open = false;
            }
            if (!current) return;
            if (node.tag == tags::TOOL_NAME) current->name = std::string(node.text);
            if (!xml && node.tag == tags::TOOL_ARGS && !node.text.empty() && node.text.front() == '{') {
                current->arguments = std::string(node.text); value_end = node.end;
                auto tail = std::string_view(ctx.input).substr(node.end);
                if (node.is_partial && incomplete_utf8(tail)) { current->arguments += tail; value_end = ctx.input.size(); }
            }
            if (xml && node.tag == tags::TOOL_ARG_NAME) {
                if (fields++) current->arguments += ",";
                current->arguments += common_json(std::string(node.text)).dump() + ":";
            }
            if (xml && node.tag == tags::TOOL_ARG_STRING_VALUE) {
                current->arguments += "\"" + json_prefix(node.text);
                string_open = true; value_end = node.end;
                auto tail = std::string_view(ctx.input).substr(node.end);
                if (node.is_partial && incomplete_utf8(tail)) { current->arguments += tail; value_end = ctx.input.size(); }
            }
            if (xml && node.tag == tags::TOOL_ARG_VALUE) current->arguments += std::string(node.text);
            if (xml && node.tag == tags::TOOL_ARG_CLOSE && string_open) { current->arguments += "\""; string_open = false; }
            if (node.tag == tags::TOOL_CLOSE) {
                if (xml) current->arguments += "}";
                current = nullptr; string_open = false;
            }
        });
        fast_string = false;
        if (!out.empty() && value_end == ctx.input.size()) {
            if (xml) fast_string = string_open;
            else {
                arguments probe;
                probe.feed(out.back().arguments);
                fast_string = probe.in == arguments::field_string && probe.string.complete_character();
            }
        }
        last = std::move(out);
        return last;
    }
};
inline std::shared_ptr<mask> make_native_mask(const common_json & config, const common_chat_session & session, const llama_vocab * vocab) {
    if (config.at("version").get<int>() != 1) throw std::invalid_argument("unsupported oldString mask version");
    // These formats expose argument strings incrementally through the native
    // mapper. Gemma4 and content-only formats do not provide that guarantee.
    if (session.format() != COMMON_CHAT_FORMAT_PEG_NATIVE && session.format() != COMMON_CHAT_FORMAT_PEG_SIMPLE) {
        throw std::invalid_argument("oldString mask: unsupported native tool-call format");
    }
    // Support raw JSON and the native Qwen XML AST. Other constructed formats
    // require an explicit adapter because their mapper may synthesize bytes.
    bool raw_args = false;
    bool qwen_xml = session.parser().has_rule("xml-arg-string");
    for (size_t i = 0; i < session.parser().size(); ++i) {
        if (auto tag = std::get_if<common_peg_tag_parser>(&session.parser().get(i))) {
            if (tag->tag == common_chat_peg_builder::TOOL_ARGS) raw_args = true;
            if (!qwen_xml && (tag->tag == common_chat_peg_builder::TOOL_ARG_OPEN || tag->tag == common_chat_peg_builder::TOOL_ARG_VALUE ||
                tag->tag == common_chat_peg_builder::TOOL_ARG_STRING_VALUE)) {
                throw std::invalid_argument("oldString mask: constructed tool arguments are unsupported; raw JSON captures required");
            }
        }
    }
    if (!raw_args) throw std::invalid_argument("oldString mask: native parser has no argument capture");
    auto field = config.at("path_field").get<std::string>();
    if (field != "filePath" && field != "path") throw std::invalid_argument("invalid oldString mask path field");
    auto store = std::make_shared<file_store>(config.at("root").get<std::string>(), config.at("files").get<std::vector<std::string>>());
    auto logger = [id = config.at("request_id").get<std::string>()](const event & value) {
        common_json data = {{"request_id", id}, {"status", value.status}, {"reason", value.reason}, {"call_index", value.call_index},
            {"files", value.files}, {"content_bytes", value.content_bytes}, {"index_bytes", value.index_bytes},
            {"masked_tokens", value.masked_tokens}};
        LOG_INF("OLLAMA_EDIT_MASK %s\n", data.dump().c_str());
    };
    return std::make_shared<mask>(std::make_unique<native_parser>(session), store, field, logger,
        [vocab](int32_t token) { return common_token_to_piece(vocab, token, true); });
}
} // namespace ollama_edit
