#pragma once

// The matcher has no llama.cpp dependency, so its byte/JSON/filesystem behavior
// can be tested independently of a model and of the native payload build.
#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace ollama_edit {
constexpr size_t file_limit = 16u << 20;
constexpr size_t union_limit = 128u << 20;

inline void utf8_append(std::string & out, uint32_t cp) {
    if (cp < 0x80) out += char(cp);
    else if (cp < 0x800) { out += char(0xc0 | (cp >> 6)); out += char(0x80 | (cp & 63)); }
    else if (cp < 0x10000) {
        out += char(0xe0 | (cp >> 12)); out += char(0x80 | ((cp >> 6) & 63)); out += char(0x80 | (cp & 63));
    } else {
        out += char(0xf0 | (cp >> 18)); out += char(0x80 | ((cp >> 12) & 63));
        out += char(0x80 | ((cp >> 6) & 63)); out += char(0x80 | (cp & 63));
    }
}
inline bool text_file(std::string_view s) {
    uint32_t cp = 0, minimum = 0; int remaining = 0;
    for (unsigned char c : s) {
        if (!remaining) {
            if (c == 0) return false;
            if (c < 0x80) continue;
            if (c >= 0xc2 && c <= 0xdf) { remaining = 1; cp = c & 31; minimum = 0x80; }
            else if (c >= 0xe0 && c <= 0xef) { remaining = 2; cp = c & 15; minimum = 0x800; }
            else if (c >= 0xf0 && c <= 0xf4) { remaining = 3; cp = c & 7; minimum = 0x10000; }
            else return false;
        } else {
            if ((c & 0xc0) != 0x80) return false;
            cp = (cp << 6) | (c & 63);
            if (--remaining == 0 && (cp < minimum || cp > 0x10ffff || (cp >= 0xd800 && cp <= 0xdfff))) return false;
        }
    }
    return remaining == 0;
}

// One suffix array per file. No separator or concatenated-file approximation;
// even a NUL in generated JSON cannot match across a file boundary.
struct substring_index {
    std::string text;
    std::vector<uint32_t> suffix;
    explicit substring_index(std::string value) : text(std::move(value)), suffix(text.size()) {
        const uint32_t n = uint32_t(text.size());
        if (!n) return;
        std::vector<uint32_t> rank(n), next(n), tmp(n), counts(std::max<uint32_t>(n + 1, 258));
        std::iota(suffix.begin(), suffix.end(), 0);
        for (uint32_t i = 0; i < n; ++i) rank[i] = uint8_t(text[i]) + 1;
        uint32_t classes = 256;
        for (uint32_t width = 1; width < n; width *= 2) {
            auto pass = [&](bool second) {
                std::fill(counts.begin(), counts.begin() + classes + 1, 0);
                auto key = [&](uint32_t i) { return second ? (i + width < n ? rank[i + width] : 0) : rank[i]; };
                for (uint32_t i : suffix) ++counts[key(i)];
                uint32_t sum = 0;
                for (uint32_t i = 0; i <= classes; ++i) { uint32_t c = counts[i]; counts[i] = sum; sum += c; }
                for (uint32_t i : suffix) tmp[counts[key(i)]++] = i;
                suffix.swap(tmp);
            };
            pass(true); pass(false);
            uint32_t count = 1;
            next[suffix[0]] = count;
            for (uint32_t i = 1; i < n; ++i) {
                uint32_t a = suffix[i-1], b = suffix[i];
                if (rank[a] != rank[b] || (a + width < n ? rank[a + width] : 0) != (b + width < n ? rank[b + width] : 0)) ++count;
                next[b] = count;
            }
            rank.swap(next); classes = count;
            if (count == n) break;
        }
    }
    bool contains(std::string_view value) const {
        if (value.empty()) return true;
        auto it = std::lower_bound(suffix.begin(), suffix.end(), value, [&](uint32_t i, std::string_view v) {
            return std::string_view(text).substr(i).compare(0, v.size(), v) < 0;
        });
        return it != suffix.end() && std::string_view(text).substr(*it).compare(0, value.size(), value) == 0;
    }
    bool follows(std::string_view prefix, uint32_t low, uint32_t high) const {
        if (low > high) return false;
        std::string needle(prefix); utf8_append(needle, low);
        auto it = std::lower_bound(suffix.begin(), suffix.end(), needle, [&](uint32_t i, const std::string & v) {
            return std::string_view(text).substr(i).compare(0, v.size(), v) < 0;
        });
        if (it == suffix.end()) return false;
        auto tail = std::string_view(text).substr(*it);
        if (tail.size() <= prefix.size() || tail.substr(0, prefix.size()) != prefix) return false;
        tail.remove_prefix(prefix.size());
        if ((uint8_t(tail[0]) & 0xc0) == 0x80) {
            // Suffix arrays include continuation-byte positions. A JSON codepoint
            // must start at a character boundary, even for an empty oldString.
            needle = std::string(prefix); utf8_append(needle, std::max(low, 0x80u));
            it = std::lower_bound(suffix.begin(), suffix.end(), needle, [&](uint32_t i, const std::string & v) {
                return std::string_view(text).substr(i).compare(0, v.size(), v) < 0;
            });
            if (it == suffix.end()) return false;
            tail = std::string_view(text).substr(*it);
            if (tail.size() <= prefix.size() || tail.substr(0, prefix.size()) != prefix) return false;
            tail.remove_prefix(prefix.size());
        }
        uint32_t cp = uint8_t(tail[0]); size_t count = 1;
        if (cp >= 0xf0) { cp &= 7; count = 4; }
        else if (cp >= 0xe0) { cp &= 15; count = 3; }
        else if (cp >= 0xc0) { cp &= 31; count = 2; }
        if (tail.size() < count) return false;
        for (size_t i = 1; i < count; ++i) cp = (cp << 6) | (uint8_t(tail[i]) & 63);
        return cp >= low && cp <= high;
    }
    size_t bytes() const { return text.size() + suffix.size() * sizeof(uint32_t); }
};

struct snapshot { std::shared_ptr<const substring_index> index; std::string reason; };
struct file_store {
    std::filesystem::path root;
    std::set<std::string> manifest;
    std::map<std::string, snapshot> cache;
    std::vector<std::shared_ptr<const substring_index>> all;
    bool union_loaded = false;
    std::string union_reason;
    size_t content_bytes = 0, index_bytes = 0;
    size_t total_limit = union_limit, per_file_limit = file_limit;

    file_store(std::string directory, const std::vector<std::string> & files) : root(std::filesystem::canonical(directory)), manifest(files.begin(), files.end()) {}

    snapshot get(const std::string & requested) {
        namespace fs = std::filesystem;
        fs::path path = fs::path(requested).is_absolute() ? fs::path(requested) : root / requested;
        path = path.lexically_normal();
        auto relative = path.lexically_relative(root);
        if (relative.empty() || relative == "." || *relative.begin() == "..") return {{}, "outside_root"};
        for (const auto & part : relative) if (part == ".git") return {{}, "git_metadata"};
        auto rel = relative.generic_string();
        auto cached = cache.find(rel);
        if (cached != cache.end()) return cached->second;
        snapshot result;
        std::error_code ec;
        const auto resolved = fs::canonical(path, ec);
        if (ec) result.reason = ec == std::errc::permission_denied ? "unreadable" : "missing";
        else {
            auto inside = resolved.lexically_relative(root);
            if (inside.empty() || *inside.begin() == "..") result.reason = "outside_root_symlink";
            else if (std::find(inside.begin(), inside.end(), ".git") != inside.end()) result.reason = "git_metadata";
            else if (!manifest.count(rel)) result.reason = "ignored_or_ineligible";
            else if (!fs::is_regular_file(resolved, ec)) result.reason = "not_regular";
            else if (fs::file_size(resolved, ec) > per_file_limit) result.reason = "file_limit";
            else if (ec) result.reason = "unreadable";
            else {
                std::ifstream input(resolved, std::ios::binary);
                if (!input) result.reason = "unreadable";
                else {
                    // Bounded reads also protect against a file growing after stat.
                    std::string data; char buffer[65536];
                    while (input && data.size() <= per_file_limit) {
                        input.read(buffer, std::min<size_t>(sizeof(buffer), per_file_limit + 1 - data.size()));
                        data.append(buffer, size_t(input.gcount()));
                    }
                    if (input.bad()) result.reason = "unreadable";
                    else if (data.size() > per_file_limit) result.reason = "file_limit";
                    else if (!text_file(data)) result.reason = "binary";
                    else {
                        result.index = std::make_shared<substring_index>(std::move(data));
                        content_bytes += result.index->text.size();
                        index_bytes += result.index->bytes() - result.index->text.size();
                    }
                }
            }
        }
        cache.emplace(rel, result);
        return result;
    }

    void load_union() {
        if (union_loaded) return;
        union_loaded = true;
        size_t bytes = 0;
        for (const auto & path : manifest) {
            auto s = get(path);
            if (s.index) {
                bytes += s.index->text.size();
                if (bytes > total_limit) { union_reason = "union_limit"; all.clear(); return; }
                all.push_back(s.index);
            }
        }
    }
    bool union_contains(std::string_view value) {
        load_union();
        for (const auto & index : all) if (index->contains(value)) return true;
        return false;
    }
};

// Incremental JSON string decoder. The grammar owns general JSON validity;
// incomplete escapes and UTF-8 are retained until enough bytes are emitted.
struct json_string {
    std::string decoded;
    bool escaped = false, done = false, invalid = false;
    int hex_left = 0, utf_left = 0;
    uint32_t code = 0, high = 0, utf_cp = 0, utf_min = 0;
    std::string utf_bytes;
    bool low_slash = false, low_u = false;

    void put(unsigned char c) {
        if (done || invalid) { invalid = true; return; }
        if (utf_left) {
            if ((c & 0xc0) != 0x80) { invalid = true; return; }
            utf_bytes += char(c); utf_cp = (utf_cp << 6) | (c & 63);
            if (--utf_left == 0) {
                if (utf_cp < utf_min || utf_cp > 0x10ffff || (utf_cp >= 0xd800 && utf_cp <= 0xdfff)) invalid = true;
                else decoded += utf_bytes;
                utf_bytes.clear();
            }
            return;
        }
        if (hex_left) {
            unsigned digit;
            if (c >= '0' && c <= '9') digit = c - '0';
            else if (c >= 'a' && c <= 'f') digit = c - 'a' + 10;
            else if (c >= 'A' && c <= 'F') digit = c - 'A' + 10;
            else { invalid = true; return; }
            code = (code << 4) | digit;
            if (--hex_left == 0) {
                if (high) {
                    if (code < 0xdc00 || code > 0xdfff) invalid = true;
                    else utf8_append(decoded, 0x10000 + ((high - 0xd800) << 10) + code - 0xdc00);
                    high = 0;
                } else if (code >= 0xd800 && code <= 0xdbff) { high = code; low_slash = true; }
                else if (code >= 0xdc00 && code <= 0xdfff) invalid = true;
                else utf8_append(decoded, code);
            }
            return;
        }
        if (low_slash) { low_slash = false; low_u = true; if (c != '\\') invalid = true; return; }
        if (low_u) { low_u = false; if (c != 'u') invalid = true; else { hex_left = 4; code = 0; } return; }
        if (escaped) {
            escaped = false;
            switch (c) {
                case '"': case '\\': case '/': decoded += char(c); break;
                case 'b': decoded += '\b'; break;
                case 'f': decoded += '\f'; break;
                case 'n': decoded += '\n'; break;
                case 'r': decoded += '\r'; break;
                case 't': decoded += '\t'; break;
                case 'u': hex_left = 4; code = 0; break;
                default: invalid = true;
            }
        } else if (c == '"') done = true;
        else if (c == '\\') escaped = true;
        else if (c < 0x20) invalid = true;
        else if (c < 0x80) decoded += char(c);
        else {
            utf_bytes = std::string(1, char(c));
            if (c >= 0xc2 && c <= 0xdf) { utf_left = 1; utf_cp = c & 31; utf_min = 0x80; }
            else if (c >= 0xe0 && c <= 0xef) { utf_left = 2; utf_cp = c & 15; utf_min = 0x800; }
            else if (c >= 0xf0 && c <= 0xf4) { utf_left = 3; utf_cp = c & 7; utf_min = 0x10000; }
            else invalid = true;
        }
    }
    bool complete_character() const { return !escaped && !hex_left && !high && !low_slash && !low_u && !utf_left; }
};

struct arguments {
    enum phase { object, key, colon, value, after, finished } at = object;
    enum string_kind { none, key_string, field_string, ignored_string } in = none;
    json_string string;
    std::string field, path_field, path, old;
    bool path_done = false, old_seen = false, old_done = false, invalid_old = false;
    bool path_before_old = false;
    size_t depth = 0;
    bool scalar = false;
    json_string old_decoder;

    explicit arguments(std::string name = "filePath") : path_field(std::move(name)) {}
    void feed(std::string_view delta) {
        for (unsigned char c : delta) {
            if (in != none) {
                string.put(c);
                if (in == field_string && field == "oldString") {
                    old_decoder = string; old = string.decoded; invalid_old = string.invalid;
                }
                if (!string.done) continue;
                if (in == key_string) { field = string.decoded; at = colon; }
                else if (in == field_string) {
                    if (field == path_field) { path = string.decoded; path_done = !string.invalid; }
                    if (field == "oldString") old_done = !string.invalid;
                    at = after;
                }
                in = none; continue;
            }
            if (depth) {
                if (c == '"') { in = ignored_string; string = {}; }
                else if (c == '{' || c == '[') ++depth;
                else if (c == '}' || c == ']') { if (--depth == 0) at = after; }
                continue;
            }
            if (scalar) {
                if (c != ',' && c != '}') continue;
                scalar = false; at = after;
            }
            if (c == ' ' || c == '\t' || c == '\r' || c == '\n') continue;
            switch (at) {
                case object: if (c == '{') at = key; break;
                case key:
                    if (c == '"') { in = key_string; string = {}; }
                    else if (c == '}') at = finished;
                    break;
                case colon: if (c == ':') at = value; break;
                case value:
                    if (c == '"') {
                        in = field_string; string = {};
                        if (field == "oldString") { old_seen = true; path_before_old = path_done; old_done = false; old.clear(); old_decoder = {}; }
                    } else if (c == '{' || c == '[') depth = 1;
                    else scalar = true;
                    break;
                case after:
                    if (c == ',') { at = key; field.clear(); }
                    else if (c == '}') at = finished;
                    break;
                case finished: break;
            }
        }
    }
};

struct call { std::string name, arguments; };
// Native adapters clone the native tool-call parser/session. No scanning model
// prose for strings resembling edit calls, and no format-specific delimiters.
struct parser {
    virtual ~parser() = default;
    virtual std::unique_ptr<parser> clone() const = 0;
    virtual std::vector<call> feed(const std::string & bytes, int32_t token) = 0;
};
struct event {
    std::string status, reason;
    size_t call_index = 0, files = 0, content_bytes = 0, index_bytes = 0;
    uint64_t masked_tokens = 0;
};
struct call_state {
    std::string raw;
    arguments args;
    bool skipped = false;
    std::string reason;
    explicit call_state(const std::string & field) : args(field) {}
};

class mask {
    std::shared_ptr<const parser> initial;
    std::unique_ptr<parser> current;
    std::shared_ptr<file_store> files;
    std::string path_field;
    std::vector<call_state> states;
    std::function<void(const event &)> report;
    std::function<std::string(int32_t)> decode;
    std::set<std::string> reported;
    uint64_t masked_tokens = 0;

    bool viable_escape(const arguments & a) {
        const auto & d = a.old_decoder;
        if (d.complete_character()) return true;
        auto matches = [&](uint32_t low, uint32_t high) {
            if (low > high) return false;
            if (a.path_done) return files->get(a.path).index->follows(a.old, low, high);
            for (const auto & index : files->all) if (index->follows(a.old, low, high)) return true;
            return false;
        };
        if (d.high) {
            uint32_t low = 0xdc00, high = 0xdfff;
            if (d.hex_left) {
                low = std::max(low, d.code << (4 * d.hex_left));
                high = std::min(high, (d.code << (4 * d.hex_left)) + ((1u << (4 * d.hex_left)) - 1));
            }
            if (low > high) return false;
            uint32_t base = 0x10000 + ((d.high - 0xd800) << 10);
            return matches(base + low - 0xdc00, base + high - 0xdc00);
        }
        if (d.hex_left) {
            uint32_t low = d.code << (4 * d.hex_left), high = low + ((1u << (4 * d.hex_left)) - 1);
            if (matches(low, std::min(high, 0xd7ffu))) return true;
            if (matches(std::max(low, 0xe000u), high)) return true;
            uint32_t hlow = std::max(low, 0xd800u), hhigh = std::min(high, 0xdbffu);
            return hlow <= hhigh && matches(0x10000 + ((hlow - 0xd800) << 10), 0x10000 + ((hhigh - 0xd800 + 1) << 10) - 1);
        }
        if (d.escaped) return matches(0, 0xd7ff) || matches(0xe000, 0x10ffff);
        return true; // raw UTF-8 prefixes are checked by byte below
    }

    bool evaluate(const std::vector<call> & calls) {
        for (size_t i = 0; i < states.size(); ++i) {
            const auto & s = states[i];
            if (!s.skipped && s.args.old_seen && !s.args.old_done && (i >= calls.size() || calls[i].name != "edit")) return false;
        }
        while (states.size() < calls.size()) states.emplace_back(path_field);
        for (size_t i = 0; i < calls.size(); ++i) {
            if (calls[i].name != "edit") continue;
            auto & s = states[i];
            const auto & raw = calls[i].arguments;
            if (raw.compare(0, s.raw.size(), s.raw) != 0) {
                // A supported native parser must expose monotone argument bytes.
                // Silent weakening would invalidate the experiment.
                throw std::runtime_error("oldString mask: unsupported non-monotone tool-call arguments");
            }
            s.args.feed(std::string_view(raw).substr(s.raw.size())); s.raw = raw;
            const auto & a = s.args;
            if (!a.old_seen) continue;
            if (s.skipped && a.path_done && (s.reason == "union_limit" || s.reason == "no_eligible_text_files")) {
                s.skipped = false; s.reason.clear();
            }
            if (s.skipped) continue;
            if (a.path_done) {
                const auto selected = files->get(a.path);
                if (!selected.index) { s.skipped = true; s.reason = selected.reason; continue; }
                if (!selected.index->contains(a.old)) {
                    if (!a.path_before_old) { s.skipped = true; s.reason = "late_path_conflict"; continue; }
                    return false;
                }
            } else {
                files->load_union();
                if (!files->union_reason.empty()) { s.skipped = true; s.reason = files->union_reason; continue; }
                if (files->all.empty()) { s.skipped = true; s.reason = "no_eligible_text_files"; continue; }
                if (!files->union_contains(a.old)) return false;
            }
            if (a.invalid_old) return false;
            if (!viable_escape(a)) return false;
            // Reject an impossible partial UTF-8 continuation now; JSON escapes
            // are checked as soon as they decode into a complete codepoint.
            if (!a.old_decoder.utf_bytes.empty()) {
                std::string prefix = a.old + a.old_decoder.utf_bytes;
                bool found = a.path_done ? files->get(a.path).index->contains(prefix) : files->union_contains(prefix);
                if (!found) return false;
            }
        }
        return true;
    }
    void emit() {
        for (size_t i = 0; i < states.size(); ++i) {
            const auto & s = states[i];
            if (!s.args.old_seen) continue;
            std::string status = s.skipped ? "skipped" : "active";
            std::string reason = s.skipped ? s.reason : (s.args.path_done ? "selected_file" : "eligible_file_union");
            std::string key = std::to_string(i) + ":" + status + ":" + reason;
            if (reported.insert(key).second && report) report({status, reason, i, files->manifest.size(), files->content_bytes, files->index_bytes, masked_tokens});
        }
    }
public:
    mask(std::unique_ptr<parser> p, std::shared_ptr<file_store> store, std::string field, std::function<void(const event &)> logger = {}, std::function<std::string(int32_t)> decoder = {})
        : initial(p->clone()), current(std::move(p)), files(std::move(store)), path_field(std::move(field)), report(std::move(logger)), decode(std::move(decoder)) {}
    mask(const mask & other) : initial(other.initial), current(other.current->clone()), files(other.files), path_field(other.path_field), states(other.states), report(other.report), decode(other.decode), reported(other.reported), masked_tokens(other.masked_tokens) {}
    std::shared_ptr<mask> clone() const { return std::make_shared<mask>(*this); }
    bool allows(const std::string & extension, int32_t token, bool eog = false) const {
        mask temp(*this);
        if (eog) {
            for (const auto & s : states) if (!s.skipped && s.args.old_seen && !s.args.old_done) return false;
        }
        return temp.evaluate(temp.current->feed(extension, token));
    }
    void accept(const std::string & piece, int32_t token) {
        if (!evaluate(current->feed(piece, token))) throw std::runtime_error("oldString mask: accepted a rejected token");
        emit();
        if (report) report({"counts", "sample", 0, files->manifest.size(), files->content_bytes, files->index_bytes, masked_tokens});
    }
    void masked(uint64_t count) { masked_tokens += count; }
    void grammar_retry() {
        if (report) report({"grammar_retry", "retry", 0, files->manifest.size(), files->content_bytes, files->index_bytes, masked_tokens});
    }
    void accept_token(int32_t token) {
        if (!decode) throw std::runtime_error("oldString mask: no native token decoder");
        accept(decode(token), token);
    }
    void reset() { current = initial->clone(); states.clear(); reported.clear(); masked_tokens = 0; }
};
} // namespace ollama_edit
