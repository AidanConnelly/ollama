#include "matcher.h"
#include <cassert>
#include <iostream>
#include <random>

using namespace ollama_edit;
namespace fs = std::filesystem;

// This fixture exposes raw argument deltas. Native parser integration has its
// own tests linked against llama-common; these tests isolate the mask engine.
struct raw_parser : parser {
    std::string text;
    std::unique_ptr<parser> clone() const override { return std::make_unique<raw_parser>(*this); }
    std::vector<call> feed(const std::string & bytes, int32_t) override {
        text += bytes;
        std::vector<call> result;
        size_t start = 0;
        while (true) {
            auto end = text.find('\x1e', start);
            result.push_back({"edit", text.substr(start, end == std::string::npos ? end : end - start)});
            if (end == std::string::npos) break;
            start = end + 1;
        }
        return result;
    }
};
void write(const fs::path & path, const std::string & value) { std::ofstream(path, std::ios::binary) << value; }
std::unique_ptr<mask> make(const std::shared_ptr<file_store> & files, std::vector<event> * events = nullptr) {
    return std::make_unique<mask>(std::make_unique<raw_parser>(), files, "filePath", [events](const event & e) { if(events) events->push_back(e); });
}
void accept(mask & m, const std::string & piece) { assert(m.allows(piece, 0)); m.accept(piece, 0); }
void index_test() {
    std::mt19937 random(42);
    for (size_t n = 0; n < 300; ++n) {
        std::string value;
        for (size_t i=0;i<n;++i) value += char(random()%256);
        substring_index index(value);
        for (size_t i=0;i<=n;++i) {
            for (size_t len=0;len<20;++len) assert(index.contains(value.substr(i,len)));
            std::string needle;
            for (size_t k=0;k<12;++k) needle += char(random()%256);
            assert(index.contains(needle) == (value.find(needle) != std::string::npos));
        }
    }
    substring_index repeated(std::string(20000,'x'));
    assert(repeated.contains(std::string(19999,'x')));
    assert(!repeated.contains(std::string(20001,'x')));
}
void json_test() {
    const std::string raw = R"({"newString":{"nested":["oldString",{"oldString":"fake"}]},"filePath":"a","old\u0053tring":"\"\\\n\t\u263a\ud83d\ude00"})";
    for (size_t split=0;split<=raw.size();++split) {
        arguments a;
        a.feed(raw.substr(0,split)); a.feed(raw.substr(split));
        assert(a.path_done && a.path=="a" && a.path_before_old);
        assert(a.old_seen && a.old_done && !a.invalid_old);
        assert(a.old == "\"\\\n\t☺😀");
    }
    arguments a;
    a.feed(R"({"oldString":"\ud83d)"); assert(!a.invalid_old && !a.old_done);
    a.feed(R"(\u0041)"); assert(a.invalid_old);
}
int main(int argc, char ** argv) {
    static_assert(file_limit == 16777216 && union_limit == 134217728);
    index_test(); json_test();
    fs::path root = argc>1 ? argv[1] : fs::temp_directory_path()/"oldstring-mask-test";
    fs::create_directories(root);
    write(root/"a", "alpha repeated repeated  \t\nquote \"slash \\☺😀\nline\nline\nline\nline\nline\nline\nline\nline\nline\nline\n");
    write(root/"b", "beta");
    auto files = std::make_shared<file_store>(root.string(),std::vector<std::string>{"a","b"});
    {
        auto m=make(files); accept(*m,R"({"filePath":"a","oldString":")");
        assert(!m->allows("XYZ",0)); assert(m->allows("pha",0));
        accept(*m,"repeated repeated");
        assert(!m->allows("XYZ\",\"newString\":\"anything\"}",0));
        accept(*m,R"(","newString":"XYZ"})");
        accept(*m,std::string(1,'\x1e')+R"({"filePath":"b","oldString":"bet)");
        assert(!m->allows("z",0)); accept(*m,R"(a","newString":"ZZZ"})");
    }
    {
        auto m=make(files); accept(*m,R"({"filePath":"a","oldString":")");
        auto clone=m->clone(); accept(*m,"alpha"); accept(*clone,"quote");
        assert(!m->allows("q",0)); assert(clone->allows(" ",0));
        m->reset(); accept(*m,R"({"filePath":"b","oldString":"beta"})");
    }
    {
        auto m=make(files); accept(*m,R"({"oldString":"alpha)");
        assert(!m->allows("beta",0)); // no matches across a and b
        accept(*m,R"(","filePath":"a","newString":"anything"})");
        std::vector<event> events;
        m=make(files,&events); accept(*m,R"({"oldString":"alpha)");
        accept(*m,R"(","filePath":"b","newString":"anything"})");
        assert(std::any_of(events.begin(),events.end(),[](const event & e){return e.reason=="late_path_conflict";}));
    }
    {
        auto m=make(files); accept(*m,R"({"filePath":"a","oldString":"quote \)");
        assert(m->allows("\"",0)); assert(!m->allows("q",0));
        accept(*m,"\"slash "); accept(*m,"\\"); accept(*m,"\\");
        accept(*m,R"(\u263)"); accept(*m,"a");
        accept(*m,R"(\ud83d)"); accept(*m,R"(\ude00)");
        assert(!m->allows("",0,true)); accept(*m,"\"");
    }
    {
        std::vector<event> events;
        auto small=std::make_shared<file_store>(root.string(),std::vector<std::string>{"a","b"}); small->total_limit=4;
        auto m=make(small,&events); accept(*m,R"({"oldString":"any text)");
        accept(*m,R"(","filePath":"a","newString":"foo"})");
        assert(std::any_of(events.begin(),events.end(),[](const event &e){return e.reason=="union_limit";}));
        assert(std::any_of(events.begin(),events.end(),[](const event &e){return e.reason=="late_path_conflict";}));
    }
    {
        write(root/"binary",std::string("x\0y",3));
        auto bad=std::make_shared<file_store>(root.string(),std::vector<std::string>{"binary","big","unreadable"});
        write(root/"big", "x"); fs::resize_file(root/"big", file_limit+1);
        write(root/"unreadable","x"); fs::permissions(root/"unreadable",fs::perms::none);
        assert(bad->get("binary").reason=="binary"); assert(bad->get("big").reason=="file_limit");
        assert(bad->get("../outside").reason=="outside_root");
        assert(bad->get("missing").reason=="missing");
        assert(bad->get("a").reason=="ignored_or_ineligible");
        if (bad->get("unreadable").index) std::cerr<<"unreadable test skipped (privileged process)\n";
        else assert(bad->get("unreadable").reason=="unreadable");
        fs::permissions(root/"unreadable",fs::perms::owner_all);
        auto m=make(bad); accept(*m,R"({"filePath":"binary","oldString":"anything"})");
    }
    {
        // Larger than the former 256 KiB file cap, with a match well beyond
        // the former enumeration budget, and a span longer than eight lines.
        std::string content(300000,'x');
        content += "targetPARTword\n";
        for(int i=0;i<16;++i) content += "line\n";
        write(root/"long",content);
        auto long_files=std::make_shared<file_store>(root.string(),std::vector<std::string>{"long"});
        auto m=make(long_files);
        accept(*m,R"({"filePath":"long","oldString":")");
        accept(*m,"PARTword");
        for(int i=0;i<12;++i) accept(*m,R"(\nline)");
        accept(*m,R"(","newString":"free"})");
        m->reset();
        accept(*m,std::string(R"({"filePath":"long","oldString":")")+std::string(12000,'x')+R"("})");
        m->reset(); assert(!m->allows(R"({"filePath":"long","oldString":" ")",0));
    }
    {
        write(root/"left","left"); write(root/"right","right");
        auto separate=std::make_shared<file_store>(root.string(),std::vector<std::string>{"left","right"});
        auto m=make(separate); accept(*m,R"({"oldString":")");
        assert(m->allows("left",0)); assert(m->allows("right",0));
        assert(!m->allows("leftright",0));
        auto before=separate->get("left"); write(root/"left","refreshed");
        assert(before.index->contains("left")); assert(separate->get("left").index->contains("left"));
        auto fresh=std::make_shared<file_store>(root.string(),std::vector<std::string>{"left"});
        assert(!fresh->get("left").index->contains("left")); assert(fresh->get("left").index->contains("refreshed"));
        auto outside=fs::path(root.string()+"-outside"); write(outside,"outside");
        fs::remove(root/"outside-alias"); fs::create_symlink(outside,root/"outside-alias");
        auto symlink=std::make_shared<file_store>(root.string(),std::vector<std::string>{"outside-alias"});
        assert(symlink->get("outside-alias").reason=="outside_root_symlink");
    }
    if (argc>2) {
        std::ifstream input(argv[2]); std::string sled((std::istreambuf_iterator<char>(input)),{});
        substring_index index(sled);
        assert(index.contains("//! `sled` is a high-performance embedded database with"));
        auto pos=sled.find("//! `sled`"); assert(index.contains(sled.substr(pos+2,800)));
    }
    std::cout << "oldString matcher tests passed\n";
}
