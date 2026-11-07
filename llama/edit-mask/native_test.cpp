#include "native.h"
#include "sampling.h"
#include "ggml-backend.h"
#include <cassert>
#include <iostream>
#include <fstream>
#include <chrono>

using namespace ollama_edit;
namespace fs = std::filesystem;

void accept_piece(mask & value, const std::string & text) {
    if (!value.allows(text, LLAMA_TOKEN_NULL)) { std::cerr << "rejected piece: " << text << '\n'; std::abort(); }
    value.accept(text, LLAMA_TOKEN_NULL);
}
std::string load(const char * path) { std::ifstream input(path); return {(std::istreambuf_iterator<char>(input)),{}}; }
int main(int argc, char ** argv) {
    if(argc<3) { std::cerr << "usage: native-test QWEN_TEMPLATE FIXTURE_DIR [MODEL]\n"; return 2; }
    fs::path root=argv[2]; fs::create_directories(root);
    std::ofstream(root/"a") << "alpha repeated repeated \t\nquote \"slash \\☺😀\nline\nline\nline\n";
    std::ofstream(root/"b") << "beta";
    auto tmpls=common_chat_templates_init(nullptr,load(argv[1]),"<|im_start|>","<|im_end|>");
    common_chat_templates_inputs inputs;
    inputs.messages.push_back({"user","please edit"});
    inputs.tools={{"edit","edit text",R"({"type":"object","properties":{"filePath":{"type":"string"},"oldString":{"type":"string"},"newString":{"type":"string"}},"required":["filePath","oldString","newString"]})"},
                  {"read","read text",R"({"type":"object","properties":{"filePath":{"type":"string"}},"required":["filePath"]})"}};
    inputs.parallel_tool_calls=true; inputs.enable_thinking=false;
    common_chat_session session(tmpls.get(),nullptr,inputs);
    common_json config={{"version",1},{"request_id","native-test"},{"root",root.string()},{"files",std::vector<std::string>{"a","b"}},{"path_field","filePath"}};
    const std::string begin="<tool_call>\n<function=edit>\n<parameter=filePath>\na\n</parameter>\n<parameter=oldString>\n";
    const std::string close="\n</parameter>\n<parameter=newString>\nanything absent\n</parameter>\n</function>\n</tool_call>";
    auto value=make_native_mask(config,session,nullptr);
    // Prose and non-edit tools must remain free.
    accept_piece(*value,"I can mention oldString and edit in ordinary prose.\n");
    accept_piece(*value,"<tool_call>\n<function=read>\n<parameter=filePath>\nmissing\n</parameter>\n</function>\n</tool_call>\n");
    accept_piece(*value,begin);
    assert(!value->allows("not in file",LLAMA_TOKEN_NULL));
    accept_piece(*value,"alpha");
    assert(!value->allows("X"+close,LLAMA_TOKEN_NULL));
    // Split every byte of the closing marker, including its initial newline.
    for(char c:close) accept_piece(*value,std::string(1,c));
    value->reset();
    // Candidate enters oldString and attempts an invalid prefix in one token.
    assert(!value->allows(begin+"INVALID",LLAMA_TOKEN_NULL));
    assert(value->allows(begin+"repeated repeated"+close,LLAMA_TOKEN_NULL));
    // Check every partition of a string with whitespace, escapes and Unicode.
    const std::string old="quote \"slash \\☺😀";
    for(size_t split=0;split<=old.size();++split) {
        value->reset(); accept_piece(*value,begin);
        accept_piece(*value,old.substr(0,split)); accept_piece(*value,old.substr(split)); accept_piece(*value,close);
    }
    // Preview must never mutate state; clones and resets are independent.
    value->reset(); accept_piece(*value,begin);
    auto copy=value->clone(); accept_piece(*value,"alpha"); accept_piece(*copy,"repeated");
    assert(!value->allows("X",LLAMA_TOKEN_NULL));
    value->reset(); accept_piece(*value,begin+"alpha"+close);
    value->reset();
    const std::string late="<tool_call>\n<function=edit>\n<parameter=oldString>\nalpha\n</parameter>\n<parameter=filePath>\nb\n</parameter>\n<parameter=newString>\nfree\n</parameter>\n</function>\n</tool_call>";
    accept_piece(*value,late); // late incompatible paths are never rejected
    // Validate the optimized parser against full native AST parsing for every
    // byte prefix, including boundaries that begin/end in the same token.
    native_parser incremental(session);
    const auto output=begin+old+close;
    for(size_t i=0;i<output.size();++i) {
        auto got=incremental.feed(output.substr(i,1),LLAMA_TOKEN_NULL);
        native_parser fresh(session); auto want=fresh.feed(output.substr(0,i+1),LLAMA_TOKEN_NULL);
        assert(got.size()==want.size());
        for(size_t j=0;j<got.size();++j) { assert(got[j].name==want[j].name); if (got[j].arguments!=want[j].arguments) { std::cerr << "prefix="<<i<<" got="<<got[j].arguments<<" want="<<want[j].arguments<<"\n"; std::abort(); } }
    }
    if(argc>4) {
        auto json_tmpls=common_chat_templates_init(nullptr,load(argv[4]),"<|im_start|>","<|im_end|>");
        common_chat_session json_session(json_tmpls.get(),nullptr,inputs);
        auto json_mask=make_native_mask(config,json_session,nullptr);
        const std::string json_begin="<tool_call>\n{\"name\":\"edit\",\"arguments\":{\"filePath\":\"a\",\"oldString\":\"";
        accept_piece(*json_mask,json_begin);
        assert(!json_mask->allows("INVALID",LLAMA_TOKEN_NULL));
        const std::string escaped=R"(quote \"slash \\\u263a\ud83d\ude00)";
        for(char c:escaped) accept_piece(*json_mask,std::string(1,c));
        accept_piece(*json_mask,"\",\"newString\":\"free\"}}\n</tool_call>");
    }
    if(argc>3) {
        const bool test_sampling=std::getenv("OLLAMA_EDIT_MASK_SAMPLING_TEST") != nullptr;
        ggml_backend_load_all_from_path((fs::absolute(argv[0]).parent_path()/"bin").string().c_str());
        auto mp=llama_model_default_params(); mp.vocab_only=!test_sampling; mp.n_gpu_layers=0;
        auto * model=llama_model_load_from_file(argv[3],mp); assert(model);
        const auto * vocab=llama_model_get_vocab(model);
        common_chat_session token_session(tmpls.get(),vocab,inputs);
        common_params_sampling sampling; sampling.edit_mask=make_native_mask(config,token_session,vocab);
        auto * sampler=common_sampler_init(model,sampling); assert(sampler);
        auto feed=[&](common_sampler * smpl,const std::string & text) { for(auto token:common_tokenize(vocab,text,false,true)) common_sampler_accept(smpl,token,true); };
        feed(sampler,begin);
        auto * clone=common_sampler_clone(sampler);
        feed(sampler,"alpha"); feed(clone,"alpha");
        common_sampler_reset(sampler); feed(sampler,begin);
        common_sampler_copy(sampler,clone);
        feed(sampler,"alpha"); feed(clone,"alpha");
        common_sampler_free(sampler); common_sampler_free(clone);
        auto probe=make_native_mask(config,token_session,vocab);
        for(auto token:common_tokenize(vocab,begin,false,true)) probe->accept_token(token);
        const auto started=std::chrono::steady_clock::now();
        size_t rejected=0;
        for(int32_t token=0;token<llama_vocab_n_tokens(vocab);++token) {
            if(!probe->allows(common_token_to_piece(vocab,token,true),token,llama_vocab_is_eog(vocab,token))) ++rejected;
        }
        std::cout << "vocabulary candidates="<<llama_vocab_n_tokens(vocab)<<" rejected="<<rejected
                  <<" elapsed_ms="<<std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now()-started).count()<<"\n";
        assert(rejected>0);

        if (test_sampling) {
            auto cp=llama_context_default_params(); cp.n_ctx=512; cp.n_batch=512; cp.n_ubatch=512; cp.n_threads=6; cp.n_threads_batch=6;
            auto * ctx=llama_init_from_model(model,cp); assert(ctx);
            llama_token input_token=common_tokenize(vocab,"hello",false,false).front();
            assert(llama_decode(ctx,llama_batch_get_one(&input_token,1))==0);
            for (bool grammar_first : {false,true}) {
                std::vector<event> events;
                auto state=std::make_shared<mask>(std::make_unique<native_parser>(token_session),
                    std::make_shared<file_store>(root.string(),std::vector<std::string>{"a","b"}),"filePath",
                    [&](const event & e) { events.push_back(e); },[vocab](int32_t token){return common_token_to_piece(vocab,token,true);});
                for(auto token:common_tokenize(vocab,begin,false,true)) state->accept_token(token);
                auto force=common_tokenize(vocab,"repeated",false,false).front();
                assert(state->allows(common_token_to_piece(vocab,force,true),force));
                common_params_sampling constrained;
                constrained.edit_mask=state;
                constrained.grammar={COMMON_GRAMMAR_TYPE_USER,"root ::= \"alpha\""};
                constrained.samplers={COMMON_SAMPLER_TYPE_TOP_K}; constrained.top_k=1;
                constrained.logit_bias.push_back({force,1000.0f});
                auto * smpl=common_sampler_init(model,constrained);
                auto token=common_sampler_sample(smpl,ctx,-1,grammar_first);
                auto piece=common_token_to_piece(vocab,token,true);
                assert(!piece.empty() && std::string("alpha").find(piece)==0);
                common_sampler_accept(smpl,token,true);
                auto retry=std::find_if(events.begin(),events.end(),[](const event & e){return e.status=="grammar_retry";});
                assert((retry==events.end())==grammar_first);
                assert(events.back().masked_tokens>0);
                if(!grammar_first) assert(events.back().masked_tokens==2*retry->masked_tokens);
                common_sampler_free(smpl);
            }
            llama_free(ctx);
        }
        llama_model_free(model);
    }
    std::cout << "native parser and sampler tests passed\n";
}
