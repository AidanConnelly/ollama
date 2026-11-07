#!/usr/bin/env python3
"""Save native render/token equality evidence for counterfactual mask conditions."""
import argparse
import copy
import hashlib
import json
import time
import urllib.request
from pathlib import Path

parser=argparse.ArgumentParser()
parser.add_argument('--url',required=True)
parser.add_argument('--root',required=True)
parser.add_argument('--out',type=Path,required=True)
args=parser.parse_args()

def post(path,body):
    request=urllib.request.Request(args.url+path,json.dumps(body,ensure_ascii=False).encode(),{'Content-Type':'application/json'})
    with urllib.request.urlopen(request,timeout=120) as response:return json.load(response)

for attempt in range(180):
    try:
        with urllib.request.urlopen(args.url+'/health',timeout=1) as response:
            if json.load(response).get('status')=='ok':break
    except (OSError,ValueError):pass
    time.sleep(1)
else:raise RuntimeError('native runner did not become ready')

tools=[{'type':'function','function':{'name':'edit','description':'Replace exact text in a file','parameters':{
    'type':'object','properties':{key:{'type':'string'} for key in ['newString','oldString','filePath']},'required':['newString','oldString','filePath']}}},
    {'type':'function','function':{'name':'read','parameters':{'type':'object','properties':{'filePath':{'type':'string'},'offset':{'type':'integer'},'limit':{'type':'integer'}},'required':['filePath']}}}]
messages=[{'role':'system','content':f'<env>\nWorking directory: {args.root}\n</env>'},{'role':'user','content':'Inspect src/lib.rs, then insert the requested documentation comment.'}]
evidence={'implementation':'oldString-mask-v1','comparisons':[]}
for after_read in [False,True]:
    if after_read:
        messages.extend([{'role':'assistant','content':'','tool_calls':[{'id':'read-1','type':'function','function':{'name':'read','arguments':json.dumps({'filePath':str(Path(args.root)/'src/lib.rs'),'offset':1,'limit':100})}}]},
                         {'role':'tool','tool_call_id':'read-1','name':'read','content':'\n'.join((Path(args.root)/'src/lib.rs').read_text().splitlines()[:100])}])
    body={'messages':copy.deepcopy(messages),'tools':tools,'chat_template_kwargs':{'enable_thinking':False},'backend_sampling':False}
    conditions=[]
    for enabled in [False,True]:
        request=copy.deepcopy(body)
        if enabled:request['_ollama_oldstring_mask']={'version':1,'request_id':'equality','root':args.root,'files':['src/lib.rs'],'path_field':'filePath'}
        rendered=post('/apply-template',request)['prompt']
        tokens=post('/tokenize',{'content':rendered,'add_special':True,'parse_special':True})['tokens']
        conditions.append({'condition':'constrained' if enabled else 'control','request':request,'rendered_prompt':rendered,'prompt_token_ids':tokens,'prompt_sha256':hashlib.sha256(rendered.encode()).hexdigest()})
    a,b=conditions
    checks={'messages_equal':a['request']['messages']==b['request']['messages'],'tools_equal':a['request']['tools']==b['request']['tools'],'rendered_prompts_equal':a['rendered_prompt']==b['rendered_prompt'],'prompt_token_ids_equal':a['prompt_token_ids']==b['prompt_token_ids']}
    assert all(checks.values()),checks
    evidence['comparisons'].append({'after_read':after_read,**checks,'conditions':conditions})
args.out.parent.mkdir(parents=True,exist_ok=True)
args.out.write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+'\n')
print(json.dumps([{'after_read':item['after_read'],'tokens':len(item['conditions'][0]['prompt_token_ids']),'all_equal':True} for item in evidence['comparisons']]))
