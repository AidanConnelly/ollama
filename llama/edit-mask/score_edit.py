#!/usr/bin/env python3
"""Score the requested sled edit independently of an assistant's self-report."""
import argparse
import json
import subprocess
from pathlib import Path

parser=argparse.ArgumentParser()
parser.add_argument('experiment',type=Path)
parser.add_argument('--repo',type=Path,required=True)
args=parser.parse_args()
metadata=dict(line.split('=',1) for line in (args.experiment/'metadata.txt').read_text().splitlines() if '=' in line)
original=subprocess.check_output(['git','-C',str(args.repo),'show',metadata['baseline_commit']+':src/lib.rs'])
needle=b'//! `sled` is a high-performance embedded database with'
assert original.count(needle)==1
expected=original.replace(needle,b'// The crate-level documentation begins below.\n'+needle,1)
results=[]
for trial in sorted((args.experiment/'trials').glob('pair-*')):
    worktree=args.experiment/'worktrees'/trial.name
    actual=(worktree/'src/lib.rs').read_bytes()
    changes=subprocess.check_output(['git','-C',str(worktree),'diff','--name-only','HEAD']).decode().splitlines()
    calls=[]
    transcript=trial/'opencode.stdout.jsonl'
    for line in transcript.read_text().splitlines():
        try:event=json.loads(line)
        except ValueError:continue
        if event.get('type')=='tool_use':
            part=event.get('part',{})
            calls.append({'tool':part.get('tool'),'state':part.get('state',{})})
    reads=[i for i,call in enumerate(calls) if call['tool']=='read' and call['state'].get('status')=='completed' and call['state'].get('input',{}).get('offset')==1 and call['state'].get('input',{}).get('limit')==100]
    edits=[i for i,call in enumerate(calls) if call['tool']=='edit' and call['state'].get('status')=='completed']
    mask_events=[]
    for line in (trial/'ollama.log').read_text(errors='replace').splitlines():
        if 'OLLAMA_EDIT_MASK ' in line:
            try:mask_events.append(json.loads(line.split('OLLAMA_EDIT_MASK ',1)[1]))
            except ValueError:pass
    successes=[call for call in calls if call['tool']=='edit' and call['state'].get('status')=='completed']
    read_sequence=any(a<edit<b for a in reads for edit in edits for b in reads)
    confirmed=any(edit<b and '// The crate-level documentation begins below.' in str(calls[b]['state'].get('output','')) for edit in edits for b in reads)
    results.append({'trial':trial.name,'exact_requested_edit':actual==expected,'only_requested_file_modified':changes==['src/lib.rs'],
                    'read_edit_read_sequence':read_sequence,'confirmed_in_read_output':confirmed,'actual_task_success':actual==expected and changes==['src/lib.rs'] and read_sequence and confirmed,
                    'edit_calls':len([call for call in calls if call['tool']=='edit']),'completed_edit_calls':len(successes),
                    'mask_active':any(e['status']=='active' for e in mask_events),'fallbacks':[e for e in mask_events if e['status']=='skipped'],
                    'masked_tokens_total':sum(max((e.get('masked_tokens',0) for e in mask_events if e.get('request_id')==req),default=0) for req in {e.get('request_id') for e in mask_events}),
                    'mask_events':mask_events})
output={'implementation':'oldString-mask-v1','note':'Self-report and process success are retained separately in results.csv. Fallbacks are not evidence of successful masking.','trials':results}
(args.experiment/'actual-edit-score.json').write_text(json.dumps(output,indent=2)+'\n')
print(json.dumps([{k:v for k,v in r.items() if k!='mask_events'} for r in results],indent=2))
