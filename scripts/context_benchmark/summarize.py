import json,sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from rpent.context.trajectory import read_response, write_trajectory

def summarize(out):
    out=Path(out); audit=out/'requests'; selections={}
    if (out/'selection.jsonl').exists():
        for line in (out/'selection.jsonl').read_text().splitlines():
            v=json.loads(line);selections[v.get('request_id')]=v
    records=[]; previous={m:set() for m in ('image','text','action')}
    for path in sorted(audit.glob('*.before.json')):
        rid=path.name.split('.')[0];before=json.loads(path.read_text());timing_path=audit/(rid+'.timing.json')
        timing=json.loads(timing_path.read_text()) if timing_path.exists() else {}
        usage,outputs,response_id=read_response(audit/(rid+'.response'),timing.get('content_encoding',''))
        selection=selections.get(rid,{})
        refs={};latest=None
        for item in before.get('input',[]):
            snapshot=None
            for i,b in enumerate(item.get('output',[]) if isinstance(item.get('output'),list) else []):
                if b.get('type')=='input_text':
                    try:v=json.loads(b.get('text',''))
                    except ValueError:v=None
                    snapshot=v if isinstance(v,dict) and 'step' in v and 'state' in v else None
                if snapshot:
                    latest=max(latest or 0,snapshot['step']);refs[f"{item.get('id',item.get('call_id'))}:{i}"]=snapshot['step']
        deleted={}
        for m in previous:
            values=selection.get('records',[]) if m=='image' else selection.get(m,{}).get('records',[])
            current={r['id'] for r in values if not r['selected']}
            deleted[m]=[dict(r,source_step=refs.get(r['id'],r.get('step'))) for r in values if r['id'] in current-previous[m]]
            previous[m]=current
        records.append(dict(response_id=response_id,request_id=rid,latest_step=latest,usage=usage,timing=timing,newly_omitted=deleted,model_output=outputs,selection=selection))
    (out/'per-request.json').write_text(json.dumps(records,ensure_ascii=False,indent=2))
    usage_records=[r['usage'] for r in records if r['usage']]
    summary=dict(model_requests=len(records),usage_reported_requests=len(usage_records),input_tokens=sum(u.get('input_tokens',0) for u in usage_records),output_tokens=sum(u.get('output_tokens',0) for u in usage_records),cached_input_tokens=sum(u.get('input_tokens_details',{}).get('cached_tokens',0) for u in usage_records),model_request_latency_s=sum(r['timing'].get('latency_s',0) for r in records))
    states_path=out/'states.json'
    if states_path.exists():
        states=json.loads(states_path.read_text());summary['states']=states
    for path in out.glob('transcript*.json'):
        t=json.loads(path.read_text());summary['task_result']={k:v for k,v in t.items() if k!='messages'}
    if (out/'timing.json').exists():
        summary['timing']=json.loads((out/'timing.json').read_text())
    if states_path.exists():
        steps=states.get('steps',[])
        summary['action_steps']=sum(bool(x.get('command')) for x in steps)
    baseline_path = out/"network-baseline.json"
    if baseline_path.exists():
        summary["network_baseline"] = json.loads(baseline_path.read_text())
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    lines=['# 逐请求记录','','token 来自服务返回；缺失记为未返回，不估算。latency 为代理收到请求到响应流结束，包含插件和网络时间。工具执行结果见 states.json 和 tool-results.jsonl。','','| 请求 | 最新步骤 | 输入 token | 输出 token | latency 秒 | 新省略 Image / Text / Action 数量 |','| --- | --- | --- | --- | --- | --- |']
    for r in records:
        u=r['usage'] or {};lines.append(f"| {r['request_id']} | {r['latest_step']} | {u.get('input_tokens','未返回')} | {u.get('output_tokens','未返回')} | {r['timing'].get('latency_s','进行中')} | {' / '.join(str(len(r['newly_omitted'][m])) for m in previous)} |")
    for r in records:
        lines.extend(['',f"## 请求 {r['request_id']}，最新步骤 {r['latest_step']}",''])
        for m,values in r['newly_omitted'].items():
            lines.append(f'### 新省略 {m}')
            lines.extend(['','```json',json.dumps(values,ensure_ascii=False,indent=2),'```',''])
        if 'memory' in r['selection']:
            lines.extend(['### 本轮 Memory 查询、保留及省略内容','','```json',json.dumps(r['selection']['memory'],ensure_ascii=False,indent=2),'```',''])
        lines.extend(['### 模型随后输出的文字和工具调用','','```json',json.dumps(r['model_output'],ensure_ascii=False,indent=2),'```'])
    (out/'STEPS.zh-CN.md').write_text('\n'.join(lines)+'\n')
    write_trajectory(out,records,summary)
    return summary
if __name__=='__main__':summarize(sys.argv[1])
