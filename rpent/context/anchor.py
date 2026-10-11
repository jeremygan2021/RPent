"""Keep image anchors while allowing old robot execution details to be selected."""
import copy
import json
from dataclasses import replace
from .grouped import GroupedAdapter
from .codex import action_summary

class AnchorAdapter(GroupedAdapter):
    def extract_action(self, request):
        records = super().extract_action(request)
        if not records:
            return records
        items = request['input']
        outputs = [x.get('call_id') for x in items if x.get('type') in ('custom_tool_call_output', 'function_call_output')]
        latest = max(r.snapshot['step'] for r in records)
        result = []
        for r in records:
            outcome = r.snapshot['log'].get('result') or {}
            protected = (items[r.item].get('call_id') in outputs[-2:]
                         or r.snapshot['step'] == latest
                         or bool(outcome.get('error'))
                         or outcome.get('success') is False
                         or bool(r.snapshot.get('terminated'))
                         or bool(r.snapshot.get('truncated')))
            result.append(replace(r, protected=protected))
        return result

    def render_action(self, request, records, keep):
        result = super().render_action(request, records, keep)
        for r in records:
            if r.id in keep:
                continue
            blocks = request['input'][r.item]['output']
            if not any(b.get('type') == 'input_image' for b in blocks[r.block+1:]):
                continue
            if result is request:
                result = copy.deepcopy(request)
            snapshot = r.snapshot
            outcome = snapshot['log'].get('result') or {}
            command = snapshot['log'].get('command') or {}
            # Preserve the full state and artifact mapping explaining retained images.
            value = {k: snapshot[k] for k in ('step','state','artifacts','images','task_language','terminated','truncated','task_progress','success','vla_desync') if k in snapshot}
            value['context_omitted'] = True
            value['log'] = action_summary(snapshot)
            result['input'][r.item]['output'][r.block]['text'] = json.dumps(value, ensure_ascii=False)
        return result
