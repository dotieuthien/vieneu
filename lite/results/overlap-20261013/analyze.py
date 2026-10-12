import json, hashlib, math
from pathlib import Path
root=Path('/Users/dotieuthien/Documents/rnd/vieneu')
out=root/'lite/results/overlap-20261013'
def quant(a,p):
 a=sorted(a); x=(len(a)-1)*p; i=int(x); return a[i]+(a[min(i+1,len(a)-1)]-a[i])*(x-i)
def stats(rows):
 return {'n':len(rows), **{k:{'p50':quant([r[k] for r in rows],.5),'p90':quant([r[k] for r in rows],.9)} for k in ['ttfa_s','rtf','total_s','audio_s']}}
groups={}; blocks=[]; hosts=set(); gpus=set()
for d in sorted(out.glob('block-*')):
 name=d.name.split('-')[-1]; source=root/('baseline' if name=='baseline' else 'lite')
 rows=[]
 for c in [1,16]:
  f=d/f'bench-{c}.jsonl'
  if not f.exists():continue
  rs=[json.loads(s) for s in f.read_text().splitlines()]
  assert len(rs)==(3 if c==1 else 16) and all(r['ok'] for r in rs)
  meta=json.loads((d/f'bench-{c}.gpu.json').read_text()); assert meta['idle_pass']
  hosts.add(meta['hostname']); gpus.update(s['uuid'] for s in meta['before'])
  for path,h in meta['source_sha256'].items():
   local=source/path
   if path=='bootstrap.py':continue
   assert local.exists() and hashlib.sha256(local.read_bytes()).hexdigest()==h, (d,path)
  groups.setdefault((name,c),[]).extend(rs); rows.extend(rs)
  blocks.append({'block':d.name,'c':c,**stats(rs)})
 if (d/'server-requests.jsonl').exists():
  traces={r['request_id']:r for r in map(json.loads,(d/'server-requests.jsonl').read_text().splitlines())}
  assert set(traces)=={r['request_id'] for r in rows}
  assert all(abs(r['audio_s']-traces[r['request_id']]['audio_s'])<1e-8 for r in rows)
assert len(hosts)<=1 and len(gpus)<=1
report={'groups':[{'name':n,'c':c,**stats(rs)} for (n,c),rs in groups.items()], 'blocks':blocks,'hostnames':list(hosts),'gpu_uuids':list(gpus)}
(out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
for x in report['groups']:
 print(x['name'],x['c'],x['n'],'ttfa',round(1000*x['ttfa_s']['p50'],1),'rtf',round(x['rtf']['p50'],3))
