from pathlib import Path
import json,hashlib,statistics
root=Path('/Users/dotieuthien/Documents/rnd/vieneu')
def percentile(values,p):
 v=sorted(values);idx=(len(v)-1)*p;lo=int(idx);hi=min(lo+1,len(v)-1);return v[lo]+(v[hi]-v[lo])*(idx-lo)
def summary(rows):
 return dict(n=len(rows),client_ttfa_p50_ms=percentile([r['ttfa_s']*1000 for r in rows],.5),client_ttfa_p90_ms=percentile([r['ttfa_s']*1000 for r in rows],.9),server_ttfa_p50_ms=percentile([r['server_ttfa_s']*1000 for r in rows],.5),server_ttfa_p90_ms=percentile([r['server_ttfa_s']*1000 for r in rows],.9),rtf_p50=percentile([r['rtf'] for r in rows],.5),rtf_p90=percentile([r['rtf'] for r in rows],.9),rtf_max=max(r['rtf'] for r in rows),audio_s_min=min(r['audio_s'] for r in rows),audio_s_max=max(r['audio_s'] for r in rows),audio_s_p50=percentile([r['audio_s'] for r in rows],.5))
reports={}
for name in ['baseline','lite']:
 d=root/name/'results/abba-20261013';rows=[];blocks=[];gpu=[];hosts=set();uuids=set();preflights=[]
 hashes=json.loads((d/'source-sha256.json').read_text())
 for k,v in hashes.items():assert hashlib.sha256((root/name/k).read_bytes()).hexdigest()==v,(name,k)
 for block in sorted(d.glob('block-*')):
  number=int(block.name.split('-')[1]);slots=int(block.name.split('-')[-1]);local=[]
  server={r['request_id']:r for r in map(json.loads,(block/'server-requests.jsonl').read_text().splitlines())}
  pre=json.loads((block/'gpu-preflight.json').read_text());assert pre['idle_pass'];preflights.append(pre)
  for file in sorted(block.glob('bench-*.jsonl')):
   meta=json.loads(file.with_suffix('.gpu.json').read_text());assert meta['idle_pass'];hosts.add(meta['hostname'])
   for k,v in meta['source_sha256'].items():assert hashes[k]==v,(name,block,k)
   for snap in meta['during']:
    if 'uuid' in snap:gpu.append(snap);uuids.add(snap['uuid'])
   for r in map(json.loads,file.read_text().splitlines()):
    assert r['ok'];s=server[r['request_id']];assert abs(s['audio_s']-r['audio_s'])<1e-8
    r.update(server_ttfa_s=s['server_ttfa_s'],server_total_s=s['server_total_s'],block=number,slots=slots)
    rows.append(r);local.append(r)
  blocks.append(dict(block=number,slots=slots,cases={str(c):summary([r for r in local if r['concurrency']==c]) for c in sorted({r['concurrency'] for r in local})}))
 assert len(rows)==630,len(rows)
 cases={str(c):summary([r for r in rows if r['concurrency']==c]) for c in [1,4,8,16,32]}
 assert len(hosts)==len(uuids)==1
 (d/'requests-joined.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
 report=dict(date_local='2026-10-13',implementation=name,requests_ok=len(rows),errors=0,cases=cases,blocks=blocks,hostname=next(iter(hosts)),gpu_uuid=next(iter(uuids)),source_sha256=hashes,rental=dict(id='vk_7fba8887e3fa2c37',hourly_usd=0.077778,released=True),method=dict(order=['baseline','lite','lite','baseline'],slots={str(c):16 if c<=16 else 32 for c in [1,4,8,16,32]},measured_repetitions_per_block={'1':15,'4':5,'8':5,'16':5,'32':5},warmup_bursts_per_case=2,seed='42 + repetition index, reset before each burst; scheduling can still change generated audio',gpu_idle='<=5% utilization for 5 consecutive samples before model load and before each concurrency case',clock_target_mhz=1500,percentile='linear interpolation at (n-1)*p over individual requests; no median of medians',benchmark='Instrumented upstream PCM client, adding per-request JSONL, barrier, seed and repetitions',client_location='same container HTTP loopback http://127.0.0.1:8000',server_ttfa='speech handler after validation to first float audio chunk; includes scheduler/preparation, excludes PCM conversion/socket',comparison='descriptive ABBA; two deployment blocks per implementation, not a statistical equivalence test'),gpu=dict(preflights=preflights,driver_versions=sorted({g['driver_version'] for g in gpu}),power_limits_w=sorted({float(g['power.limit']) for g in gpu}),graphics_clock_range_mhz=[min(float(g['clocks.current.graphics']) for g in gpu),max(float(g['clocks.current.graphics']) for g in gpu)],temperature_range_c=[min(float(g['temperature.gpu']) for g in gpu),max(float(g['temperature.gpu']) for g in gpu)]))
 active=[g for g in gpu if float(g['utilization.gpu']) >= 50]
 report['gpu']['active_graphics_clock_range_mhz']=[min(float(g['clocks.current.graphics']) for g in active),max(float(g['clocks.current.graphics']) for g in active)]
 report['gpu']['clock_locked']=all(p['clock_lock']['returncode']==0 for p in preflights)
 reports[name]=report;(root/name/'results/report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
assert reports['baseline']['hostname']==reports['lite']['hostname'] and reports['baseline']['gpu_uuid']==reports['lite']['gpu_uuid']
rows=[];serverrows=[];lengthrows=[]
for c in ['1','4','8','16','32']:
 b=reports['baseline']['cases'][c];l=reports['lite']['cases'][c]
 rows.append(f"| {c} | {16 if int(c)<=16 else 32} | {b['n']} | {b['client_ttfa_p50_ms']:.1f} / {b['client_ttfa_p90_ms']:.1f} | {l['client_ttfa_p50_ms']:.1f} / {l['client_ttfa_p90_ms']:.1f} | {b['rtf_p50']:.3f} / {b['rtf_p90']:.3f} | {l['rtf_p50']:.3f} / {l['rtf_p90']:.3f} |")
 serverrows.append(f"| {c} | {b['server_ttfa_p50_ms']:.1f} / {b['server_ttfa_p90_ms']:.1f} | {l['server_ttfa_p50_ms']:.1f} / {l['server_ttfa_p90_ms']:.1f} |")
 lengthrows.append(f"| {c} | {b['audio_s_min']:.2f}–{b['audio_s_max']:.2f} | {l['audio_s_min']:.2f}–{l['audio_s_max']:.2f} |")
table='| Clients | Slots | n mỗi bản | Baseline TTFA p50/p90 (ms) | Lite TTFA p50/p90 (ms) | Baseline RTF p50/p90 | Lite RTF p50/p90 |\n|---:|---:|---:|---:|---:|---:|---:|\n'+'\n'.join(rows)
server_table='| Clients | Baseline server TTFA p50/p90 (ms) | Lite server TTFA p50/p90 (ms) |\n|---:|---:|---:|\n'+'\n'.join(serverrows)
audio_table='| Clients | Baseline audio min–max (s) | Lite audio min–max (s) |\n|---:|---:|---:|\n'+'\n'.join(lengthrows)
(root/'baseline/results/abba-20261013/tables.md').write_text(table+'\n\n'+server_table+'\n\n'+audio_table+'\n')
print(table);print('\n'+server_table)
for name,r in reports.items():print(name,'GPU',json.dumps({k:v for k,v in r['gpu'].items() if k!='preflights'}),'locks',[(p['clock_lock']['returncode'],p['clock_lock']['stdout'].strip()) for p in r['gpu']['preflights']])
