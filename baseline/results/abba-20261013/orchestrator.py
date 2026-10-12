"""Local orchestration only; all timed requests run in remote benchmark subprocess."""
import argparse,json,re,subprocess,time,urllib.request,signal,threading
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('instance');ap.add_argument('--first-base');a=ap.parse_args()
root=Path('/Users/dotieuthien/Documents/rnd/vieneu');stage=Path('/tmp/vieneu-abba')
def fetch(base,path):
 with urllib.request.urlopen(base+path,timeout=30) as r:return r.read()
def post(base,path):
 with urllib.request.urlopen(urllib.request.Request(base+path,data=b'',method='POST'),timeout=30) as r:return json.load(r)
proc=None
try:
 for block,name in enumerate(['baseline','lite','lite','baseline'],1):
  for slots,counts in [(16,[(1,15),(4,5),(8,5),(16,5)]),(32,[(32,5)])]:
   out=root/name/'results/abba-20261013'/f'block-{block}-slots-{slots}';out.mkdir(parents=True,exist_ok=True)
   config=stage/name/'vkong.yaml';config.write_text(re.sub(r'MAX_STREAMS=\d+',f'MAX_STREAMS={slots}',config.read_text()))
   (out/'vkong-deployed.yaml').write_text(config.read_text())
   print(f'START block={block} backend={name} slots={slots}',flush=True)
   if block==1 and slots==16 and a.first_base:
    base=a.first_base
   else:
    proc=subprocess.Popen(['vkong','attach',a.instance,'--update','-C',str(stage/name)],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
    base=None
    with (out/'attach.log').open('w') as log:
     for line in proc.stdout:
      log.write(line);log.flush()
      if 'failed' in line or 'Error' in line:print(line.strip(),flush=True)
      m=re.search(r'http://127\.0\.0\.1:\d+',line)
      if m:
       base=m.group();break
    if base is None:raise RuntimeError(f'Attach failed: {out}')
    def drain(pipe, path):
     with path.open('a') as log:
      for line in pipe:
       log.write(line);log.flush()
    threading.Thread(target=drain,args=(proc.stdout,out/'attach.log'),daemon=True).start()
   health=json.loads(fetch(base,'/health'));assert health['max_streams']==slots,health
   assert health['backend']==('pytorch' if name=='baseline' else 'pytorch-lite'),health
   (out/'runtime.json').write_text(json.dumps(health,indent=2)+'\n')
   (out/'gpu-preflight.json').write_bytes(fetch(base,'/results/gpu-preflight.json'))
   pre=json.loads((out/'gpu-preflight.json').read_text());assert pre['idle_pass']
   print('PREFLIGHT',json.dumps(pre['clock_lock']),flush=True)
   ids=set()
   for c,repeats in counts:
    job=post(base,f'/benchmark?concurrency={c}&repeats={repeats}&seed=42')
    deadline=time.monotonic()+300
    while True:
     state=json.loads(fetch(base,'/benchmark'))
     if not state['running']:break
     if time.monotonic()>deadline:raise TimeoutError((block,slots,c))
     time.sleep(2)
    log=fetch(base,job['results']);(out/f'bench-{c}.log').write_bytes(log)
    for ext in ['jsonl','gpu.json']:(out/f'bench-{c}.{ext}').write_bytes(fetch(base,f'/results/bench-{c}.{ext}'))
    assert state['exit_code']==0,(state,log.decode())
    rows=[json.loads(x) for x in (out/f'bench-{c}.jsonl').read_text().splitlines()]
    assert len(rows)==c*repeats and all(r['ok'] for r in rows)
    ids.update(r['request_id'] for r in rows)
    print(f'DONE block={block} {name} slots={slots} c={c} measured={len(rows)}',flush=True)
   time.sleep(1)
   traces=[json.loads(x) for x in fetch(base,'/results/server-requests.jsonl').decode().splitlines()]
   traces=[r for r in traces if r['request_id'] in ids]
   assert len(traces)==len(ids),(len(traces),len(ids))
   (out/'server-requests.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in traces))
   (out/'health-final.json').write_bytes(fetch(base,'/health'))
   if proc:
    proc.send_signal(signal.SIGINT)
    try:proc.wait(timeout=15)
    except subprocess.TimeoutExpired:proc.terminate();proc.wait(timeout=10)
    proc=None
 print('ABBA COMPLETE: 1260 measured requests, all succeeded',flush=True)
finally:
 if proc and proc.poll() is None:
  proc.send_signal(signal.SIGINT)
