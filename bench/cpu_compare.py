"""Alternating sequential/pipeline measurements; no claim of guaranteed speedup."""
from pathlib import Path
import hashlib,json,subprocess,time,os
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results';OUT.mkdir(exist_ok=True)
binary=ROOT/'build-local/yolo_detector'
environment=dict(os.environ,YOLO_ORT_INTRA_OP_THREADS='3')
record={'binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),
        'frames_per_run':30,'warmup':2,'queue_capacity':2,'threads':3,'runs':[]}
for repeat in range(3):
    modes=['sequential','pipeline'] if repeat%2==0 else ['pipeline','sequential']
    for mode in modes:
        label=f'compare-{repeat}-{mode}';path=OUT/(label+'.json');peak=0
        command=[str(binary),'--video','test.mp4','--headless','--mode',mode,'--queue-policy','block',
            '--queue-size','2','--max-frames','30','--warmup','2','--report',str(path)]
        with (OUT/(label+'.log')).open('w') as log:
            p=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=log,env=environment)
            begin=time.monotonic()
            while p.poll() is None:
                if time.monotonic()-begin>120:p.kill();p.wait();raise RuntimeError('comparison timeout')
                try:
                    for line in Path(f'/proc/{p.pid}/status').read_text().splitlines():
                        if line.startswith('VmHWM:'):peak=max(peak,int(line.split()[1]))
                except FileNotFoundError:pass
                time.sleep(.05)
            assert p.returncode==0,(label,p.returncode)
        row=json.loads(path.read_text());assert row['output_frames']==row['inferred_frames']==row['read_frames']==30
        row.update(repeat=repeat,peak_rss_kib_sampled=peak);record['runs'].append(row)
        print(label,'fps',round(row['output_fps'],3),flush=True)
(OUT/'controlled-comparison.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
# Complete the existing full local clip, not just a prefix.
subprocess.run(['bash','scripts/demo.sh'],cwd=ROOT,check=True,stdout=(OUT/'demo.log').open('w'),stderr=subprocess.STDOUT,timeout=180,env=environment)
demo=json.loads((OUT/'demo.json').read_text());assert demo['read_frames']==demo['inferred_frames']==demo['output_frames']==150
print('full 150-frame clip passed',flush=True)
