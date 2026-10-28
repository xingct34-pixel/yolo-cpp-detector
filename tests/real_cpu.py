from pathlib import Path
import subprocess,json,os,hashlib,tempfile,signal,time
r=Path(__file__).resolve().parents[1];out=r/'results';out.mkdir(exist_ok=True);exe=r/'build-local/yolo_detector'
base=[str(exe),'--model','models/yolo11n.onnx','--classes','data/coco.txt','--headless','--warmup','2']
records=[]
def run(name,args,timeout=180):
    with (out/(name+'.log')).open('w') as log:p=subprocess.run(base+args,cwd=r,stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
    if p.returncode:raise RuntimeError(name+': '+(out/(name+'.log')).read_text()[-2000:])
    print(name,'passed',flush=True)
for mode in ['sequential','pipeline']:
    name='real-'+mode
    run(name,['--video','test.mp4','--mode',mode,'--max-frames','30','--queue-size','2','--report',str(out/(name+'.json'))])
    d=json.loads((out/(name+'.json')).read_text());assert d['read_frames']==d['inferred_frames']==d['output_frames']==30
    assert d['input_dropped']==d['result_dropped']==0;records.append(d)
run('real-drop',['--video','test.mp4','--queue-policy','drop-oldest','--queue-size','1','--report',str(out/'real-drop.json')])
d=json.loads((out/'real-drop.json').read_text());assert d['read_frames']==d['inferred_frames']+d['input_dropped'];assert d['inferred_frames']==d['output_frames']+d['result_dropped'];records.append(d)
with tempfile.TemporaryDirectory() as folder:
    folder=Path(folder);image=folder/'input.ppm';image.write_bytes(b'P6\n320 240\n255\n'+bytes([80,120,160])*320*240)
    run('real-image',['--image',str(image),'--output',str(folder/'output.png'),'--report',str(out/'real-image.json')])
    assert (folder/'output.png').stat().st_size>0;records.append(json.loads((out/'real-image.json').read_text()))
    # A real worker shutdown test using only our own child process.
    with (out/'real-sigint.log').open('w') as log:
        p=subprocess.Popen(base+['--video','test.mp4','--queue-size','1'],cwd=r,stdout=log,stderr=log)
        time.sleep(2);assert p.poll() is None;p.send_signal(signal.SIGINT);code=p.wait(15);assert code==0
record={'binary_sha256':hashlib.sha256(exe.read_bytes()).hexdigest(),'model_sha256':hashlib.sha256((r/'models/yolo11n.onnx').read_bytes()).hexdigest(),
        'video_sha256':hashlib.sha256((r/'test.mp4').read_bytes()).hexdigest(),'cases':records,'sigint_exit_code':code}
(out/'local-verification.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(record,ensure_ascii=False),flush=True)
