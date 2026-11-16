# YOLO 检测部署 | tests/real_cpu.py
# 功能：用本机的真实 ONNX 和视频调用已编译的检测程序，检查顺序、流水线、丢帧和单图路径。
# 例如读入帧数应等于推理帧数加输入丢帧数；最后还向测试子进程发送 SIGINT 验证退出。
# 目的：把队列、图像解码和真实推理连起来验收，检查合成测试看不到的协作问题。


from pathlib import Path
import subprocess,json,os,hashlib,tempfile,signal,time
r=Path(__file__).resolve().parents[1];out=r/'results';out.mkdir(exist_ok=True);exe=r/'build-local/yolo_detector'
base=[str(exe),'--model','models/yolo11n.onnx','--classes','data/coco.txt','--headless','--warmup','2']
records=[]
def run(name,args,timeout=180):
    with (out/(name+'.log')).open('w') as log:p=subprocess.run(base+args,cwd=r,stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
    if p.returncode:raise RuntimeError(name+': '+(out/(name+'.log')).read_text()[-2000:])
    print(name,'passed',flush=True)


# 模块：核对完整处理
# 顺序和流水线输入同样多的帧，阻塞模式下三段计数应一致。
# 先固定阻塞队列，让读取、推理、输出都完成规定帧数，才具备比较两种运行模式的前提。
for mode in ['sequential','pipeline']:
    name='real-'+mode
    run(name,['--video','test.mp4','--mode',mode,'--max-frames','30','--queue-size','2','--report',str(out/(name+'.json'))])
    d=json.loads((out/(name+'.json')).read_text());assert d['read_frames']==d['inferred_frames']==d['output_frames']==30
    assert d['input_dropped']==d['result_dropped']==0;records.append(d)


# 模块：核对被丢弃的帧
# 把两个队列分别算账，哪条队列丢了帧，差值就应出现在它自己的计数里。
# DropOldest 的实际丢帧数取决于线程运行速度。回归更适合核对“读入、推理、显示和两处丢帧”的数量关系，固定要求每次都丢同样多的帧会让测试不可靠。
run('real-drop',['--video','test.mp4','--queue-policy','drop-oldest','--queue-size','1','--report',str(out/'real-drop.json')])
d=json.loads((out/'real-drop.json').read_text());assert d['read_frames']==d['inferred_frames']+d['input_dropped'];assert d['inferred_frames']==d['output_frames']+d['result_dropped'];records.append(d)
with tempfile.TemporaryDirectory() as folder:
    folder=Path(folder);image=folder/'input.ppm';image.write_bytes(b'P6\n320 240\n255\n'+bytes([80,120,160])*320*240)
    run('real-image',['--image',str(image),'--output',str(folder/'output.png'),'--report',str(out/'real-image.json')])
    assert (folder/'output.png').stat().st_size>0;records.append(json.loads((out/'real-image.json').read_text()))
    # A real worker shutdown test using only our own child process.


    # 信号场景
    # 单独启动子进程，在它运行期间发信号；这样才能测线程能否退出，而不只是输入文件自然结束。
    with (out/'real-sigint.log').open('w') as log:
        p=subprocess.Popen(base+['--video','test.mp4','--queue-size','1'],cwd=r,stdout=log,stderr=log)
        time.sleep(2);assert p.poll() is None;p.send_signal(signal.SIGINT);code=p.wait(15);assert code==0
record={'binary_sha256':hashlib.sha256(exe.read_bytes()).hexdigest(),'model_sha256':hashlib.sha256((r/'models/yolo11n.onnx').read_bytes()).hexdigest(),
        'video_sha256':hashlib.sha256((r/'test.mp4').read_bytes()).hexdigest(),'cases':records,'sigint_exit_code':code}
(out/'local-verification.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(record,ensure_ascii=False),flush=True)
