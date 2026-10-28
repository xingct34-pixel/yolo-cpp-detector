import subprocess,sys
exe=sys.argv[1]
cases=[['--queue-size','65'],['--queue-size','9999999999999999999999'],['--max-frames','-1'],
       ['--warmup','101'],['--mode','bad'],['--queue-policy','bad'],['--image','missing.ppm'],['--model']]
for args in cases:
    r=subprocess.run([exe]+args,capture_output=True,timeout=3)
    assert r.returncode==1,(args,r.returncode)
assert subprocess.run([exe,'--help'],capture_output=True,timeout=3).returncode==0
print('8 invalid CLI cases and help passed')
