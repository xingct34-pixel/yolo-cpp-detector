# YOLO 检测部署 | tests/test_cli.py
# 从用户使用的命令行入口传入超范围数字、缺失路径等参数，核对失败与帮助行为。
# 这里测试 main.cpp 的入口约束，错误参数应该能在模型推理前被识别或报告。


import subprocess,sys
exe=sys.argv[1]


# 模块：非法输入样本
# 把边界值、溢出、缺少参数等作为用户输入传给真正的命令行程序。


cases=[['--queue-size','65'],['--queue-size','9999999999999999999999'],['--max-frames','-1'],
       ['--warmup','101'],['--mode','bad'],['--queue-policy','bad'],['--image','missing.ppm'],['--model']]
for args in cases:
    r=subprocess.run([exe]+args,capture_output=True,timeout=3)
    assert r.returncode==1,(args,r.returncode)
assert subprocess.run([exe,'--help'],capture_output=True,timeout=3).returncode==0
print('8 invalid CLI cases and help passed')
