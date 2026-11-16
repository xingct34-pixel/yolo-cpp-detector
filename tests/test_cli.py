# YOLO 检测部署 | tests/test_cli.py
# 功能：从用户使用的命令行入口传入超范围数字、缺失路径等参数，核对失败与帮助行为。
# 这里测试 main.cpp 的入口约束，错误参数应该能在模型推理前被识别或报告。
# 目的：从可执行程序入口检查坏参数的退出行为，避免错误输入悄悄进入推理。


import subprocess,sys
exe=sys.argv[1]


# 模块：非法输入样本
# 把边界值、溢出、缺少参数等作为用户输入传给真正的命令行程序。
# 65 比允许的上限 64 大一格，用来检查业务边界；那串很长的 9 还会超出整数可表示的范围。两种输入都应失败，但失败原因不同。
cases=[['--queue-size','65'],['--queue-size','9999999999999999999999'],['--max-frames','-1'],
       ['--warmup','101'],['--mode','bad'],['--queue-policy','bad'],['--image','missing.ppm'],['--model']]


# 模块：遍历坏参数
# capture_output 收集输出，timeout 防止错误输入意外进入长时间视频处理。每轮核对退出码。
for args in cases:
    r=subprocess.run([exe]+args,capture_output=True,timeout=3)
    assert r.returncode==1,(args,r.returncode)


# 模块：正向对照
# --help 应正常退出，防止程序因为根本无法运行而让所有坏输入看起来都被拒绝。
assert subprocess.run([exe,'--help'],capture_output=True,timeout=3).returncode==0
print('8 invalid CLI cases and help passed')
