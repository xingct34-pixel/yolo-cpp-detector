# CPU 稳定版架构

main 先校验参数，再选择图片模式或视频 Pipeline。Detector 独占 InferenceBackend，模型只加载一次。固定模型合同在 ORT 构造阶段检查，不匹配立即退出。

视频模式默认三个角色：读取线程、推理线程、主线程结果消费。顺序模式使用相同 Detector 和处理函数，便于建立基线。读取/推理/输出计数分别由对应角色写入，join 后汇总，避免并发读取非原子计数。

ThreadSafeQueue 的 mutex 保护 closed、容器与计数；not_empty 唤醒消费者，not_full 唤醒阻塞生产者。close 幂等并通知两类等待者。正常 EOF 逐级关闭并排空；异常或用户取消设置停止标志，关闭两端，join 后重新抛出首个异常。中断情况下不保证处理所有已读帧。

默认 block 策略保持离线帧完整性；drop-oldest 策略统计丢弃数量。正常结束时 read=inferred+input_dropped，inferred=output+result_dropped；取消时可有未处理余量。报告区分真实输出 FPS 与模型推理耗时倒数。

纯处理模块不依赖 ORT：prepare_yolo 负责 Letterbox、实际横纵缩放比例、RGB/CHW；decode_yolo 验证固定输出大小，过滤 NaN/Inf/负尺寸，浮点裁剪后转整数，并按类别分别 NMS。Detector 负责组合这些函数、调用后端及绘制。

帧队列和统计缓冲均有容量限制，但本项目没有为所有分辨率给出统一进程内存硬上限。统计分位保留最近一万帧，计数/均值/极值覆盖全程。端到端帧计时从解码完成开始，不包含模型加载、预热和此前采集时间。

GPU/TensorRT 路径保留为未验证实验代码；不用于 CPU 验收结论。
