// YOLO 检测部署 | include/thread_safe_queue.h
// 视频各阶段速度不同，这个模板把“暂存数据”和“跨线程交接”放在一起。
// 例如容量为 2，第三帧到达时，Block 会等空位，DropOldest 会腾掉最早的未处理项。
// pop 在持锁时取元素，调用者拿到元素后可以在锁外推理。close 会唤醒等待者。


#pragma once
#include <algorithm>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <utility>


// 模块：容量策略：按任务目标选择阻塞等待或丢弃旧数据
// Block：满时生产者等待，适合保留离线帧；DropOldest：丢最旧待处理项，偏向实时新鲜度。


enum class QueuePolicy { DropOldest, Block };


// 模块：队列接口：通过互斥锁和条件变量保护共享数据


template <class T> class ThreadSafeQueue {
public:


    // 模块：初始化：设定容量及满队列处理策略


    explicit ThreadSafeQueue(size_t capacity, QueuePolicy policy = QueuePolicy::DropOldest)
        : capacity_(capacity), policy_(policy) {
        if (!capacity)
            throw std::invalid_argument("queue capacity must be positive");
    }


    // 模块：生产者入队：按策略处理满队列并通知消费者
    // 锁只保护队列、关闭状态和计数；通知是唤醒信号，数据仍通过队列交接。


    bool push(T item) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (policy_ == QueuePolicy::Block)
            not_full_.wait(lock, [&] {
                return closed_ || queue_.size() < capacity_;
            });
        if (closed_)
            return false;
        if (queue_.size() == capacity_) {
            queue_.pop();
            ++dropped_;
        }
        queue_.push(std::move(item));
        peak_ = std::max(peak_, queue_.size());
        lock.unlock();
        not_empty_.notify_one();
        return true;
    }


    // 模块：消费者出队：等待数据，取出后立即释放锁


    bool pop(T& item) {
        std::unique_lock<std::mutex> lock(mutex_);
        // wait睡眠时释放锁，醒来重新加锁并检查条件，避免空转及虚假唤醒误取数据。
        not_empty_.wait(lock, [&] {
            return closed_ || !queue_.empty();
        });
        if (queue_.empty())
            return false;
        // front只访问队头，pop才移除；取出后立即解锁，调用者的推理/显示不占此锁。
        item = std::move(queue_.front());
        queue_.pop();
        lock.unlock();
        not_full_.notify_one();
        return true;
    }


    // 模块：关闭通知：拒绝新数据，唤醒等待者并允许排空旧数据
    // close不清空队列；关闭且非空仍可取出旧数据，关闭且为空时pop才返回false。
    // 两类等待者都要唤醒，避免生产者卡在满队列、消费者卡在空队列，导致join一直等待。


    void close() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            closed_ = true;
        }
        not_full_.notify_all();
        not_empty_.notify_all();
    }


    // 模块：只读统计：查询丢弃数量和队列峰值


    size_t dropped() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return dropped_;
    }

    size_t peak() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return peak_;
    }

private:


    // 模块：内部状态：共享容器、同步原语与容量计数


    std::queue<T> queue_;
    mutable std::mutex mutex_;
    std::condition_variable not_empty_, not_full_;
    size_t capacity_, dropped_ = 0, peak_ = 0;
    QueuePolicy policy_;
    bool closed_ = false;
};
