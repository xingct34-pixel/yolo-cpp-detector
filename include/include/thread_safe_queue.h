#pragma once

#include <condition_variable>
#include <cstddef>
#include <mutex>
#include <queue>
#include <utility>

// ==================== 线程安全队列模块 ====================
//
// ThreadSafeQueue：多个线程之间安全传递数据的“中转仓库”
//
// mutex          → 仓库门锁
// condition_variable → 仓库铃铛
// queue          → 仓库货架
//
// 本项目采用实时视频处理策略：
// 队列满时丢掉最旧的数据，而不是阻塞生产者。

template<typename T>
class ThreadSafeQueue
{
public:
    explicit ThreadSafeQueue(
        size_t max_size)
        : max_size_(max_size)
    {
    }

    // ==================== 写入模块 ====================

    bool push(T item)
    {
        {
            std::lock_guard<std::mutex> lock(
                mutex_);

            // 队列已经关闭
            if (closed_)
            {
                return false;
            }

            // ==================== 实时丢帧策略 ====================
            //
            // 队列满：
            // 丢掉最旧的数据，给最新数据腾位置。
            //
            // 对实时视频来说：
            // 最新画面 > 旧画面
            if (queue_.size() >= max_size_)
            {
                queue_.pop();
            }

            queue_.push(
                std::move(item));
        }

        // 通知等待中的消费者
        not_empty_.notify_one();

        return true;
    }

    // ==================== 读取模块 ====================

    bool pop(T& item)
    {
        std::unique_lock<std::mutex> lock(
            mutex_);

        // ==================== 条件等待模块 ====================
        //
        // 队列为空：
        // 当前线程睡眠等待。
        //
        // 有数据：
        // 醒来继续执行。
        //
        // 队列关闭：
        // 即使没有数据，也必须醒来退出。
        not_empty_.wait(
            lock,
            [this]
            {
                return !queue_.empty()
                    || closed_;
            });

        // ==================== 退出判断 ====================

        if (queue_.empty() && closed_)
        {
            return false;
        }

        // ==================== 数据取出模块 ====================

        item =
            std::move(queue_.front());

        queue_.pop();

        return true;
    }

    // ==================== 队列关闭模块 ====================

    void close()
    {
        {
            std::lock_guard<std::mutex> lock(
                mutex_);

            closed_ = true;
        }

        // 唤醒所有等待中的消费者
        not_empty_.notify_all();
    }

private:
    // ==================== 数据结构模块 ====================

    std::queue<T> queue_;

    // ==================== 同步模块 ====================

    std::mutex mutex_;
    std::condition_variable not_empty_;

    // ==================== 队列控制模块 ====================

    size_t max_size_;
    bool closed_ = false;
};
