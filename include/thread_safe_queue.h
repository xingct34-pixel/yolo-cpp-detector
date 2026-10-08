#pragma once

#include <condition_variable>
#include <cstddef>
#include <mutex>
#include <queue>
#include <utility>

/*
 * ============================================================
 * ThreadSafeQueue
 *
 * 功能：
 * 1. 多线程安全地进行 push / pop
 * 2. 队列为空时，消费者阻塞等待
 * 3. 队列满时丢弃最旧数据
 * 4. close() 后通知消费者退出
 *
 * 应用场景：
 *
 * read thread
 *      ↓
 * frame_queue
 *      ↓
 * inference thread
 *      ↓
 * result_queue
 *      ↓
 * display
 *
 * 这里的 mutex（门锁）只保护队列本身，
 * 不保护真正耗时的推理计算。
 * ============================================================
 */

template<typename T>
class ThreadSafeQueue
{
public:

    // ==================== 构造模块 ====================

    explicit ThreadSafeQueue(std::size_t max_size)
        : max_size_(max_size)
    {
    }

    // ==================== 生产数据模块 ====================

    void push(T item)
    {
        {
            // mutex（门锁）只保护队列操作
            std::lock_guard<std::mutex> lock(mutex_);

            /*
             * 实时视频场景下：
             *
             * 如果消费者处理不过来，
             * 不应该无限等待生产者。
             *
             * 队列满时丢弃最旧帧，
             * 保证处理的数据尽可能接近当前画面。
             */
            if (queue_.size() >= max_size_)
            {
                queue_.pop();
            }

            queue_.push(std::move(item));
        }

        /*
         * condition_variable（叫醒铃）
         *
         * 告诉正在等待数据的消费者：
         * “队列里有新数据了。”
         */
        not_empty_.notify_one();
    }

    // ==================== 消费数据模块 ====================

    bool pop(T& item)
    {
        std::unique_lock<std::mutex> lock(mutex_);

        /*
         * condition_variable（叫醒铃）：
         *
         * 队列为空时不轮询，
         * 而是主动睡眠等待通知。
         *
         * 两种情况会被唤醒：
         *
         * 1. 队列出现数据
         * 2. 队列被 close()
         */
        not_empty_.wait(lock, [this]
        {
            return !queue_.empty() || closed_;
        });

        /*
         * 队列为空 + 已关闭：
         *
         * 说明生产者已经彻底结束，
         * 不会再产生任何数据。
         */
        if (queue_.empty() && closed_)
        {
            return false;
        }

        // 取出最早进入队列的数据
        item = std::move(queue_.front());

        queue_.pop();

        return true;
    }

    // ==================== 队列关闭模块 ====================

    void close()
    {
        {
            std::lock_guard<std::mutex> lock(mutex_);

            closed_ = true;
        }

        /*
         * 唤醒所有等待中的消费者。
         *
         * 否则消费者可能永远阻塞在 pop()。
         */
        not_empty_.notify_all();
    }

private:

    // ==================== 队列数据 ====================

    std::queue<T> queue_;

    // ==================== 并发控制 ====================

    std::mutex mutex_;

    std::condition_variable not_empty_;

    // ==================== 队列配置 ====================

    std::size_t max_size_;

    // ==================== 生命周期状态 ====================

    bool closed_ = false;
};
