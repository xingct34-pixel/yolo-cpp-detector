#pragma once

#include <queue>
#include <mutex>
#include <condition_variable>

template<typename T>
class ThreadSafeQueue {
public:
    explicit ThreadSafeQueue(size_t max_size)
        : max_size_(max_size) {}

    void push(T item)
    {
        {
            std::lock_guard<std::mutex> lock(mutex_);

            // 队列满了：丢掉最旧的数据
            if (queue_.size() >= max_size_)
            {
                queue_.pop();
            }

            queue_.push(std::move(item));
        }

        // 通知等待中的消费者
        not_empty_.notify_one();
    }

    bool pop(T& item)
    {
        std::unique_lock<std::mutex> lock(mutex_);

        // 队列为空就等待；队列关闭后也要醒来
        not_empty_.wait(lock, [this] {
            return !queue_.empty() || closed_;
        });

        // 没有数据，并且队列已经关闭
        if (queue_.empty() && closed_)
        {
            return false;
        }

        item = std::move(queue_.front());
        queue_.pop();

        return true;
    }

    void close()
    {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            closed_ = true;
        }

        // 唤醒所有等待中的消费者
        not_empty_.notify_all();
    }

private:
    std::queue<T> queue_;
    std::mutex mutex_;
    std::condition_variable not_empty_;

    size_t max_size_;
    bool closed_ = false;
};
