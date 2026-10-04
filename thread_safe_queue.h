#pragma once

#include <queue>
#include <mutex>
#include <condition_variable>

template<typename T>
class ThreadSafeQueue {
public:
    explicit ThreadSafeQueue(size_t max_size)
        : max_size_(max_size) {}

    // 生产者：队列满时丢掉最旧的数据
    void push(T item) {
        {
            std::lock_guard<std::mutex> lock(mutex_);

            if (queue_.size() >= max_size_) {
                queue_.pop();
            }

            queue_.push(std::move(item));
        }

        not_empty_.notify_one();
    }

    // 消费者：没有数据时等待
    // 如果队列已经关闭并且没有数据，返回 false
    bool pop(T& item) {
        std::unique_lock<std::mutex> lock(mutex_);

        not_empty_.wait(lock, [this] {
            return !queue_.empty() || closed_;
        });

        if (queue_.empty() && closed_) {
            return false;
        }

        item = std::move(queue_.front());
        queue_.pop();

        lock.unlock();
        not_full_.notify_one();

        return true;
    }

    // 告诉消费者：不会再有新数据了
    void close() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            closed_ = true;
        }

        not_empty_.notify_all();
    }

    bool empty() {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.empty();
    }

private:
    std::queue<T> queue_;

    std::mutex mutex_;

    std::condition_variable not_full_;
    std::condition_variable not_empty_;

    size_t max_size_;

    bool closed_ = false;
};
