#pragma once
#include <queue>
#include <mutex>
#include <condition_variable>

template<typename T>
class ThreadSafeQueue {
public:
    explicit ThreadSafeQueue(size_t max_size) : max_size_(max_size) {}

    void push(T item) {
        std::unique_lock<std::mutex> lock(mutex_);

        //第一个问题，no_full是条件变量，给生产者线程用，执行逻辑：先运行lambda，拿到返回值(true/false)，返回 false → 生产者线程阻塞休眠，释放互斥锁，
         // 让消费者可以拿到锁，等待被再次唤醒。如果被唤醒，拿到互斥锁，再次执行lamda判断，以防虚假唤醒；返回 true → 不阻塞，生产者继续生产，
        not_full_.wait(lock, [this] { return queue_.size() < max_size_; });
        queue_.push(std::move(item));
        lock.unlock();
        not_empty_.notify_one();
    }

    T pop() {
        std::unique_lock<std::mutex> lock(mutex_);
        not_empty_.wait(lock, [this] { return !queue_.empty(); });
        T item = std::move(queue_.front());           //queue_front队列第一个元素
        queue_.pop();
        lock.unlock();
        not_full_.notify_one();
        return item;
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
};
