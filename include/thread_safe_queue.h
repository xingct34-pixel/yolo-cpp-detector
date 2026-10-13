//condition_variable 被封装在 ThreadSafeQueue

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


/*
1. push() 为什么要加锁
size()、pop()、push() 是三个独立操作。生产者和消费者同时改它，内部指针会被破坏。锁保证"判满、丢旧、放新"这一整段不被打断。

2. pop() 为什么要加锁
两个消费者可能取到同一个元素

3. 为什么锁不能一直持有
push 里用了额外的花括号，notify_one() 放在锁外。锁持有越久，别的线程越要排队。如果在锁内通知，被唤醒的消费者还要立刻去抢这把锁，白白多一次阻塞。pop 里 wait 睡眠时会自动释放锁，生产者才进得来。

4. unique_lock 与 lock_guard

lock_guard：构造时加锁，析构时解锁，中途不能手动解锁或重新加锁。push、close 只需要"进来改一下就走"，用它最轻。
unique_lock：可以手动 unlock() 和 lock()。condition_variable::wait 需要在睡眠时解锁、醒来时重新加锁，所以必须传 unique_lock。pop 因此用它。

5. condition_variable 为什么要配合 mutex
条件（队列非空或已关闭）本身是共享数据，必须由锁保护。否则会出现"丢失唤醒"：消费者检查到队列空，准备睡的瞬间，生产者放入数据并通知，消费者随后才睡下，就永远错过了这次通知。mutex 保证"检查条件"和"进入睡眠"之间不会插进别的线程。

6. wait() 到底发生了什么
持有锁的情况下调用 pred()。如果条件已经满足，直接返回，根本不睡。
条件不满足，就执行 wait(lock)。这一步做两件事：释放锁，并让当前线程进入睡眠。这两件事是原子的，不会有空隙。
别的线程拿到锁，修改共享数据，然后调用 notify_one() 或 notify_all()。
本线程被唤醒，但此时还不能返回。它必须先重新抢到锁，抢到后 wait(lock) 才返回。
回到循环顶部，再次用 pred() 检查条件。满足则退出，不满足则重新睡眠。

7. 为什么 wait() 要重新检查条件

虚假唤醒：系统可能无缘无故唤醒线程。
抢占：多个消费者被唤醒，第一个取走了数据，第二个醒来时队列又空了。
close() 也会唤醒，此时队列可能是空的。


9. 为什么 close() 要 notify_all()
关闭时可能有多个消费者线程都在 wait 里睡着。notify_one 只叫醒一个，其余线程会永远卡住，程序退出时 join 就会死锁。notify_all 让所有消费者都醒来，发现 closed_ 为真后依次退出。或者关闭后数据没处理干净，通知线程检查一下

10. 为什么 close() 后还允许消费者取完剩余数据
看 pop 的退出判断：

cpp
if (queue_.empty() && closed_)
{
    return false;
}

只有"队列空且已关闭"才返回 false。关闭只表示"不会再有新数据"，货架上已有的数据仍然有效，不应该丢弃。


