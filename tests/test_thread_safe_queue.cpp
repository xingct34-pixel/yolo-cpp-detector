#include "thread_safe_queue.h"

#include <atomic>
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <thread>

int main()
{
    // 1. 零容量必须被拒绝。
    bool rejected_zero_capacity = false;
    try
    {
        ThreadSafeQueue<int> invalid_queue(0);
    }
    catch (const std::invalid_argument&)
    {
        rejected_zero_capacity = true;
    }
    assert(rejected_zero_capacity);
    std::cout << "[通过] 零容量校验\n";

    // 2. 队列满时丢弃最旧的数据。
    ThreadSafeQueue<int> bounded_queue(2);
    assert(bounded_queue.push(1));
    assert(bounded_queue.push(2));
    assert(bounded_queue.push(3));

    int item = 0;
    assert(bounded_queue.pop(item) && item == 2);
    assert(bounded_queue.pop(item) && item == 3);
    bounded_queue.close();
    assert(!bounded_queue.pop(item));
    assert(!bounded_queue.push(4));
    std::cout << "[通过] 满队列策略与关闭后的读写行为\n";

    // 3. close() 必须唤醒正在等待数据的消费者。
    ThreadSafeQueue<int> waiting_queue(1);
    std::atomic<bool> consumer_finished{false};

    std::thread consumer([&]
    {
        int value = 0;
        const bool received = waiting_queue.pop(value);
        assert(!received);
        consumer_finished.store(true);
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    waiting_queue.close();
    consumer.join();

    assert(consumer_finished.load());
    std::cout << "[通过] close() 唤醒阻塞消费者\n";

    // 4. 关闭后仍应允许消费者排空已有数据。
    ThreadSafeQueue<int> drain_queue(2);
    assert(drain_queue.push(42));
    drain_queue.close();
    assert(drain_queue.pop(item) && item == 42);
    assert(!drain_queue.pop(item));
    std::cout << "[通过] 关闭队列可排空已有数据\n";


    ThreadSafeQueue<int> blocking(1,QueuePolicy::Block);
    assert(blocking.push(1));std::atomic<bool> started{false},accepted{true};
    std::thread producer([&]{started=true;accepted=blocking.push(2);});
    while(!started)std::this_thread::yield();
    blocking.close();producer.join();assert(!accepted);assert(blocking.pop(item)&&item==1);
    assert(blocking.dropped()==0&&blocking.peak()==1);
    ThreadSafeQueue<int> transfer(2,QueuePolicy::Block);
    std::thread writer([&]{for(int k=0;k<1000;++k)assert(transfer.push(k));transfer.close();});
    int expected=0;while(transfer.pop(item))assert(item==expected++);writer.join();assert(expected==1000);
    return 0;
}
