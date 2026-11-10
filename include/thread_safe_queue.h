#pragma once
#include <algorithm>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <utility>
// Block：满时生产者等待，适合保留离线帧；DropOldest：丢最旧待处理项，偏向实时新鲜度。
enum class QueuePolicy { DropOldest,Block };
template<class T> class ThreadSafeQueue {
public:
    explicit ThreadSafeQueue(size_t capacity,QueuePolicy policy=QueuePolicy::DropOldest):capacity_(capacity),policy_(policy){
        if(!capacity)throw std::invalid_argument("queue capacity must be positive");}
    // 锁只保护队列、关闭状态和计数；通知是唤醒信号，数据仍通过队列交接。
    bool push(T item){
        std::unique_lock<std::mutex> lock(mutex_);
        if(policy_==QueuePolicy::Block)not_full_.wait(lock,[&]{return closed_||queue_.size()<capacity_;});
        if(closed_)return false;
        if(queue_.size()==capacity_){queue_.pop();++dropped_;}
        queue_.push(std::move(item));peak_=std::max(peak_,queue_.size());
        lock.unlock();not_empty_.notify_one();return true;
    }
    bool pop(T& item){
        std::unique_lock<std::mutex> lock(mutex_);
        // wait睡眠时释放锁，醒来重新加锁并检查条件，避免空转及虚假唤醒误取数据。
        not_empty_.wait(lock,[&]{return closed_||!queue_.empty();});
        if(queue_.empty())return false;
        // front只访问队头，pop才移除；取出后立即解锁，调用者的推理/显示不占此锁。
        item=std::move(queue_.front());queue_.pop();lock.unlock();not_full_.notify_one();return true;
    }
    // close不清空队列；关闭且非空仍可取出旧数据，关闭且为空时pop才返回false。
    // 两类等待者都要唤醒，避免生产者卡在满队列、消费者卡在空队列，导致join一直等待。
    void close(){ {std::lock_guard<std::mutex> lock(mutex_);closed_=true;}not_full_.notify_all();not_empty_.notify_all();}
    size_t dropped()const{std::lock_guard<std::mutex> lock(mutex_);return dropped_;}
    size_t peak()const{std::lock_guard<std::mutex> lock(mutex_);return peak_;}
private:
    std::queue<T> queue_;
    mutable std::mutex mutex_;
    std::condition_variable not_empty_,not_full_;
    size_t capacity_,dropped_=0,peak_=0;
    QueuePolicy policy_;
    bool closed_=false;
};
