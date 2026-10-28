#pragma once
#include <algorithm>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <utility>
enum class QueuePolicy { DropOldest,Block };
template<class T> class ThreadSafeQueue {
public:
    explicit ThreadSafeQueue(size_t capacity,QueuePolicy policy=QueuePolicy::DropOldest):capacity_(capacity),policy_(policy){
        if(!capacity)throw std::invalid_argument("queue capacity must be positive");}
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
        not_empty_.wait(lock,[&]{return closed_||!queue_.empty();});
        if(queue_.empty())return false;
        item=std::move(queue_.front());queue_.pop();lock.unlock();not_full_.notify_one();return true;
    }
    // Closing wakes producers and consumers; consumers may drain existing items.
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
