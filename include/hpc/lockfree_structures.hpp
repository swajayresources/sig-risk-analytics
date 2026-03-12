/**
 * Lock-Free Data Structures and Memory Optimization
 *
 * High-performance concurrent data structures for ultra-low latency
 * risk analytics with optimized memory access patterns
 */

#pragma once

#include <atomic>
#include <memory>
#include <array>
#include <type_traits>
#include <cstdint>
#include <cstring>
#include <immintrin.h>

#ifdef __linux__
#include <sys/mman.h>
#include <numa.h>
#include <numaif.h>
#endif

namespace risk_analytics {
namespace hpc {
namespace lockfree {

/**
 * Memory alignment utilities for cache-line optimization
 */
constexpr size_t CACHE_LINE_SIZE = 64;
constexpr size_t PAGE_SIZE = 4096;
constexpr size_t HUGE_PAGE_SIZE = 2 * 1024 * 1024; // 2MB

#define CACHE_ALIGNED alignas(CACHE_LINE_SIZE)
#define PAGE_ALIGNED alignas(PAGE_SIZE)

/**
 * Memory ordering utilities for different consistency models
 */
namespace memory_order {
    constexpr std::memory_order relaxed = std::memory_order_relaxed;
    constexpr std::memory_order consume = std::memory_order_consume;
    constexpr std::memory_order acquire = std::memory_order_acquire;
    constexpr std::memory_order release = std::memory_order_release;
    constexpr std::memory_order acq_rel = std::memory_order_acq_rel;
    constexpr std::memory_order seq_cst = std::memory_order_seq_cst;
}

/**
 * NUMA-aware memory allocator
 */
class NumaAllocator {
private:
    int numa_node_;
    size_t allocated_bytes_;

public:
    explicit NumaAllocator(int numa_node = -1) : numa_node_(numa_node), allocated_bytes_(0) {
        #ifdef __linux__
        if (numa_node_ == -1) {
            numa_node_ = numa_node_of_cpu(sched_getcpu());
        }
        #endif
    }

    void* allocate(size_t size, size_t alignment = CACHE_LINE_SIZE) {
        void* ptr = nullptr;

        #ifdef __linux__
        if (numa_available() >= 0) {
            ptr = numa_alloc_onnode(size, numa_node_);
            if (ptr) {
                allocated_bytes_ += size;
                return ptr;
            }
        }
        #endif

        // Fallback to aligned allocation
        if (posix_memalign(&ptr, alignment, size) == 0) {
            allocated_bytes_ += size;
            return ptr;
        }

        return nullptr;
    }

    void deallocate(void* ptr, size_t size) {
        if (ptr) {
            #ifdef __linux__
            if (numa_available() >= 0) {
                numa_free(ptr, size);
            } else
            #endif
            {
                free(ptr);
            }
            allocated_bytes_ -= size;
        }
    }

    size_t get_allocated_bytes() const { return allocated_bytes_; }
    int get_numa_node() const { return numa_node_; }
};

/**
 * Cache-line padded atomic for avoiding false sharing
 */
template<typename T>
struct CACHE_ALIGNED PaddedAtomic {
    std::atomic<T> value;
    char padding[CACHE_LINE_SIZE - sizeof(std::atomic<T>)];

    PaddedAtomic() : value{} {}
    explicit PaddedAtomic(T val) : value(val) {}

    T load(std::memory_order order = memory_order::acquire) const {
        return value.load(order);
    }

    void store(T val, std::memory_order order = memory_order::release) {
        value.store(val, order);
    }

    T exchange(T val, std::memory_order order = memory_order::acq_rel) {
        return value.exchange(val, order);
    }

    bool compare_exchange_weak(T& expected, T desired,
                              std::memory_order success = memory_order::acq_rel,
                              std::memory_order failure = memory_order::acquire) {
        return value.compare_exchange_weak(expected, desired, success, failure);
    }

    bool compare_exchange_strong(T& expected, T desired,
                               std::memory_order success = memory_order::acq_rel,
                               std::memory_order failure = memory_order::acquire) {
        return value.compare_exchange_strong(expected, desired, success, failure);
    }

    T fetch_add(T arg, std::memory_order order = memory_order::acq_rel) {
        return value.fetch_add(arg, order);
    }

    T fetch_sub(T arg, std::memory_order order = memory_order::acq_rel) {
        return value.fetch_sub(arg, order);
    }

    operator T() const { return load(); }
    T operator=(T val) { store(val); return val; }
    T operator++() { return fetch_add(1) + 1; }
    T operator++(int) { return fetch_add(1); }
    T operator--() { return fetch_sub(1) - 1; }
    T operator--(int) { return fetch_sub(1); }
};

/**
 * Lock-free single-producer single-consumer (SPSC) queue
 * Optimized for ultra-low latency with cache-friendly memory layout
 */
template<typename T, size_t Capacity>
class SPSCQueue {
    static_assert((Capacity & (Capacity - 1)) == 0, "Capacity must be power of 2");

private:
    static constexpr size_t MASK = Capacity - 1;

    struct alignas(CACHE_LINE_SIZE) ProducerData {
        size_t head = 0;
        size_t cached_tail = 0;
    };

    struct alignas(CACHE_LINE_SIZE) ConsumerData {
        size_t tail = 0;
        size_t cached_head = 0;
    };

    // Separate cache lines for producer and consumer to avoid false sharing
    ProducerData producer_data_;
    ConsumerData consumer_data_;

    // Ring buffer storage with cache-line aligned elements
    alignas(CACHE_LINE_SIZE) std::array<T, Capacity> buffer_;

public:
    SPSCQueue() = default;

    // Non-copyable, non-movable for safety
    SPSCQueue(const SPSCQueue&) = delete;
    SPSCQueue& operator=(const SPSCQueue&) = delete;
    SPSCQueue(SPSCQueue&&) = delete;
    SPSCQueue& operator=(SPSCQueue&&) = delete;

    /**
     * Try to enqueue an element (producer side)
     */
    bool try_push(const T& item) {
        const size_t head = producer_data_.head;
        const size_t next_head = (head + 1) & MASK;

        // Check if queue is full using cached tail
        if (next_head == producer_data_.cached_tail) {
            // Refresh cached tail
            producer_data_.cached_tail = consumer_data_.tail;
            if (next_head == producer_data_.cached_tail) {
                return false; // Queue is full
            }
        }

        // Store element
        buffer_[head] = item;

        // Memory barrier to ensure item is written before head is updated
        std::atomic_thread_fence(memory_order::release);

        // Update head
        producer_data_.head = next_head;

        return true;
    }

    /**
     * Try to enqueue with move semantics
     */
    bool try_push(T&& item) {
        const size_t head = producer_data_.head;
        const size_t next_head = (head + 1) & MASK;

        if (next_head == producer_data_.cached_tail) {
            producer_data_.cached_tail = consumer_data_.tail;
            if (next_head == producer_data_.cached_tail) {
                return false;
            }
        }

        buffer_[head] = std::move(item);
        std::atomic_thread_fence(memory_order::release);
        producer_data_.head = next_head;

        return true;
    }

    /**
     * Try to dequeue an element (consumer side)
     */
    bool try_pop(T& item) {
        const size_t tail = consumer_data_.tail;

        // Check if queue is empty using cached head
        if (tail == consumer_data_.cached_head) {
            // Refresh cached head
            consumer_data_.cached_head = producer_data_.head;
            if (tail == consumer_data_.cached_head) {
                return false; // Queue is empty
            }
        }

        // Memory barrier to ensure we read the item after checking head
        std::atomic_thread_fence(memory_order::acquire);

        // Load element
        item = buffer_[tail];

        // Update tail
        consumer_data_.tail = (tail + 1) & MASK;

        return true;
    }

    /**
     * Get current queue size (approximate)
     */
    size_t size() const {
        const size_t head = producer_data_.head;
        const size_t tail = consumer_data_.tail;
        return (head - tail) & MASK;
    }

    /**
     * Check if queue is empty
     */
    bool empty() const {
        return producer_data_.head == consumer_data_.tail;
    }

    /**
     * Check if queue is full
     */
    bool full() const {
        const size_t next_head = (producer_data_.head + 1) & MASK;
        return next_head == consumer_data_.tail;
    }

    static constexpr size_t capacity() { return Capacity; }
};

/**
 * Lock-free multiple-producer single-consumer (MPSC) queue
 * Uses atomic operations with exponential backoff
 */
template<typename T, size_t Capacity>
class MPSCQueue {
    static_assert((Capacity & (Capacity - 1)) == 0, "Capacity must be power of 2");

private:
    static constexpr size_t MASK = Capacity - 1;

    struct Node {
        std::atomic<T*> data{nullptr};
        PaddedAtomic<size_t> next{0};
    };

    CACHE_ALIGNED PaddedAtomic<size_t> head_{0};
    CACHE_ALIGNED PaddedAtomic<size_t> tail_{0};
    CACHE_ALIGNED std::array<Node, Capacity> nodes_;

    // Exponential backoff for contention management
    void backoff(int& attempt) const {
        if (attempt < 10) {
            for (int i = 0; i < (1 << attempt); ++i) {
                _mm_pause(); // CPU pause instruction
            }
            ++attempt;
        } else {
            std::this_thread::yield();
        }
    }

public:
    MPSCQueue() {
        // Initialize nodes
        for (size_t i = 0; i < Capacity; ++i) {
            nodes_[i].next.store(i + 1);
        }
        nodes_[Capacity - 1].next.store(0); // Circular
    }

    ~MPSCQueue() {
        // Clean up any remaining data
        T* item;
        while (try_pop(item)) {
            delete item;
        }
    }

    /**
     * Try to enqueue an element (multiple producers)
     */
    bool try_push(T* item) {
        int backoff_count = 0;

        while (true) {
            const size_t head = head_.load(memory_order::acquire);
            const size_t next_head = (head + 1) & MASK;

            // Check if queue is full
            if (next_head == tail_.load(memory_order::acquire)) {
                return false;
            }

            // Try to claim the slot
            if (head_.compare_exchange_weak(const_cast<size_t&>(head), next_head,
                                          memory_order::acq_rel, memory_order::acquire)) {
                // Successfully claimed slot, store the item
                nodes_[head].data.store(item, memory_order::release);
                return true;
            }

            // Failed to claim, backoff and retry
            backoff(backoff_count);
        }
    }

    /**
     * Try to dequeue an element (single consumer)
     */
    bool try_pop(T*& item) {
        const size_t tail = tail_.load(memory_order::relaxed);

        // Check if queue is empty
        if (tail == head_.load(memory_order::acquire)) {
            return false;
        }

        // Load the item
        item = nodes_[tail].data.exchange(nullptr, memory_order::acq_rel);

        if (item == nullptr) {
            return false; // Slot not ready yet
        }

        // Update tail
        tail_.store((tail + 1) & MASK, memory_order::release);

        return true;
    }

    size_t size() const {
        const size_t head = head_.load(memory_order::acquire);
        const size_t tail = tail_.load(memory_order::acquire);
        return (head - tail) & MASK;
    }

    bool empty() const {
        return head_.load(memory_order::acquire) == tail_.load(memory_order::acquire);
    }
};

/**
 * Lock-free hash map optimized for financial data lookups
 * Uses Robin Hood hashing with atomic operations
 */
template<typename Key, typename Value, size_t Capacity = 65536>
class LockFreeHashMap {
    static_assert((Capacity & (Capacity - 1)) == 0, "Capacity must be power of 2");

private:
    static constexpr size_t MASK = Capacity - 1;
    static constexpr uint32_t EMPTY_HASH = 0;
    static constexpr uint32_t DELETED_HASH = 1;

    struct Entry {
        PaddedAtomic<uint32_t> hash{EMPTY_HASH};
        Key key;
        Value value;
        PaddedAtomic<uint32_t> probe_distance{0};
    };

    CACHE_ALIGNED std::array<Entry, Capacity> entries_;

    // Hash function optimized for financial symbols
    uint32_t hash_function(const Key& key) const {
        if constexpr (std::is_same_v<Key, std::string>) {
            // FNV-1a hash for strings
            uint32_t hash = 2166136261u;
            for (char c : key) {
                hash ^= static_cast<uint32_t>(c);
                hash *= 16777619u;
            }
            return hash == EMPTY_HASH || hash == DELETED_HASH ? hash + 2 : hash;
        } else {
            // Simple hash for other types
            std::hash<Key> hasher;
            uint32_t hash = static_cast<uint32_t>(hasher(key));
            return hash == EMPTY_HASH || hash == DELETED_HASH ? hash + 2 : hash;
        }
    }

    size_t get_index(uint32_t hash) const {
        return static_cast<size_t>(hash) & MASK;
    }

public:
    LockFreeHashMap() = default;

    /**
     * Insert or update a key-value pair
     */
    bool insert(const Key& key, const Value& value) {
        const uint32_t hash = hash_function(key);
        size_t index = get_index(hash);
        uint32_t probe_distance = 0;

        while (probe_distance < Capacity) {
            Entry& entry = entries_[index];
            uint32_t entry_hash = entry.hash.load(memory_order::acquire);

            if (entry_hash == EMPTY_HASH) {
                // Try to claim empty slot
                uint32_t expected = EMPTY_HASH;
                if (entry.hash.compare_exchange_strong(expected, hash,
                                                     memory_order::acq_rel,
                                                     memory_order::acquire)) {
                    entry.key = key;
                    entry.value = value;
                    entry.probe_distance.store(probe_distance, memory_order::release);
                    return true;
                }
                // Someone else claimed it, continue probing
            } else if (entry_hash == hash && entry.key == key) {
                // Update existing entry
                entry.value = value;
                return true;
            } else if (entry_hash == DELETED_HASH) {
                // Try to claim deleted slot
                uint32_t expected = DELETED_HASH;
                if (entry.hash.compare_exchange_strong(expected, hash,
                                                     memory_order::acq_rel,
                                                     memory_order::acquire)) {
                    entry.key = key;
                    entry.value = value;
                    entry.probe_distance.store(probe_distance, memory_order::release);
                    return true;
                }
            }

            // Robin Hood hashing: if our probe distance is greater than the entry's,
            // we should displace it
            uint32_t entry_probe_distance = entry.probe_distance.load(memory_order::acquire);
            if (probe_distance > entry_probe_distance && entry_hash != EMPTY_HASH && entry_hash != DELETED_HASH) {
                // Try to displace the entry (this is a simplified version)
                // Full Robin Hood implementation would be more complex
            }

            index = (index + 1) & MASK;
            ++probe_distance;
        }

        return false; // Table full
    }

    /**
     * Lookup a value by key
     */
    bool find(const Key& key, Value& value) const {
        const uint32_t hash = hash_function(key);
        size_t index = get_index(hash);
        uint32_t probe_distance = 0;

        while (probe_distance < Capacity) {
            const Entry& entry = entries_[index];
            uint32_t entry_hash = entry.hash.load(memory_order::acquire);

            if (entry_hash == EMPTY_HASH) {
                return false; // Not found
            }

            if (entry_hash == hash && entry.key == key) {
                value = entry.value;
                return true;
            }

            // If we've probed further than the entry's probe distance,
            // the key doesn't exist
            uint32_t entry_probe_distance = entry.probe_distance.load(memory_order::acquire);
            if (probe_distance > entry_probe_distance) {
                return false;
            }

            index = (index + 1) & MASK;
            ++probe_distance;
        }

        return false;
    }

    /**
     * Remove a key-value pair
     */
    bool erase(const Key& key) {
        const uint32_t hash = hash_function(key);
        size_t index = get_index(hash);
        uint32_t probe_distance = 0;

        while (probe_distance < Capacity) {
            Entry& entry = entries_[index];
            uint32_t entry_hash = entry.hash.load(memory_order::acquire);

            if (entry_hash == EMPTY_HASH) {
                return false; // Not found
            }

            if (entry_hash == hash && entry.key == key) {
                // Mark as deleted
                entry.hash.store(DELETED_HASH, memory_order::release);
                return true;
            }

            uint32_t entry_probe_distance = entry.probe_distance.load(memory_order::acquire);
            if (probe_distance > entry_probe_distance) {
                return false;
            }

            index = (index + 1) & MASK;
            ++probe_distance;
        }

        return false;
    }

    /**
     * Get approximate size (not exact due to concurrent modifications)
     */
    size_t size() const {
        size_t count = 0;
        for (const auto& entry : entries_) {
            uint32_t hash = entry.hash.load(memory_order::acquire);
            if (hash != EMPTY_HASH && hash != DELETED_HASH) {
                ++count;
            }
        }
        return count;
    }

    /**
     * Check if map is empty
     */
    bool empty() const {
        return size() == 0;
    }

    static constexpr size_t capacity() { return Capacity; }
};

/**
 * Memory pool with lock-free allocation
 * Optimized for frequent small allocations in trading systems
 */
template<size_t BlockSize, size_t NumBlocks>
class LockFreeMemoryPool {
private:
    struct Block {
        alignas(CACHE_LINE_SIZE) char data[BlockSize];
    };

    struct FreeNode {
        std::atomic<FreeNode*> next;
    };

    // Memory storage
    PAGE_ALIGNED std::array<Block, NumBlocks> blocks_;

    // Free list head
    CACHE_ALIGNED PaddedAtomic<FreeNode*> free_head_{nullptr};

    // Allocation statistics
    CACHE_ALIGNED PaddedAtomic<size_t> allocated_count_{0};
    CACHE_ALIGNED PaddedAtomic<size_t> allocation_attempts_{0};

public:
    LockFreeMemoryPool() {
        // Initialize free list
        for (size_t i = 0; i < NumBlocks - 1; ++i) {
            FreeNode* node = reinterpret_cast<FreeNode*>(&blocks_[i]);
            node->next.store(reinterpret_cast<FreeNode*>(&blocks_[i + 1]),
                           memory_order::relaxed);
        }

        // Last block points to nullptr
        FreeNode* last_node = reinterpret_cast<FreeNode*>(&blocks_[NumBlocks - 1]);
        last_node->next.store(nullptr, memory_order::relaxed);

        // Set head to first block
        free_head_.store(reinterpret_cast<FreeNode*>(&blocks_[0]), memory_order::release);
    }

    /**
     * Allocate a block from the pool
     */
    void* allocate() {
        allocation_attempts_.fetch_add(1, memory_order::relaxed);

        FreeNode* head = free_head_.load(memory_order::acquire);

        while (head != nullptr) {
            FreeNode* next = head->next.load(memory_order::acquire);

            // Try to update head to next
            if (free_head_.compare_exchange_weak(head, next,
                                               memory_order::acq_rel,
                                               memory_order::acquire)) {
                allocated_count_.fetch_add(1, memory_order::relaxed);
                return reinterpret_cast<void*>(head);
            }
            // head was updated by the failed CAS, retry
        }

        return nullptr; // Pool exhausted
    }

    /**
     * Deallocate a block back to the pool
     */
    void deallocate(void* ptr) {
        if (ptr == nullptr) return;

        FreeNode* node = reinterpret_cast<FreeNode*>(ptr);
        FreeNode* head = free_head_.load(memory_order::acquire);

        do {
            node->next.store(head, memory_order::relaxed);
        } while (!free_head_.compare_exchange_weak(head, node,
                                                 memory_order::acq_rel,
                                                 memory_order::acquire));

        allocated_count_.fetch_sub(1, memory_order::relaxed);
    }

    /**
     * Get allocation statistics
     */
    struct Statistics {
        size_t allocated_blocks;
        size_t free_blocks;
        size_t total_blocks;
        size_t allocation_attempts;
        double allocation_success_rate;
        size_t block_size;
    };

    Statistics get_statistics() const {
        const size_t allocated = allocated_count_.load(memory_order::acquire);
        const size_t attempts = allocation_attempts_.load(memory_order::acquire);

        return Statistics{
            .allocated_blocks = allocated,
            .free_blocks = NumBlocks - allocated,
            .total_blocks = NumBlocks,
            .allocation_attempts = attempts,
            .allocation_success_rate = attempts > 0 ? double(allocated) / attempts : 0.0,
            .block_size = BlockSize
        };
    }

    static constexpr size_t block_size() { return BlockSize; }
    static constexpr size_t num_blocks() { return NumBlocks; }
    static constexpr size_t total_memory() { return BlockSize * NumBlocks; }
};

/**
 * Specialized memory pool for financial position data
 */
using PositionPool = LockFreeMemoryPool<sizeof(Position), 1000000>; // 1M positions

/**
 * Lock-free stack for high-frequency operations
 */
template<typename T>
class LockFreeStack {
private:
    struct Node {
        T data;
        std::atomic<Node*> next;

        Node(const T& item) : data(item), next(nullptr) {}
        Node(T&& item) : data(std::move(item)), next(nullptr) {}
    };

    CACHE_ALIGNED PaddedAtomic<Node*> head_{nullptr};

public:
    LockFreeStack() = default;

    ~LockFreeStack() {
        while (Node* node = head_.load()) {
            head_.store(node->next.load());
            delete node;
        }
    }

    void push(const T& item) {
        Node* new_node = new Node(item);
        Node* old_head = head_.load(memory_order::acquire);

        do {
            new_node->next.store(old_head, memory_order::relaxed);
        } while (!head_.compare_exchange_weak(old_head, new_node,
                                            memory_order::acq_rel,
                                            memory_order::acquire));
    }

    void push(T&& item) {
        Node* new_node = new Node(std::move(item));
        Node* old_head = head_.load(memory_order::acquire);

        do {
            new_node->next.store(old_head, memory_order::relaxed);
        } while (!head_.compare_exchange_weak(old_head, new_node,
                                            memory_order::acq_rel,
                                            memory_order::acquire));
    }

    bool pop(T& item) {
        Node* old_head = head_.load(memory_order::acquire);

        while (old_head != nullptr) {
            Node* next = old_head->next.load(memory_order::acquire);

            if (head_.compare_exchange_weak(old_head, next,
                                          memory_order::acq_rel,
                                          memory_order::acquire)) {
                item = std::move(old_head->data);
                delete old_head;
                return true;
            }
            // old_head was updated by the failed CAS, retry
        }

        return false; // Stack is empty
    }

    bool empty() const {
        return head_.load(memory_order::acquire) == nullptr;
    }
};

/**
 * Cache-optimized array for hot financial data
 */
template<typename T, size_t Size>
class CacheOptimizedArray {
private:
    // Ensure array fits in L1 cache if possible
    static_assert(sizeof(T) * Size <= 32768, "Array too large for L1 cache");

    CACHE_ALIGNED std::array<T, Size> data_;

public:
    CacheOptimizedArray() = default;

    T& operator[](size_t index) {
        return data_[index];
    }

    const T& operator[](size_t index) const {
        return data_[index];
    }

    T* data() { return data_.data(); }
    const T* data() const { return data_.data(); }

    constexpr size_t size() const { return Size; }

    // Prefetch data for better cache performance
    void prefetch(size_t index) const {
        if (index < Size) {
            _mm_prefetch(reinterpret_cast<const char*>(&data_[index]), _MM_HINT_T0);
        }
    }

    void prefetch_range(size_t start, size_t count) const {
        for (size_t i = start; i < std::min(start + count, Size); ++i) {
            _mm_prefetch(reinterpret_cast<const char*>(&data_[i]), _MM_HINT_T0);
        }
    }
};

} // namespace lockfree
} // namespace hpc
} // namespace risk_analytics