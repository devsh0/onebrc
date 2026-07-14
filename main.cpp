#include <iostream>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <stdlib.h>
#include <vector>
#include <algorithm>
#include <cstring>
#include <sched.h>
#include <stdint.h>
#include <pthread.h>
#include <sys/mman.h>
#include <immintrin.h>
#include <time.h>

#define BENCH_START u64 __start_time = micros()
#define BENCH_END u64 __total_time = micros() - __start_time
#define BENCH_ACCUMULATE(val) (val += micros() - __start_time)
#define BENCH_REPORT std::printf("Time = %lu micros (%lu ms)\n", __total_time, (__total_time / 1000))
#define BENCH_END_AND_REPORT BENCH_END; BENCH_REPORT

static constexpr int N_WORKERS = 64;
static constexpr size_t N_BUCKETS = 2017;
static constexpr int NAME_LENGTH = 39;

using u8 = uint8_t;
using u16 = uint16_t;
using u32 = uint32_t;
using u64 = uint64_t;
using i8 = int8_t;
using i16 = int16_t;
using i32 = int32_t;
using i64 = int64_t;

struct CityNameMap;

u64 nanos() {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (ts.tv_sec * 1000000000) + ts.tv_nsec;
}

u64 micros() {
    return nanos() / 1000;
}

u64 ms() {
    return micros() / 1000;
}

u64 min(u64 a, u64 b) {
    return a < b ? a : b;
}

struct CombinationTask {
    CityNameMap* a;
    CityNameMap* b;
};

struct CombinationTaskQueue {
    bool is_locked = false;
    int size = 0;
    CityNameMap* maps[N_WORKERS];

    void lock() {
        while (true) {
            bool expected = false;
            bool desired = true;
            bool success = __atomic_compare_exchange_n(&is_locked, &expected, desired, false, __ATOMIC_ACQUIRE,
                                                       __ATOMIC_RELAXED);
            if (success) {
                return;
            }
            while (__atomic_load_n(&is_locked, __ATOMIC_RELAXED)) {
                _mm_pause();
            }
        }
    }

    void unlock() {
        __atomic_store_n(&is_locked, false, __ATOMIC_RELEASE);
    }

    void publish_map(CityNameMap* map) {
        maps[size++] = map;
    }

    bool has_task() {
        return size >= 2;
    }

    CombinationTask claim_task() {
        CombinationTask t;
        t.a = maps[--size];
        t.b = maps[--size];
        return t;
    }
};

struct TaskData {
    CombinationTaskQueue* combination_task_queue = nullptr;
    u8* beg = nullptr;
    u8* end = nullptr;
    int cpuid = -1;
};

struct Entry {
    char city_name[NAME_LENGTH];
    i8 city_name_length = 0;
    int total_temperature = 0;
    int occurrences = 0;
    short min = 0;
    short max = 0;
    short bucket_index = -1;

    void print() {
        printf("%s=%.1f/%.1f/%.1f", city_name, min / 10.f, (total_temperature / 10.f) / (float) occurrences, max / 10.f);
    }
};


struct CityNameBucket {
    static constexpr int factor = 2;

    Entry* entries = nullptr;
    u8 current_capacity = 0;
    u8 size = 0;

    Entry* add_new_entry(u8* city_beg, u8* city_end, int temperature, int bucket_index) {
        if (__builtin_expect(size == current_capacity, 0)) {
            if (current_capacity == 0) {
                // Most buckets only hold a single entry. So start out thin.
                current_capacity = 1;
            } else {
                current_capacity *= factor;
            }
            entries = (Entry *) realloc(entries, current_capacity * sizeof(Entry));
        }

        if (__builtin_expect(city_end - city_beg + 2 > NAME_LENGTH, 0)) {
            printf("FIXME: can't handle city names > %d characters\n", NAME_LENGTH - 1);
            exit(-1);
        }

        Entry& e = entries[size];
        e.bucket_index = bucket_index;

        // At least for the default dataset, this isn't in the hotpath.
        // Copying one byte per iteration is fine.
        int i;
        for (i = 0; city_beg <= city_end; i++) {
            e.city_name[i] = *city_beg;
            city_beg += 1;
        }
        e.city_name[i] = '\0';

        e.city_name_length = i;
        e.occurrences = 1;
        e.total_temperature = temperature;
        e.min = temperature;
        e.max = temperature;
        size += 1;
        return &e;
    }

    Entry* city_in_bucket_avx2(__m256i& city_vec) {
        for (int i = 0; i < size; i++) {
            Entry* e = entries + i;
            __m256i entry_vec = _mm256_loadu_si256((__m256i const *) e->city_name);
            __m256i cmp_res = _mm256_cmpeq_epi8(city_vec, entry_vec);
            u32 cmp_mask = (u32) _mm256_movemask_epi8(cmp_res);
            cmp_mask = ~cmp_mask;
            int count = __builtin_ctz(cmp_mask);
            if (count >= e->city_name_length) {
                return e;
            }
        }
        return nullptr;
    }

    Entry* city_in_bucket(u8* city_beg, u8* city_end) {
        for (int i = 0; i < size; i++) {
            Entry* e = entries + i;
            if (strncmp((char *) city_beg, e->city_name, (city_end - city_beg) + 1) == 0) {
                return e;
            }
        }
        return nullptr;
    }

    void upsert(u8* city_beg, u8* city_end, int temperature, int bucket_index) {
        Entry* e = city_in_bucket(city_beg, city_end);

        if (e == nullptr) {
            add_new_entry(city_beg, city_end, temperature, bucket_index);
            return;
        }

        e->total_temperature += temperature;
        e->occurrences += 1;
        e->min = e->min > temperature ? temperature : e->min;
        e->max = e->max < temperature ? temperature : e->max;
    }

    void upsert_avx2(__m256i& city_vec, u8* city_beg, u8* city_end, int temperature, int bucket_index) {
        Entry* e = city_in_bucket_avx2(city_vec);

        if (e == nullptr) {
            add_new_entry(city_beg, city_end, temperature, bucket_index);
            return;
        }

        e->total_temperature += temperature;
        e->occurrences += 1;
        e->min = e->min > temperature ? temperature : e->min;
        e->max = e->max < temperature ? temperature : e->max;
    }

    void merge(Entry& input) {
        u8* city_beg = (u8 *) &input.city_name;
        __m256i city_vec = _mm256_loadu_si256((__m256i const *) city_beg);
        Entry* e = city_in_bucket_avx2(city_vec);

        if (e == nullptr) {
            u8* city_end = (u8 *) (input.city_name + input.city_name_length);
            e = add_new_entry(city_beg, city_end, 0, input.bucket_index);
            e->min = input.min;
            e->max = input.max;
            e->total_temperature = input.total_temperature;
            e->occurrences = input.occurrences;
            return;
        }

        e->total_temperature += input.total_temperature;
        e->occurrences += input.occurrences;
        e->min = e->min > input.min ? input.min : e->min;
        e->max = e->max < input.max ? input.max : e->max;
    }

    void print() {
        for (int i = 0; i < size; i++) {
            entries[i].print();
            printf(", ");
        }
    }
};

struct CityNameMap {
    CityNameBucket buckets[N_BUCKETS];

    // Some city names are 3-byte long. But what follows is a semicolon so we're good.
    u32 hash_and_prefetch(u8* city_beg) {
        u32 index = *((u32 *) city_beg);
        index = index % N_BUCKETS;
        u8* entries = (u8 *) buckets[index].entries;
        _mm_prefetch(entries, _MM_HINT_T0);
        return index;
    }

    void insert(u8* city_beg, u8* city_end, int temperature, u32 bucket_index) {
        buckets[bucket_index].upsert(city_beg, city_end, temperature, bucket_index);
    }

    void insert_avx2(__m256i& city_vec, u8* city_beg, u8* city_end, int temperature, u32 bucket_index) {
        buckets[bucket_index].upsert_avx2(city_vec, city_beg, city_end, temperature, bucket_index);
    }

    void merge(Entry& e) {
        CityNameBucket& bucket = buckets[e.bucket_index];
        bucket.merge(e);
    }

    void print() {
        printf("{");
        for (int i = 0; i < N_BUCKETS; i++) {
            buckets[i].print();
        }
        printf("}");
    }
};

CityNameMap* merge_task_helper(CombinationTask& ct) {
    CityNameMap* merge_map = new CityNameMap;
    CityNameMap* map_a = ct.a;

    // Merge all entires from map_a.
    for (int i = 0; i < N_BUCKETS; i++) {
        CityNameBucket& bucket = map_a->buckets[i];
        for (int j = 0; j < bucket.size; j++) {
            Entry& e = bucket.entries[j];
            merge_map->merge(e);
        }
    }
    delete map_a;

    // Merge all entries from map_b.
    CityNameMap* map_b = ct.b;
    for (int i = 0; i < N_BUCKETS; i++) {
        CityNameBucket& bucket = map_b->buckets[i];
        for (int j = 0; j < bucket.size; j++) {
            Entry& e = bucket.entries[j];
            merge_map->merge(e);
        }
    }
    delete map_b;

    return merge_map;
}

void do_merge(CityNameMap* map, CombinationTaskQueue* combination_task_queue) {
    combination_task_queue->lock();
    combination_task_queue->publish_map(map);
    while (combination_task_queue->has_task()) {
        CombinationTask ct = combination_task_queue->claim_task();
        combination_task_queue->unlock();
        CityNameMap* merge_map = merge_task_helper(ct);
        combination_task_queue->lock();
        combination_task_queue->publish_map(merge_map);
    }
    combination_task_queue->unlock();
}

// Branchless temperature parsing. This is in the hotpath.
// Introducing even a single branch costs 80-100ms.
int parse_temperature(u8* num_beg, int& len) {
    int sign = 1 - 2 * (num_beg[0] == '-');
    int pos = (num_beg[0] == '-');
    u32 uvalue;
    memcpy(&uvalue, num_beg + pos, 4);
    int shift = 8 * (num_beg[pos + 1] == '.');
    uvalue <<= shift;
    constexpr u64 C = 1 + (10 << 16) + (100 << 24);
    uvalue &= 0x0f000f0f;
    uvalue = ((uvalue * C) >> 24) & 0x3ff;
    len = pos + 4 - (shift >> 3);
    return (int) uvalue * sign;
}

void parse_helper(u8* beg, u8* end, CityNameMap* map) {
    while (beg <= end) {
        u8* city_beg = beg;
        while (*beg != ';') {
            beg += 1;
        }

        u8* city_end = beg - 1;
        beg += 1; // skip semicolon

        int num_len = 0;
        int temperature = parse_temperature(beg, num_len);
        u32 index = map->hash_and_prefetch(city_beg);
        map->insert(city_beg, city_end, temperature, index);
        beg += num_len + 1; // also skip the new line.
    }
}

// Both beg and end are 0-indexed. Dereferencing either is fine.
// The chunk starts at `beg` and ends at `end`.
void parse_avx2(u8* beg, u8* end, CombinationTaskQueue* combination_task_queue) {
    CityNameMap* map = new CityNameMap;
    __m256i semicolon_vec = _mm256_set1_epi8(';');
    __m256i newline_vec = _mm256_set1_epi8('\n');

    while (beg + 64 <= end) {
        // Find the first newline to locate line 2 start.
        __m256i window = _mm256_loadu_si256((__m256i const *) beg);
        __m256i nl_cmp = _mm256_cmpeq_epi8(window, newline_vec);
        u32 nl_mask = (u32) _mm256_movemask_epi8(nl_cmp);
        if (nl_mask == 0) {
            // No newline in first 32 bytes. Process this line alone.
            __m256i semi_cmp = _mm256_cmpeq_epi8(window, semicolon_vec);
            u32 semi_mask = (u32) _mm256_movemask_epi8(semi_cmp);
            if (semi_mask == 0) {
                printf("FIXME: didn't find semicolon!\n");
                exit(1);
            }
            u32 bucket_index = map->hash_and_prefetch(beg);
            u8* semi = beg + __builtin_ctz(semi_mask);
            int nl = 0;
            int temperature = parse_temperature(semi + 1, nl);
            map->insert_avx2(window, beg, semi - 1, temperature, bucket_index);
            beg = semi + 1 + nl + 1;
            continue;
        }

        u8* city_beg1 = beg;
        u8* city_beg2 = beg + __builtin_ctz(nl_mask) + 1;

        // Both line-starts are known at this point. Hash and prefetch corresponding
        // hash-bucket entries early so that we don't miss L1 when we need it.
        u32 bucket_index1 = map->hash_and_prefetch(city_beg1);
        u32 bucket_index2 = map->hash_and_prefetch(city_beg2);

        // Line 1: semicolon search (reuses the already-loaded window).
        __m256i semi_cmp1 = _mm256_cmpeq_epi8(window, semicolon_vec);
        u32 semi_mask1 = (u32) _mm256_movemask_epi8(semi_cmp1);
        if (semi_mask1 == 0) {
            printf("FIXME: didn't find semicolon!\n");
            exit(1);
        }
        u8* semi1 = beg + __builtin_ctz(semi_mask1);
        u8* city_end1 = semi1 - 1;
        int num_len1 = 0;
        int temp1 = parse_temperature(semi1 + 1, num_len1);

        // Line 2: load + semicolon search.
        __m256i entry_vec2 = _mm256_loadu_si256((__m256i const *) city_beg2);
        __m256i semi_cmp2 = _mm256_cmpeq_epi8(entry_vec2, semicolon_vec);
        u32 semi_mask2 = (u32) _mm256_movemask_epi8(semi_cmp2);
        if (semi_mask2 == 0) {
            printf("FIXME: didn't find semicolon!\n");
            exit(1);
        }
        u8* semi2 = city_beg2 + __builtin_ctz(semi_mask2);
        u8* city_end2 = semi2 - 1;
        int num_len2 = 0;
        int temp2 = parse_temperature(semi2 + 1, num_len2);

        // Insert both
        map->insert_avx2(window, city_beg1, city_end1, temp1, bucket_index1);
        map->insert_avx2(entry_vec2, city_beg2, city_end2, temp2, bucket_index2);
        beg = semi2 + 1 + num_len2 + 1;
    }

    while (beg + 32 <= end) {
        u8* city_beg = beg;
        // Hash early to find the bucket, then prefetch the entries to prevent L1d load miss.
        u32 bucket_index = map->hash_and_prefetch(city_beg);
        __m256i entry_vec = _mm256_loadu_si256((__m256i const *) city_beg);
        __m256i semicolon_cmp_res = _mm256_cmpeq_epi8(entry_vec, semicolon_vec);
        u32 semicolon_mask = (u32) _mm256_movemask_epi8(semicolon_cmp_res);

        if (semicolon_mask == 0) {
            printf("FIXME: didn't find semicolon!\n");
            exit(1);
        }

        u8* semicolon_at = beg + __builtin_ctz(semicolon_mask);
        u8* city_end = semicolon_at - 1;
        u8* num_beg = semicolon_at + 1;
        int num_len = 0;
        int temperature = parse_temperature(num_beg, num_len);
        map->insert_avx2(entry_vec, city_beg, city_end, temperature, bucket_index);
        beg = num_beg + num_len + 1;
    }

    parse_helper(beg, end, map);
    do_merge(map, combination_task_queue);
}

void pinme(int cpuid) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpuid, &cpuset);

    if (sched_setaffinity(0, sizeof(cpu_set_t), &cpuset) == -1) {
        perror("sched_setaffinity");
        return;
    }
}

void* read_task(void* vdata) {
    TaskData* task = (TaskData *) vdata;
    pinme(task->cpuid);
    parse_avx2(task->beg, task->end, task->combination_task_queue);
    return NULL;
}

size_t newline_distance_from(u8* end) {
    u8* old = end;
    while (*end != '\n') {
        end += 1;
    }
    return (size_t) (end - old);
}

CityNameMap* start_workers(u8* data, size_t file_size) {
    pthread_t workers[N_WORKERS];
    size_t bytes_left = file_size;
    size_t min_chunk_size = file_size / N_WORKERS;
    u8* chunk_beg = data;
    u8* chunk_end = chunk_beg + min_chunk_size - 1;
    CombinationTaskQueue combination_task_queue;
    TaskData tasks[N_WORKERS];

    for (int i = 0; i < N_WORKERS; i++) {
        size_t newline_offset = newline_distance_from(chunk_end);
        chunk_end += newline_offset; // include the newline character

        TaskData& task = tasks[i];
        task.combination_task_queue = &combination_task_queue;
        task.cpuid = i;
        task.beg = chunk_beg;
        task.end = chunk_end;
        pthread_create(&workers[i], NULL, read_task, &task);

        bytes_left -= (chunk_end - chunk_beg) + 1;
        chunk_beg = chunk_end + 1;
        chunk_end += min(bytes_left, min_chunk_size);
    }

    for (int i = 0; i < N_WORKERS; i++) {
        pthread_join(workers[i], NULL);
    }

    return combination_task_queue.maps[0];;
}

void map_file(const char* path, u8*& data, size_t& file_size) {
    int fd = open(path, O_RDONLY);
    struct stat status;
    fstat(fd, &status);
    file_size = status.st_size;
    data = (u8 *) mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (data == MAP_FAILED) {
        perror("mmap");
        exit(-1);
    }
    close(fd);
}

std::vector<Entry> sort_entries(CityNameMap* map) {
    std::vector<Entry> vec;
    vec.reserve(N_BUCKETS);
    int cur = 0;
    for (int i = 0; i < N_BUCKETS; i++) {
        CityNameBucket& bucket = map->buckets[i];
        for (int j = 0; j < bucket.size; j++) {
            Entry& e = bucket.entries[j];
            vec.push_back(e);
        }
    }

    std::sort(vec.begin(), vec.end(), [](const Entry& a, const Entry& b) {
        return std::strcmp(a.city_name, b.city_name) < 0;
    });

    return vec;
}

void report(std::vector<Entry>& entries) {
    printf("{");
    for (int i = 0; i < entries.size(); i++) {
        Entry& e = entries[i];
        e.print();
        const char* suffix = i == entries.size() - 1 ? "" : ", ";
        printf("%s", suffix);
    }
    printf("}");
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <measurements.txt>\n", argv[0]);
        return 1;
    }

    const char* path = argv[1];
    u8* data = nullptr;
    size_t file_size = 0;
    map_file(path, data, file_size);

    BENCH_START;
    CityNameMap* result = start_workers(data, file_size);
    std::vector<Entry> entries = sort_entries(result);
    report(entries);
    printf("\n\n");
    BENCH_END_AND_REPORT;

    munmap(data, file_size);
    delete result;

    return 0;
}
