#include <sys/mman.h>
#include <sys/syscall.h>
#include <sched.h>
#include <unistd.h>
#include <cstdio>
#include <cstdint>
#include <thread>
#include <vector>
int main() {
    const uint64_t blob = 2764800, n = 22000;           // 1024 experts, like one layer x2
    const uint64_t bytes = blob * n;
    uint8_t* base = (uint8_t*) mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    const uint64_t rb[3] = {921600, 921600, 921600};
    std::vector<std::pair<uint64_t,uint64_t>> r[2];
    for (uint64_t e = 0; e < n; ++e) {
        uint64_t start = e * blob;
        for (int k = 0; k < 3; ++k) {
            uint64_t mid = start + rb[k] / 2, end = start + rb[k];
            r[0].push_back({start / 4096 * 4096, (mid + 4095) / 4096 * 4096});
            uint64_t lo = (mid + 4095) / 4096 * 4096, hi = end / 4096 * 4096;
            if (hi > lo) r[1].push_back({lo, hi});
            start = end;
        }
    }
    auto toucher = [&](int k) {
        cpu_set_t s; CPU_ZERO(&s);
        for (int c = 0; c < 72; ++c) if ((k == 0) == ((c % 36) < 18)) CPU_SET(c, &s);
        sched_setaffinity(0, sizeof s, &s);
        unsigned long mask[4] = {1ul << k, 0, 0, 0};
        long rc = syscall(SYS_set_mempolicy, 2, mask, 256);
        if (rc) perror("set_mempolicy");
        for (auto& x : r[k]) for (uint64_t at = x.first; at < x.second; at += 4096) ((volatile uint8_t*) base)[at] = 0;
    };
    std::vector<std::thread> th; for (int t = 0; t < 2; ++t) th.emplace_back(toucher, t); for (auto& x : th) x.join();
    for (int k = 0; k < 2; ++k) {
        std::vector<void*> pages; for (auto& x : r[k]) for (uint64_t at = x.first; at < x.second; at += 4096) pages.push_back(base + at);
        std::vector<int> st(pages.size(), -99);
        long rc = syscall(SYS_move_pages, 0, (unsigned long) pages.size(), pages.data(), nullptr, st.data(), 0);
        size_t on[2] = {0, 0}, other = 0;
        for (int v : st) if (v == 0 || v == 1) on[v]++; else other++;
        printf("slice %d: rc %ld, %zu pages: node0 %zu node1 %zu other %zu (first status %d)\n", k, rc, pages.size(), on[0], on[1], other, st[0]);
    }
}
