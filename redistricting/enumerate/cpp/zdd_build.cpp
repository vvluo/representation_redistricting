// Frontier-based construction of the p-partition ZDD (C++ backend).
//
// A line-for-line port of PartitionZDD._build in zdd.py.  States are processed in
// the same order (by node id within a level, 0-arc child before 1-arc child) and
// canonicalised the same way (labels renumbered by first occurrence along the
// frontier), so the node numbering -- and therefore every lo/hi array -- is
// identical to the Python builder's.  tests/test_zdd_cpp.py checks that.
//
// State key (bytes):  canon labels of the frontier (1 byte each)
//                   | forbidden pairs over those labels, packed upper triangle
//                   | dcc (1 byte)
// Child codes: 0 = 0-terminal, 1 = 1-terminal, k >= 2 = node k-2 of the next level.
//
// C interface (ctypes):
//   void*  zdd_build(n, m, eu, ev, p, max_nodes, max_seconds)
//   int    zdd_status(h)      0 done, 1 node budget, 2 time budget,
//                             -2 isolated vertex, -3 frontier wider than 62
//   int    zdd_levels(h)      levels built
//   long long zdd_width(h, l)
//   void   zdd_copy(h, l, int32* lo, int32* hi)
//   long long zdd_nodes(h)
//   void   zdd_free(h)
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

constexpr int MAXL = 62;   // labels fit in one uint64 row of the forbidden matrix

struct Handle {
    std::vector<std::vector<int32_t>> lo, hi;
    long long nodes = 0;
    int status = 0;
};

inline int npairs_bytes(int k) { return (k * (k - 1) / 2 + 7) / 8; }

}  // namespace

extern "C" {

void* zdd_build(int n, int m, const int* eu, const int* ev, int p,
                long long max_nodes, double max_seconds) {
    auto* H = new Handle();
    auto t0 = std::chrono::steady_clock::now();
    std::vector<int> first(n, -1), last(n, -1);
    for (int l = 0; l < m; ++l)
        for (int w : {eu[l], ev[l]}) {
            if (first[w] < 0) first[w] = l;
            last[w] = l;
        }
    for (int w = 0; w < n; ++w)
        if (first[w] < 0) { H->status = -2; return H; }

    std::vector<int> Fprev;
    int entered = 0;
    std::vector<std::string> cur{std::string(1, '\0')};   // empty frontier, no pairs, dcc 0

    for (int l = 0; l < m; ++l) {
        const int u = eu[l], v = ev[l];
        std::vector<int> enter, leave;
        for (int w : {u, v}) if (first[w] == l) enter.push_back(w);
        for (int w : {u, v}) if (last[w] == l) leave.push_back(w);
        entered += (int)enter.size();
        const int unentered = n - entered;
        std::vector<int> Fext = Fprev;
        Fext.insert(Fext.end(), enter.begin(), enter.end());
        const int nf = (int)Fprev.size(), ne = (int)enter.size(), K = nf + ne;
        if (K > MAXL) { H->status = -3; return H; }
        int iu = -1, iv = -1;
        for (int i = 0; i < K; ++i) { if (Fext[i] == u) iu = i; if (Fext[i] == v) iv = i; }
        std::vector<int> keep_idx, leave_idx;
        for (int i = 0; i < K; ++i)
            if (std::find(leave.begin(), leave.end(), Fext[i]) == leave.end()) keep_idx.push_back(i);
        for (int w : leave)
            for (int i = 0; i < K; ++i) if (Fext[i] == w) leave_idx.push_back(i);
        const bool is_last = (l == m - 1);
        const int nk = (int)keep_idx.size();

        const size_t nlev = cur.size();
        std::vector<int32_t> lo(nlev, 0), hi(nlev, 0);
        std::unordered_map<std::string, int32_t> nxt;
        nxt.reserve(nlev * 2 + 16);
        std::vector<std::string> nxt_keys;

        int lab0[MAXL + 2], lab[MAXL + 2], newlab[MAXL + 2], mp[MAXL + 2], old_of[MAXL + 2];
        uint64_t fb0[MAXL + 2], fb[MAXL + 2];
        std::string key;

        for (size_t sid = 0; sid < nlev; ++sid) {
            if ((sid & 0xFFFF) == 0 && sid) {          // budgets inside a wide level too
                if (max_nodes > 0 && H->nodes + (long long)nlev + (long long)nxt_keys.size() > max_nodes) {
                    H->status = 1; return H;
                }
                double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                if (max_seconds > 0 && el > max_seconds) { H->status = 2; return H; }
            }
            const std::string& s = cur[sid];
            // ---- decode
            int k0 = 0;
            for (int i = 0; i < nf; ++i) { lab0[i] = (uint8_t)s[i]; k0 = std::max(k0, lab0[i] + 1); }
            for (int i = 0; i < K; ++i) fb0[i] = 0;
            {
                int bit = 0; const int base = nf;
                for (int i = 0; i < k0; ++i)
                    for (int j = i + 1; j < k0; ++j, ++bit)
                        if (((uint8_t)s[base + bit / 8] >> (bit % 8)) & 1) {
                            fb0[i] |= 1ULL << j; fb0[j] |= 1ULL << i;
                        }
            }
            const int dcc0 = (uint8_t)s[nf + npairs_bytes(k0)];
            for (int i = 0; i < ne; ++i) lab0[nf + i] = k0 + i;
            const int a = lab0[iu], b = lab0[iv];

            for (int take = 0; take < 2; ++take) {
                std::copy(lab0, lab0 + K, lab);
                std::copy(fb0, fb0 + k0 + ne, fb);
                const int nlab = k0 + ne;
                int d = dcc0;
                if (take) {
                    if (a != b) {
                        const int x = std::min(a, b), y = std::max(a, b);
                        if ((fb[x] >> y) & 1) continue;
                        for (int i = 0; i < K; ++i) if (lab[i] == y) lab[i] = x;
                        fb[x] |= fb[y]; fb[y] = 0;
                        for (int i = 0; i < nlab; ++i)
                            if ((fb[i] >> y) & 1) { fb[i] &= ~(1ULL << y); fb[i] |= 1ULL << x; }
                        fb[x] &= ~(1ULL << x);
                    }
                } else {
                    if (a == b) continue;
                    fb[a] |= 1ULL << b; fb[b] |= 1ULL << a;
                }
                uint64_t alive = 0;
                for (int i = 0; i < nk; ++i) { newlab[i] = lab[keep_idx[i]]; alive |= 1ULL << newlab[i]; }
                if (!leave_idx.empty()) {
                    uint64_t closed = 0;
                    for (int i : leave_idx) {
                        const int c = lab[i];
                        if (!((alive >> c) & 1)) closed |= 1ULL << c;
                    }
                    if (closed) {
                        d += __builtin_popcountll(closed);
                        if (d > p) continue;
                        for (int c = 0; c < nlab; ++c) {
                            if ((closed >> c) & 1) fb[c] = 0;
                            else fb[c] &= ~closed;
                        }
                    }
                }
                int32_t code;
                if (is_last) {
                    code = (d == p) ? 1 : 0;
                } else {
                    if (d == p) continue;
                    const int distinct = __builtin_popcountll(alive);
                    if (d + distinct + unentered < p) continue;
                    for (int i = 0; i < nlab; ++i) mp[i] = -1;
                    int k = 0;
                    key.assign(nk, '\0');
                    for (int i = 0; i < nk; ++i) {
                        int o = newlab[i];
                        if (mp[o] < 0) { mp[o] = k; old_of[k] = o; ++k; }
                        key[i] = (char)mp[o];
                    }
                    const int pb = npairs_bytes(k);
                    key.append(pb, '\0');
                    int bit = 0;
                    for (int i = 0; i < k; ++i)
                        for (int j = i + 1; j < k; ++j, ++bit)
                            if ((fb[old_of[i]] >> old_of[j]) & 1)
                                key[nk + bit / 8] = (char)((uint8_t)key[nk + bit / 8] | (1u << (bit % 8)));
                    key.push_back((char)d);
                    auto it = nxt.find(key);
                    int32_t j;
                    if (it == nxt.end()) {
                        j = (int32_t)nxt_keys.size();
                        nxt.emplace(key, j); nxt_keys.push_back(key);
                    } else j = it->second;
                    code = j + 2;
                }
                (take ? hi : lo)[sid] = code;
            }
        }
        H->lo.push_back(std::move(lo)); H->hi.push_back(std::move(hi));
        H->nodes += (long long)nlev;
        if (max_nodes > 0 && H->nodes + (long long)nxt_keys.size() > max_nodes) { H->status = 1; return H; }
        double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (max_seconds > 0 && secs > max_seconds) { H->status = 2; return H; }
        Fprev.clear();
        for (int i : keep_idx) Fprev.push_back(Fext[i]);
        cur.swap(nxt_keys);
    }
    return H;
}

int zdd_status(void* h) { return static_cast<Handle*>(h)->status; }
int zdd_levels(void* h) { return (int)static_cast<Handle*>(h)->lo.size(); }
long long zdd_nodes(void* h) { return static_cast<Handle*>(h)->nodes; }
long long zdd_width(void* h, int l) { return (long long)static_cast<Handle*>(h)->lo[l].size(); }
void zdd_copy(void* h, int l, int32_t* lo, int32_t* hi) {
    auto* H = static_cast<Handle*>(h);
    std::copy(H->lo[l].begin(), H->lo[l].end(), lo);
    std::copy(H->hi[l].begin(), H->hi[l].end(), hi);
}
void zdd_free(void* h) { delete static_cast<Handle*>(h); }

}  // extern "C"
