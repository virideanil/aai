// Exact BF16 words carried by binary fields and five balanced-ternary planes.
// The block payload follows the existing BF5P exponent/sign/fraction rule.
// This container adds per-block raw fallback and independently scheduled tiles.
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <Accelerate/Accelerate.h>

namespace {
constexpr size_t block_words = 8192;
thread_local std::string last_error;
void need(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
void put16(std::vector<uint8_t>& v, uint16_t x) { v.push_back(uint8_t(x)); v.push_back(uint8_t(x >> 8)); }
uint16_t get16(const uint8_t* p) { return uint16_t(p[0]) | uint16_t(p[1]) << 8; }
float widen(uint16_t word) { uint32_t bits = uint32_t(word) << 16; float f; std::memcpy(&f, &bits, 4); return f; }
struct Block { uint8_t kind; uint16_t count; std::vector<uint8_t> data; };

void bits_put(std::vector<uint8_t>& data, size_t index, unsigned width, unsigned value) {
    const size_t bit = index * width, byte = bit / 8;
    const unsigned v = value << (bit % 8);
    data[byte] |= uint8_t(v);
    if (byte + 1 < data.size()) data[byte + 1] |= uint8_t(v >> 8);
}
unsigned bits_get(const uint8_t* data, size_t bytes, size_t index, unsigned width) {
    const size_t bit = index * width, byte = bit / 8;
    unsigned v = data[byte];
    if (byte + 1 < bytes) v |= unsigned(data[byte + 1]) << 8;
    return (v >> (bit % 8)) & ((1U << width) - 1);
}

Block encode_window(const uint16_t* words, size_t n, uint8_t kind) {
    // Eight normal exponents plus escape fit two trits. The binary controls
    // use seven or fifteen normal exponents plus an escape, respectively.
    const bool trit_exponents = kind == 3 || kind == 5;
    const unsigned levels = trit_exponents ? 8 : kind == 2 ? 15 : 7;
    const unsigned width = kind == 2 ? 4 : 3;
    std::array<unsigned, 256> frequency{};
    for (size_t j = 0; j < n; ++j) frequency[(words[j] >> 7) & 255]++;
    unsigned base = 1, best = 0;
    for (unsigned lo = 1; lo + levels <= 255; ++lo) {
        unsigned covered = 0;
        for (unsigned e = lo; e < lo + levels; ++e) covered += frequency[e];
        if (covered > best) { best = covered; base = lo; }
    }
    const size_t groups = (n + 4) / 5;
    std::vector<uint8_t> escapes, exponents(trit_exponents ? groups * 2 : (n * width + 7) / 8),
        signs(kind == 5 ? 0 : (n + 7) / 8), fractions(kind == 5 ? n : (n * 7 + 7) / 8);
    const unsigned powers[5] = {1, 3, 9, 27, 81};
    for (size_t j = 0; j < n; ++j) {
        const uint16_t word = words[j];
        const unsigned e = (word >> 7) & 255;
        unsigned code = levels;
        if (e >= base && e < base + levels) code = e - base;
        else { put16(escapes, uint16_t(j)); put16(escapes, word); }
        if (trit_exponents) {
            exponents[j / 5] += uint8_t((code % 3) * powers[j % 5]);
            exponents[groups + j / 5] += uint8_t((code / 3) * powers[j % 5]);
        } else bits_put(exponents, j, width, code);
        if (kind == 5) fractions[j] = uint8_t(((word >> 8) & 128) | (word & 127));
        else {
            signs[j / 8] |= uint8_t((word >> 15) << (j % 8));
            bits_put(fractions, j, 7, word & 127);
        }
    }
    Block result{kind, uint16_t(n), {uint8_t(base)}};
    put16(result.data, uint16_t(escapes.size() / 4));
    for (const auto* part : {&escapes, &exponents, &signs, &fractions}) result.data.insert(result.data.end(), part->begin(), part->end());
    return result;
}

Block encode(const uint16_t* words, size_t n, int mode) {
    need(n > 0 && n <= block_words && mode >= 0 && mode <= 6, "Block extent or packing mode");
    if (mode >= 3) return encode_window(words, n, uint8_t(mode - 1));
    Block raw{0, uint16_t(n), {}};
    raw.data.reserve(2 * n);
    for (size_t j = 0; j < n; ++j) put16(raw.data, words[j]);
    if (mode == 0) return raw;
    int max_e = 1;
    for (size_t j = 0; j < n; ++j) {
        int e = (words[j] >> 7) & 255;
        if (e > 0 && e < 255) max_e = std::max(max_e, e);
    }
    const size_t groups = (n + 4) / 5;
    std::vector<uint8_t> escapes, exponents((n + 1) / 2), signs((n + 7) / 8), planes(groups * 5, 121);
    // 121 is the five-zero-trit byte: each stored base-three digit is trit+1.
    const unsigned powers[5] = {1, 3, 9, 27, 81};
    for (size_t j = 0; j < n; ++j) {
        const uint16_t w = words[j];
        const int e = (w >> 7) & 255;
        int code = e - (max_e - 14);
        if (e == 0 || e == 255 || code < 0 || code > 14) {
            code = 15; put16(escapes, uint16_t(j)); put16(escapes, w);
        } else {
            int q = (w & 127) - 64;
            for (int k = 0; k < 5; ++k) {
                int digit = ((q % 3) + 3) % 3;
                if (digit == 2) digit = -1;
                const size_t index = size_t(4 - k) * groups + j / 5;
                planes[index] = uint8_t(int(planes[index]) + digit * int(powers[j % 5]));
                q = (q - digit) / 3;
            }
            need(q == 0, "Fraction representation overflow");
        }
        exponents[j / 2] |= uint8_t(code << (4 * (j % 2)));
        signs[j / 8] |= uint8_t((w >> 15) << (j % 8));
    }
    Block packed{1, uint16_t(n), {uint8_t(max_e)}};
    put16(packed.data, uint16_t(escapes.size() / 4));
    for (const auto* part : {&escapes, &exponents, &signs, &planes}) packed.data.insert(packed.data.end(), part->begin(), part->end());
    if (mode == 1) return packed;
    Block best = packed.data.size() < raw.data.size() ? std::move(packed) : std::move(raw);
    for (uint8_t kind : {2, 3, 4, 5}) {
        Block candidate = encode_window(words, n, kind);
        if (candidate.data.size() < best.data.size() || (kind == 5 && candidate.data.size() == best.data.size()))
            best = std::move(candidate);
    }
    return best;
}

constexpr auto trit_table() {
    std::array<std::array<int8_t, 5>, 243> table{};
    for (unsigned code = 0; code < 243; ++code) {
        unsigned q = code;
        for (unsigned j = 0; j < 5; ++j) { table[code][j] = int8_t(int(q % 3) - 1); q /= 3; }
    }
    return table;
}
constexpr auto digits = trit_table();

// Two base-three bytes describe five exponent codes. Groupwise lookup keeps
// the stored alphabet while avoiding division and separate sign/fraction
// extraction for every weight. The table has 59,049 five-byte entries.
const auto& exponent_groups() {
    static const auto table = [] {
        std::array<std::array<uint8_t, 5>, 243 * 243> result{};
        for (unsigned high = 0; high < 243; ++high) for (unsigned low = 0; low < 243; ++low)
            for (unsigned j = 0; j < 5; ++j)
                result[high * 243 + low][j] = uint8_t(digits[low][j] + 1 + 3 * (digits[high][j] + 1));
        return result;
    }();
    return table;
}

void decode(const Block& block, uint16_t* words) {
    const size_t n = block.count;
    need(n > 0 && n <= block_words, "Decoded block extent");
    const auto& d = block.data;
    if (block.kind == 0) {
        need(d.size() == 2 * n, "Raw block byte extent");
        for (size_t j = 0; j < n; ++j) words[j] = get16(d.data() + j * 2);
        return;
    }
    need(block.kind >= 1 && block.kind <= 5 && d.size() >= 3, "Block kind or header");
    if (block.kind == 5) {
        const unsigned base = d[0];
        const size_t escaped = get16(d.data() + 1), groups = (n + 4) / 5;
        need(base >= 1 && base + 8 <= 255 && escaped <= n, "Grouped exponent header");
        const size_t exponent = 3 + 4 * escaped, fields = exponent + 2 * groups;
        need(d.size() == fields + n, "Grouped exponent byte extent");
        std::array<uint8_t, block_words> seen{};
        for (size_t j = 0; j < escaped; ++j) {
            const size_t index = get16(d.data() + 3 + j * 4);
            need(index < n && !seen[index], "Grouped escape index or duplication");
            seen[index] = 1; words[index] = get16(d.data() + 5 + j * 4);
        }
        const auto& table = exponent_groups();
        for (size_t g = 0; g < groups; ++g) {
            const unsigned low = d[exponent + g], high = d[exponent + groups + g];
            need(low < 243 && high < 243, "Invalid grouped exponent trit byte");
            const auto& codes = table[high * 243 + low];
            for (size_t j = 0; j < std::min<size_t>(5, n - g * 5); ++j) {
                const size_t at = g * 5 + j;
                const unsigned code = codes[j];
                need((code == 8) == bool(seen[at]), "Grouped escape marker");
                if (code == 8) continue;
                const unsigned field = d[fields + at];
                words[at] = uint16_t(((field & 128) << 8) | ((base + code) << 7) | (field & 127));
            }
        }
        return;
    }
    if (block.kind >= 2) {
        const unsigned levels = block.kind == 3 ? 8 : block.kind == 2 ? 15 : 7;
        const unsigned width = block.kind == 2 ? 4 : 3;
        const unsigned base = d[0];
        const size_t escaped = get16(d.data() + 1), groups = (n + 4) / 5;
        need(base >= 1 && base + levels <= 255 && escaped <= n, "Window header");
        const size_t exp_bytes = block.kind == 3 ? 2 * groups : (n * width + 7) / 8;
        const size_t exp = 3 + 4 * escaped, sign = exp + exp_bytes, fraction = sign + (n + 7) / 8;
        const size_t fraction_bytes = (n * 7 + 7) / 8;
        need(d.size() == fraction + fraction_bytes, "Window byte extent");
        std::array<uint8_t, block_words> seen{};
        for (size_t j = 0; j < escaped; ++j) {
            const size_t index = get16(d.data() + 3 + j * 4);
            need(index < n && !seen[index], "Window escape index or duplication");
            seen[index] = 1; words[index] = get16(d.data() + 5 + j * 4);
        }
        for (size_t j = 0; j < n; ++j) {
            unsigned code;
            if (block.kind == 3) {
                const unsigned low = d[exp + j / 5], high = d[exp + groups + j / 5];
                need(low < 243 && high < 243, "Invalid exponent trit byte");
                code = unsigned(digits[low][j % 5] + 1) + 3 * unsigned(digits[high][j % 5] + 1);
            } else code = bits_get(d.data() + exp, exp_bytes, j, width);
            need(code <= levels && (code == levels) == bool(seen[j]), "Window escape marker");
            if (code == levels) continue;
            words[j] = uint16_t((((d[sign + j / 8] >> (j % 8)) & 1) << 15) | ((base + code) << 7)
                | bits_get(d.data() + fraction, fraction_bytes, j, 7));
        }
        return;
    }
    const int max_e = d[0];
    need(max_e >= 1 && max_e <= 254, "Exponent base");
    const size_t escaped = get16(d.data() + 1), groups = (n + 4) / 5;
    need(escaped <= n, "Escape count");
    const size_t nib = 3 + 4 * escaped, sign = nib + (n + 1) / 2, plane = sign + (n + 7) / 8;
    need(d.size() == plane + 5 * groups, "Packed block byte extent");
    std::array<uint8_t, block_words> seen{};
    for (size_t j = 0; j < escaped; ++j) {
        const size_t index = get16(d.data() + 3 + 4 * j);
        need(index < n && !seen[index], "Escape index or duplication");
        seen[index] = 1; words[index] = get16(d.data() + 5 + 4 * j);
    }
    std::array<int16_t, block_words + 4> fractions{};
    constexpr int scales[5] = {81, 27, 9, 3, 1};
    for (size_t p = 0; p < 5; ++p) for (size_t g = 0; g < groups; ++g) {
        const unsigned code = d[plane + p * groups + g];
        need(code < 243, "Invalid ternary byte");
        for (size_t j = 0; j < 5; ++j) fractions[g * 5 + j] += int16_t(scales[p] * digits[code][j]);
    }
    for (size_t j = 0; j < n; ++j) {
        const unsigned code = (d[nib + j / 2] >> (4 * (j % 2))) & 15;
        need((code == 15) == bool(seen[j]), "Escape marker mismatch");
        if (code == 15) continue;
        const int e = max_e - 14 + int(code), f = fractions[j] + 64;
        need(e >= 1 && e <= 254 && f >= 0 && f <= 127, "Decoded exponent or fraction");
        words[j] = uint16_t((((d[sign + j / 8] >> (j % 8)) & 1) << 15) | (e << 7) | f);
    }
}

template<class F> void parallel(size_t count, unsigned threads, F work) {
    need(threads >= 1 && threads <= 16, "Worker count");
    if (threads == 1 || count < 2) { for (size_t j = 0; j < count; ++j) work(j); return; }
    std::atomic<size_t> cursor{0};
    std::vector<std::thread> workers;
    std::exception_ptr error;
    std::mutex error_mutex;
    auto worker = [&] {
        try {
            while (true) { const size_t j = cursor.fetch_add(1); if (j >= count) break; work(j); }
        } catch (...) { std::lock_guard<std::mutex> lock(error_mutex); if (!error) error = std::current_exception(); }
    };
    // Join every created worker if thread construction itself fails.
    try { for (unsigned k = 1; k < std::min<size_t>(threads, count); ++k) workers.emplace_back(worker); }
    catch (...) { for (auto& w : workers) w.join(); throw; }
    worker(); for (auto& w : workers) w.join(); if (error) std::rethrow_exception(error);
}

struct Matrix {
    size_t rows, columns, rows_per_block;
    bool finite = true;
    std::vector<Block> blocks;
    std::vector<float> cache;
    Matrix(const uint16_t* words, size_t r, size_t c, int mode): rows(r), columns(c), rows_per_block(block_words / c) {
        need(r > 0 && c > 0 && c <= block_words && r <= 1048576 && r <= SIZE_MAX / c, "Matrix shape");
        for (size_t i = 0; i < r * c; ++i) if (((words[i] >> 7) & 255) == 255) finite = false;
        for (size_t first = 0; first < r; first += rows_per_block)
            blocks.push_back(encode(words + first * c, std::min(rows_per_block, r - first) * c, mode));
    }
    void unpack(uint16_t* words, unsigned threads) const {
        parallel(blocks.size(), threads, [&](size_t b) { decode(blocks[b], words + b * rows_per_block * columns); });
    }
    void prepare_cache(unsigned threads) {
        if (!cache.empty()) return;
        cache.resize(rows * columns);
        parallel(blocks.size(), threads, [&](size_t b) {
            std::array<uint16_t, block_words> w{}; decode(blocks[b], w.data());
            for (size_t j = 0; j < blocks[b].count; ++j) cache[b * rows_per_block * columns + j] = widen(w[j]);
        });
    }
    void project(const float* x, float* y, size_t batch, unsigned threads, int cache_mode) const {
        need(finite && batch > 0 && batch <= 256, "Projection requires finite weights and a bounded batch");
        need(cache_mode >= 0 && cache_mode <= 2 && threads >= 1 && threads <= 16, "Projection route");
        for (size_t i = 0; i < batch * columns; ++i) need(std::isfinite(x[i]), "Projection requires finite inputs");
        if (cache_mode) {
            need(cache.size() == rows * columns, "Decoded cache has not been prepared");
        }
        if (cache_mode == 2) {
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, int(batch), int(rows), int(columns),
                1, x, int(columns), cache.data(), int(columns), 0, y, int(rows));
            return;
        }
        const size_t blocks_per_tile = std::max<size_t>(1, 64 / rows_per_block);
        const size_t tile_count = (blocks.size() + blocks_per_tile - 1) / blocks_per_tile;
        parallel(tile_count, threads, [&](size_t tile) {
            const size_t first_block = tile * blocks_per_tile, end = std::min(first_block + blocks_per_tile, blocks.size());
            const size_t first_row = first_block * rows_per_block, tile_rows = std::min(rows, end * rows_per_block) - first_row;
            std::vector<float> weights;
            const float* data = cache_mode ? cache.data() + first_row * columns : nullptr;
            if (!cache_mode) {
                weights.resize(tile_rows * columns);
                std::array<uint16_t, block_words> w{};
                for (size_t b = first_block; b < end; ++b) {
                    decode(blocks[b], w.data());
                    for (size_t j = 0; j < blocks[b].count; ++j) weights[(b - first_block) * rows_per_block * columns + j] = widen(w[j]);
                }
                data = weights.data();
            }
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, int(batch), int(tile_rows), int(columns),
                1, x, int(columns), data, int(columns), 0, y + first_row, int(rows));
        });
    }
};

using File = std::unique_ptr<FILE, decltype(&std::fclose)>;
void write(FILE* f, const void* p, size_t n) { need(std::fwrite(p, 1, n, f) == n, "File write failed"); }
void read(FILE* f, void* p, size_t n) { need(std::fread(p, 1, n, f) == n, "Truncated container"); }
void integer(FILE* f, uint64_t value, unsigned bytes) { uint8_t p[8]; for (unsigned k = 0; k < bytes; ++k) p[k] = uint8_t(value >> (8 * k)); write(f, p, bytes); }
uint64_t integer(FILE* f, unsigned bytes) { uint8_t p[8]; read(f, p, bytes); uint64_t n = 0; for (unsigned k = 0; k < bytes; ++k) n |= uint64_t(p[k]) << (8 * k); return n; }

void pack_file(const char* source, const char* target, int mode) {
    File in(std::fopen(source, "rb"), &std::fclose), out(std::fopen(target, "wbx"), &std::fclose);
    need(bool(in) && bool(out), "Readable source and new output path required");
    need(std::fseek(in.get(), 0, SEEK_END) == 0, "Source seek");
    long bytes = std::ftell(in.get()); need(bytes >= 0, "Source size"); std::rewind(in.get());
    write(out.get(), "NBH1", 4); integer(out.get(), uint64_t(bytes), 8); integer(out.get(), block_words, 4);
    std::array<uint8_t, block_words * 2> raw{}; std::array<uint16_t, block_words> words{};
    for (uint64_t used = 0; used < uint64_t(bytes);) {
        const size_t count = size_t(std::min<uint64_t>(raw.size(), uint64_t(bytes) - used));
        raw.fill(0); read(in.get(), raw.data(), count);
        const size_t n = (count + 1) / 2;
        for (size_t j = 0; j < n; ++j) words[j] = get16(raw.data() + 2 * j);
        Block b = encode(words.data(), n, mode);
        integer(out.get(), b.kind, 1); integer(out.get(), b.count, 2); integer(out.get(), b.data.size(), 4);
        write(out.get(), b.data.data(), b.data.size()); used += count;
    }
    need(std::fflush(out.get()) == 0, "Container flush");
}

void unpack_file(const char* source, const char* target) {
    File in(std::fopen(source, "rb"), &std::fclose), out(std::fopen(target, "wbx"), &std::fclose);
    need(bool(in) && bool(out), "Readable container and new output path required");
    char magic[4]; read(in.get(), magic, 4); need(std::memcmp(magic, "NBH1", 4) == 0, "Container magic");
    const uint64_t bytes = integer(in.get(), 8);
    need(integer(in.get(), 4) == block_words, "Container block size");
    std::array<uint16_t, block_words> words{}; std::array<uint8_t, block_words * 2> raw{};
    for (uint64_t done = 0; done < bytes;) {
        Block b{uint8_t(integer(in.get(), 1)), uint16_t(integer(in.get(), 2)), {}};
        const uint64_t size = integer(in.get(), 4);
        const size_t wanted = size_t(std::min<uint64_t>(block_words * 2, bytes - done));
        need(b.count == (wanted + 1) / 2 && size <= 7 * block_words, "Container block extent");
        b.data.resize(size_t(size)); read(in.get(), b.data.data(), b.data.size()); decode(b, words.data());
        for (size_t j = 0; j < b.count; ++j) { raw[j * 2] = uint8_t(words[j]); raw[j * 2 + 1] = uint8_t(words[j] >> 8); }
        if (wanted % 2) need(raw[wanted] == 0, "Noncanonical final padding");
        write(out.get(), raw.data(), wanted); done += wanted;
    }
    need(std::fgetc(in.get()) == EOF && !std::ferror(in.get()), "Container trailing data or read error");
    need(std::fflush(out.get()) == 0, "Restored file flush");
}
} // namespace

extern "C" {
int harmony_abi_version() { return 2; }
const char* harmony_error() { return last_error.c_str(); }
void* harmony_matrix(const uint16_t* words, size_t rows, size_t columns, int mode) {
    try { need(words && columns > 0, "Source words and shape required"); return new Matrix(words, rows, columns, mode); }
    catch (const std::exception& e) { last_error = e.what(); return nullptr; }
}
void harmony_free(void* handle) { delete static_cast<Matrix*>(handle); }
int harmony_stats(void* handle, uint64_t* stats) {
    try {
        need(handle && stats, "Stats arguments"); auto& m = *static_cast<Matrix*>(handle);
        std::fill(stats, stats + 10, uint64_t(0));
        for (const auto& b : m.blocks) { stats[0] += b.data.size() + 7; stats[1 + b.kind]++; }
        stats[7] = m.blocks.size() * sizeof(Block); stats[8] = m.cache.size() * sizeof(float); stats[9] = m.rows * m.columns * 2;
        return 0;
    } catch (const std::exception& e) { last_error = e.what(); return -1; }
}
int harmony_unpack(void* handle, uint16_t* words, unsigned threads) {
    try { need(handle && words, "Unpack arguments"); static_cast<Matrix*>(handle)->unpack(words, threads); return 0; }
    catch (const std::exception& e) { last_error = e.what(); return -1; }
}
int harmony_cache(void* handle, unsigned threads) {
    try { need(handle, "Cache argument"); static_cast<Matrix*>(handle)->prepare_cache(threads); return 0; }
    catch (const std::exception& e) { last_error = e.what(); return -1; }
}
int harmony_project(void* handle, const float* x, float* y, size_t batch, unsigned threads, int cached) {
    try { need(handle && x && y, "Projection arguments"); static_cast<Matrix*>(handle)->project(x, y, batch, threads, cached); return 0; }
    catch (const std::exception& e) { last_error = e.what(); return -1; }
}
int harmony_pack_file(const char* source, const char* target, int mode) {
    try { need(source && target, "File paths required"); pack_file(source, target, mode); return 0; }
    catch (const std::exception& e) { last_error = e.what(); return -1; }
}
int harmony_unpack_file(const char* source, const char* target) {
    try { need(source && target, "File paths required"); unpack_file(source, target); return 0; }
    catch (const std::exception& e) { last_error = e.what(); return -1; }
}
}
