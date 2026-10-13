// NMT1 consumer: finite byte law, SHAKE256 artifact checks and exact byte spans.
// Build: C++20 plus OpenSSL 3 libcrypto. stdin carries one hex byte string per line.
#include <openssl/evp.h>
#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace nanoai {
constexpr std::size_t max_input = 131072;
constexpr std::size_t max_piece = 128;
constexpr std::size_t max_active = 131072;
constexpr std::size_t max_work = 16777216;
constexpr double neg_inf = -std::numeric_limits<double>::infinity();

std::string hex(std::string_view raw) {
    constexpr char digits[] = "0123456789abcdef";
    std::string result;
    result.reserve(2 * raw.size());
    for (const unsigned char byte : raw) {
        result.push_back(digits[byte >> 4]);
        result.push_back(digits[byte & 15]);
    }
    return result;
}

std::string unhex(std::string_view text) {
    if (text.size() > 2 * max_input || text.size() % 2 != 0) throw std::runtime_error("input extent");
    auto digit = [](char c) -> unsigned {
        if (c >= '0' && c <= '9') return static_cast<unsigned>(c - '0');
        if (c >= 'a' && c <= 'f') return 10U + static_cast<unsigned>(c - 'a');
        if (c >= 'A' && c <= 'F') return 10U + static_cast<unsigned>(c - 'A');
        throw std::runtime_error("input hex encoding");
    };
    std::string result;
    result.reserve(text.size() / 2);
    for (std::size_t i = 0; i < text.size(); i += 2)
        result.push_back(static_cast<char>((digit(text[i]) << 4) | digit(text[i + 1])));
    return result;
}

std::string shake(std::string_view raw) {
    std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> context(EVP_MD_CTX_new(), EVP_MD_CTX_free);
    std::array<unsigned char, 64> result{};
    if (!context || EVP_DigestInit_ex(context.get(), EVP_shake256(), nullptr) != 1 ||
        EVP_DigestUpdate(context.get(), raw.data(), raw.size()) != 1 ||
        EVP_DigestFinalXOF(context.get(), result.data(), result.size()) != 1)
        throw std::runtime_error("SHAKE256 failed");
    return {reinterpret_cast<const char*>(result.data()), result.size()};
}

struct Reader {
    std::string_view data;
    std::size_t cursor = 0;
    std::string_view take(std::size_t size) {
        if (size > data.size() - cursor) throw std::runtime_error("truncated artifact");
        auto result = data.substr(cursor, size);
        cursor += size;
        return result;
    }
    std::uint64_t integer(std::size_t size) {
        const auto bytes = take(size);
        std::uint64_t value = 0;
        for (std::size_t i = 0; i < size; ++i)
            value |= static_cast<std::uint64_t>(static_cast<unsigned char>(bytes[i])) << (8 * i);
        return value;
    }
    double number() { return std::bit_cast<double>(integer(8)); }
};

struct Piece { std::uint32_t id; std::string bytes; double logq; };
struct Span { std::uint32_t id; std::size_t slot; std::size_t start; std::size_t end; };
struct Encoding {
    std::string tokenizer_identity;
    std::string vocabulary_identity;
    std::string byte_identity;
    std::vector<Span> spans;
    double forward;
    double viterbi;
    double selected;
};
struct StringHash {
    using is_transparent = void;
    std::size_t operator()(std::string_view value) const { return std::hash<std::string_view>{}(value); }
};

class MasterTokenizer {
    std::vector<Piece> pieces_;
    std::unordered_map<std::string, std::size_t, StringHash, std::equal_to<>> slots_;
    double emit_ = 0;
    double stop_ = 0;
    std::size_t max_length_ = 0;
    std::string identity_;
    std::string vocabulary_identity_;
    std::string inventory_identity_;
public:
    const std::string& identity() const { return identity_; }
    const std::string& vocabulary_identity() const { return vocabulary_identity_; }
    const std::string& inventory_identity() const { return inventory_identity_; }

    static std::string read_artifact(const std::string& path) {
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        if (!file) throw std::runtime_error("artifact open failed");
        const auto size = file.tellg();
        if (size < 208 || size > 24000000) throw std::runtime_error("artifact extent");
        std::string data(static_cast<std::size_t>(size), '\0');
        file.seekg(0);
        if (!file.read(data.data(), static_cast<std::streamsize>(data.size())) || file.peek() != EOF)
            throw std::runtime_error("artifact read extent changed");
        return data;
    }

    explicit MasterTokenizer(const std::string& path) : MasterTokenizer(read_artifact(path), true) {}

    MasterTokenizer(std::string_view data, bool) {
        if (data.size() < 208 || data.size() > 24000000) throw std::runtime_error("artifact extent");
        const std::string_view body(data.data(), data.size() - 64);
        const auto digest = shake(body);
        if (digest != std::string_view(data).substr(data.size() - 64)) throw std::runtime_error("artifact digest differs");
        identity_ = "shake256:" + hex(digest);
        Reader reader{body};
        if (reader.take(4) != "NMT1") throw std::runtime_error("artifact magic");
        const auto count = reader.integer(4);
        if (count < 256 || count > max_active) throw std::runtime_error("active vocabulary extent");
        const double p_end = reader.number();
        if (!(p_end > 0 && p_end < 1) || !std::isfinite(p_end)) throw std::runtime_error("stop probability");
        emit_ = std::log1p(-p_end);
        stop_ = std::log(p_end);
        vocabulary_identity_ = "shake256:" + hex(reader.take(64));
        inventory_identity_ = "shake256:" + hex(reader.take(64));
        pieces_.reserve(static_cast<std::size_t>(count));
        slots_.reserve(static_cast<std::size_t>(count));
        long double sum = 0;
        std::array<bool, 256> fallback{};
        for (std::size_t i = 0; i < count; ++i) {
            const auto id = reader.integer(4);
            const auto length = reader.integer(2);
            const double q = reader.number();
            if (id >= 2000000 || length < 1 || length > max_piece || !std::isfinite(q) || q <= 0)
                throw std::runtime_error("piece record");
            std::string bytes(reader.take(static_cast<std::size_t>(length)));
            if (!pieces_.empty() && (id <= pieces_.back().id || bytes <= pieces_.back().bytes))
                throw std::runtime_error("piece or ID ordering");
            if (length == 1) fallback[static_cast<unsigned char>(bytes[0])] = true;
            sum += q;
            max_length_ = std::max(max_length_, bytes.size());
            slots_.emplace(bytes, i);
            pieces_.push_back({static_cast<std::uint32_t>(id), std::move(bytes), std::log(q)});
        }
        if (reader.cursor != body.size() || std::abs(sum - 1) > 1e-10L ||
            !std::all_of(fallback.begin(), fallback.end(), [](bool value) { return value; }))
            throw std::runtime_error("finite byte law or trailing artifact bytes");
    }

    void check_input(std::string_view raw, const std::vector<std::size_t>& boundaries) const {
        if (raw.size() > max_input || raw.size() * max_length_ > max_work) throw std::runtime_error("input work extent");
        std::size_t previous = 0;
        for (const auto boundary : boundaries) {
            if (boundary <= previous || boundary >= raw.size()) throw std::runtime_error("boundary extent or order");
            previous = boundary;
        }
    }

    Encoding encode(std::string_view raw, bool minimum = false, const std::vector<std::size_t>& boundaries = {}) const {
        check_input(raw, boundaries);
        std::vector<double> forward(raw.size() + 1, neg_inf), best(raw.size() + 1, neg_inf);
        std::vector<double> selected(raw.size() + 1, neg_inf);
        std::vector<std::size_t> counts(raw.size() + 1, max_input + 1);
        std::vector<std::size_t> back(raw.size() + 1, 0);
        forward[0] = 0; best[0] = 0; selected[0] = 0; counts[0] = 0;
        std::size_t floor = 0, boundary_index = 0;
        for (std::size_t end = 1; end <= raw.size(); ++end) {
            while (boundary_index < boundaries.size() && boundaries[boundary_index] < end)
                floor = boundaries[boundary_index++];
            for (std::size_t length = 1; length <= std::min(max_length_, end - floor); ++length) {
                const auto found = slots_.find(raw.substr(end - length, length));
                if (found == slots_.end()) continue;
                const std::size_t slot = found->second, start = end - length;
                const double candidate = forward[start] + emit_ + pieces_[slot].logq;
                if (forward[end] == neg_inf) forward[end] = candidate;
                else {
                    const double high = std::max(forward[end], candidate), low = std::min(forward[end], candidate);
                    forward[end] = high + std::log1p(std::exp(low - high));
                }
                const double score = best[start] + emit_ + pieces_[slot].logq;
                if (score > best[end]) best[end] = score;
                const auto n = counts[start] + 1;
                const double path_score = selected[start] + emit_ + pieces_[slot].logq;
                if ((!minimum && path_score > selected[end]) ||
                    (minimum && (n < counts[end] || (n == counts[end] && path_score > selected[end])))) {
                    selected[end] = path_score; counts[end] = n; back[end] = slot;
                }
            }
        }
        Encoding result{identity_, vocabulary_identity_, "shake256:" + hex(shake(raw)), {}, forward.back() + stop_, best.back() + stop_, selected.back() + stop_};
        for (std::size_t end = raw.size(); end > 0;) {
            const auto slot = back[end];
            const auto start = end - pieces_[slot].bytes.size();
            result.spans.push_back({pieces_[slot].id, slot, start, end});
            end = start;
        }
        std::reverse(result.spans.begin(), result.spans.end());
        return result;
    }

    std::pair<std::vector<double>, double> expected_counts(std::string_view raw) const {
        check_input(raw, {});
        std::vector<double> forward(raw.size() + 1, neg_inf), backward(raw.size() + 1, neg_inf);
        std::vector<double> counts(pieces_.size(), 0);
        forward[0] = 0; backward.back() = 0;
        auto add = [](double a, double b) {
            if (a == neg_inf) return b;
            const double high = std::max(a, b), low = std::min(a, b);
            return high + std::log1p(std::exp(low - high));
        };
        for (std::size_t end = 1; end <= raw.size(); ++end) {
            for (std::size_t length = 1; length <= std::min(max_length_, end); ++length) {
                const auto found = slots_.find(raw.substr(end - length, length));
                if (found != slots_.end())
                    forward[end] = add(forward[end], forward[end-length] + emit_ + pieces_[found->second].logq);
            }
        }
        for (std::size_t end = raw.size(); end > 0; --end) {
            for (std::size_t length = 1; length <= std::min(max_length_, end); ++length) {
                const auto found = slots_.find(raw.substr(end - length, length));
                if (found == slots_.end()) continue;
                const auto start = end - length, slot = found->second;
                const double edge = emit_ + pieces_[slot].logq;
                backward[start] = add(backward[start], edge + backward[end]);
                counts[slot] += std::exp(forward[start] + edge + backward[end] - forward.back());
            }
        }
        return {std::move(counts), forward.back() + stop_};
    }

    std::string decode(const Encoding& encoded) const {
        if (encoded.tokenizer_identity != identity_ || encoded.vocabulary_identity != vocabulary_identity_ || encoded.spans.size() > max_input)
            throw std::runtime_error("token record identity or extent");
        std::string result;
        for (const auto& span : encoded.spans) {
            if (span.slot >= pieces_.size()) throw std::runtime_error("token slot extent");
            const auto& piece = pieces_[span.slot];
            if (span.id != piece.id || span.start != result.size() || span.end != result.size() + piece.bytes.size() || span.end > max_input)
                throw std::runtime_error("token span or ID");
            result += piece.bytes;
        }
        if ("shake256:" + hex(shake(result)) != encoded.byte_identity) throw std::runtime_error("token byte identity");
        return result;
    }
};
} // namespace nanoai

#ifndef NANOAI_MASTER_LIBRARY_ONLY
int main(int argc, char** argv) {
    try {
        if (argc < 2 || argc > 3) throw std::runtime_error("usage: master_tokenizer ARTIFACT [EXPECTED_SHAKE256]");
        const nanoai::MasterTokenizer tokenizer(argv[1]);
        if (argc == 3 && tokenizer.identity() != argv[2]) throw std::runtime_error("expected artifact version differs");
        std::cout << std::setprecision(17);
        std::string line;
        char c = 0;
        auto process = [&]() {
            const auto raw = nanoai::unhex(line);
            const auto start = std::chrono::steady_clock::now();
            const auto encoded = tokenizer.encode(raw);
            const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count();
            const auto decoded = tokenizer.decode(encoded);
            if (decoded != raw) throw std::runtime_error("decoded bytes differ");
            std::cout << "{\"schema\":\"nanoai.master-tokens.v1\",\"tokenizer_identity\":\"" << encoded.tokenizer_identity
                      << "\",\"vocabulary_identity\":\"" << encoded.vocabulary_identity
                      << "\",\"byte_identity\":\"" << encoded.byte_identity
                      << "\",\"decoded_hex\":\"" << nanoai::hex(decoded) << "\",\"forward_logp\":" << encoded.forward
                      << ",\"viterbi_logp\":" << encoded.viterbi << ",\"encode_ns\":" << elapsed << ",\"spans\":[";
            bool first = true;
            for (const auto& span : encoded.spans) {
                if (!first) std::cout << ',';
                first = false;
                std::cout << "{\"id\":" << span.id << ",\"active_id\":" << span.slot << ",\"start\":" << span.start << ",\"end\":" << span.end << '}';
            }
            std::cout << "]}\n";
            line.clear();
        };
        while (std::cin.get(c)) {
            if (c == '\n') process();
            else {
                if (line.size() >= 2 * nanoai::max_input) throw std::runtime_error("input line extent");
                line.push_back(c);
            }
        }
        if (!line.empty()) process();
        if (!std::cin.eof()) throw std::runtime_error("input read failure");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "master_tokenizer: " << error.what() << '\n';
        return 2;
    }
}
#endif
