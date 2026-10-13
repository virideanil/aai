#define NANOAI_MASTER_LIBRARY_ONLY
#include "../native/master_tokenizer.cpp"

int main(int argc, char** argv) {
    if (argc != 2) return 1;
    const nanoai::MasterTokenizer tokenizer(argv[1]);
    std::string raw("a\0b\xff\n", 5);
    const auto original = raw;
    const auto encoded = tokenizer.encode(raw);
    raw.clear();
    if (tokenizer.decode(encoded) != original) return 2;
    std::size_t rejected = 0;
    for (int mutation = 0; mutation < 4; ++mutation) {
        auto changed = encoded;
        if (mutation == 0) changed.tokenizer_identity = "foreign";
        if (mutation == 1) changed.spans[0].slot = nanoai::max_active;
        if (mutation == 2) changed.spans[0].end += 1;
        if (mutation == 3) changed.byte_identity = "foreign";
        try { const auto result = tokenizer.decode(changed); (void)result; }
        catch (const std::runtime_error&) { ++rejected; }
    }
    if (rejected != 4) return 3;
    const std::string boundary(nanoai::max_input, 'x');
    if (tokenizer.decode(tokenizer.encode(boundary)) != boundary) return 4;
    try { const auto result = tokenizer.encode(boundary + "x"); (void)result; return 5; }
    catch (const std::runtime_error&) {}
    std::cout << "native API ownership, identity, malformed spans and input boundary passed\n";
}
