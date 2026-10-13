#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define NANOAI_MASTER_LIBRARY_ONLY
#include "master_tokenizer.cpp"

namespace {
constexpr const char* capsule_name = "nanoai_tokens.NMT1";
struct RestoreGIL {
    PyThreadState* state = PyEval_SaveThread();
    ~RestoreGIL() { PyEval_RestoreThread(state); }
};
struct Decref { void operator()(PyObject* value) const { Py_XDECREF(value); } };
using Object = std::unique_ptr<PyObject, Decref>;

nanoai::MasterTokenizer* model(PyObject* capsule) {
    auto* result = static_cast<nanoai::MasterTokenizer*>(PyCapsule_GetPointer(capsule, capsule_name));
    if (!result) throw std::runtime_error("native tokenizer handle");
    return result;
}
void destroy(PyObject* capsule) {
    auto* value = static_cast<nanoai::MasterTokenizer*>(PyCapsule_GetPointer(capsule, capsule_name));
    if (value) delete value;
    else PyErr_Clear();
}
PyObject* fail(const char* message) {
    if (!PyErr_Occurred()) PyErr_SetString(PyExc_ValueError, message);
    return nullptr;
}

PyObject* create(PyObject*, PyObject* args) {
    const char* data = nullptr; Py_ssize_t size = 0;
    if (!PyArg_ParseTuple(args, "y#", &data, &size)) return nullptr;
    try {
        std::unique_ptr<nanoai::MasterTokenizer> tokenizer;
        { RestoreGIL gil; tokenizer = std::make_unique<nanoai::MasterTokenizer>(std::string_view(data, static_cast<std::size_t>(size)), true); }
        PyObject* result = PyCapsule_New(tokenizer.get(), capsule_name, destroy);
        if (result) tokenizer.release();
        return result;
    } catch (const std::exception& error) { return fail(error.what()); }
    catch (...) { return fail("native tokenizer construction failed"); }
}

PyObject* encode(PyObject*, PyObject* args) {
    PyObject *capsule = nullptr, *boundary_arg = nullptr;
    const char* data = nullptr; Py_ssize_t size = 0; int minimum = 0;
    if (!PyArg_ParseTuple(args, "Oy#pO", &capsule, &data, &size, &minimum, &boundary_arg)) return nullptr;
    try {
        const auto* tokenizer = model(capsule);
        Object sequence(PySequence_Fast(boundary_arg, "boundaries require a sequence"));
        if (!sequence) return nullptr;
        const auto count = PySequence_Fast_GET_SIZE(sequence.get());
        if (count > size) return fail("boundary count");
        std::vector<std::size_t> boundaries;
        boundaries.reserve(static_cast<std::size_t>(count));
        for (Py_ssize_t i = 0; i < count; ++i) {
            PyObject* value = PySequence_Fast_GET_ITEM(sequence.get(), i);
            if (!PyLong_CheckExact(value)) return fail("integer byte boundary required");
            const auto boundary = PyLong_AsSize_t(value);
            if (PyErr_Occurred()) return nullptr;
            boundaries.push_back(boundary);
        }
        nanoai::Encoding encoded;
        { RestoreGIL gil; encoded = tokenizer->encode(std::string_view(data, static_cast<std::size_t>(size)), minimum != 0, boundaries); }
        Object spans(PyList_New(static_cast<Py_ssize_t>(encoded.spans.size())));
        if (!spans) return nullptr;
        for (std::size_t i = 0; i < encoded.spans.size(); ++i) {
            const auto& span = encoded.spans[i];
            PyObject* row = Py_BuildValue("{sI,sK,sK,sK}", "id", span.id,
                "active_id", static_cast<unsigned long long>(span.slot), "start", static_cast<unsigned long long>(span.start),
                "end", static_cast<unsigned long long>(span.end));
            if (!row) return nullptr;
            PyList_SET_ITEM(spans.get(), static_cast<Py_ssize_t>(i), row);
        }
        return Py_BuildValue("{ss,ss,ss,ss,sO,sd,sd,sd}", "schema", "nanoai.master-tokens.v1",
            "tokenizer_identity", encoded.tokenizer_identity.c_str(), "vocabulary_identity", encoded.vocabulary_identity.c_str(),
            "byte_identity", encoded.byte_identity.c_str(), "spans", spans.get(), "forward_logp", encoded.forward,
            "viterbi_logp", encoded.viterbi, "selected_logp", encoded.selected);
    } catch (const std::exception& error) { return fail(error.what()); }
    catch (...) { return fail("native tokenization failed"); }
}

PyObject* expected_counts(PyObject*, PyObject* args) {
    PyObject* capsule = nullptr; const char* data = nullptr; Py_ssize_t size = 0;
    if (!PyArg_ParseTuple(args, "Oy#", &capsule, &data, &size)) return nullptr;
    try {
        const auto* tokenizer = model(capsule);
        std::pair<std::vector<double>, double> result;
        { RestoreGIL gil; result = tokenizer->expected_counts(std::string_view(data, static_cast<std::size_t>(size))); }
        Object counts(PyList_New(static_cast<Py_ssize_t>(result.first.size())));
        if (!counts) return nullptr;
        for (std::size_t i = 0; i < result.first.size(); ++i) {
            PyObject* value = PyFloat_FromDouble(result.first[i]);
            if (!value) return nullptr;
            PyList_SET_ITEM(counts.get(), static_cast<Py_ssize_t>(i), value);
        }
        return Py_BuildValue("(Od)", counts.get(), result.second);
    } catch (const std::exception& error) { return fail(error.what()); }
    catch (...) { return fail("native lexical expectation failed"); }
}

PyMethodDef methods[] = {
    {"create", create, METH_VARARGS, "Own and verify an immutable NMT1 artifact."},
    {"encode", encode, METH_VARARGS, "Native complete byte-law segmentation."},
    {"expected_counts", expected_counts, METH_VARARGS, "Native lexical expected counts."},
    {nullptr, nullptr, 0, nullptr}
};
PyModuleDef module = {PyModuleDef_HEAD_INIT, "_native", "Native byte token core.", -1, methods, nullptr, nullptr, nullptr, nullptr};
}
PyMODINIT_FUNC PyInit__native() { return PyModule_Create(&module); }
