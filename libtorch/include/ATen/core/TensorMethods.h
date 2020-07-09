#pragma once

#include <c10/core/Scalar.h>
#include <c10/core/MemoryFormat.h>
#include <c10/core/QScheme.h>
#include <c10/macros/Macros.h>
#include <c10/core/TensorOptions.h>
#include <c10/util/intrusive_ptr.h>
#include <ATen/core/DeprecatedTypeProperties.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/NamedTensor.h>
#include <ATen/core/LegacyTypeDispatch.h>

#ifdef USE_STATIC_DISPATCH
#include <ATen/TypeDefault.h>
#include <ATen/CPUType.h>
#include <ATen/QuantizedCPUType.h>
#endif

namespace at {

struct Quantizer;
// This is temporary typedef to enable Quantizer in aten native function API
// we'll remove them when we are actually exposing Quantizer class
// to frontend
using ConstQuantizerPtr = const c10::intrusive_ptr<Quantizer>&;

inline Tensor Tensor::cpu() const {
  return to(options().device(DeviceType::CPU), /*non_blocking*/ false, /*copy*/ false);
}

// TODO: The Python version also accepts arguments
inline Tensor Tensor::cuda() const {
  return to(options().device(DeviceType::CUDA), /*non_blocking*/ false, /*copy*/ false);
}

inline Tensor Tensor::hip() const {
  return to(options().device(DeviceType::HIP), /*non_blocking*/ false, /*copy*/ false);
}

inline Tensor Tensor::toType(ScalarType t) const {
  return to(options().dtype(t), /*non_blocking*/ false, /*copy*/ false);
}

// TODO: Deprecate me
inline Tensor Tensor::toBackend(Backend b) const {
  return to(options().device(backendToDeviceType(b)).layout(layout_from_backend(b)), /*non_blocking*/ false, /*copy*/ false);
}

inline TensorOptions Tensor::options() const {
  return TensorOptions().dtype(dtype())
                        .device(device())
                        .layout(layout());
}

// all static inline to allow for inlining of the non-dynamic part of dispatch

// aten::backward(Tensor self, Tensor? gradient=None, bool? retain_graph=None, bool create_graph=False) -> ()
inline void Tensor::backward(const Tensor & gradient, c10::optional<bool> retain_graph, bool create_graph) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
     TypeDefault::backward(const_cast<Tensor&>(*this), gradient, retain_graph, create_graph);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::backward", "");
    return op.callUnboxed<void, const Tensor &, const Tensor &, c10::optional<bool>, bool>(const_cast<Tensor&>(*this), gradient, retain_graph, create_graph);
#endif
}

// aten::set_data(Tensor(a!) self, Tensor new_data) -> ()
inline void Tensor::set_data(const Tensor & new_data) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
     TypeDefault::set_data(const_cast<Tensor&>(*this), new_data);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::set_data", "");
    return op.callUnboxed<void, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), new_data);
#endif
}

// aten::data(Tensor self) -> Tensor
inline Tensor Tensor::data() const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::data(const_cast<Tensor&>(*this));
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::data", "");
    return op.callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}

// aten::is_leaf(Tensor self) -> bool
inline bool Tensor::is_leaf() const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::is_leaf(const_cast<Tensor&>(*this));
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::is_leaf", "");
    return op.callUnboxed<bool, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}

// aten::output_nr(Tensor self) -> int
inline int64_t Tensor::output_nr() const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::output_nr(const_cast<Tensor&>(*this));
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::output_nr", "");
    return op.callUnboxed<int64_t, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}

// aten::_version(Tensor self) -> int
inline int64_t Tensor::_version() const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::_version(const_cast<Tensor&>(*this));
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::_version", "");
    return op.callUnboxed<int64_t, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}

// aten::requires_grad_(Tensor(a!) self, bool requires_grad=True) -> Tensor(a!)
inline Tensor & Tensor::requires_grad_(bool requires_grad) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::requires_grad_(const_cast<Tensor&>(*this), requires_grad);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::requires_grad_", "");
    return op.callUnboxed<Tensor &, Tensor &, bool>(const_cast<Tensor&>(*this), requires_grad);
#endif
}

// aten::retain_grad(Tensor(a!) self) -> ()
inline void Tensor::retain_grad() const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
     TypeDefault::retain_grad(const_cast<Tensor&>(*this));
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::retain_grad", "");
    return op.callUnboxed<void, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}

// aten::rename_(Tensor(a!) self, Dimname[]? names) -> Tensor(a!)
inline Tensor & Tensor::rename_(c10::optional<DimnameList> names) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::rename_(const_cast<Tensor&>(*this), names);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::rename_", "");
    return op.callUnboxed<Tensor &, Tensor &, c10::optional<DimnameList>>(const_cast<Tensor&>(*this), names);
#endif
}

// aten::rename(Tensor(a) self, Dimname[]? names) -> Tensor(a)
inline Tensor Tensor::rename(c10::optional<DimnameList> names) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::rename(const_cast<Tensor&>(*this), names);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::rename", "");
    return op.callUnboxed<Tensor, const Tensor &, c10::optional<DimnameList>>(const_cast<Tensor&>(*this), names);
#endif
}

// aten::align_to(Tensor(a) self, Dimname[] names) -> Tensor(a)
inline Tensor Tensor::align_to(DimnameList names) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::align_to(const_cast<Tensor&>(*this), names);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::align_to", "");
    return op.callUnboxed<Tensor, const Tensor &, DimnameList>(const_cast<Tensor&>(*this), names);
#endif
}

// aten::align_to.ellipsis_idx(Tensor(a) self, Dimname[] order, int ellipsis_idx) -> Tensor(a)
inline Tensor Tensor::align_to(DimnameList order, int64_t ellipsis_idx) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::align_to_ellipsis_idx(const_cast<Tensor&>(*this), order, ellipsis_idx);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::align_to", "ellipsis_idx");
    return op.callUnboxed<Tensor, const Tensor &, DimnameList, int64_t>(const_cast<Tensor&>(*this), order, ellipsis_idx);
#endif
}

// aten::align_as(Tensor self, Tensor other) -> Tensor
inline Tensor Tensor::align_as(const Tensor & other) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::align_as(const_cast<Tensor&>(*this), other);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::align_as", "");
    return op.callUnboxed<Tensor, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), other);
#endif
}

// aten::refine_names(Tensor(a) self, Dimname[] names) -> Tensor(a)
inline Tensor Tensor::refine_names(DimnameList names) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::refine_names(const_cast<Tensor&>(*this), names);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::refine_names", "");
    return op.callUnboxed<Tensor, const Tensor &, DimnameList>(const_cast<Tensor&>(*this), names);
#endif
}

// aten::unflatten.Dimname(Tensor self, Dimname dim, int[] sizes, Dimname[] names) -> Tensor
inline Tensor Tensor::unflatten(Dimname dim, IntArrayRef sizes, DimnameList names) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::unflatten_Dimname(const_cast<Tensor&>(*this), dim, sizes, names);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::unflatten", "Dimname");
    return op.callUnboxed<Tensor, const Tensor &, Dimname, IntArrayRef, DimnameList>(const_cast<Tensor&>(*this), dim, sizes, names);
#endif
}

// aten::unflatten.int(Tensor self, int dim, int[] sizes, Dimname[] names) -> Tensor
inline Tensor Tensor::unflatten(int64_t dim, IntArrayRef sizes, DimnameList names) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::unflatten_int(const_cast<Tensor&>(*this), dim, sizes, names);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::unflatten", "int");
    return op.callUnboxed<Tensor, const Tensor &, int64_t, IntArrayRef, DimnameList>(const_cast<Tensor&>(*this), dim, sizes, names);
#endif
}

// aten::abs(Tensor self) -> Tensor
inline Tensor Tensor::abs() const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::abs(const_cast<Tensor&>(*this));
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::abs", "");
    return op.callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}

// aten::abs_(Tensor(a!) self) -> Tensor(a!)
inline Tensor & Tensor::abs_() const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::abs_(const_cast<Tensor&>(*this));
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::abs_", "");
    return op.callUnboxed<Tensor &, Tensor &>(const_cast<Tensor&>(*this));
#endif
}

// aten::absolute(Tensor self) -> Tensor
inline Tensor Tensor::absolute() const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(*this);
    DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
    DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);
    switch (dispatchKeyToBackend(_dk)) {
        case Backend::CPU:
            return CPUType::absolute(const_cast<Tensor&>(*this));
            break;
        default:
            AT_ERROR("absolute not implemented for ", at::toString(_dk));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::absolute", "");
    return op.callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}

// aten::absolute_(Tensor(a!) self) -> Tensor(a!)
inline Tensor & Tensor::absolute_() const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(*this);
    DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
    DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);
    switch (dispatchKeyToBackend(_dk)) {
        case Backend::CPU:
            return CPUType::absolute_(const_cast<Tensor&>(*this));
            break;
        default:
            AT_ERROR("absolute_ not implemented for ", at::toString(_dk));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::absolute_", "");
    return op.callUnboxed<Tensor &, Tensor &>(const_cast<Tensor&>(*this));
#endif
}

// aten::angle(Tensor self) -> Tensor
inline Tensor Tensor::angle() const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::angle(const_cast<Tensor&>(*this));
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::angle", "");
    return op.callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}

// aten::copy_real(Tensor self) -> Tensor
inline Tensor Tensor::copy_real() const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::copy_real(const_cast<Tensor&>(*this));
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::copy_real", "");
    return op.callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}

// aten::copy_imag(Tensor self) -> Tensor
inline Tensor Tensor::copy_imag() const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::copy_imag(const_cast<Tensor&>(*this));
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::copy_imag", "");
    return op.callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}

// aten::conj(Tensor self) -> Tensor
inline Tensor Tensor::conj() const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::conj(const_cast<Tensor&>(*this));
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::conj", "");
    return op.callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}

// aten::acos(Tensor self) -> Tensor
inline Tensor Tensor::acos() const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::acos(const_cast<Tensor&>(*this));
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::acos", "");
    return op.callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}

// aten::acos_(Tensor(a!) self) -> Tensor(a!)
inline Tensor & Tensor::acos_() const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::acos_(const_cast<Tensor&>(*this));
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::acos_", "");
    return op.callUnboxed<Tensor &, Tensor &>(const_cast<Tensor&>(*this));
#endif
}

// aten::add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
inline Tensor Tensor::add(const Tensor & other, Scalar alpha) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(*this, other);
    DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
    DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);
    switch (dispatchKeyToBackend(_dk)) {
        case Backend::CPU:
            return CPUType::add_Tensor(const_cast<Tensor&>(*this), other, alpha);
            break;
        default:
            AT_ERROR("add not implemented for ", at::toString(_dk));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::add", "Tensor");
    return op.callUnboxed<Tensor, const Tensor &, const Tensor &, Scalar>(const_cast<Tensor&>(*this), other, alpha);
#endif
}

// aten::add_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> Tensor(a!)
inline Tensor & Tensor::add_(const Tensor & other, Scalar alpha) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(*this, other);
    DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
    DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);
    switch (dispatchKeyToBackend(_dk)) {
        case Backend::CPU:
            return CPUType::add__Tensor(const_cast<Tensor&>(*this), other, alpha);
            break;
        default:
            AT_ERROR("add_ not implemented for ", at::toString(_dk));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::add_", "Tensor");
    return op.callUnboxed<Tensor &, Tensor &, const Tensor &, Scalar>(const_cast<Tensor&>(*this), other, alpha);
#endif
}

// aten::add.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor
inline Tensor Tensor::add(Scalar other, Scalar alpha) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::add_Scalar(const_cast<Tensor&>(*this), other, alpha);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::add", "Scalar");
    return op.callUnboxed<Tensor, const Tensor &, Scalar, Scalar>(const_cast<Tensor&>(*this), other, alpha);
#endif
}

// aten::add_.Scalar(Tensor(a!) self, Scalar other, Scalar alpha=1) -> Tensor(a!)
inline Tensor & Tensor::add_(Scalar other, Scalar alpha) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::add__Scalar(const_cast<Tensor&>(*this), other, alpha);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::add_", "Scalar");
    return op.callUnboxed<Tensor &, Tensor &, Scalar, Scalar>(const_cast<Tensor&>(*this), other, alpha);
#endif
}

// aten::addmv(Tensor self, Tensor mat, Tensor vec, *, Scalar beta=1, Scalar alpha=1) -> Tensor
inline Tensor Tensor::addmv(const Tensor & mat, const Tensor & vec, Scalar beta, Scalar alpha) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::addmv(const_cast<Tensor&>(*this), mat, vec, beta, alpha);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::addmv", "");
    return op.callUnboxed<Tensor, const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar>(const_cast<Tensor&>(*this), mat, vec, beta, alpha);
#endif
}

// aten::addmv_(Tensor(a!) self, Tensor mat, Tensor vec, *, Scalar beta=1, Scalar alpha=1) -> Tensor(a!)
inline Tensor & Tensor::addmv_(const Tensor & mat, const Tensor & vec, Scalar beta, Scalar alpha) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::addmv_(const_cast<Tensor&>(*this), mat, vec, beta, alpha);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::addmv_", "");
    return op.callUnboxed<Tensor &, Tensor &, const Tensor &, const Tensor &, Scalar, Scalar>(const_cast<Tensor&>(*this), mat, vec, beta, alpha);
#endif
}

// aten::addr(Tensor self, Tensor vec1, Tensor vec2, *, Scalar beta=1, Scalar alpha=1) -> Tensor
inline Tensor Tensor::addr(const Tensor & vec1, const Tensor & vec2, Scalar beta, Scalar alpha) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::addr(const_cast<Tensor&>(*this), vec1, vec2, beta, alpha);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::addr", "");
    return op.callUnboxed<Tensor, const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar>(const_cast<Tensor&>(*this), vec1, vec2, beta, alpha);
#endif
}

// aten::addr_(Tensor(a!) self, Tensor vec1, Tensor vec2, *, Scalar beta=1, Scalar alpha=1) -> Tensor(a!)
inline Tensor & Tensor::addr_(const Tensor & vec1, const Tensor & vec2, Scalar beta, Scalar alpha) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::addr_(const_cast<Tensor&>(*this), vec1, vec2, beta, alpha);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::addr_", "");
    return op.callUnboxed<Tensor &, Tensor &, const Tensor &, const Tensor &, Scalar, Scalar>(const_cast<Tensor&>(*this), vec1, vec2, beta, alpha);
#endif
}

// aten::all.dim(Tensor self, int dim, bool keepdim=False) -> Tensor
inline Tensor Tensor::all(int64_t dim, bool keepdim) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::all_dim(const_cast<Tensor&>(*this), dim, keepdim);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::all", "dim");
    return op.callUnboxed<Tensor, const Tensor &, int64_t, bool>(const_cast<Tensor&>(*this), dim, keepdim);
#endif
}

// aten::all.dimname(Tensor self, Dimname dim, bool keepdim=False) -> Tensor
inline Tensor Tensor::all(Dimname dim, bool keepdim) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::all_dimname(const_cast<Tensor&>(*this), dim, keepdim);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::all", "dimname");
    return op.callUnboxed<Tensor, const Tensor &, Dimname, bool>(const_cast<Tensor&>(*this), dim, keepdim);
#endif
}

// aten::allclose(Tensor self, Tensor other, float rtol=1e-05, float atol=1e-08, bool equal_nan=False) -> bool
inline bool Tensor::allclose(const Tensor & other, double rtol, double atol, bool equal_nan) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::allclose(const_cast<Tensor&>(*this), other, rtol, atol, equal_nan);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::allclose", "");
    return op.callUnboxed<bool, const Tensor &, const Tensor &, double, double, bool>(const_cast<Tensor&>(*this), other, rtol, atol, equal_nan);
#endif
}

// aten::any.dim(Tensor self, int dim, bool keepdim=False) -> Tensor
inline Tensor Tensor::any(int64_t dim, bool keepdim) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::any_dim(const_cast<Tensor&>(*this), dim, keepdim);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::any", "dim");
    return op.callUnboxed<Tensor, const Tensor &, int64_t, bool>(const_cast<Tensor&>(*this), dim, keepdim);
#endif
}

// aten::any.dimname(Tensor self, Dimname dim, bool keepdim=False) -> Tensor
inline Tensor Tensor::any(Dimname dim, bool keepdim) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::any_dimname(const_cast<Tensor&>(*this), dim, keepdim);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::any", "dimname");
    return op.callUnboxed<Tensor, const Tensor &, Dimname, bool>(const_cast<Tensor&>(*this), dim, keepdim);
#endif
}

// aten::argmax(Tensor self, int? dim=None, bool keepdim=False) -> Tensor
inline Tensor Tensor::argmax(c10::optional<int64_t> dim, bool keepdim) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(*this);
    DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
    DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);
    switch (dispatchKeyToBackend(_dk)) {
        case Backend::CPU:
            return CPUType::argmax(const_cast<Tensor&>(*this), dim, keepdim);
            break;
        default:
            AT_ERROR("argmax not implemented for ", at::toString(_dk));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::argmax", "");
    return op.callUnboxed<Tensor, const Tensor &, c10::optional<int64_t>, bool>(const_cast<Tensor&>(*this), dim, keepdim);
#endif
}

// aten::argmin(Tensor self, int? dim=None, bool keepdim=False) -> Tensor
inline Tensor Tensor::argmin(c10::optional<int64_t> dim, bool keepdim) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(*this);
    DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
    DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);
    switch (dispatchKeyToBackend(_dk)) {
        case Backend::CPU:
            return CPUType::argmin(const_cast<Tensor&>(*this), dim, keepdim);
            break;
        default:
            AT_ERROR("argmin not implemented for ", at::toString(_dk));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::argmin", "");
    return op.callUnboxed<Tensor, const Tensor &, c10::optional<int64_t>, bool>(const_cast<Tensor&>(*this), dim, keepdim);
#endif
}

// aten::as_strided(Tensor(a) self, int[] size, int[] stride, int? storage_offset=None) -> Tensor(a)
inline Tensor Tensor::as_strided(IntArrayRef size, IntArrayRef stride, c10::optional<int64_t> storage_offset) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(*this);
    DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
    DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);
    switch (dispatchKeyToBackend(_dk)) {
        case Backend::CPU:
            return CPUType::as_strided(const_cast<Tensor&>(*this), size, stride, storage_offset);
            break;
        case Backend::QuantizedCPU:
            return QuantizedCPUType::as_strided(const_cast<Tensor&>(*this), size, stride, storage_offset);
            break;
        default:
            AT_ERROR("as_strided not implemented for ", at::toString(_dk));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::as_strided", "");
    return op.callUnboxed<Tensor, const Tensor &, IntArrayRef, IntArrayRef, c10::optional<int64_t>>(const_cast<Tensor&>(*this), size, stride, storage_offset);
#endif
}

// aten::as_strided_(Tensor(a!) self, int[] size, int[] stride, int? storage_offset=None) -> Tensor(a!)
inline Tensor & Tensor::as_strided_(IntArrayRef size, IntArrayRef stride, c10::optional<int64_t> storage_offset) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::as_strided_(const_cast<Tensor&>(*this), size, stride, storage_offset);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::as_strided_", "");
    return op.callUnboxed<Tensor &, Tensor &, IntArrayRef, IntArrayRef, c10::optional<int64_t>>(const_cast<Tensor&>(*this), size, stride, storage_offset);
#endif
}

// aten::asin(Tensor self) -> Tensor
inline Tensor Tensor::asin() const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::asin(const_cast<Tensor&>(*this));
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::asin", "");
    return op.callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}

// aten::asin_(Tensor(a!) self) -> Tensor(a!)
inline Tensor & Tensor::asin_() const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::asin_(const_cast<Tensor&>(*this));
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::asin_", "");
    return op.callUnboxed<Tensor &, Tensor &>(const_cast<Tensor&>(*this));
#endif
}

// aten::atan(Tensor self) -> Tensor
inline Tensor Tensor::atan() const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::atan(const_cast<Tensor&>(*this));
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::atan", "");
    return op.callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}

// aten::atan_(Tensor(a!) self) -> Tensor(a!)
inline Tensor & Tensor::atan_() const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(*this);
    DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
    DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);
    switch (dispatchKeyToBackend(_dk)) {
        case Backend::CPU:
            return CPUType::atan_(const_cast<Tensor&>(*this));
            break;
        default:
            AT_ERROR("atan_ not implemented for ", at::toString(_dk));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::atan_", "");
    return op.callUnboxed<Tensor &, Tensor &>(const_cast<Tensor&>(*this));
#endif
}

// aten::baddbmm(Tensor self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1) -> Tensor
inline Tensor Tensor::baddbmm(const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(*this, batch1, batch2);
    DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
    DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);
    switch (dispatchKeyToBackend(_dk)) {
        case Backend::CPU:
            return CPUType::baddbmm(const_cast<Tensor&>(*this), batch1, batch2, beta, alpha);
            break;
        default:
            AT_ERROR("baddbmm not implemented for ", at::toString(_dk));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::baddbmm", "");
    return op.callUnboxed<Tensor, const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar>(const_cast<Tensor&>(*this), batch1, batch2, beta, alpha);
#endif
}

// aten::baddbmm_(Tensor(a!) self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1) -> Tensor(a!)
inline Tensor & Tensor::baddbmm_(const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(*this, batch1, batch2);
    DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
    DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);
    switch (dispatchKeyToBackend(_dk)) {
        case Backend::CPU:
            return CPUType::baddbmm_(const_cast<Tensor&>(*this), batch1, batch2, beta, alpha);
            break;
        default:
            AT_ERROR("baddbmm_ not implemented for ", at::toString(_dk));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::baddbmm_", "");
    return op.callUnboxed<Tensor &, Tensor &, const Tensor &, const Tensor &, Scalar, Scalar>(const_cast<Tensor&>(*this), batch1, batch2, beta, alpha);
#endif
}

// aten::bernoulli(Tensor self, *, Generator? generator=None) -> Tensor
inline Tensor Tensor::bernoulli(c10::optional<Generator> generator) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::bernoulli(const_cast<Tensor&>(*this), generator);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::bernoulli", "");
    return op.callUnboxed<Tensor, const Tensor &, c10::optional<Generator>>(const_cast<Tensor&>(*this), generator);
#endif
}

// aten::bernoulli_.Tensor(Tensor(a!) self, Tensor p, *, Generator? generator=None) -> Tensor(a!)
inline Tensor & Tensor::bernoulli_(const Tensor & p, c10::optional<Generator> generator) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(*this, p);
    DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
    DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);
    switch (dispatchKeyToBackend(_dk)) {
        case Backend::CPU:
            return CPUType::bernoulli__Tensor(const_cast<Tensor&>(*this), p, generator);
            break;
        default:
            AT_ERROR("bernoulli_ not implemented for ", at::toString(_dk));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::bernoulli_", "Tensor");
    return op.callUnboxed<Tensor &, Tensor &, const Tensor &, c10::optional<Generator>>(const_cast<Tensor&>(*this), p, generator);
#endif
}

// aten::bernoulli_.float(Tensor(a!) self, float p=0.5, *, Generator? generator=None) -> Tensor(a!)
inline Tensor & Tensor::bernoulli_(double p, c10::optional<Generator> generator) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(*this);
    DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
    DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);
    switch (dispatchKeyToBackend(_dk)) {
        case Backend::CPU:
            return CPUType::bernoulli__float(const_cast<Tensor&>(*this), p, generator);
            break;
        default:
            AT_ERROR("bernoulli_ not implemented for ", at::toString(_dk));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::bernoulli_", "float");
    return op.callUnboxed<Tensor &, Tensor &, double, c10::optional<Generator>>(const_cast<Tensor&>(*this), p, generator);
#endif
}

// aten::bernoulli.p(Tensor self, float p, *, Generator? generator=None) -> Tensor
inline Tensor Tensor::bernoulli(double p, c10::optional<Generator> generator) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::bernoulli_p(const_cast<Tensor&>(*this), p, generator);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::bernoulli", "p");
    return op.callUnboxed<Tensor, const Tensor &, double, c10::optional<Generator>>(const_cast<Tensor&>(*this), p, generator);
#endif
}

// aten::bincount(Tensor self, Tensor? weights=None, int minlength=0) -> Tensor
inline Tensor Tensor::bincount(const Tensor & weights, int64_t minlength) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(*this, weights);
    DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
    DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);
    switch (dispatchKeyToBackend(_dk)) {
        case Backend::CPU:
            return CPUType::bincount(const_cast<Tensor&>(*this), weights, minlength);
            break;
        default:
            AT_ERROR("bincount not implemented for ", at::toString(_dk));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::bincount", "");
    return op.callUnboxed<Tensor, const Tensor &, const Tensor &, int64_t>(const_cast<Tensor&>(*this), weights, minlength);
#endif
}

// aten::bitwise_not(Tensor self) -> Tensor
inline Tensor Tensor::bitwise_not() const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::bitwise_not(const_cast<Tensor&>(*this));
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::bitwise_not", "");
    return op.callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}

// aten::bitwise_not_(Tensor(a!) self) -> Tensor(a!)
inline Tensor & Tensor::bitwise_not_() const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::bitwise_not_(const_cast<Tensor&>(*this));
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::bitwise_not_", "");
    return op.callUnboxed<Tensor &, Tensor &>(const_cast<Tensor&>(*this));
#endif
}

// aten::logical_not(Tensor self) -> Tensor
inline Tensor Tensor::logical_not() const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::logical_not(const_cast<Tensor&>(*this));
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::logical_not", "");
    return op.callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}

// aten::logical_not_(Tensor(a!) self) -> Tensor(a!)
inline Tensor & Tensor::logical_not_() const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::logical_not_(const_cast<Tensor&>(*this));
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::logical_not_", "");
    return op.callUnboxed<Tensor &, Tensor &>(const_cast<Tensor&>(*this));
#endif
}

// aten::logical_xor(Tensor self, Tensor other) -> Tensor
inline Tensor Tensor::logical_xor(const Tensor & other) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::logical_xor(const_cast<Tensor&>(*this), other);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::logical_xor", "");
    return op.callUnboxed<Tensor, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), other);
#endif
}

// aten::logical_xor_(Tensor(a!) self, Tensor other) -> Tensor(a!)
inline Tensor & Tensor::logical_xor_(const Tensor & other) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::logical_xor_(const_cast<Tensor&>(*this), other);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::logical_xor_", "");
    return op.callUnboxed<Tensor &, Tensor &, const Tensor &>(const_cast<Tensor&>(*this), other);
#endif
}

// aten::logical_and(Tensor self, Tensor other) -> Tensor
inline Tensor Tensor::logical_and(const Tensor & other) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::logical_and(const_cast<Tensor&>(*this), other);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::logical_and", "");
    return op.callUnboxed<Tensor, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), other);
#endif
}

// aten::logical_and_(Tensor(a!) self, Tensor other) -> Tensor(a!)
inline Tensor & Tensor::logical_and_(const Tensor & other) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::logical_and_(const_cast<Tensor&>(*this), other);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::logical_and_", "");
    return op.callUnboxed<Tensor &, Tensor &, const Tensor &>(const_cast<Tensor&>(*this), other);
#endif
}

// aten::logical_or(Tensor self, Tensor other) -> Tensor
inline Tensor Tensor::logical_or(const Tensor & other) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::logical_or(const_cast<Tensor&>(*this), other);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::logical_or", "");
    return op.callUnboxed<Tensor, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), other);
#endif
}

// aten::logical_or_(Tensor(a!) self, Tensor other) -> Tensor(a!)
inline Tensor & Tensor::logical_or_(const Tensor & other) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::logical_or_(const_cast<Tensor&>(*this), other);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::logical_or_", "");
    return op.callUnboxed<Tensor &, Tensor &, const Tensor &>(const_cast<Tensor&>(*this), other);
#endif
}

// aten::bmm(Tensor self, Tensor mat2) -> Tensor
inline Tensor Tensor::bmm(const Tensor & mat2) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(*this, mat2);
    DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
    DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);
    switch (dispatchKeyToBackend(_dk)) {
        case Backend::CPU:
            return CPUType::bmm(const_cast<Tensor&>(*this), mat2);
            break;
        default:
            AT_ERROR("bmm not implemented for ", at::toString(_dk));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::bmm", "");
    return op.callUnboxed<Tensor, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), mat2);
#endif
}

// aten::ceil(Tensor self) -> Tensor
inline Tensor Tensor::ceil() const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::ceil(const_cast<Tensor&>(*this));
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::ceil", "");
    return op.callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}

// aten::ceil_(Tensor(a!) self) -> Tensor(a!)
inline Tensor & Tensor::ceil_() const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::ceil_(const_cast<Tensor&>(*this));
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::ceil_", "");
    return op.callUnboxed<Tensor &, Tensor &>(const_cast<Tensor&>(*this));
#endif
}

// aten::chunk(Tensor(a) self, int chunks, int dim=0) -> Tensor(a)[]
inline std::vector<Tensor> Tensor::chunk(int64_t chunks, int64_t dim) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::chunk(const_cast<Tensor&>(*this), chunks, dim);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::chunk", "");
    return op.callUnboxed<std::vector<Tensor>, const Tensor &, int64_t, int64_t>(const_cast<Tensor&>(*this), chunks, dim);
#endif
}

// aten::clamp(Tensor self, Scalar? min=None, Scalar? max=None) -> Tensor
inline Tensor Tensor::clamp(c10::optional<Scalar> min, c10::optional<Scalar> max) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(*this);
    DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
    DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);
    switch (dispatchKeyToBackend(_dk)) {
        case Backend::CPU:
            return CPUType::clamp(const_cast<Tensor&>(*this), min, max);
            break;
        case Backend::QuantizedCPU:
            return QuantizedCPUType::clamp(const_cast<Tensor&>(*this), min, max);
            break;
        default:
            AT_ERROR("clamp not implemented for ", at::toString(_dk));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::clamp", "");
    return op.callUnboxed<Tensor, const Tensor &, c10::optional<Scalar>, c10::optional<Scalar>>(const_cast<Tensor&>(*this), min, max);
#endif
}

// aten::clamp_(Tensor(a!) self, Scalar? min=None, Scalar? max=None) -> Tensor(a!)
inline Tensor & Tensor::clamp_(c10::optional<Scalar> min, c10::optional<Scalar> max) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(*this);
    DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
    DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);
    switch (dispatchKeyToBackend(_dk)) {
        case Backend::CPU:
            return CPUType::clamp_(const_cast<Tensor&>(*this), min, max);
            break;
        default:
            AT_ERROR("clamp_ not implemented for ", at::toString(_dk));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::clamp_", "");
    return op.callUnboxed<Tensor &, Tensor &, c10::optional<Scalar>, c10::optional<Scalar>>(const_cast<Tensor&>(*this), min, max);
#endif
}

// aten::clamp_max(Tensor self, Scalar max) -> Tensor
inline Tensor Tensor::clamp_max(Scalar max) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::clamp_max(const_cast<Tensor&>(*this), max);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::clamp_max", "");
    return op.callUnboxed<Tensor, const Tensor &, Scalar>(const_cast<Tensor&>(*this), max);
#endif
}

// aten::clamp_max_(Tensor(a!) self, Scalar max) -> Tensor(a!)
inline Tensor & Tensor::clamp_max_(Scalar max) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(*this);
    DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
    DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);
    switch (dispatchKeyToBackend(_dk)) {
        case Backend::CPU:
            return CPUType::clamp_max_(const_cast<Tensor&>(*this), max);
            break;
        default:
            AT_ERROR("clamp_max_ not implemented for ", at::toString(_dk));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::clamp_max_", "");
    return op.callUnboxed<Tensor &, Tensor &, Scalar>(const_cast<Tensor&>(*this), max);
#endif
}

// aten::clamp_min(Tensor self, Scalar min) -> Tensor
inline Tensor Tensor::clamp_min(Scalar min) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::clamp_min(const_cast<Tensor&>(*this), min);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::clamp_min", "");
    return op.callUnboxed<Tensor, const Tensor &, Scalar>(const_cast<Tensor&>(*this), min);
#endif
}

// aten::clamp_min_(Tensor(a!) self, Scalar min) -> Tensor(a!)
inline Tensor & Tensor::clamp_min_(Scalar min) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(*this);
    DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
    DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);
    switch (dispatchKeyToBackend(_dk)) {
        case Backend::CPU:
            return CPUType::clamp_min_(const_cast<Tensor&>(*this), min);
            break;
        default:
            AT_ERROR("clamp_min_ not implemented for ", at::toString(_dk));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::clamp_min_", "");
    return op.callUnboxed<Tensor &, Tensor &, Scalar>(const_cast<Tensor&>(*this), min);
#endif
}

// aten::contiguous(Tensor self, *, MemoryFormat memory_format=contiguous_format) -> Tensor
inline Tensor Tensor::contiguous(MemoryFormat memory_format) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::contiguous(const_cast<Tensor&>(*this), memory_format);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::contiguous", "");
    return op.callUnboxed<Tensor, const Tensor &, MemoryFormat>(const_cast<Tensor&>(*this), memory_format);
#endif
}

// aten::copy_(Tensor(a!) self, Tensor src, bool non_blocking=False) -> Tensor(a!)
inline Tensor & Tensor::copy_(const Tensor & src, bool non_blocking) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::copy_(const_cast<Tensor&>(*this), src, non_blocking);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::copy_", "");
    return op.callUnboxed<Tensor &, Tensor &, const Tensor &, bool>(const_cast<Tensor&>(*this), src, non_blocking);
#endif
}

// aten::cos(Tensor self) -> Tensor
inline Tensor Tensor::cos() const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::cos(const_cast<Tensor&>(*this));
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::cos", "");
    return op.callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}

// aten::cos_(Tensor(a!) self) -> Tensor(a!)
inline Tensor & Tensor::cos_() const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::cos_(const_cast<Tensor&>(*this));
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::cos_", "");
    return op.callUnboxed<Tensor &, Tensor &>(const_cast<Tensor&>(*this));
#endif
}

// aten::cosh(Tensor self) -> Tensor
inline Tensor Tensor::cosh() const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::cosh(const_cast<Tensor&>(*this));
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::cosh", "");
    return op.callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}

// aten::cosh_(Tensor(a!) self) -> Tensor(a!)
inline Tensor & Tensor::cosh_() const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::cosh_(const_cast<Tensor&>(*this));
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::cosh_", "");
    return op.callUnboxed<Tensor &, Tensor &>(const_cast<Tensor&>(*this));
#endif
}

// aten::cummax(Tensor self, int dim) -> (Tensor values, Tensor indices)
inline std::tuple<Tensor,Tensor> Tensor::cummax(int64_t dim) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::cummax(const_cast<Tensor&>(*this), dim);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::cummax", "");
    return op.callUnboxed<std::tuple<Tensor,Tensor>, const Tensor &, int64_t>(const_cast<Tensor&>(*this), dim);
#endif
}

// aten::cummax.dimname(Tensor self, Dimname dim) -> (Tensor values, Tensor indices)
inline std::tuple<Tensor,Tensor> Tensor::cummax(Dimname dim) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::cummax_dimname(const_cast<Tensor&>(*this), dim);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::cummax", "dimname");
    return op.callUnboxed<std::tuple<Tensor,Tensor>, const Tensor &, Dimname>(const_cast<Tensor&>(*this), dim);
#endif
}

// aten::cummin(Tensor self, int dim) -> (Tensor values, Tensor indices)
inline std::tuple<Tensor,Tensor> Tensor::cummin(int64_t dim) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::cummin(const_cast<Tensor&>(*this), dim);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::cummin", "");
    return op.callUnboxed<std::tuple<Tensor,Tensor>, const Tensor &, int64_t>(const_cast<Tensor&>(*this), dim);
#endif
}

// aten::cummin.dimname(Tensor self, Dimname dim) -> (Tensor values, Tensor indices)
inline std::tuple<Tensor,Tensor> Tensor::cummin(Dimname dim) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::cummin_dimname(const_cast<Tensor&>(*this), dim);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::cummin", "dimname");
    return op.callUnboxed<std::tuple<Tensor,Tensor>, const Tensor &, Dimname>(const_cast<Tensor&>(*this), dim);
#endif
}

// aten::cumprod(Tensor self, int dim, *, ScalarType? dtype=None) -> Tensor
inline Tensor Tensor::cumprod(int64_t dim, c10::optional<ScalarType> dtype) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::cumprod(const_cast<Tensor&>(*this), dim, dtype);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::cumprod", "");
    return op.callUnboxed<Tensor, const Tensor &, int64_t, c10::optional<ScalarType>>(const_cast<Tensor&>(*this), dim, dtype);
#endif
}

// aten::cumprod.dimname(Tensor self, Dimname dim, *, ScalarType? dtype=None) -> Tensor
inline Tensor Tensor::cumprod(Dimname dim, c10::optional<ScalarType> dtype) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::cumprod_dimname(const_cast<Tensor&>(*this), dim, dtype);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::cumprod", "dimname");
    return op.callUnboxed<Tensor, const Tensor &, Dimname, c10::optional<ScalarType>>(const_cast<Tensor&>(*this), dim, dtype);
#endif
}

// aten::cumsum(Tensor self, int dim, *, ScalarType? dtype=None) -> Tensor
inline Tensor Tensor::cumsum(int64_t dim, c10::optional<ScalarType> dtype) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::cumsum(const_cast<Tensor&>(*this), dim, dtype);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::cumsum", "");
    return op.callUnboxed<Tensor, const Tensor &, int64_t, c10::optional<ScalarType>>(const_cast<Tensor&>(*this), dim, dtype);
#endif
}

// aten::cumsum.dimname(Tensor self, Dimname dim, *, ScalarType? dtype=None) -> Tensor
inline Tensor Tensor::cumsum(Dimname dim, c10::optional<ScalarType> dtype) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::cumsum_dimname(const_cast<Tensor&>(*this), dim, dtype);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::cumsum", "dimname");
    return op.callUnboxed<Tensor, const Tensor &, Dimname, c10::optional<ScalarType>>(const_cast<Tensor&>(*this), dim, dtype);
#endif
}

// aten::det(Tensor self) -> Tensor
inline Tensor Tensor::det() const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::det(const_cast<Tensor&>(*this));
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::det", "");
    return op.callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}

// aten::diag_embed(Tensor self, int offset=0, int dim1=-2, int dim2=-1) -> Tensor
inline Tensor Tensor::diag_embed(int64_t offset, int64_t dim1, int64_t dim2) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::diag_embed(const_cast<Tensor&>(*this), offset, dim1, dim2);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::diag_embed", "");
    return op.callUnboxed<Tensor, const Tensor &, int64_t, int64_t, int64_t>(const_cast<Tensor&>(*this), offset, dim1, dim2);
#endif
}

// aten::diagflat(Tensor self, int offset=0) -> Tensor
inline Tensor Tensor::diagflat(int64_t offset) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::diagflat(const_cast<Tensor&>(*this), offset);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::diagflat", "");
    return op.callUnboxed<Tensor, const Tensor &, int64_t>(const_cast<Tensor&>(*this), offset);
#endif
}

// aten::diagonal(Tensor(a) self, int offset=0, int dim1=0, int dim2=1) -> Tensor(a)
inline Tensor Tensor::diagonal(int64_t offset, int64_t dim1, int64_t dim2) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::diagonal(const_cast<Tensor&>(*this), offset, dim1, dim2);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::diagonal", "");
    return op.callUnboxed<Tensor, const Tensor &, int64_t, int64_t, int64_t>(const_cast<Tensor&>(*this), offset, dim1, dim2);
#endif
}

// aten::diagonal.Dimname(Tensor(a) self, *, Dimname outdim, Dimname dim1, Dimname dim2, int offset=0) -> Tensor(a)
inline Tensor Tensor::diagonal(Dimname outdim, Dimname dim1, Dimname dim2, int64_t offset) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::diagonal_Dimname(const_cast<Tensor&>(*this), outdim, dim1, dim2, offset);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::diagonal", "Dimname");
    return op.callUnboxed<Tensor, const Tensor &, Dimname, Dimname, Dimname, int64_t>(const_cast<Tensor&>(*this), outdim, dim1, dim2, offset);
#endif
}

// aten::fill_diagonal_(Tensor(a!) self, Scalar fill_value, bool wrap=False) -> Tensor(a!)
inline Tensor & Tensor::fill_diagonal_(Scalar fill_value, bool wrap) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::fill_diagonal_(const_cast<Tensor&>(*this), fill_value, wrap);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::fill_diagonal_", "");
    return op.callUnboxed<Tensor &, Tensor &, Scalar, bool>(const_cast<Tensor&>(*this), fill_value, wrap);
#endif
}

// aten::div.Tensor(Tensor self, Tensor other) -> Tensor
inline Tensor Tensor::div(const Tensor & other) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(*this, other);
    DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
    DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);
    switch (dispatchKeyToBackend(_dk)) {
        case Backend::CPU:
            return CPUType::div_Tensor(const_cast<Tensor&>(*this), other);
            break;
        default:
            AT_ERROR("div not implemented for ", at::toString(_dk));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::div", "Tensor");
    return op.callUnboxed<Tensor, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), other);
#endif
}

// aten::div_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
inline Tensor & Tensor::div_(const Tensor & other) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(*this, other);
    DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
    DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);
    switch (dispatchKeyToBackend(_dk)) {
        case Backend::CPU:
            return CPUType::div__Tensor(const_cast<Tensor&>(*this), other);
            break;
        default:
            AT_ERROR("div_ not implemented for ", at::toString(_dk));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::div_", "Tensor");
    return op.callUnboxed<Tensor &, Tensor &, const Tensor &>(const_cast<Tensor&>(*this), other);
#endif
}

// aten::div.Scalar(Tensor self, Scalar other) -> Tensor
inline Tensor Tensor::div(Scalar other) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::div_Scalar(const_cast<Tensor&>(*this), other);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::div", "Scalar");
    return op.callUnboxed<Tensor, const Tensor &, Scalar>(const_cast<Tensor&>(*this), other);
#endif
}

// aten::div_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
inline Tensor & Tensor::div_(Scalar other) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::div__Scalar(const_cast<Tensor&>(*this), other);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::div_", "Scalar");
    return op.callUnboxed<Tensor &, Tensor &, Scalar>(const_cast<Tensor&>(*this), other);
#endif
}

// aten::dot(Tensor self, Tensor tensor) -> Tensor
inline Tensor Tensor::dot(const Tensor & tensor) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(*this, tensor);
    DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
    DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);
    switch (dispatchKeyToBackend(_dk)) {
        case Backend::CPU:
            return CPUType::dot(const_cast<Tensor&>(*this), tensor);
            break;
        default:
            AT_ERROR("dot not implemented for ", at::toString(_dk));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::dot", "");
    return op.callUnboxed<Tensor, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), tensor);
#endif
}

// aten::new_empty(Tensor self, int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
inline Tensor Tensor::new_empty(IntArrayRef size, const TensorOptions & options) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::new_empty(const_cast<Tensor&>(*this), size, options);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::new_empty", "");
    return op.callUnboxed<Tensor, const Tensor &, IntArrayRef, const TensorOptions &>(const_cast<Tensor&>(*this), size, options);
#endif
}

// aten::new_full(Tensor self, int[] size, Scalar fill_value, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
inline Tensor Tensor::new_full(IntArrayRef size, Scalar fill_value, const TensorOptions & options) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::new_full(const_cast<Tensor&>(*this), size, fill_value, options);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::new_full", "");
    return op.callUnboxed<Tensor, const Tensor &, IntArrayRef, Scalar, const TensorOptions &>(const_cast<Tensor&>(*this), size, fill_value, options);
#endif
}

// aten::new_zeros(Tensor self, int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
inline Tensor Tensor::new_zeros(IntArrayRef size, const TensorOptions & options) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::new_zeros(const_cast<Tensor&>(*this), size, options);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::new_zeros", "");
    return op.callUnboxed<Tensor, const Tensor &, IntArrayRef, const TensorOptions &>(const_cast<Tensor&>(*this), size, options);
#endif
}

// aten::resize_(Tensor(a!) self, int[] size, *, MemoryFormat? memory_format=None) -> Tensor(a!)
inline Tensor & Tensor::resize_(IntArrayRef size, c10::optional<MemoryFormat> memory_format) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::resize_(const_cast<Tensor&>(*this), size, memory_format);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::resize_", "");
    return op.callUnboxed<Tensor &, Tensor &, IntArrayRef, c10::optional<MemoryFormat>>(const_cast<Tensor&>(*this), size, memory_format);
#endif
}

// aten::erf(Tensor self) -> Tensor
inline Tensor Tensor::erf() const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::erf(const_cast<Tensor&>(*this));
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::erf", "");
    return op.callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}

// aten::erf_(Tensor(a!) self) -> Tensor(a!)
inline Tensor & Tensor::erf_() const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(*this);
    DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
    DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);
    switch (dispatchKeyToBackend(_dk)) {
        case Backend::CPU:
            return CPUType::erf_(const_cast<Tensor&>(*this));
            break;
        default:
            AT_ERROR("erf_ not implemented for ", at::toString(_dk));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::erf_", "");
    return op.callUnboxed<Tensor &, Tensor &>(const_cast<Tensor&>(*this));
#endif
}

// aten::erfc(Tensor self) -> Tensor
inline Tensor Tensor::erfc() const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::erfc(const_cast<Tensor&>(*this));
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::erfc", "");
    return op.callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}

// aten::erfc_(Tensor(a!) self) -> Tensor(a!)
inline Tensor & Tensor::erfc_() const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(*this);
    DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
    DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);
    switch (dispatchKeyToBackend(_dk)) {
        case Backend::CPU:
            return CPUType::erfc_(const_cast<Tensor&>(*this));
            break;
        default:
            AT_ERROR("erfc_ not implemented for ", at::toString(_dk));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::erfc_", "");
    return op.callUnboxed<Tensor &, Tensor &>(const_cast<Tensor&>(*this));
#endif
}

// aten::exp(Tensor self) -> Tensor
inline Tensor Tensor::exp() const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::exp(const_cast<Tensor&>(*this));
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::exp", "");
    return op.callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}

// aten::exp_(Tensor(a!) self) -> Tensor(a!)
inline Tensor & Tensor::exp_() const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(*this);
    DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
    DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);
    switch (dispatchKeyToBackend(_dk)) {
        case Backend::CPU:
            return CPUType::exp_(const_cast<Tensor&>(*this));
            break;
        default:
            AT_ERROR("exp_ not implemented for ", at::toString(_dk));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::exp_", "");
    return op.callUnboxed<Tensor &, Tensor &>(const_cast<Tensor&>(*this));
#endif
}

// aten::expm1(Tensor self) -> Tensor
inline Tensor Tensor::expm1() const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::expm1(const_cast<Tensor&>(*this));
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::expm1", "");
    return op.callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}

// aten::expm1_(Tensor(a!) self) -> Tensor(a!)
inline Tensor & Tensor::expm1_() const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::expm1_(const_cast<Tensor&>(*this));
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::expm1_", "");
    return op.callUnboxed<Tensor &, Tensor &>(const_cast<Tensor&>(*this));
#endif
}

// aten::expand(Tensor(a) self, int[] size, *, bool implicit=False) -> Tensor(a)
inline Tensor Tensor::expand(IntArrayRef size, bool implicit) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::expand(const_cast<Tensor&>(*this), size, implicit);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::expand", "");
    return op.callUnboxed<Tensor, const Tensor &, IntArrayRef, bool>(const_cast<Tensor&>(*this), size, implicit);
#endif
}

// aten::expand_as(Tensor self, Tensor other) -> Tensor
inline Tensor Tensor::expand_as(const Tensor & other) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::expand_as(const_cast<Tensor&>(*this), other);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::expand_as", "");
    return op.callUnboxed<Tensor, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), other);
#endif
}

// aten::flatten.using_ints(Tensor self, int start_dim=0, int end_dim=-1) -> Tensor
inline Tensor Tensor::flatten(int64_t start_dim, int64_t end_dim) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::flatten_using_ints(const_cast<Tensor&>(*this), start_dim, end_dim);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::flatten", "using_ints");
    return op.callUnboxed<Tensor, const Tensor &, int64_t, int64_t>(const_cast<Tensor&>(*this), start_dim, end_dim);
#endif
}

// aten::flatten.named_out_dim(Tensor self, int start_dim, int end_dim, Dimname out_dim) -> Tensor
inline Tensor Tensor::flatten(int64_t start_dim, int64_t end_dim, Dimname out_dim) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::flatten_named_out_dim(const_cast<Tensor&>(*this), start_dim, end_dim, out_dim);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::flatten", "named_out_dim");
    return op.callUnboxed<Tensor, const Tensor &, int64_t, int64_t, Dimname>(const_cast<Tensor&>(*this), start_dim, end_dim, out_dim);
#endif
}

// aten::flatten.using_names(Tensor self, Dimname start_dim, Dimname end_dim, Dimname out_dim) -> Tensor
inline Tensor Tensor::flatten(Dimname start_dim, Dimname end_dim, Dimname out_dim) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::flatten_using_names(const_cast<Tensor&>(*this), start_dim, end_dim, out_dim);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::flatten", "using_names");
    return op.callUnboxed<Tensor, const Tensor &, Dimname, Dimname, Dimname>(const_cast<Tensor&>(*this), start_dim, end_dim, out_dim);
#endif
}

// aten::flatten.DimnameList(Tensor self, Dimname[] dims, Dimname out_dim) -> Tensor
inline Tensor Tensor::flatten(DimnameList dims, Dimname out_dim) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::flatten_DimnameList(const_cast<Tensor&>(*this), dims, out_dim);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::flatten", "DimnameList");
    return op.callUnboxed<Tensor, const Tensor &, DimnameList, Dimname>(const_cast<Tensor&>(*this), dims, out_dim);
#endif
}

// aten::fill_.Scalar(Tensor(a!) self, Scalar value) -> Tensor(a!)
inline Tensor & Tensor::fill_(Scalar value) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::fill__Scalar(const_cast<Tensor&>(*this), value);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::fill_", "Scalar");
    return op.callUnboxed<Tensor &, Tensor &, Scalar>(const_cast<Tensor&>(*this), value);
#endif
}

// aten::fill_.Tensor(Tensor(a!) self, Tensor value) -> Tensor(a!)
inline Tensor & Tensor::fill_(const Tensor & value) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::fill__Tensor(const_cast<Tensor&>(*this), value);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::fill_", "Tensor");
    return op.callUnboxed<Tensor &, Tensor &, const Tensor &>(const_cast<Tensor&>(*this), value);
#endif
}

// aten::floor(Tensor self) -> Tensor
inline Tensor Tensor::floor() const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::floor(const_cast<Tensor&>(*this));
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::floor", "");
    return op.callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}

// aten::floor_(Tensor(a!) self) -> Tensor(a!)
inline Tensor & Tensor::floor_() const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::floor_(const_cast<Tensor&>(*this));
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::floor_", "");
    return op.callUnboxed<Tensor &, Tensor &>(const_cast<Tensor&>(*this));
#endif
}

// aten::floor_divide(Tensor self, Tensor other) -> Tensor
inline Tensor Tensor::floor_divide(const Tensor & other) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(*this, other);
    DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
    DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);
    switch (dispatchKeyToBackend(_dk)) {
        case Backend::CPU:
            return CPUType::floor_divide(const_cast<Tensor&>(*this), other);
            break;
        default:
            AT_ERROR("floor_divide not implemented for ", at::toString(_dk));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::floor_divide", "");
    return op.callUnboxed<Tensor, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), other);
#endif
}

// aten::floor_divide_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
inline Tensor & Tensor::floor_divide_(const Tensor & other) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(*this, other);
    DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
    DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);
    switch (dispatchKeyToBackend(_dk)) {
        case Backend::CPU:
            return CPUType::floor_divide__Tensor(const_cast<Tensor&>(*this), other);
            break;
        default:
            AT_ERROR("floor_divide_ not implemented for ", at::toString(_dk));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::floor_divide_", "Tensor");
    return op.callUnboxed<Tensor &, Tensor &, const Tensor &>(const_cast<Tensor&>(*this), other);
#endif
}

// aten::floor_divide.Scalar(Tensor self, Scalar other) -> Tensor
inline Tensor Tensor::floor_divide(Scalar other) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::floor_divide_Scalar(const_cast<Tensor&>(*this), other);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::floor_divide", "Scalar");
    return op.callUnboxed<Tensor, const Tensor &, Scalar>(const_cast<Tensor&>(*this), other);
#endif
}

// aten::floor_divide_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
inline Tensor & Tensor::floor_divide_(Scalar other) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::floor_divide__Scalar(const_cast<Tensor&>(*this), other);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::floor_divide_", "Scalar");
    return op.callUnboxed<Tensor &, Tensor &, Scalar>(const_cast<Tensor&>(*this), other);
#endif
}

// aten::frac(Tensor self) -> Tensor
inline Tensor Tensor::frac() const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::frac(const_cast<Tensor&>(*this));
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::frac", "");
    return op.callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}

// aten::frac_(Tensor(a!) self) -> Tensor(a!)
inline Tensor & Tensor::frac_() const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::frac_(const_cast<Tensor&>(*this));
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::frac_", "");
    return op.callUnboxed<Tensor &, Tensor &>(const_cast<Tensor&>(*this));
#endif
}

// aten::ger(Tensor self, Tensor vec2) -> Tensor
inline Tensor Tensor::ger(const Tensor & vec2) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::ger(const_cast<Tensor&>(*this), vec2);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::ger", "");
    return op.callUnboxed<Tensor, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), vec2);
#endif
}

// aten::fft(Tensor self, int signal_ndim, bool normalized=False) -> Tensor
inline Tensor Tensor::fft(int64_t signal_ndim, bool normalized) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::fft(const_cast<Tensor&>(*this), signal_ndim, normalized);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::fft", "");
    return op.callUnboxed<Tensor, const Tensor &, int64_t, bool>(const_cast<Tensor&>(*this), signal_ndim, normalized);
#endif
}

// aten::ifft(Tensor self, int signal_ndim, bool normalized=False) -> Tensor
inline Tensor Tensor::ifft(int64_t signal_ndim, bool normalized) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::ifft(const_cast<Tensor&>(*this), signal_ndim, normalized);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::ifft", "");
    return op.callUnboxed<Tensor, const Tensor &, int64_t, bool>(const_cast<Tensor&>(*this), signal_ndim, normalized);
#endif
}

// aten::rfft(Tensor self, int signal_ndim, bool normalized=False, bool onesided=True) -> Tensor
inline Tensor Tensor::rfft(int64_t signal_ndim, bool normalized, bool onesided) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::rfft(const_cast<Tensor&>(*this), signal_ndim, normalized, onesided);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::rfft", "");
    return op.callUnboxed<Tensor, const Tensor &, int64_t, bool, bool>(const_cast<Tensor&>(*this), signal_ndim, normalized, onesided);
#endif
}

// aten::irfft(Tensor self, int signal_ndim, bool normalized=False, bool onesided=True, int[] signal_sizes=[]) -> Tensor
inline Tensor Tensor::irfft(int64_t signal_ndim, bool normalized, bool onesided, IntArrayRef signal_sizes) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::irfft(const_cast<Tensor&>(*this), signal_ndim, normalized, onesided, signal_sizes);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::irfft", "");
    return op.callUnboxed<Tensor, const Tensor &, int64_t, bool, bool, IntArrayRef>(const_cast<Tensor&>(*this), signal_ndim, normalized, onesided, signal_sizes);
#endif
}

// aten::index.Tensor(Tensor self, Tensor?[] indices) -> Tensor
inline Tensor Tensor::index(TensorList indices) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::index_Tensor(const_cast<Tensor&>(*this), indices);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::index", "Tensor");
    return op.callUnboxed<Tensor, const Tensor &, TensorList>(const_cast<Tensor&>(*this), indices);
#endif
}

// aten::index_copy_(Tensor(a!) self, int dim, Tensor index, Tensor source) -> Tensor(a!)
inline Tensor & Tensor::index_copy_(int64_t dim, const Tensor & index, const Tensor & source) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::index_copy_(const_cast<Tensor&>(*this), dim, index, source);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::index_copy_", "");
    return op.callUnboxed<Tensor &, Tensor &, int64_t, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), dim, index, source);
#endif
}

// aten::index_copy(Tensor self, int dim, Tensor index, Tensor source) -> Tensor
inline Tensor Tensor::index_copy(int64_t dim, const Tensor & index, const Tensor & source) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::index_copy(const_cast<Tensor&>(*this), dim, index, source);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::index_copy", "");
    return op.callUnboxed<Tensor, const Tensor &, int64_t, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), dim, index, source);
#endif
}

// aten::index_copy_.dimname(Tensor(a!) self, Dimname dim, Tensor index, Tensor source) -> Tensor(a!)
inline Tensor & Tensor::index_copy_(Dimname dim, const Tensor & index, const Tensor & source) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::index_copy__dimname(const_cast<Tensor&>(*this), dim, index, source);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::index_copy_", "dimname");
    return op.callUnboxed<Tensor &, Tensor &, Dimname, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), dim, index, source);
#endif
}

// aten::index_copy.dimname(Tensor self, Dimname dim, Tensor index, Tensor source) -> Tensor
inline Tensor Tensor::index_copy(Dimname dim, const Tensor & index, const Tensor & source) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::index_copy_dimname(const_cast<Tensor&>(*this), dim, index, source);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::index_copy", "dimname");
    return op.callUnboxed<Tensor, const Tensor &, Dimname, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), dim, index, source);
#endif
}

// aten::index_put_(Tensor(a!) self, Tensor?[] indices, Tensor values, bool accumulate=False) -> Tensor(a!)
inline Tensor & Tensor::index_put_(TensorList indices, const Tensor & values, bool accumulate) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::index_put_(const_cast<Tensor&>(*this), indices, values, accumulate);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::index_put_", "");
    return op.callUnboxed<Tensor &, Tensor &, TensorList, const Tensor &, bool>(const_cast<Tensor&>(*this), indices, values, accumulate);
#endif
}

// aten::index_put(Tensor self, Tensor?[] indices, Tensor values, bool accumulate=False) -> Tensor
inline Tensor Tensor::index_put(TensorList indices, const Tensor & values, bool accumulate) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::index_put(const_cast<Tensor&>(*this), indices, values, accumulate);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::index_put", "");
    return op.callUnboxed<Tensor, const Tensor &, TensorList, const Tensor &, bool>(const_cast<Tensor&>(*this), indices, values, accumulate);
#endif
}

// aten::inverse(Tensor self) -> Tensor
inline Tensor Tensor::inverse() const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::inverse(const_cast<Tensor&>(*this));
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::inverse", "");
    return op.callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}

// aten::isclose(Tensor self, Tensor other, float rtol=1e-05, float atol=1e-08, bool equal_nan=False) -> Tensor
inline Tensor Tensor::isclose(const Tensor & other, double rtol, double atol, bool equal_nan) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::isclose(const_cast<Tensor&>(*this), other, rtol, atol, equal_nan);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::isclose", "");
    return op.callUnboxed<Tensor, const Tensor &, const Tensor &, double, double, bool>(const_cast<Tensor&>(*this), other, rtol, atol, equal_nan);
#endif
}

// aten::is_distributed(Tensor self) -> bool
inline bool Tensor::is_distributed() const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::is_distributed(const_cast<Tensor&>(*this));
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::is_distributed", "");
    return op.callUnboxed<bool, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}

// aten::is_floating_point(Tensor self) -> bool
inline bool Tensor::is_floating_point() const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::is_floating_point(const_cast<Tensor&>(*this));
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::is_floating_point", "");
    return op.callUnboxed<bool, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}

// aten::is_complex(Tensor self) -> bool
inline bool Tensor::is_complex() const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::is_complex(const_cast<Tensor&>(*this));
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::is_complex", "");
    return op.callUnboxed<bool, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}

// aten::is_nonzero(Tensor self) -> bool
inline bool Tensor::is_nonzero() const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::is_nonzero(const_cast<Tensor&>(*this));
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::is_nonzero", "");
    return op.callUnboxed<bool, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}

// aten::is_same_size(Tensor self, Tensor other) -> bool
inline bool Tensor::is_same_size(const Tensor & other) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::is_same_size(const_cast<Tensor&>(*this), other);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::is_same_size", "");
    return op.callUnboxed<bool, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), other);
#endif
}

// aten::is_signed(Tensor self) -> bool
inline bool Tensor::is_signed() const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::is_signed(const_cast<Tensor&>(*this));
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::is_signed", "");
    return op.callUnboxed<bool, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}

// aten::kthvalue(Tensor self, int k, int dim=-1, bool keepdim=False) -> (Tensor values, Tensor indices)
inline std::tuple<Tensor,Tensor> Tensor::kthvalue(int64_t k, int64_t dim, bool keepdim) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::kthvalue(const_cast<Tensor&>(*this), k, dim, keepdim);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::kthvalue", "");
    return op.callUnboxed<std::tuple<Tensor,Tensor>, const Tensor &, int64_t, int64_t, bool>(const_cast<Tensor&>(*this), k, dim, keepdim);
#endif
}

// aten::kthvalue.dimname(Tensor self, int k, Dimname dim, bool keepdim=False) -> (Tensor values, Tensor indices)
inline std::tuple<Tensor,Tensor> Tensor::kthvalue(int64_t k, Dimname dim, bool keepdim) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::kthvalue_dimname(const_cast<Tensor&>(*this), k, dim, keepdim);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::kthvalue", "dimname");
    return op.callUnboxed<std::tuple<Tensor,Tensor>, const Tensor &, int64_t, Dimname, bool>(const_cast<Tensor&>(*this), k, dim, keepdim);
#endif
}

// aten::log(Tensor self) -> Tensor
inline Tensor Tensor::log() const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::log(const_cast<Tensor&>(*this));
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::log", "");
    return op.callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}

// aten::log_(Tensor(a!) self) -> Tensor(a!)
inline Tensor & Tensor::log_() const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::log_(const_cast<Tensor&>(*this));
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::log_", "");
    return op.callUnboxed<Tensor &, Tensor &>(const_cast<Tensor&>(*this));
#endif
}

// aten::log10(Tensor self) -> Tensor
inline Tensor Tensor::log10() const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::log10(const_cast<Tensor&>(*this));
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::log10", "");
    return op.callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}

// aten::log10_(Tensor(a!) self) -> Tensor(a!)
inline Tensor & Tensor::log10_() const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::log10_(const_cast<Tensor&>(*this));
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::log10_", "");
    return op.callUnboxed<Tensor &, Tensor &>(const_cast<Tensor&>(*this));
#endif
}

// aten::log1p(Tensor self) -> Tensor
inline Tensor Tensor::log1p() const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::log1p(const_cast<Tensor&>(*this));
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::log1p", "");
    return op.callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}

// aten::log1p_(Tensor(a!) self) -> Tensor(a!)
inline Tensor & Tensor::log1p_() const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(*this);
    DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
    DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);
    switch (dispatchKeyToBackend(_dk)) {
        case Backend::CPU:
            return CPUType::log1p_(const_cast<Tensor&>(*this));
            break;
        default:
            AT_ERROR("log1p_ not implemented for ", at::toString(_dk));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::log1p_", "");
    return op.callUnboxed<Tensor &, Tensor &>(const_cast<Tensor&>(*this));
#endif
}

// aten::log2(Tensor self) -> Tensor
inline Tensor Tensor::log2() const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::log2(const_cast<Tensor&>(*this));
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::log2", "");
    return op.callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}

// aten::log2_(Tensor(a!) self) -> Tensor(a!)
inline Tensor & Tensor::log2_() const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::log2_(const_cast<Tensor&>(*this));
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::log2_", "");
    return op.callUnboxed<Tensor &, Tensor &>(const_cast<Tensor&>(*this));
#endif
}

// aten::logdet(Tensor self) -> Tensor
inline Tensor Tensor::logdet() const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::logdet(const_cast<Tensor&>(*this));
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::logdet", "");
    return op.callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}

// aten::log_softmax.int(Tensor self, int dim, ScalarType? dtype=None) -> Tensor
inline Tensor Tensor::log_softmax(int64_t dim, c10::optional<ScalarType> dtype) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::log_softmax_int(const_cast<Tensor&>(*this), dim, dtype);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::log_softmax", "int");
    return op.callUnboxed<Tensor, const Tensor &, int64_t, c10::optional<ScalarType>>(const_cast<Tensor&>(*this), dim, dtype);
#endif
}

// aten::log_softmax.Dimname(Tensor self, Dimname dim, *, ScalarType? dtype=None) -> Tensor
inline Tensor Tensor::log_softmax(Dimname dim, c10::optional<ScalarType> dtype) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::log_softmax_Dimname(const_cast<Tensor&>(*this), dim, dtype);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::log_softmax", "Dimname");
    return op.callUnboxed<Tensor, const Tensor &, Dimname, c10::optional<ScalarType>>(const_cast<Tensor&>(*this), dim, dtype);
#endif
}

// aten::logsumexp(Tensor self, int[1] dim, bool keepdim=False) -> Tensor
inline Tensor Tensor::logsumexp(IntArrayRef dim, bool keepdim) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::logsumexp(const_cast<Tensor&>(*this), dim, keepdim);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::logsumexp", "");
    return op.callUnboxed<Tensor, const Tensor &, IntArrayRef, bool>(const_cast<Tensor&>(*this), dim, keepdim);
#endif
}

// aten::logsumexp.names(Tensor self, Dimname[1] dim, bool keepdim=False) -> Tensor
inline Tensor Tensor::logsumexp(DimnameList dim, bool keepdim) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::logsumexp_names(const_cast<Tensor&>(*this), dim, keepdim);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::logsumexp", "names");
    return op.callUnboxed<Tensor, const Tensor &, DimnameList, bool>(const_cast<Tensor&>(*this), dim, keepdim);
#endif
}

// aten::matmul(Tensor self, Tensor other) -> Tensor
inline Tensor Tensor::matmul(const Tensor & other) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::matmul(const_cast<Tensor&>(*this), other);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::matmul", "");
    return op.callUnboxed<Tensor, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), other);
#endif
}

// aten::matrix_power(Tensor self, int n) -> Tensor
inline Tensor Tensor::matrix_power(int64_t n) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::matrix_power(const_cast<Tensor&>(*this), n);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::matrix_power", "");
    return op.callUnboxed<Tensor, const Tensor &, int64_t>(const_cast<Tensor&>(*this), n);
#endif
}

// aten::max.dim(Tensor self, int dim, bool keepdim=False) -> (Tensor values, Tensor indices)
inline std::tuple<Tensor,Tensor> Tensor::max(int64_t dim, bool keepdim) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::max_dim(const_cast<Tensor&>(*this), dim, keepdim);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::max", "dim");
    return op.callUnboxed<std::tuple<Tensor,Tensor>, const Tensor &, int64_t, bool>(const_cast<Tensor&>(*this), dim, keepdim);
#endif
}

// aten::max_values(Tensor self, int[1] dim, bool keepdim=False) -> Tensor
inline Tensor Tensor::max_values(IntArrayRef dim, bool keepdim) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::max_values(const_cast<Tensor&>(*this), dim, keepdim);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::max_values", "");
    return op.callUnboxed<Tensor, const Tensor &, IntArrayRef, bool>(const_cast<Tensor&>(*this), dim, keepdim);
#endif
}

// aten::max.names_dim(Tensor self, Dimname dim, bool keepdim=False) -> (Tensor values, Tensor indices)
inline std::tuple<Tensor,Tensor> Tensor::max(Dimname dim, bool keepdim) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::max_names_dim(const_cast<Tensor&>(*this), dim, keepdim);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::max", "names_dim");
    return op.callUnboxed<std::tuple<Tensor,Tensor>, const Tensor &, Dimname, bool>(const_cast<Tensor&>(*this), dim, keepdim);
#endif
}

// aten::max_values.names(Tensor self, Dimname[1] dim, bool keepdim=False) -> Tensor
inline Tensor Tensor::max_values(DimnameList dim, bool keepdim) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::max_values_names(const_cast<Tensor&>(*this), dim, keepdim);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::max_values", "names");
    return op.callUnboxed<Tensor, const Tensor &, DimnameList, bool>(const_cast<Tensor&>(*this), dim, keepdim);
#endif
}

// aten::mean(Tensor self, *, ScalarType? dtype=None) -> Tensor
inline Tensor Tensor::mean(c10::optional<ScalarType> dtype) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(*this);
    DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
    DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);
    switch (dispatchKeyToBackend(_dk)) {
        case Backend::CPU:
            return CPUType::mean(const_cast<Tensor&>(*this), dtype);
            break;
        case Backend::QuantizedCPU:
            return QuantizedCPUType::mean(const_cast<Tensor&>(*this), dtype);
            break;
        default:
            AT_ERROR("mean not implemented for ", at::toString(_dk));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::mean", "");
    return op.callUnboxed<Tensor, const Tensor &, c10::optional<ScalarType>>(const_cast<Tensor&>(*this), dtype);
#endif
}

// aten::mean.dim(Tensor self, int[1] dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
inline Tensor Tensor::mean(IntArrayRef dim, bool keepdim, c10::optional<ScalarType> dtype) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(*this);
    DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
    DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);
    switch (dispatchKeyToBackend(_dk)) {
        case Backend::CPU:
            return CPUType::mean_dim(const_cast<Tensor&>(*this), dim, keepdim, dtype);
            break;
        case Backend::QuantizedCPU:
            return QuantizedCPUType::mean_dim(const_cast<Tensor&>(*this), dim, keepdim, dtype);
            break;
        default:
            AT_ERROR("mean not implemented for ", at::toString(_dk));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::mean", "dim");
    return op.callUnboxed<Tensor, const Tensor &, IntArrayRef, bool, c10::optional<ScalarType>>(const_cast<Tensor&>(*this), dim, keepdim, dtype);
#endif
}

// aten::mean.names_dim(Tensor self, Dimname[1] dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
inline Tensor Tensor::mean(DimnameList dim, bool keepdim, c10::optional<ScalarType> dtype) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::mean_names_dim(const_cast<Tensor&>(*this), dim, keepdim, dtype);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::mean", "names_dim");
    return op.callUnboxed<Tensor, const Tensor &, DimnameList, bool, c10::optional<ScalarType>>(const_cast<Tensor&>(*this), dim, keepdim, dtype);
#endif
}

// aten::median.dim(Tensor self, int dim, bool keepdim=False) -> (Tensor values, Tensor indices)
inline std::tuple<Tensor,Tensor> Tensor::median(int64_t dim, bool keepdim) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::median_dim(const_cast<Tensor&>(*this), dim, keepdim);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::median", "dim");
    return op.callUnboxed<std::tuple<Tensor,Tensor>, const Tensor &, int64_t, bool>(const_cast<Tensor&>(*this), dim, keepdim);
#endif
}

// aten::median.names_dim(Tensor self, Dimname dim, bool keepdim=False) -> (Tensor values, Tensor indices)
inline std::tuple<Tensor,Tensor> Tensor::median(Dimname dim, bool keepdim) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::median_names_dim(const_cast<Tensor&>(*this), dim, keepdim);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::median", "names_dim");
    return op.callUnboxed<std::tuple<Tensor,Tensor>, const Tensor &, Dimname, bool>(const_cast<Tensor&>(*this), dim, keepdim);
#endif
}

// aten::min.dim(Tensor self, int dim, bool keepdim=False) -> (Tensor values, Tensor indices)
inline std::tuple<Tensor,Tensor> Tensor::min(int64_t dim, bool keepdim) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::min_dim(const_cast<Tensor&>(*this), dim, keepdim);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::min", "dim");
    return op.callUnboxed<std::tuple<Tensor,Tensor>, const Tensor &, int64_t, bool>(const_cast<Tensor&>(*this), dim, keepdim);
#endif
}

// aten::min_values(Tensor self, int[1] dim, bool keepdim=False) -> Tensor
inline Tensor Tensor::min_values(IntArrayRef dim, bool keepdim) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::min_values(const_cast<Tensor&>(*this), dim, keepdim);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::min_values", "");
    return op.callUnboxed<Tensor, const Tensor &, IntArrayRef, bool>(const_cast<Tensor&>(*this), dim, keepdim);
#endif
}

// aten::min.names_dim(Tensor self, Dimname dim, bool keepdim=False) -> (Tensor values, Tensor indices)
inline std::tuple<Tensor,Tensor> Tensor::min(Dimname dim, bool keepdim) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::min_names_dim(const_cast<Tensor&>(*this), dim, keepdim);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::min", "names_dim");
    return op.callUnboxed<std::tuple<Tensor,Tensor>, const Tensor &, Dimname, bool>(const_cast<Tensor&>(*this), dim, keepdim);
#endif
}

// aten::min_values.names(Tensor self, Dimname[1] dim, bool keepdim=False) -> Tensor
inline Tensor Tensor::min_values(DimnameList dim, bool keepdim) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::min_values_names(const_cast<Tensor&>(*this), dim, keepdim);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::min_values", "names");
    return op.callUnboxed<Tensor, const Tensor &, DimnameList, bool>(const_cast<Tensor&>(*this), dim, keepdim);
#endif
}

// aten::mm(Tensor self, Tensor mat2) -> Tensor
inline Tensor Tensor::mm(const Tensor & mat2) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(*this, mat2);
    DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
    DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);
    switch (dispatchKeyToBackend(_dk)) {
        case Backend::CPU:
            return CPUType::mm(const_cast<Tensor&>(*this), mat2);
            break;
        default:
            AT_ERROR("mm not implemented for ", at::toString(_dk));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::mm", "");
    return op.callUnboxed<Tensor, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), mat2);
#endif
}

// aten::mode(Tensor self, int dim=-1, bool keepdim=False) -> (Tensor values, Tensor indices)
inline std::tuple<Tensor,Tensor> Tensor::mode(int64_t dim, bool keepdim) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::mode(const_cast<Tensor&>(*this), dim, keepdim);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::mode", "");
    return op.callUnboxed<std::tuple<Tensor,Tensor>, const Tensor &, int64_t, bool>(const_cast<Tensor&>(*this), dim, keepdim);
#endif
}

// aten::mode.dimname(Tensor self, Dimname dim, bool keepdim=False) -> (Tensor values, Tensor indices)
inline std::tuple<Tensor,Tensor> Tensor::mode(Dimname dim, bool keepdim) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::mode_dimname(const_cast<Tensor&>(*this), dim, keepdim);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::mode", "dimname");
    return op.callUnboxed<std::tuple<Tensor,Tensor>, const Tensor &, Dimname, bool>(const_cast<Tensor&>(*this), dim, keepdim);
#endif
}

// aten::mul.Tensor(Tensor self, Tensor other) -> Tensor
inline Tensor Tensor::mul(const Tensor & other) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(*this, other);
    DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
    DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);
    switch (dispatchKeyToBackend(_dk)) {
        case Backend::CPU:
            return CPUType::mul_Tensor(const_cast<Tensor&>(*this), other);
            break;
        default:
            AT_ERROR("mul not implemented for ", at::toString(_dk));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::mul", "Tensor");
    return op.callUnboxed<Tensor, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), other);
#endif
}

// aten::mul_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
inline Tensor & Tensor::mul_(const Tensor & other) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(*this, other);
    DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
    DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);
    switch (dispatchKeyToBackend(_dk)) {
        case Backend::CPU:
            return CPUType::mul__Tensor(const_cast<Tensor&>(*this), other);
            break;
        default:
            AT_ERROR("mul_ not implemented for ", at::toString(_dk));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::mul_", "Tensor");
    return op.callUnboxed<Tensor &, Tensor &, const Tensor &>(const_cast<Tensor&>(*this), other);
#endif
}

// aten::mul.Scalar(Tensor self, Scalar other) -> Tensor
inline Tensor Tensor::mul(Scalar other) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::mul_Scalar(const_cast<Tensor&>(*this), other);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::mul", "Scalar");
    return op.callUnboxed<Tensor, const Tensor &, Scalar>(const_cast<Tensor&>(*this), other);
#endif
}

// aten::mul_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
inline Tensor & Tensor::mul_(Scalar other) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::mul__Scalar(const_cast<Tensor&>(*this), other);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::mul_", "Scalar");
    return op.callUnboxed<Tensor &, Tensor &, Scalar>(const_cast<Tensor&>(*this), other);
#endif
}

// aten::mv(Tensor self, Tensor vec) -> Tensor
inline Tensor Tensor::mv(const Tensor & vec) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(*this, vec);
    DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
    DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);
    switch (dispatchKeyToBackend(_dk)) {
        case Backend::CPU:
            return CPUType::mv(const_cast<Tensor&>(*this), vec);
            break;
        default:
            AT_ERROR("mv not implemented for ", at::toString(_dk));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::mv", "");
    return op.callUnboxed<Tensor, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), vec);
#endif
}

// aten::mvlgamma(Tensor self, int p) -> Tensor
inline Tensor Tensor::mvlgamma(int64_t p) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::mvlgamma(const_cast<Tensor&>(*this), p);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::mvlgamma", "");
    return op.callUnboxed<Tensor, const Tensor &, int64_t>(const_cast<Tensor&>(*this), p);
#endif
}

// aten::mvlgamma_(Tensor(a!) self, int p) -> Tensor(a!)
inline Tensor & Tensor::mvlgamma_(int64_t p) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::mvlgamma_(const_cast<Tensor&>(*this), p);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::mvlgamma_", "");
    return op.callUnboxed<Tensor &, Tensor &, int64_t>(const_cast<Tensor&>(*this), p);
#endif
}

// aten::narrow_copy(Tensor self, int dim, int start, int length) -> Tensor
inline Tensor Tensor::narrow_copy(int64_t dim, int64_t start, int64_t length) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(*this);
    DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
    DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);
    switch (dispatchKeyToBackend(_dk)) {
        case Backend::CPU:
            return CPUType::narrow_copy(const_cast<Tensor&>(*this), dim, start, length);
            break;
        default:
            AT_ERROR("narrow_copy not implemented for ", at::toString(_dk));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::narrow_copy", "");
    return op.callUnboxed<Tensor, const Tensor &, int64_t, int64_t, int64_t>(const_cast<Tensor&>(*this), dim, start, length);
#endif
}

// aten::narrow(Tensor(a) self, int dim, int start, int length) -> Tensor(a)
inline Tensor Tensor::narrow(int64_t dim, int64_t start, int64_t length) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::narrow(const_cast<Tensor&>(*this), dim, start, length);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::narrow", "");
    return op.callUnboxed<Tensor, const Tensor &, int64_t, int64_t, int64_t>(const_cast<Tensor&>(*this), dim, start, length);
#endif
}

// aten::narrow.Tensor(Tensor(a) self, int dim, Tensor start, int length) -> Tensor(a)
inline Tensor Tensor::narrow(int64_t dim, const Tensor & start, int64_t length) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::narrow_Tensor(const_cast<Tensor&>(*this), dim, start, length);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::narrow", "Tensor");
    return op.callUnboxed<Tensor, const Tensor &, int64_t, const Tensor &, int64_t>(const_cast<Tensor&>(*this), dim, start, length);
#endif
}

// aten::permute(Tensor(a) self, int[] dims) -> Tensor(a)
inline Tensor Tensor::permute(IntArrayRef dims) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::permute(const_cast<Tensor&>(*this), dims);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::permute", "");
    return op.callUnboxed<Tensor, const Tensor &, IntArrayRef>(const_cast<Tensor&>(*this), dims);
#endif
}

// aten::numpy_T(Tensor(a) self) -> Tensor(a)
inline Tensor Tensor::numpy_T() const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::numpy_T(const_cast<Tensor&>(*this));
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::numpy_T", "");
    return op.callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}

// aten::is_pinned(Tensor self) -> bool
inline bool Tensor::is_pinned() const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::is_pinned(const_cast<Tensor&>(*this));
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::is_pinned", "");
    return op.callUnboxed<bool, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}

// aten::pin_memory(Tensor self) -> Tensor
inline Tensor Tensor::pin_memory() const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::pin_memory(const_cast<Tensor&>(*this));
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::pin_memory", "");
    return op.callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}

// aten::pinverse(Tensor self, float rcond=1e-15) -> Tensor
inline Tensor Tensor::pinverse(double rcond) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::pinverse(const_cast<Tensor&>(*this), rcond);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::pinverse", "");
    return op.callUnboxed<Tensor, const Tensor &, double>(const_cast<Tensor&>(*this), rcond);
#endif
}

// aten::reciprocal(Tensor self) -> Tensor
inline Tensor Tensor::reciprocal() const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::reciprocal(const_cast<Tensor&>(*this));
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::reciprocal", "");
    return op.callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}

// aten::reciprocal_(Tensor(a!) self) -> Tensor(a!)
inline Tensor & Tensor::reciprocal_() const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::reciprocal_(const_cast<Tensor&>(*this));
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::reciprocal_", "");
    return op.callUnboxed<Tensor &, Tensor &>(const_cast<Tensor&>(*this));
#endif
}

// aten::neg(Tensor self) -> Tensor
inline Tensor Tensor::neg() const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::neg(const_cast<Tensor&>(*this));
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::neg", "");
    return op.callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}

// aten::neg_(Tensor(a!) self) -> Tensor(a!)
inline Tensor & Tensor::neg_() const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::neg_(const_cast<Tensor&>(*this));
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::neg_", "");
    return op.callUnboxed<Tensor &, Tensor &>(const_cast<Tensor&>(*this));
#endif
}

// aten::repeat(Tensor self, int[] repeats) -> Tensor
inline Tensor Tensor::repeat(IntArrayRef repeats) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::repeat(const_cast<Tensor&>(*this), repeats);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::repeat", "");
    return op.callUnboxed<Tensor, const Tensor &, IntArrayRef>(const_cast<Tensor&>(*this), repeats);
#endif
}

// aten::repeat_interleave.self_Tensor(Tensor self, Tensor repeats, int? dim=None) -> Tensor
inline Tensor Tensor::repeat_interleave(const Tensor & repeats, c10::optional<int64_t> dim) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::repeat_interleave_self_Tensor(const_cast<Tensor&>(*this), repeats, dim);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::repeat_interleave", "self_Tensor");
    return op.callUnboxed<Tensor, const Tensor &, const Tensor &, c10::optional<int64_t>>(const_cast<Tensor&>(*this), repeats, dim);
#endif
}

// aten::repeat_interleave.self_int(Tensor self, int repeats, int? dim=None) -> Tensor
inline Tensor Tensor::repeat_interleave(int64_t repeats, c10::optional<int64_t> dim) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::repeat_interleave_self_int(const_cast<Tensor&>(*this), repeats, dim);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::repeat_interleave", "self_int");
    return op.callUnboxed<Tensor, const Tensor &, int64_t, c10::optional<int64_t>>(const_cast<Tensor&>(*this), repeats, dim);
#endif
}

// aten::reshape(Tensor self, int[] shape) -> Tensor
inline Tensor Tensor::reshape(IntArrayRef shape) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::reshape(const_cast<Tensor&>(*this), shape);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::reshape", "");
    return op.callUnboxed<Tensor, const Tensor &, IntArrayRef>(const_cast<Tensor&>(*this), shape);
#endif
}

// aten::reshape_as(Tensor self, Tensor other) -> Tensor
inline Tensor Tensor::reshape_as(const Tensor & other) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::reshape_as(const_cast<Tensor&>(*this), other);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::reshape_as", "");
    return op.callUnboxed<Tensor, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), other);
#endif
}

// aten::round(Tensor self) -> Tensor
inline Tensor Tensor::round() const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::round(const_cast<Tensor&>(*this));
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::round", "");
    return op.callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}

// aten::round_(Tensor(a!) self) -> Tensor(a!)
inline Tensor & Tensor::round_() const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::round_(const_cast<Tensor&>(*this));
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::round_", "");
    return op.callUnboxed<Tensor &, Tensor &>(const_cast<Tensor&>(*this));
#endif
}

// aten::relu(Tensor self) -> Tensor
inline Tensor Tensor::relu() const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(*this);
    DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
    DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);
    switch (dispatchKeyToBackend(_dk)) {
        case Backend::CPU:
            return CPUType::relu(const_cast<Tensor&>(*this));
            break;
        case Backend::QuantizedCPU:
            return QuantizedCPUType::relu(const_cast<Tensor&>(*this));
            break;
        default:
            AT_ERROR("relu not implemented for ", at::toString(_dk));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::relu", "");
    return op.callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}

// aten::relu_(Tensor(a!) self) -> Tensor(a!)
inline Tensor & Tensor::relu_() const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(*this);
    DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
    DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);
    switch (dispatchKeyToBackend(_dk)) {
        case Backend::CPU:
            return CPUType::relu_(const_cast<Tensor&>(*this));
            break;
        case Backend::QuantizedCPU:
            return QuantizedCPUType::relu_(const_cast<Tensor&>(*this));
            break;
        default:
            AT_ERROR("relu_ not implemented for ", at::toString(_dk));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::relu_", "");
    return op.callUnboxed<Tensor &, Tensor &>(const_cast<Tensor&>(*this));
#endif
}

// aten::prelu(Tensor self, Tensor weight) -> Tensor
inline Tensor Tensor::prelu(const Tensor & weight) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(*this, weight);
    DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
    DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);
    switch (dispatchKeyToBackend(_dk)) {
        case Backend::CPU:
            return CPUType::prelu(const_cast<Tensor&>(*this), weight);
            break;
        default:
            AT_ERROR("prelu not implemented for ", at::toString(_dk));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::prelu", "");
    return op.callUnboxed<Tensor, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), weight);
#endif
}

// aten::prelu_backward(Tensor grad_output, Tensor self, Tensor weight) -> (Tensor, Tensor)
inline std::tuple<Tensor,Tensor> Tensor::prelu_backward(const Tensor & grad_output, const Tensor & weight) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(grad_output, *this, weight);
    DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
    DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);
    switch (dispatchKeyToBackend(_dk)) {
        case Backend::CPU:
            return CPUType::prelu_backward(grad_output, const_cast<Tensor&>(*this), weight);
            break;
        default:
            AT_ERROR("prelu_backward not implemented for ", at::toString(_dk));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::prelu_backward", "");
    return op.callUnboxed<std::tuple<Tensor,Tensor>, const Tensor &, const Tensor &, const Tensor &>(grad_output, const_cast<Tensor&>(*this), weight);
#endif
}

// aten::hardshrink(Tensor self, Scalar lambd=0.5) -> Tensor
inline Tensor Tensor::hardshrink(Scalar lambd) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::hardshrink(const_cast<Tensor&>(*this), lambd);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::hardshrink", "");
    return op.callUnboxed<Tensor, const Tensor &, Scalar>(const_cast<Tensor&>(*this), lambd);
#endif
}

// aten::hardshrink_backward(Tensor grad_out, Tensor self, Scalar lambd) -> Tensor
inline Tensor Tensor::hardshrink_backward(const Tensor & grad_out, Scalar lambd) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::hardshrink_backward(grad_out, const_cast<Tensor&>(*this), lambd);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::hardshrink_backward", "");
    return op.callUnboxed<Tensor, const Tensor &, const Tensor &, Scalar>(grad_out, const_cast<Tensor&>(*this), lambd);
#endif
}

// aten::rsqrt(Tensor self) -> Tensor
inline Tensor Tensor::rsqrt() const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::rsqrt(const_cast<Tensor&>(*this));
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::rsqrt", "");
    return op.callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}

// aten::rsqrt_(Tensor(a!) self) -> Tensor(a!)
inline Tensor & Tensor::rsqrt_() const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::rsqrt_(const_cast<Tensor&>(*this));
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::rsqrt_", "");
    return op.callUnboxed<Tensor &, Tensor &>(const_cast<Tensor&>(*this));
#endif
}

// aten::select.Dimname(Tensor(a) self, Dimname dim, int index) -> Tensor(a)
inline Tensor Tensor::select(Dimname dim, int64_t index) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::select_Dimname(const_cast<Tensor&>(*this), dim, index);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::select", "Dimname");
    return op.callUnboxed<Tensor, const Tensor &, Dimname, int64_t>(const_cast<Tensor&>(*this), dim, index);
#endif
}

// aten::select.int(Tensor(a) self, int dim, int index) -> Tensor(a)
inline Tensor Tensor::select(int64_t dim, int64_t index) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::select_int(const_cast<Tensor&>(*this), dim, index);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::select", "int");
    return op.callUnboxed<Tensor, const Tensor &, int64_t, int64_t>(const_cast<Tensor&>(*this), dim, index);
#endif
}

// aten::sigmoid(Tensor self) -> Tensor
inline Tensor Tensor::sigmoid() const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(*this);
    DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
    DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);
    switch (dispatchKeyToBackend(_dk)) {
        case Backend::CPU:
            return CPUType::sigmoid(const_cast<Tensor&>(*this));
            break;
        case Backend::QuantizedCPU:
            return QuantizedCPUType::sigmoid(const_cast<Tensor&>(*this));
            break;
        default:
            AT_ERROR("sigmoid not implemented for ", at::toString(_dk));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::sigmoid", "");
    return op.callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}

// aten::sigmoid_(Tensor(a!) self) -> Tensor(a!)
inline Tensor & Tensor::sigmoid_() const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(*this);
    DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
    DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);
    switch (dispatchKeyToBackend(_dk)) {
        case Backend::CPU:
            return CPUType::sigmoid_(const_cast<Tensor&>(*this));
            break;
        default:
            AT_ERROR("sigmoid_ not implemented for ", at::toString(_dk));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::sigmoid_", "");
    return op.callUnboxed<Tensor &, Tensor &>(const_cast<Tensor&>(*this));
#endif
}

// aten::sin(Tensor self) -> Tensor
inline Tensor Tensor::sin() const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::sin(const_cast<Tensor&>(*this));
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::sin", "");
    return op.callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}

// aten::sin_(Tensor(a!) self) -> Tensor(a!)
inline Tensor & Tensor::sin_() const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::sin_(const_cast<Tensor&>(*this));
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::sin_", "");
    return op.callUnboxed<Tensor &, Tensor &>(const_cast<Tensor&>(*this));
#endif
}

// aten::sinh(Tensor self) -> Tensor
inline Tensor Tensor::sinh() const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::sinh(const_cast<Tensor&>(*this));
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::sinh", "");
    return op.callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}

// aten::sinh_(Tensor(a!) self) -> Tensor(a!)
inline Tensor & Tensor::sinh_() const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::sinh_(const_cast<Tensor&>(*this));
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::sinh_", "");
    return op.callUnboxed<Tensor &, Tensor &>(const_cast<Tensor&>(*this));
#endif
}

// aten::detach(Tensor self) -> Tensor
inline Tensor Tensor::detach() const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::detach(const_cast<Tensor&>(*this));
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::detach", "");
    return op.callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}

// aten::detach_(Tensor(a!) self) -> Tensor(a!)
inline Tensor & Tensor::detach_() const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::detach_(const_cast<Tensor&>(*this));
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::detach_", "");
    return op.callUnboxed<Tensor &, Tensor &>(const_cast<Tensor&>(*this));
#endif
}

// aten::size.int(Tensor self, int dim) -> int
inline int64_t Tensor::size(int64_t dim) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::size_int(const_cast<Tensor&>(*this), dim);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::size", "int");
    return op.callUnboxed<int64_t, const Tensor &, int64_t>(const_cast<Tensor&>(*this), dim);
#endif
}

// aten::size.Dimname(Tensor self, Dimname dim) -> int
inline int64_t Tensor::size(Dimname dim) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::size_Dimname(const_cast<Tensor&>(*this), dim);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::size", "Dimname");
    return op.callUnboxed<int64_t, const Tensor &, Dimname>(const_cast<Tensor&>(*this), dim);
#endif
}

// aten::slice.Tensor(Tensor(a) self, int dim=0, int start=0, int end=9223372036854775807, int step=1) -> Tensor(a)
inline Tensor Tensor::slice(int64_t dim, int64_t start, int64_t end, int64_t step) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::slice_Tensor(const_cast<Tensor&>(*this), dim, start, end, step);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::slice", "Tensor");
    return op.callUnboxed<Tensor, const Tensor &, int64_t, int64_t, int64_t, int64_t>(const_cast<Tensor&>(*this), dim, start, end, step);
#endif
}

// aten::slogdet(Tensor self) -> (Tensor sign, Tensor logabsdet)
inline std::tuple<Tensor,Tensor> Tensor::slogdet() const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::slogdet(const_cast<Tensor&>(*this));
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::slogdet", "");
    return op.callUnboxed<std::tuple<Tensor,Tensor>, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}

// aten::smm(Tensor self, Tensor mat2) -> Tensor
inline Tensor Tensor::smm(const Tensor & mat2) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::smm(const_cast<Tensor&>(*this), mat2);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::smm", "");
    return op.callUnboxed<Tensor, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), mat2);
#endif
}

// aten::softmax.int(Tensor self, int dim, ScalarType? dtype=None) -> Tensor
inline Tensor Tensor::softmax(int64_t dim, c10::optional<ScalarType> dtype) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::softmax_int(const_cast<Tensor&>(*this), dim, dtype);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::softmax", "int");
    return op.callUnboxed<Tensor, const Tensor &, int64_t, c10::optional<ScalarType>>(const_cast<Tensor&>(*this), dim, dtype);
#endif
}

// aten::softmax.Dimname(Tensor self, Dimname dim, *, ScalarType? dtype=None) -> Tensor
inline Tensor Tensor::softmax(Dimname dim, c10::optional<ScalarType> dtype) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::softmax_Dimname(const_cast<Tensor&>(*this), dim, dtype);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::softmax", "Dimname");
    return op.callUnboxed<Tensor, const Tensor &, Dimname, c10::optional<ScalarType>>(const_cast<Tensor&>(*this), dim, dtype);
#endif
}

// aten::split.Tensor(Tensor(a) self, int split_size, int dim=0) -> Tensor(a)[]
inline std::vector<Tensor> Tensor::split(int64_t split_size, int64_t dim) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::split_Tensor(const_cast<Tensor&>(*this), split_size, dim);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::split", "Tensor");
    return op.callUnboxed<std::vector<Tensor>, const Tensor &, int64_t, int64_t>(const_cast<Tensor&>(*this), split_size, dim);
#endif
}

// aten::split_with_sizes(Tensor self, int[] split_sizes, int dim=0) -> Tensor[]
inline std::vector<Tensor> Tensor::split_with_sizes(IntArrayRef split_sizes, int64_t dim) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::split_with_sizes(const_cast<Tensor&>(*this), split_sizes, dim);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::split_with_sizes", "");
    return op.callUnboxed<std::vector<Tensor>, const Tensor &, IntArrayRef, int64_t>(const_cast<Tensor&>(*this), split_sizes, dim);
#endif
}

// aten::squeeze(Tensor(a) self) -> Tensor(a)
inline Tensor Tensor::squeeze() const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::squeeze(const_cast<Tensor&>(*this));
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::squeeze", "");
    return op.callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}

// aten::squeeze.dim(Tensor(a) self, int dim) -> Tensor(a)
inline Tensor Tensor::squeeze(int64_t dim) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::squeeze_dim(const_cast<Tensor&>(*this), dim);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::squeeze", "dim");
    return op.callUnboxed<Tensor, const Tensor &, int64_t>(const_cast<Tensor&>(*this), dim);
#endif
}

// aten::squeeze.dimname(Tensor(a) self, Dimname dim) -> Tensor(a)
inline Tensor Tensor::squeeze(Dimname dim) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::squeeze_dimname(const_cast<Tensor&>(*this), dim);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::squeeze", "dimname");
    return op.callUnboxed<Tensor, const Tensor &, Dimname>(const_cast<Tensor&>(*this), dim);
#endif
}

// aten::squeeze_(Tensor(a!) self) -> Tensor(a!)
inline Tensor & Tensor::squeeze_() const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::squeeze_(const_cast<Tensor&>(*this));
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::squeeze_", "");
    return op.callUnboxed<Tensor &, Tensor &>(const_cast<Tensor&>(*this));
#endif
}

// aten::squeeze_.dim(Tensor(a!) self, int dim) -> Tensor(a!)
inline Tensor & Tensor::squeeze_(int64_t dim) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::squeeze__dim(const_cast<Tensor&>(*this), dim);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::squeeze_", "dim");
    return op.callUnboxed<Tensor &, Tensor &, int64_t>(const_cast<Tensor&>(*this), dim);
#endif
}

// aten::squeeze_.dimname(Tensor(a!) self, Dimname dim) -> Tensor(a!)
inline Tensor & Tensor::squeeze_(Dimname dim) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::squeeze__dimname(const_cast<Tensor&>(*this), dim);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::squeeze_", "dimname");
    return op.callUnboxed<Tensor &, Tensor &, Dimname>(const_cast<Tensor&>(*this), dim);
#endif
}

// aten::sspaddmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1) -> Tensor
inline Tensor Tensor::sspaddmm(const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::sspaddmm(const_cast<Tensor&>(*this), mat1, mat2, beta, alpha);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::sspaddmm", "");
    return op.callUnboxed<Tensor, const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar>(const_cast<Tensor&>(*this), mat1, mat2, beta, alpha);
#endif
}

// aten::stft(Tensor self, int n_fft, int? hop_length=None, int? win_length=None, Tensor? window=None, bool normalized=False, bool onesided=True) -> Tensor
inline Tensor Tensor::stft(int64_t n_fft, c10::optional<int64_t> hop_length, c10::optional<int64_t> win_length, const Tensor & window, bool normalized, bool onesided) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::stft(const_cast<Tensor&>(*this), n_fft, hop_length, win_length, window, normalized, onesided);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::stft", "");
    return op.callUnboxed<Tensor, const Tensor &, int64_t, c10::optional<int64_t>, c10::optional<int64_t>, const Tensor &, bool, bool>(const_cast<Tensor&>(*this), n_fft, hop_length, win_length, window, normalized, onesided);
#endif
}

// aten::istft(Tensor self, int n_fft, int? hop_length=None, int? win_length=None, Tensor? window=None, bool center=True, bool normalized=False, bool onesided=True, int? length=None) -> Tensor
inline Tensor Tensor::istft(int64_t n_fft, c10::optional<int64_t> hop_length, c10::optional<int64_t> win_length, const Tensor & window, bool center, bool normalized, bool onesided, c10::optional<int64_t> length) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::istft(const_cast<Tensor&>(*this), n_fft, hop_length, win_length, window, center, normalized, onesided, length);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::istft", "");
    return op.callUnboxed<Tensor, const Tensor &, int64_t, c10::optional<int64_t>, c10::optional<int64_t>, const Tensor &, bool, bool, bool, c10::optional<int64_t>>(const_cast<Tensor&>(*this), n_fft, hop_length, win_length, window, center, normalized, onesided, length);
#endif
}

// aten::stride.int(Tensor self, int dim) -> int
inline int64_t Tensor::stride(int64_t dim) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::stride_int(const_cast<Tensor&>(*this), dim);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::stride", "int");
    return op.callUnboxed<int64_t, const Tensor &, int64_t>(const_cast<Tensor&>(*this), dim);
#endif
}

// aten::stride.Dimname(Tensor self, Dimname dim) -> int
inline int64_t Tensor::stride(Dimname dim) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::stride_Dimname(const_cast<Tensor&>(*this), dim);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::stride", "Dimname");
    return op.callUnboxed<int64_t, const Tensor &, Dimname>(const_cast<Tensor&>(*this), dim);
#endif
}

// aten::sum(Tensor self, *, ScalarType? dtype=None) -> Tensor
inline Tensor Tensor::sum(c10::optional<ScalarType> dtype) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::sum(const_cast<Tensor&>(*this), dtype);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::sum", "");
    return op.callUnboxed<Tensor, const Tensor &, c10::optional<ScalarType>>(const_cast<Tensor&>(*this), dtype);
#endif
}

// aten::sum.dim_IntList(Tensor self, int[1] dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
inline Tensor Tensor::sum(IntArrayRef dim, bool keepdim, c10::optional<ScalarType> dtype) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::sum_dim_IntList(const_cast<Tensor&>(*this), dim, keepdim, dtype);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::sum", "dim_IntList");
    return op.callUnboxed<Tensor, const Tensor &, IntArrayRef, bool, c10::optional<ScalarType>>(const_cast<Tensor&>(*this), dim, keepdim, dtype);
#endif
}

// aten::sum.dim_DimnameList(Tensor self, Dimname[1] dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
inline Tensor Tensor::sum(DimnameList dim, bool keepdim, c10::optional<ScalarType> dtype) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::sum_dim_DimnameList(const_cast<Tensor&>(*this), dim, keepdim, dtype);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::sum", "dim_DimnameList");
    return op.callUnboxed<Tensor, const Tensor &, DimnameList, bool, c10::optional<ScalarType>>(const_cast<Tensor&>(*this), dim, keepdim, dtype);
#endif
}

// aten::sum_to_size(Tensor self, int[] size) -> Tensor
inline Tensor Tensor::sum_to_size(IntArrayRef size) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::sum_to_size(const_cast<Tensor&>(*this), size);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::sum_to_size", "");
    return op.callUnboxed<Tensor, const Tensor &, IntArrayRef>(const_cast<Tensor&>(*this), size);
#endif
}

// aten::sqrt(Tensor self) -> Tensor
inline Tensor Tensor::sqrt() const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::sqrt(const_cast<Tensor&>(*this));
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::sqrt", "");
    return op.callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}

// aten::sqrt_(Tensor(a!) self) -> Tensor(a!)
inline Tensor & Tensor::sqrt_() const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::sqrt_(const_cast<Tensor&>(*this));
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::sqrt_", "");
    return op.callUnboxed<Tensor &, Tensor &>(const_cast<Tensor&>(*this));
#endif
}

// aten::square(Tensor self) -> Tensor
inline Tensor Tensor::square() const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::square(const_cast<Tensor&>(*this));
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::square", "");
    return op.callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}

// aten::square_(Tensor(a!) self) -> Tensor(a!)
inline Tensor & Tensor::square_() const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::square_(const_cast<Tensor&>(*this));
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::square_", "");
    return op.callUnboxed<Tensor &, Tensor &>(const_cast<Tensor&>(*this));
#endif
}

// aten::std(Tensor self, bool unbiased=True) -> Tensor
inline Tensor Tensor::std(bool unbiased) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::std(const_cast<Tensor&>(*this), unbiased);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::std", "");
    return op.callUnboxed<Tensor, const Tensor &, bool>(const_cast<Tensor&>(*this), unbiased);
#endif
}

// aten::std.dim(Tensor self, int[1] dim, bool unbiased=True, bool keepdim=False) -> Tensor
inline Tensor Tensor::std(IntArrayRef dim, bool unbiased, bool keepdim) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::std_dim(const_cast<Tensor&>(*this), dim, unbiased, keepdim);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::std", "dim");
    return op.callUnboxed<Tensor, const Tensor &, IntArrayRef, bool, bool>(const_cast<Tensor&>(*this), dim, unbiased, keepdim);
#endif
}

// aten::std.names_dim(Tensor self, Dimname[1] dim, bool unbiased=True, bool keepdim=False) -> Tensor
inline Tensor Tensor::std(DimnameList dim, bool unbiased, bool keepdim) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::std_names_dim(const_cast<Tensor&>(*this), dim, unbiased, keepdim);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::std", "names_dim");
    return op.callUnboxed<Tensor, const Tensor &, DimnameList, bool, bool>(const_cast<Tensor&>(*this), dim, unbiased, keepdim);
#endif
}

// aten::prod(Tensor self, *, ScalarType? dtype=None) -> Tensor
inline Tensor Tensor::prod(c10::optional<ScalarType> dtype) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::prod(const_cast<Tensor&>(*this), dtype);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::prod", "");
    return op.callUnboxed<Tensor, const Tensor &, c10::optional<ScalarType>>(const_cast<Tensor&>(*this), dtype);
#endif
}

// aten::prod.dim_int(Tensor self, int dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
inline Tensor Tensor::prod(int64_t dim, bool keepdim, c10::optional<ScalarType> dtype) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::prod_dim_int(const_cast<Tensor&>(*this), dim, keepdim, dtype);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::prod", "dim_int");
    return op.callUnboxed<Tensor, const Tensor &, int64_t, bool, c10::optional<ScalarType>>(const_cast<Tensor&>(*this), dim, keepdim, dtype);
#endif
}

// aten::prod.dim_Dimname(Tensor self, Dimname dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
inline Tensor Tensor::prod(Dimname dim, bool keepdim, c10::optional<ScalarType> dtype) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::prod_dim_Dimname(const_cast<Tensor&>(*this), dim, keepdim, dtype);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::prod", "dim_Dimname");
    return op.callUnboxed<Tensor, const Tensor &, Dimname, bool, c10::optional<ScalarType>>(const_cast<Tensor&>(*this), dim, keepdim, dtype);
#endif
}

// aten::t(Tensor(a) self) -> Tensor(a)
inline Tensor Tensor::t() const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::t(const_cast<Tensor&>(*this));
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::t", "");
    return op.callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}

// aten::t_(Tensor(a!) self) -> Tensor(a!)
inline Tensor & Tensor::t_() const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::t_(const_cast<Tensor&>(*this));
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::t_", "");
    return op.callUnboxed<Tensor &, Tensor &>(const_cast<Tensor&>(*this));
#endif
}

// aten::tan(Tensor self) -> Tensor
inline Tensor Tensor::tan() const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::tan(const_cast<Tensor&>(*this));
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::tan", "");
    return op.callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}

// aten::tan_(Tensor(a!) self) -> Tensor(a!)
inline Tensor & Tensor::tan_() const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::tan_(const_cast<Tensor&>(*this));
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::tan_", "");
    return op.callUnboxed<Tensor &, Tensor &>(const_cast<Tensor&>(*this));
#endif
}

// aten::tanh(Tensor self) -> Tensor
inline Tensor Tensor::tanh() const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(*this);
    DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
    DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);
    switch (dispatchKeyToBackend(_dk)) {
        case Backend::CPU:
            return CPUType::tanh(const_cast<Tensor&>(*this));
            break;
        case Backend::QuantizedCPU:
            return QuantizedCPUType::tanh(const_cast<Tensor&>(*this));
            break;
        default:
            AT_ERROR("tanh not implemented for ", at::toString(_dk));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::tanh", "");
    return op.callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}

// aten::tanh_(Tensor(a!) self) -> Tensor(a!)
inline Tensor & Tensor::tanh_() const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::tanh_(const_cast<Tensor&>(*this));
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::tanh_", "");
    return op.callUnboxed<Tensor &, Tensor &>(const_cast<Tensor&>(*this));
#endif
}

// aten::transpose.int(Tensor(a) self, int dim0, int dim1) -> Tensor(a)
inline Tensor Tensor::transpose(int64_t dim0, int64_t dim1) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::transpose_int(const_cast<Tensor&>(*this), dim0, dim1);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::transpose", "int");
    return op.callUnboxed<Tensor, const Tensor &, int64_t, int64_t>(const_cast<Tensor&>(*this), dim0, dim1);
#endif
}

// aten::transpose.Dimname(Tensor(a) self, Dimname dim0, Dimname dim1) -> Tensor(a)
inline Tensor Tensor::transpose(Dimname dim0, Dimname dim1) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::transpose_Dimname(const_cast<Tensor&>(*this), dim0, dim1);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::transpose", "Dimname");
    return op.callUnboxed<Tensor, const Tensor &, Dimname, Dimname>(const_cast<Tensor&>(*this), dim0, dim1);
#endif
}

// aten::transpose_(Tensor(a!) self, int dim0, int dim1) -> Tensor(a!)
inline Tensor & Tensor::transpose_(int64_t dim0, int64_t dim1) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::transpose_(const_cast<Tensor&>(*this), dim0, dim1);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::transpose_", "");
    return op.callUnboxed<Tensor &, Tensor &, int64_t, int64_t>(const_cast<Tensor&>(*this), dim0, dim1);
#endif
}

// aten::flip(Tensor self, int[] dims) -> Tensor
inline Tensor Tensor::flip(IntArrayRef dims) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(*this);
    DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
    DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);
    switch (dispatchKeyToBackend(_dk)) {
        case Backend::CPU:
            return CPUType::flip(const_cast<Tensor&>(*this), dims);
            break;
        default:
            AT_ERROR("flip not implemented for ", at::toString(_dk));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::flip", "");
    return op.callUnboxed<Tensor, const Tensor &, IntArrayRef>(const_cast<Tensor&>(*this), dims);
#endif
}

// aten::roll(Tensor self, int[1] shifts, int[1] dims=[]) -> Tensor
inline Tensor Tensor::roll(IntArrayRef shifts, IntArrayRef dims) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(*this);
    DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
    DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);
    switch (dispatchKeyToBackend(_dk)) {
        case Backend::CPU:
            return CPUType::roll(const_cast<Tensor&>(*this), shifts, dims);
            break;
        default:
            AT_ERROR("roll not implemented for ", at::toString(_dk));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::roll", "");
    return op.callUnboxed<Tensor, const Tensor &, IntArrayRef, IntArrayRef>(const_cast<Tensor&>(*this), shifts, dims);
#endif
}

// aten::rot90(Tensor self, int k=1, int[] dims=[0,1]) -> Tensor
inline Tensor Tensor::rot90(int64_t k, IntArrayRef dims) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::rot90(const_cast<Tensor&>(*this), k, dims);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::rot90", "");
    return op.callUnboxed<Tensor, const Tensor &, int64_t, IntArrayRef>(const_cast<Tensor&>(*this), k, dims);
#endif
}

// aten::true_divide.Tensor(Tensor self, Tensor other) -> Tensor
inline Tensor Tensor::true_divide(const Tensor & other) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(*this, other);
    DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
    DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);
    switch (dispatchKeyToBackend(_dk)) {
        case Backend::CPU:
            return CPUType::true_divide_Tensor(const_cast<Tensor&>(*this), other);
            break;
        default:
            AT_ERROR("true_divide not implemented for ", at::toString(_dk));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::true_divide", "Tensor");
    return op.callUnboxed<Tensor, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), other);
#endif
}

// aten::true_divide_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
inline Tensor & Tensor::true_divide_(const Tensor & other) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(*this, other);
    DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
    DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);
    switch (dispatchKeyToBackend(_dk)) {
        case Backend::CPU:
            return CPUType::true_divide__Tensor(const_cast<Tensor&>(*this), other);
            break;
        default:
            AT_ERROR("true_divide_ not implemented for ", at::toString(_dk));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::true_divide_", "Tensor");
    return op.callUnboxed<Tensor &, Tensor &, const Tensor &>(const_cast<Tensor&>(*this), other);
#endif
}

// aten::true_divide.Scalar(Tensor self, Scalar other) -> Tensor
inline Tensor Tensor::true_divide(Scalar other) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::true_divide_Scalar(const_cast<Tensor&>(*this), other);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::true_divide", "Scalar");
    return op.callUnboxed<Tensor, const Tensor &, Scalar>(const_cast<Tensor&>(*this), other);
#endif
}

// aten::true_divide_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
inline Tensor & Tensor::true_divide_(Scalar other) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::true_divide__Scalar(const_cast<Tensor&>(*this), other);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::true_divide_", "Scalar");
    return op.callUnboxed<Tensor &, Tensor &, Scalar>(const_cast<Tensor&>(*this), other);
#endif
}

// aten::trunc(Tensor self) -> Tensor
inline Tensor Tensor::trunc() const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::trunc(const_cast<Tensor&>(*this));
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::trunc", "");
    return op.callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}

// aten::trunc_(Tensor(a!) self) -> Tensor(a!)
inline Tensor & Tensor::trunc_() const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::trunc_(const_cast<Tensor&>(*this));
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::trunc_", "");
    return op.callUnboxed<Tensor &, Tensor &>(const_cast<Tensor&>(*this));
#endif
}

// aten::type_as(Tensor self, Tensor other) -> Tensor
inline Tensor Tensor::type_as(const Tensor & other) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::type_as(const_cast<Tensor&>(*this), other);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::type_as", "");
    return op.callUnboxed<Tensor, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), other);
#endif
}

// aten::unsqueeze(Tensor(a) self, int dim) -> Tensor(a)
inline Tensor Tensor::unsqueeze(int64_t dim) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::unsqueeze(const_cast<Tensor&>(*this), dim);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::unsqueeze", "");
    return op.callUnboxed<Tensor, const Tensor &, int64_t>(const_cast<Tensor&>(*this), dim);
#endif
}

// aten::unsqueeze_(Tensor(a!) self, int dim) -> Tensor(a!)
inline Tensor & Tensor::unsqueeze_(int64_t dim) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::unsqueeze_(const_cast<Tensor&>(*this), dim);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::unsqueeze_", "");
    return op.callUnboxed<Tensor &, Tensor &, int64_t>(const_cast<Tensor&>(*this), dim);
#endif
}

// aten::var(Tensor self, bool unbiased=True) -> Tensor
inline Tensor Tensor::var(bool unbiased) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::var(const_cast<Tensor&>(*this), unbiased);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::var", "");
    return op.callUnboxed<Tensor, const Tensor &, bool>(const_cast<Tensor&>(*this), unbiased);
#endif
}

// aten::var.dim(Tensor self, int[1] dim, bool unbiased=True, bool keepdim=False) -> Tensor
inline Tensor Tensor::var(IntArrayRef dim, bool unbiased, bool keepdim) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::var_dim(const_cast<Tensor&>(*this), dim, unbiased, keepdim);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::var", "dim");
    return op.callUnboxed<Tensor, const Tensor &, IntArrayRef, bool, bool>(const_cast<Tensor&>(*this), dim, unbiased, keepdim);
#endif
}

// aten::var.names_dim(Tensor self, Dimname[1] dim, bool unbiased=True, bool keepdim=False) -> Tensor
inline Tensor Tensor::var(DimnameList dim, bool unbiased, bool keepdim) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::var_names_dim(const_cast<Tensor&>(*this), dim, unbiased, keepdim);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::var", "names_dim");
    return op.callUnboxed<Tensor, const Tensor &, DimnameList, bool, bool>(const_cast<Tensor&>(*this), dim, unbiased, keepdim);
#endif
}

// aten::view_as(Tensor self, Tensor other) -> Tensor
inline Tensor Tensor::view_as(const Tensor & other) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::view_as(const_cast<Tensor&>(*this), other);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::view_as", "");
    return op.callUnboxed<Tensor, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), other);
#endif
}

// aten::where.self(Tensor condition, Tensor self, Tensor other) -> Tensor
inline Tensor Tensor::where(const Tensor & condition, const Tensor & other) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::where_self(condition, const_cast<Tensor&>(*this), other);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::where", "self");
    return op.callUnboxed<Tensor, const Tensor &, const Tensor &, const Tensor &>(condition, const_cast<Tensor&>(*this), other);
#endif
}

// aten::norm.ScalarOpt_dtype(Tensor self, Scalar? p, *, ScalarType dtype) -> Tensor
inline Tensor Tensor::norm(c10::optional<Scalar> p, ScalarType dtype) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::norm_ScalarOpt_dtype(const_cast<Tensor&>(*this), p, dtype);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::norm", "ScalarOpt_dtype");
    return op.callUnboxed<Tensor, const Tensor &, c10::optional<Scalar>, ScalarType>(const_cast<Tensor&>(*this), p, dtype);
#endif
}

// aten::norm.Scalar(Tensor self, Scalar p=2) -> Tensor
inline Tensor Tensor::norm(Scalar p) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::norm_Scalar(const_cast<Tensor&>(*this), p);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::norm", "Scalar");
    return op.callUnboxed<Tensor, const Tensor &, Scalar>(const_cast<Tensor&>(*this), p);
#endif
}

// aten::norm.ScalarOpt_dim_dtype(Tensor self, Scalar? p, int[1] dim, bool keepdim, *, ScalarType dtype) -> Tensor
inline Tensor Tensor::norm(c10::optional<Scalar> p, IntArrayRef dim, bool keepdim, ScalarType dtype) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::norm_ScalarOpt_dim_dtype(const_cast<Tensor&>(*this), p, dim, keepdim, dtype);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::norm", "ScalarOpt_dim_dtype");
    return op.callUnboxed<Tensor, const Tensor &, c10::optional<Scalar>, IntArrayRef, bool, ScalarType>(const_cast<Tensor&>(*this), p, dim, keepdim, dtype);
#endif
}

// aten::norm.ScalarOpt_dim(Tensor self, Scalar? p, int[1] dim, bool keepdim=False) -> Tensor
inline Tensor Tensor::norm(c10::optional<Scalar> p, IntArrayRef dim, bool keepdim) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::norm_ScalarOpt_dim(const_cast<Tensor&>(*this), p, dim, keepdim);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::norm", "ScalarOpt_dim");
    return op.callUnboxed<Tensor, const Tensor &, c10::optional<Scalar>, IntArrayRef, bool>(const_cast<Tensor&>(*this), p, dim, keepdim);
#endif
}

// aten::norm.names_ScalarOpt_dim_dtype(Tensor self, Scalar? p, Dimname[1] dim, bool keepdim, *, ScalarType dtype) -> Tensor
inline Tensor Tensor::norm(c10::optional<Scalar> p, DimnameList dim, bool keepdim, ScalarType dtype) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::norm_names_ScalarOpt_dim_dtype(const_cast<Tensor&>(*this), p, dim, keepdim, dtype);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::norm", "names_ScalarOpt_dim_dtype");
    return op.callUnboxed<Tensor, const Tensor &, c10::optional<Scalar>, DimnameList, bool, ScalarType>(const_cast<Tensor&>(*this), p, dim, keepdim, dtype);
#endif
}

// aten::norm.names_ScalarOpt_dim(Tensor self, Scalar? p, Dimname[1] dim, bool keepdim=False) -> Tensor
inline Tensor Tensor::norm(c10::optional<Scalar> p, DimnameList dim, bool keepdim) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::norm_names_ScalarOpt_dim(const_cast<Tensor&>(*this), p, dim, keepdim);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::norm", "names_ScalarOpt_dim");
    return op.callUnboxed<Tensor, const Tensor &, c10::optional<Scalar>, DimnameList, bool>(const_cast<Tensor&>(*this), p, dim, keepdim);
#endif
}

// aten::clone(Tensor self, *, MemoryFormat? memory_format=None) -> Tensor
inline Tensor Tensor::clone(c10::optional<MemoryFormat> memory_format) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(*this);
    DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
    DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);
    switch (dispatchKeyToBackend(_dk)) {
        case Backend::CPU:
            return CPUType::clone(const_cast<Tensor&>(*this), memory_format);
            break;
        case Backend::QuantizedCPU:
            return QuantizedCPUType::clone(const_cast<Tensor&>(*this), memory_format);
            break;
        default:
            AT_ERROR("clone not implemented for ", at::toString(_dk));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::clone", "");
    return op.callUnboxed<Tensor, const Tensor &, c10::optional<MemoryFormat>>(const_cast<Tensor&>(*this), memory_format);
#endif
}

// aten::resize_as_(Tensor(a!) self, Tensor the_template, *, MemoryFormat? memory_format=None) -> Tensor(a!)
inline Tensor & Tensor::resize_as_(const Tensor & the_template, c10::optional<MemoryFormat> memory_format) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::resize_as_(const_cast<Tensor&>(*this), the_template, memory_format);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::resize_as_", "");
    return op.callUnboxed<Tensor &, Tensor &, const Tensor &, c10::optional<MemoryFormat>>(const_cast<Tensor&>(*this), the_template, memory_format);
#endif
}

// aten::pow.Tensor_Scalar(Tensor self, Scalar exponent) -> Tensor
inline Tensor Tensor::pow(Scalar exponent) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(*this);
    DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
    DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);
    switch (dispatchKeyToBackend(_dk)) {
        case Backend::CPU:
            return CPUType::pow_Tensor_Scalar(const_cast<Tensor&>(*this), exponent);
            break;
        default:
            AT_ERROR("pow not implemented for ", at::toString(_dk));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::pow", "Tensor_Scalar");
    return op.callUnboxed<Tensor, const Tensor &, Scalar>(const_cast<Tensor&>(*this), exponent);
#endif
}

// aten::zero_(Tensor(a!) self) -> Tensor(a!)
inline Tensor & Tensor::zero_() const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(*this);
    DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
    DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);
    switch (dispatchKeyToBackend(_dk)) {
        case Backend::CPU:
            return CPUType::zero_(const_cast<Tensor&>(*this));
            break;
        default:
            AT_ERROR("zero_ not implemented for ", at::toString(_dk));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::zero_", "");
    return op.callUnboxed<Tensor &, Tensor &>(const_cast<Tensor&>(*this));
#endif
}

// aten::sub.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
inline Tensor Tensor::sub(const Tensor & other, Scalar alpha) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(*this, other);
    DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
    DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);
    switch (dispatchKeyToBackend(_dk)) {
        case Backend::CPU:
            return CPUType::sub_Tensor(const_cast<Tensor&>(*this), other, alpha);
            break;
        default:
            AT_ERROR("sub not implemented for ", at::toString(_dk));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::sub", "Tensor");
    return op.callUnboxed<Tensor, const Tensor &, const Tensor &, Scalar>(const_cast<Tensor&>(*this), other, alpha);
#endif
}

// aten::sub_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> Tensor(a!)
inline Tensor & Tensor::sub_(const Tensor & other, Scalar alpha) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(*this, other);
    DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
    DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);
    switch (dispatchKeyToBackend(_dk)) {
        case Backend::CPU:
            return CPUType::sub__Tensor(const_cast<Tensor&>(*this), other, alpha);
            break;
        default:
            AT_ERROR("sub_ not implemented for ", at::toString(_dk));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::sub_", "Tensor");
    return op.callUnboxed<Tensor &, Tensor &, const Tensor &, Scalar>(const_cast<Tensor&>(*this), other, alpha);
#endif
}

// aten::sub.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor
inline Tensor Tensor::sub(Scalar other, Scalar alpha) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::sub_Scalar(const_cast<Tensor&>(*this), other, alpha);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::sub", "Scalar");
    return op.callUnboxed<Tensor, const Tensor &, Scalar, Scalar>(const_cast<Tensor&>(*this), other, alpha);
#endif
}

// aten::sub_.Scalar(Tensor(a!) self, Scalar other, Scalar alpha=1) -> Tensor(a!)
inline Tensor & Tensor::sub_(Scalar other, Scalar alpha) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::sub__Scalar(const_cast<Tensor&>(*this), other, alpha);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::sub_", "Scalar");
    return op.callUnboxed<Tensor &, Tensor &, Scalar, Scalar>(const_cast<Tensor&>(*this), other, alpha);
#endif
}

// aten::addmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1) -> Tensor
inline Tensor Tensor::addmm(const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(*this, mat1, mat2);
    DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
    DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);
    switch (dispatchKeyToBackend(_dk)) {
        case Backend::CPU:
            return CPUType::addmm(const_cast<Tensor&>(*this), mat1, mat2, beta, alpha);
            break;
        default:
            AT_ERROR("addmm not implemented for ", at::toString(_dk));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::addmm", "");
    return op.callUnboxed<Tensor, const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar>(const_cast<Tensor&>(*this), mat1, mat2, beta, alpha);
#endif
}

// aten::addmm_(Tensor(a!) self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1) -> Tensor(a!)
inline Tensor & Tensor::addmm_(const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(*this, mat1, mat2);
    DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
    DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);
    switch (dispatchKeyToBackend(_dk)) {
        case Backend::CPU:
            return CPUType::addmm_(const_cast<Tensor&>(*this), mat1, mat2, beta, alpha);
            break;
        default:
            AT_ERROR("addmm_ not implemented for ", at::toString(_dk));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::addmm_", "");
    return op.callUnboxed<Tensor &, Tensor &, const Tensor &, const Tensor &, Scalar, Scalar>(const_cast<Tensor&>(*this), mat1, mat2, beta, alpha);
#endif
}

// aten::sparse_resize_(Tensor(a!) self, int[] size, int sparse_dim, int dense_dim) -> Tensor(a!)
inline Tensor & Tensor::sparse_resize_(IntArrayRef size, int64_t sparse_dim, int64_t dense_dim) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(*this);
    DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
    DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);
    switch (dispatchKeyToBackend(_dk)) {
    
        default:
            AT_ERROR("sparse_resize_ not implemented for ", at::toString(_dk));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::sparse_resize_", "");
    return op.callUnboxed<Tensor &, Tensor &, IntArrayRef, int64_t, int64_t>(const_cast<Tensor&>(*this), size, sparse_dim, dense_dim);
#endif
}

// aten::sparse_resize_and_clear_(Tensor(a!) self, int[] size, int sparse_dim, int dense_dim) -> Tensor(a!)
inline Tensor & Tensor::sparse_resize_and_clear_(IntArrayRef size, int64_t sparse_dim, int64_t dense_dim) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(*this);
    DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
    DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);
    switch (dispatchKeyToBackend(_dk)) {
    
        default:
            AT_ERROR("sparse_resize_and_clear_ not implemented for ", at::toString(_dk));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::sparse_resize_and_clear_", "");
    return op.callUnboxed<Tensor &, Tensor &, IntArrayRef, int64_t, int64_t>(const_cast<Tensor&>(*this), size, sparse_dim, dense_dim);
#endif
}

// aten::sparse_mask(Tensor self, Tensor mask) -> Tensor
inline Tensor Tensor::sparse_mask(const Tensor & mask) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(*this, mask);
    DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
    DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);
    switch (dispatchKeyToBackend(_dk)) {
    
        default:
            AT_ERROR("sparse_mask not implemented for ", at::toString(_dk));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::sparse_mask", "");
    return op.callUnboxed<Tensor, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), mask);
#endif
}

// aten::to_dense(Tensor self) -> Tensor
inline Tensor Tensor::to_dense() const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(*this);
    DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
    DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);
    switch (dispatchKeyToBackend(_dk)) {
    
        default:
            AT_ERROR("to_dense not implemented for ", at::toString(_dk));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::to_dense", "");
    return op.callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}

// aten::sparse_dim(Tensor self) -> int
inline int64_t Tensor::sparse_dim() const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(*this);
    DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
    DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);
    switch (dispatchKeyToBackend(_dk)) {
    
        default:
            AT_ERROR("sparse_dim not implemented for ", at::toString(_dk));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::sparse_dim", "");
    return op.callUnboxed<int64_t, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}

// aten::_dimI(Tensor self) -> int
inline int64_t Tensor::_dimI() const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(*this);
    DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
    DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);
    switch (dispatchKeyToBackend(_dk)) {
    
        default:
            AT_ERROR("_dimI not implemented for ", at::toString(_dk));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::_dimI", "");
    return op.callUnboxed<int64_t, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}

// aten::dense_dim(Tensor self) -> int
inline int64_t Tensor::dense_dim() const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(*this);
    DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
    DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);
    switch (dispatchKeyToBackend(_dk)) {
    
        default:
            AT_ERROR("dense_dim not implemented for ", at::toString(_dk));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::dense_dim", "");
    return op.callUnboxed<int64_t, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}

// aten::_dimV(Tensor self) -> int
inline int64_t Tensor::_dimV() const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(*this);
    DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
    DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);
    switch (dispatchKeyToBackend(_dk)) {
    
        default:
            AT_ERROR("_dimV not implemented for ", at::toString(_dk));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::_dimV", "");
    return op.callUnboxed<int64_t, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}

// aten::_nnz(Tensor self) -> int
inline int64_t Tensor::_nnz() const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(*this);
    DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
    DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);
    switch (dispatchKeyToBackend(_dk)) {
    
        default:
            AT_ERROR("_nnz not implemented for ", at::toString(_dk));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::_nnz", "");
    return op.callUnboxed<int64_t, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}

// aten::coalesce(Tensor self) -> Tensor
inline Tensor Tensor::coalesce() const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(*this);
    DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
    DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);
    switch (dispatchKeyToBackend(_dk)) {
    
        default:
            AT_ERROR("coalesce not implemented for ", at::toString(_dk));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::coalesce", "");
    return op.callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}

// aten::is_coalesced(Tensor self) -> bool
inline bool Tensor::is_coalesced() const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(*this);
    DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
    DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);
    switch (dispatchKeyToBackend(_dk)) {
    
        default:
            AT_ERROR("is_coalesced not implemented for ", at::toString(_dk));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::is_coalesced", "");
    return op.callUnboxed<bool, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}

// aten::_indices(Tensor(a) self) -> Tensor(a)
inline Tensor Tensor::_indices() const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(*this);
    DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
    DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);
    switch (dispatchKeyToBackend(_dk)) {
    
        default:
            AT_ERROR("_indices not implemented for ", at::toString(_dk));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::_indices", "");
    return op.callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}

// aten::_values(Tensor(a) self) -> Tensor(a)
inline Tensor Tensor::_values() const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(*this);
    DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
    DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);
    switch (dispatchKeyToBackend(_dk)) {
    
        default:
            AT_ERROR("_values not implemented for ", at::toString(_dk));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::_values", "");
    return op.callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}

// aten::_coalesced_(Tensor(a!) self, bool coalesced) -> Tensor(a!)
inline Tensor & Tensor::_coalesced_(bool coalesced) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(*this);
    DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
    DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);
    switch (dispatchKeyToBackend(_dk)) {
    
        default:
            AT_ERROR("_coalesced_ not implemented for ", at::toString(_dk));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::_coalesced_", "");
    return op.callUnboxed<Tensor &, Tensor &, bool>(const_cast<Tensor&>(*this), coalesced);
#endif
}

// aten::indices(Tensor(a) self) -> Tensor(a)
inline Tensor Tensor::indices() const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(*this);
    DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
    DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);
    switch (dispatchKeyToBackend(_dk)) {
    
        default:
            AT_ERROR("indices not implemented for ", at::toString(_dk));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::indices", "");
    return op.callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}

// aten::values(Tensor(a) self) -> Tensor(a)
inline Tensor Tensor::values() const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(*this);
    DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
    DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);
    switch (dispatchKeyToBackend(_dk)) {
    
        default:
            AT_ERROR("values not implemented for ", at::toString(_dk));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::values", "");
    return op.callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}

// aten::unbind.int(Tensor(a) self, int dim=0) -> Tensor(a)[]
inline std::vector<Tensor> Tensor::unbind(int64_t dim) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::unbind_int(const_cast<Tensor&>(*this), dim);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::unbind", "int");
    return op.callUnboxed<std::vector<Tensor>, const Tensor &, int64_t>(const_cast<Tensor&>(*this), dim);
#endif
}

// aten::unbind.Dimname(Tensor(a) self, Dimname dim) -> Tensor(a)[]
inline std::vector<Tensor> Tensor::unbind(Dimname dim) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::unbind_Dimname(const_cast<Tensor&>(*this), dim);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::unbind", "Dimname");
    return op.callUnboxed<std::vector<Tensor>, const Tensor &, Dimname>(const_cast<Tensor&>(*this), dim);
#endif
}

// aten::to_sparse.sparse_dim(Tensor self, int sparse_dim) -> Tensor
inline Tensor Tensor::to_sparse(int64_t sparse_dim) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(*this);
    DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
    DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);
    switch (dispatchKeyToBackend(_dk)) {
        case Backend::CPU:
            return CPUType::to_sparse_sparse_dim(const_cast<Tensor&>(*this), sparse_dim);
            break;
        default:
            AT_ERROR("to_sparse not implemented for ", at::toString(_dk));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::to_sparse", "sparse_dim");
    return op.callUnboxed<Tensor, const Tensor &, int64_t>(const_cast<Tensor&>(*this), sparse_dim);
#endif
}

// aten::to_sparse(Tensor self) -> Tensor
inline Tensor Tensor::to_sparse() const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(*this);
    DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
    DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);
    switch (dispatchKeyToBackend(_dk)) {
        case Backend::CPU:
            return CPUType::to_sparse(const_cast<Tensor&>(*this));
            break;
        default:
            AT_ERROR("to_sparse not implemented for ", at::toString(_dk));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::to_sparse", "");
    return op.callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}

// aten::to_mkldnn(Tensor self) -> Tensor
inline Tensor Tensor::to_mkldnn() const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(*this);
    DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
    DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);
    switch (dispatchKeyToBackend(_dk)) {
        case Backend::CPU:
            return CPUType::to_mkldnn(const_cast<Tensor&>(*this));
            break;
        default:
            AT_ERROR("to_mkldnn not implemented for ", at::toString(_dk));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::to_mkldnn", "");
    return op.callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}

// aten::dequantize.self(Tensor self) -> Tensor
inline Tensor Tensor::dequantize() const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(*this);
    DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
    DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);
    switch (dispatchKeyToBackend(_dk)) {
        case Backend::QuantizedCPU:
            return QuantizedCPUType::dequantize_self(const_cast<Tensor&>(*this));
            break;
        default:
            AT_ERROR("dequantize not implemented for ", at::toString(_dk));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::dequantize", "self");
    return op.callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}

// aten::q_scale(Tensor self) -> float
inline double Tensor::q_scale() const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(*this);
    DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
    DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);
    switch (dispatchKeyToBackend(_dk)) {
        case Backend::QuantizedCPU:
            return QuantizedCPUType::q_scale(const_cast<Tensor&>(*this));
            break;
        default:
            AT_ERROR("q_scale not implemented for ", at::toString(_dk));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::q_scale", "");
    return op.callUnboxed<double, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}

// aten::q_zero_point(Tensor self) -> int
inline int64_t Tensor::q_zero_point() const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(*this);
    DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
    DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);
    switch (dispatchKeyToBackend(_dk)) {
        case Backend::QuantizedCPU:
            return QuantizedCPUType::q_zero_point(const_cast<Tensor&>(*this));
            break;
        default:
            AT_ERROR("q_zero_point not implemented for ", at::toString(_dk));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::q_zero_point", "");
    return op.callUnboxed<int64_t, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}

// aten::q_per_channel_scales(Tensor self) -> Tensor
inline Tensor Tensor::q_per_channel_scales() const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(*this);
    DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
    DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);
    switch (dispatchKeyToBackend(_dk)) {
        case Backend::QuantizedCPU:
            return QuantizedCPUType::q_per_channel_scales(const_cast<Tensor&>(*this));
            break;
        default:
            AT_ERROR("q_per_channel_scales not implemented for ", at::toString(_dk));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::q_per_channel_scales", "");
    return op.callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}

// aten::q_per_channel_zero_points(Tensor self) -> Tensor
inline Tensor Tensor::q_per_channel_zero_points() const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(*this);
    DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
    DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);
    switch (dispatchKeyToBackend(_dk)) {
        case Backend::QuantizedCPU:
            return QuantizedCPUType::q_per_channel_zero_points(const_cast<Tensor&>(*this));
            break;
        default:
            AT_ERROR("q_per_channel_zero_points not implemented for ", at::toString(_dk));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::q_per_channel_zero_points", "");
    return op.callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}

// aten::q_per_channel_axis(Tensor self) -> int
inline int64_t Tensor::q_per_channel_axis() const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(*this);
    DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
    DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);
    switch (dispatchKeyToBackend(_dk)) {
        case Backend::QuantizedCPU:
            return QuantizedCPUType::q_per_channel_axis(const_cast<Tensor&>(*this));
            break;
        default:
            AT_ERROR("q_per_channel_axis not implemented for ", at::toString(_dk));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::q_per_channel_axis", "");
    return op.callUnboxed<int64_t, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}

// aten::int_repr(Tensor self) -> Tensor
inline Tensor Tensor::int_repr() const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(*this);
    DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
    DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);
    switch (dispatchKeyToBackend(_dk)) {
        case Backend::QuantizedCPU:
            return QuantizedCPUType::int_repr(const_cast<Tensor&>(*this));
            break;
        default:
            AT_ERROR("int_repr not implemented for ", at::toString(_dk));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::int_repr", "");
    return op.callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}

// aten::qscheme(Tensor self) -> QScheme
inline QScheme Tensor::qscheme() const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(*this);
    DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
    DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);
    switch (dispatchKeyToBackend(_dk)) {
        case Backend::QuantizedCPU:
            return QuantizedCPUType::qscheme(const_cast<Tensor&>(*this));
            break;
        default:
            AT_ERROR("qscheme not implemented for ", at::toString(_dk));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::qscheme", "");
    return op.callUnboxed<QScheme, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}

// aten::to.dtype_layout(Tensor self, *, ScalarType dtype, Layout layout, Device device, bool pin_memory=False, bool non_blocking=False, bool copy=False, MemoryFormat? memory_format=None) -> Tensor
inline Tensor Tensor::to(const TensorOptions & options, bool non_blocking, bool copy, c10::optional<MemoryFormat> memory_format) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::to_dtype_layout(const_cast<Tensor&>(*this), options, non_blocking, copy, memory_format);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::to", "dtype_layout");
    return op.callUnboxed<Tensor, const Tensor &, const TensorOptions &, bool, bool, c10::optional<MemoryFormat>>(const_cast<Tensor&>(*this), options, non_blocking, copy, memory_format);
#endif
}

// aten::to.device(Tensor self, Device device, ScalarType dtype, bool non_blocking=False, bool copy=False, MemoryFormat? memory_format=None) -> Tensor
inline Tensor Tensor::to(Device device, ScalarType dtype, bool non_blocking, bool copy, c10::optional<MemoryFormat> memory_format) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::to_device(const_cast<Tensor&>(*this), device, dtype, non_blocking, copy, memory_format);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::to", "device");
    return op.callUnboxed<Tensor, const Tensor &, Device, ScalarType, bool, bool, c10::optional<MemoryFormat>>(const_cast<Tensor&>(*this), device, dtype, non_blocking, copy, memory_format);
#endif
}

// aten::to.dtype(Tensor self, ScalarType dtype, bool non_blocking=False, bool copy=False, MemoryFormat? memory_format=None) -> Tensor
inline Tensor Tensor::to(ScalarType dtype, bool non_blocking, bool copy, c10::optional<MemoryFormat> memory_format) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::to_dtype(const_cast<Tensor&>(*this), dtype, non_blocking, copy, memory_format);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::to", "dtype");
    return op.callUnboxed<Tensor, const Tensor &, ScalarType, bool, bool, c10::optional<MemoryFormat>>(const_cast<Tensor&>(*this), dtype, non_blocking, copy, memory_format);
#endif
}

// aten::to.other(Tensor self, Tensor other, bool non_blocking=False, bool copy=False, MemoryFormat? memory_format=None) -> Tensor
inline Tensor Tensor::to(const Tensor & other, bool non_blocking, bool copy, c10::optional<MemoryFormat> memory_format) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::to_other(const_cast<Tensor&>(*this), other, non_blocking, copy, memory_format);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::to", "other");
    return op.callUnboxed<Tensor, const Tensor &, const Tensor &, bool, bool, c10::optional<MemoryFormat>>(const_cast<Tensor&>(*this), other, non_blocking, copy, memory_format);
#endif
}

// aten::item(Tensor self) -> Scalar
inline Scalar Tensor::item() const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::item(const_cast<Tensor&>(*this));
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::item", "");
    return op.callUnboxed<Scalar, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}

// aten::set_.source_Storage(Tensor(a!) self, Storage source) -> Tensor(a!)
inline Tensor & Tensor::set_(Storage source) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(*this);
    DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
    DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);
    switch (dispatchKeyToBackend(_dk)) {
        case Backend::CPU:
            return CPUType::set__source_Storage(const_cast<Tensor&>(*this), source);
            break;
        default:
            AT_ERROR("set_ not implemented for ", at::toString(_dk));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::set_", "source_Storage");
    return op.callUnboxed<Tensor &, Tensor &, Storage>(const_cast<Tensor&>(*this), source);
#endif
}

// aten::set_.source_Storage_storage_offset(Tensor(a!) self, Storage source, int storage_offset, int[] size, int[] stride=[]) -> Tensor(a!)
inline Tensor & Tensor::set_(Storage source, int64_t storage_offset, IntArrayRef size, IntArrayRef stride) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(*this);
    DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
    DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);
    switch (dispatchKeyToBackend(_dk)) {
        case Backend::CPU:
            return CPUType::set__source_Storage_storage_offset(const_cast<Tensor&>(*this), source, storage_offset, size, stride);
            break;
        case Backend::QuantizedCPU:
            return QuantizedCPUType::set__source_Storage_storage_offset(const_cast<Tensor&>(*this), source, storage_offset, size, stride);
            break;
        default:
            AT_ERROR("set_ not implemented for ", at::toString(_dk));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::set_", "source_Storage_storage_offset");
    return op.callUnboxed<Tensor &, Tensor &, Storage, int64_t, IntArrayRef, IntArrayRef>(const_cast<Tensor&>(*this), source, storage_offset, size, stride);
#endif
}

// aten::set_.source_Tensor(Tensor(a!) self, Tensor source) -> Tensor(a!)
inline Tensor & Tensor::set_(const Tensor & source) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(*this, source);
    DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
    DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);
    switch (dispatchKeyToBackend(_dk)) {
        case Backend::CPU:
            return CPUType::set__source_Tensor(const_cast<Tensor&>(*this), source);
            break;
        default:
            AT_ERROR("set_ not implemented for ", at::toString(_dk));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::set_", "source_Tensor");
    return op.callUnboxed<Tensor &, Tensor &, const Tensor &>(const_cast<Tensor&>(*this), source);
#endif
}

// aten::set_(Tensor(a!) self) -> Tensor(a!)
inline Tensor & Tensor::set_() const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(*this);
    DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
    DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);
    switch (dispatchKeyToBackend(_dk)) {
        case Backend::CPU:
            return CPUType::set_(const_cast<Tensor&>(*this));
            break;
        default:
            AT_ERROR("set_ not implemented for ", at::toString(_dk));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::set_", "");
    return op.callUnboxed<Tensor &, Tensor &>(const_cast<Tensor&>(*this));
#endif
}

// aten::set_quantizer_(Tensor(a!) self, ConstQuantizerPtr quantizer) -> Tensor(a!)
inline Tensor & Tensor::set_quantizer_(ConstQuantizerPtr quantizer) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(*this);
    DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
    DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);
    switch (dispatchKeyToBackend(_dk)) {
        case Backend::QuantizedCPU:
            return QuantizedCPUType::set_quantizer_(const_cast<Tensor&>(*this), quantizer);
            break;
        default:
            AT_ERROR("set_quantizer_ not implemented for ", at::toString(_dk));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::set_quantizer_", "");
    return op.callUnboxed<Tensor &, Tensor &, ConstQuantizerPtr>(const_cast<Tensor&>(*this), quantizer);
#endif
}

// aten::is_set_to(Tensor self, Tensor tensor) -> bool
inline bool Tensor::is_set_to(const Tensor & tensor) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(*this, tensor);
    DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
    DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);
    switch (dispatchKeyToBackend(_dk)) {
        case Backend::CPU:
            return CPUType::is_set_to(const_cast<Tensor&>(*this), tensor);
            break;
        default:
            AT_ERROR("is_set_to not implemented for ", at::toString(_dk));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::is_set_to", "");
    return op.callUnboxed<bool, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), tensor);
#endif
}

// aten::masked_fill_.Scalar(Tensor(a!) self, Tensor mask, Scalar value) -> Tensor(a!)
inline Tensor & Tensor::masked_fill_(const Tensor & mask, Scalar value) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(*this, mask);
    DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
    DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);
    switch (dispatchKeyToBackend(_dk)) {
        case Backend::CPU:
            return CPUType::masked_fill__Scalar(const_cast<Tensor&>(*this), mask, value);
            break;
        default:
            AT_ERROR("masked_fill_ not implemented for ", at::toString(_dk));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::masked_fill_", "Scalar");
    return op.callUnboxed<Tensor &, Tensor &, const Tensor &, Scalar>(const_cast<Tensor&>(*this), mask, value);
#endif
}

// aten::masked_fill.Scalar(Tensor self, Tensor mask, Scalar value) -> Tensor
inline Tensor Tensor::masked_fill(const Tensor & mask, Scalar value) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::masked_fill_Scalar(const_cast<Tensor&>(*this), mask, value);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::masked_fill", "Scalar");
    return op.callUnboxed<Tensor, const Tensor &, const Tensor &, Scalar>(const_cast<Tensor&>(*this), mask, value);
#endif
}

// aten::masked_fill_.Tensor(Tensor(a!) self, Tensor mask, Tensor value) -> Tensor(a!)
inline Tensor & Tensor::masked_fill_(const Tensor & mask, const Tensor & value) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(*this, mask, value);
    DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
    DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);
    switch (dispatchKeyToBackend(_dk)) {
        case Backend::CPU:
            return CPUType::masked_fill__Tensor(const_cast<Tensor&>(*this), mask, value);
            break;
        default:
            AT_ERROR("masked_fill_ not implemented for ", at::toString(_dk));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::masked_fill_", "Tensor");
    return op.callUnboxed<Tensor &, Tensor &, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), mask, value);
#endif
}

// aten::masked_fill.Tensor(Tensor self, Tensor mask, Tensor value) -> Tensor
inline Tensor Tensor::masked_fill(const Tensor & mask, const Tensor & value) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::masked_fill_Tensor(const_cast<Tensor&>(*this), mask, value);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::masked_fill", "Tensor");
    return op.callUnboxed<Tensor, const Tensor &, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), mask, value);
#endif
}

// aten::masked_scatter_(Tensor(a!) self, Tensor mask, Tensor source) -> Tensor(a!)
inline Tensor & Tensor::masked_scatter_(const Tensor & mask, const Tensor & source) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(*this, mask, source);
    DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
    DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);
    switch (dispatchKeyToBackend(_dk)) {
        case Backend::CPU:
            return CPUType::masked_scatter_(const_cast<Tensor&>(*this), mask, source);
            break;
        default:
            AT_ERROR("masked_scatter_ not implemented for ", at::toString(_dk));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::masked_scatter_", "");
    return op.callUnboxed<Tensor &, Tensor &, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), mask, source);
#endif
}

// aten::masked_scatter(Tensor self, Tensor mask, Tensor source) -> Tensor
inline Tensor Tensor::masked_scatter(const Tensor & mask, const Tensor & source) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::masked_scatter(const_cast<Tensor&>(*this), mask, source);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::masked_scatter", "");
    return op.callUnboxed<Tensor, const Tensor &, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), mask, source);
#endif
}

// aten::view(Tensor(a) self, int[] size) -> Tensor(a)
inline Tensor Tensor::view(IntArrayRef size) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(*this);
    DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
    DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);
    switch (dispatchKeyToBackend(_dk)) {
        case Backend::CPU:
            return CPUType::view(const_cast<Tensor&>(*this), size);
            break;
        case Backend::QuantizedCPU:
            return QuantizedCPUType::view(const_cast<Tensor&>(*this), size);
            break;
        default:
            AT_ERROR("view not implemented for ", at::toString(_dk));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::view", "");
    return op.callUnboxed<Tensor, const Tensor &, IntArrayRef>(const_cast<Tensor&>(*this), size);
#endif
}

// aten::put_(Tensor(a!) self, Tensor index, Tensor source, bool accumulate=False) -> Tensor(a!)
inline Tensor & Tensor::put_(const Tensor & index, const Tensor & source, bool accumulate) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(*this, index, source);
    DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
    DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);
    switch (dispatchKeyToBackend(_dk)) {
        case Backend::CPU:
            return CPUType::put_(const_cast<Tensor&>(*this), index, source, accumulate);
            break;
        default:
            AT_ERROR("put_ not implemented for ", at::toString(_dk));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::put_", "");
    return op.callUnboxed<Tensor &, Tensor &, const Tensor &, const Tensor &, bool>(const_cast<Tensor&>(*this), index, source, accumulate);
#endif
}

// aten::index_add_(Tensor(a!) self, int dim, Tensor index, Tensor source) -> Tensor(a!)
inline Tensor & Tensor::index_add_(int64_t dim, const Tensor & index, const Tensor & source) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(*this, index, source);
    DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
    DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);
    switch (dispatchKeyToBackend(_dk)) {
        case Backend::CPU:
            return CPUType::index_add_(const_cast<Tensor&>(*this), dim, index, source);
            break;
        default:
            AT_ERROR("index_add_ not implemented for ", at::toString(_dk));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::index_add_", "");
    return op.callUnboxed<Tensor &, Tensor &, int64_t, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), dim, index, source);
#endif
}

// aten::index_add(Tensor self, int dim, Tensor index, Tensor source) -> Tensor
inline Tensor Tensor::index_add(int64_t dim, const Tensor & index, const Tensor & source) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::index_add(const_cast<Tensor&>(*this), dim, index, source);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::index_add", "");
    return op.callUnboxed<Tensor, const Tensor &, int64_t, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), dim, index, source);
#endif
}

// aten::index_add.dimname(Tensor self, Dimname dim, Tensor index, Tensor source) -> Tensor
inline Tensor Tensor::index_add(Dimname dim, const Tensor & index, const Tensor & source) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::index_add_dimname(const_cast<Tensor&>(*this), dim, index, source);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::index_add", "dimname");
    return op.callUnboxed<Tensor, const Tensor &, Dimname, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), dim, index, source);
#endif
}

// aten::index_fill_.int_Scalar(Tensor(a!) self, int dim, Tensor index, Scalar value) -> Tensor(a!)
inline Tensor & Tensor::index_fill_(int64_t dim, const Tensor & index, Scalar value) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(*this, index);
    DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
    DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);
    switch (dispatchKeyToBackend(_dk)) {
        case Backend::CPU:
            return CPUType::index_fill__int_Scalar(const_cast<Tensor&>(*this), dim, index, value);
            break;
        default:
            AT_ERROR("index_fill_ not implemented for ", at::toString(_dk));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::index_fill_", "int_Scalar");
    return op.callUnboxed<Tensor &, Tensor &, int64_t, const Tensor &, Scalar>(const_cast<Tensor&>(*this), dim, index, value);
#endif
}

// aten::index_fill.int_Scalar(Tensor self, int dim, Tensor index, Scalar value) -> Tensor
inline Tensor Tensor::index_fill(int64_t dim, const Tensor & index, Scalar value) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::index_fill_int_Scalar(const_cast<Tensor&>(*this), dim, index, value);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::index_fill", "int_Scalar");
    return op.callUnboxed<Tensor, const Tensor &, int64_t, const Tensor &, Scalar>(const_cast<Tensor&>(*this), dim, index, value);
#endif
}

// aten::index_fill_.int_Tensor(Tensor(a!) self, int dim, Tensor index, Tensor value) -> Tensor(a!)
inline Tensor & Tensor::index_fill_(int64_t dim, const Tensor & index, const Tensor & value) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(*this, index, value);
    DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
    DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);
    switch (dispatchKeyToBackend(_dk)) {
        case Backend::CPU:
            return CPUType::index_fill__int_Tensor(const_cast<Tensor&>(*this), dim, index, value);
            break;
        default:
            AT_ERROR("index_fill_ not implemented for ", at::toString(_dk));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::index_fill_", "int_Tensor");
    return op.callUnboxed<Tensor &, Tensor &, int64_t, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), dim, index, value);
#endif
}

// aten::index_fill.int_Tensor(Tensor self, int dim, Tensor index, Tensor value) -> Tensor
inline Tensor Tensor::index_fill(int64_t dim, const Tensor & index, const Tensor & value) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::index_fill_int_Tensor(const_cast<Tensor&>(*this), dim, index, value);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::index_fill", "int_Tensor");
    return op.callUnboxed<Tensor, const Tensor &, int64_t, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), dim, index, value);
#endif
}

// aten::index_fill_.Dimname_Scalar(Tensor(a!) self, Dimname dim, Tensor index, Scalar value) -> Tensor(a!)
inline Tensor & Tensor::index_fill_(Dimname dim, const Tensor & index, Scalar value) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::index_fill__Dimname_Scalar(const_cast<Tensor&>(*this), dim, index, value);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::index_fill_", "Dimname_Scalar");
    return op.callUnboxed<Tensor &, Tensor &, Dimname, const Tensor &, Scalar>(const_cast<Tensor&>(*this), dim, index, value);
#endif
}

// aten::index_fill_.Dimname_Tensor(Tensor(a!) self, Dimname dim, Tensor index, Tensor value) -> Tensor(a!)
inline Tensor & Tensor::index_fill_(Dimname dim, const Tensor & index, const Tensor & value) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::index_fill__Dimname_Tensor(const_cast<Tensor&>(*this), dim, index, value);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::index_fill_", "Dimname_Tensor");
    return op.callUnboxed<Tensor &, Tensor &, Dimname, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), dim, index, value);
#endif
}

// aten::index_fill.Dimname_Scalar(Tensor self, Dimname dim, Tensor index, Scalar value) -> Tensor
inline Tensor Tensor::index_fill(Dimname dim, const Tensor & index, Scalar value) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::index_fill_Dimname_Scalar(const_cast<Tensor&>(*this), dim, index, value);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::index_fill", "Dimname_Scalar");
    return op.callUnboxed<Tensor, const Tensor &, Dimname, const Tensor &, Scalar>(const_cast<Tensor&>(*this), dim, index, value);
#endif
}

// aten::index_fill.Dimname_Tensor(Tensor self, Dimname dim, Tensor index, Tensor value) -> Tensor
inline Tensor Tensor::index_fill(Dimname dim, const Tensor & index, const Tensor & value) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::index_fill_Dimname_Tensor(const_cast<Tensor&>(*this), dim, index, value);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::index_fill", "Dimname_Tensor");
    return op.callUnboxed<Tensor, const Tensor &, Dimname, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), dim, index, value);
#endif
}

// aten::scatter_.src(Tensor(a!) self, int dim, Tensor index, Tensor src) -> Tensor(a!)
inline Tensor & Tensor::scatter_(int64_t dim, const Tensor & index, const Tensor & src) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(*this, index, src);
    DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
    DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);
    switch (dispatchKeyToBackend(_dk)) {
        case Backend::CPU:
            return CPUType::scatter__src(const_cast<Tensor&>(*this), dim, index, src);
            break;
        default:
            AT_ERROR("scatter_ not implemented for ", at::toString(_dk));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::scatter_", "src");
    return op.callUnboxed<Tensor &, Tensor &, int64_t, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), dim, index, src);
#endif
}

// aten::scatter.src(Tensor self, int dim, Tensor index, Tensor src) -> Tensor
inline Tensor Tensor::scatter(int64_t dim, const Tensor & index, const Tensor & src) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::scatter_src(const_cast<Tensor&>(*this), dim, index, src);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::scatter", "src");
    return op.callUnboxed<Tensor, const Tensor &, int64_t, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), dim, index, src);
#endif
}

// aten::scatter_.value(Tensor(a!) self, int dim, Tensor index, Scalar value) -> Tensor(a!)
inline Tensor & Tensor::scatter_(int64_t dim, const Tensor & index, Scalar value) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(*this, index);
    DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
    DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);
    switch (dispatchKeyToBackend(_dk)) {
        case Backend::CPU:
            return CPUType::scatter__value(const_cast<Tensor&>(*this), dim, index, value);
            break;
        default:
            AT_ERROR("scatter_ not implemented for ", at::toString(_dk));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::scatter_", "value");
    return op.callUnboxed<Tensor &, Tensor &, int64_t, const Tensor &, Scalar>(const_cast<Tensor&>(*this), dim, index, value);
#endif
}

// aten::scatter.value(Tensor self, int dim, Tensor index, Scalar value) -> Tensor
inline Tensor Tensor::scatter(int64_t dim, const Tensor & index, Scalar value) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::scatter_value(const_cast<Tensor&>(*this), dim, index, value);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::scatter", "value");
    return op.callUnboxed<Tensor, const Tensor &, int64_t, const Tensor &, Scalar>(const_cast<Tensor&>(*this), dim, index, value);
#endif
}

// aten::scatter.dimname_src(Tensor self, Dimname dim, Tensor index, Tensor src) -> Tensor
inline Tensor Tensor::scatter(Dimname dim, const Tensor & index, const Tensor & src) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::scatter_dimname_src(const_cast<Tensor&>(*this), dim, index, src);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::scatter", "dimname_src");
    return op.callUnboxed<Tensor, const Tensor &, Dimname, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), dim, index, src);
#endif
}

// aten::scatter.dimname_value(Tensor self, Dimname dim, Tensor index, Scalar value) -> Tensor
inline Tensor Tensor::scatter(Dimname dim, const Tensor & index, Scalar value) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::scatter_dimname_value(const_cast<Tensor&>(*this), dim, index, value);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::scatter", "dimname_value");
    return op.callUnboxed<Tensor, const Tensor &, Dimname, const Tensor &, Scalar>(const_cast<Tensor&>(*this), dim, index, value);
#endif
}

// aten::scatter_add_(Tensor(a!) self, int dim, Tensor index, Tensor src) -> Tensor(a!)
inline Tensor & Tensor::scatter_add_(int64_t dim, const Tensor & index, const Tensor & src) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(*this, index, src);
    DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
    DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);
    switch (dispatchKeyToBackend(_dk)) {
        case Backend::CPU:
            return CPUType::scatter_add_(const_cast<Tensor&>(*this), dim, index, src);
            break;
        default:
            AT_ERROR("scatter_add_ not implemented for ", at::toString(_dk));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::scatter_add_", "");
    return op.callUnboxed<Tensor &, Tensor &, int64_t, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), dim, index, src);
#endif
}

// aten::scatter_add(Tensor self, int dim, Tensor index, Tensor src) -> Tensor
inline Tensor Tensor::scatter_add(int64_t dim, const Tensor & index, const Tensor & src) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::scatter_add(const_cast<Tensor&>(*this), dim, index, src);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::scatter_add", "");
    return op.callUnboxed<Tensor, const Tensor &, int64_t, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), dim, index, src);
#endif
}

// aten::scatter_add.dimname(Tensor self, Dimname dim, Tensor index, Tensor src) -> Tensor
inline Tensor Tensor::scatter_add(Dimname dim, const Tensor & index, const Tensor & src) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::scatter_add_dimname(const_cast<Tensor&>(*this), dim, index, src);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::scatter_add", "dimname");
    return op.callUnboxed<Tensor, const Tensor &, Dimname, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), dim, index, src);
#endif
}

// aten::lt_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
inline Tensor & Tensor::lt_(Scalar other) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::lt__Scalar(const_cast<Tensor&>(*this), other);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::lt_", "Scalar");
    return op.callUnboxed<Tensor &, Tensor &, Scalar>(const_cast<Tensor&>(*this), other);
#endif
}

// aten::lt_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
inline Tensor & Tensor::lt_(const Tensor & other) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::lt__Tensor(const_cast<Tensor&>(*this), other);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::lt_", "Tensor");
    return op.callUnboxed<Tensor &, Tensor &, const Tensor &>(const_cast<Tensor&>(*this), other);
#endif
}

// aten::gt_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
inline Tensor & Tensor::gt_(Scalar other) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::gt__Scalar(const_cast<Tensor&>(*this), other);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::gt_", "Scalar");
    return op.callUnboxed<Tensor &, Tensor &, Scalar>(const_cast<Tensor&>(*this), other);
#endif
}

// aten::gt_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
inline Tensor & Tensor::gt_(const Tensor & other) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::gt__Tensor(const_cast<Tensor&>(*this), other);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::gt_", "Tensor");
    return op.callUnboxed<Tensor &, Tensor &, const Tensor &>(const_cast<Tensor&>(*this), other);
#endif
}

// aten::le_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
inline Tensor & Tensor::le_(Scalar other) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::le__Scalar(const_cast<Tensor&>(*this), other);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::le_", "Scalar");
    return op.callUnboxed<Tensor &, Tensor &, Scalar>(const_cast<Tensor&>(*this), other);
#endif
}

// aten::le_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
inline Tensor & Tensor::le_(const Tensor & other) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::le__Tensor(const_cast<Tensor&>(*this), other);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::le_", "Tensor");
    return op.callUnboxed<Tensor &, Tensor &, const Tensor &>(const_cast<Tensor&>(*this), other);
#endif
}

// aten::ge_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
inline Tensor & Tensor::ge_(Scalar other) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::ge__Scalar(const_cast<Tensor&>(*this), other);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::ge_", "Scalar");
    return op.callUnboxed<Tensor &, Tensor &, Scalar>(const_cast<Tensor&>(*this), other);
#endif
}

// aten::ge_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
inline Tensor & Tensor::ge_(const Tensor & other) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::ge__Tensor(const_cast<Tensor&>(*this), other);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::ge_", "Tensor");
    return op.callUnboxed<Tensor &, Tensor &, const Tensor &>(const_cast<Tensor&>(*this), other);
#endif
}

// aten::eq_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
inline Tensor & Tensor::eq_(Scalar other) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::eq__Scalar(const_cast<Tensor&>(*this), other);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::eq_", "Scalar");
    return op.callUnboxed<Tensor &, Tensor &, Scalar>(const_cast<Tensor&>(*this), other);
#endif
}

// aten::eq_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
inline Tensor & Tensor::eq_(const Tensor & other) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::eq__Tensor(const_cast<Tensor&>(*this), other);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::eq_", "Tensor");
    return op.callUnboxed<Tensor &, Tensor &, const Tensor &>(const_cast<Tensor&>(*this), other);
#endif
}

// aten::ne_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
inline Tensor & Tensor::ne_(Scalar other) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::ne__Scalar(const_cast<Tensor&>(*this), other);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::ne_", "Scalar");
    return op.callUnboxed<Tensor &, Tensor &, Scalar>(const_cast<Tensor&>(*this), other);
#endif
}

// aten::ne_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
inline Tensor & Tensor::ne_(const Tensor & other) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::ne__Tensor(const_cast<Tensor&>(*this), other);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::ne_", "Tensor");
    return op.callUnboxed<Tensor &, Tensor &, const Tensor &>(const_cast<Tensor&>(*this), other);
#endif
}

// aten::bitwise_and.Scalar(Tensor self, Scalar other) -> Tensor
inline Tensor Tensor::bitwise_and(Scalar other) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::bitwise_and_Scalar(const_cast<Tensor&>(*this), other);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::bitwise_and", "Scalar");
    return op.callUnboxed<Tensor, const Tensor &, Scalar>(const_cast<Tensor&>(*this), other);
#endif
}

// aten::bitwise_and.Tensor(Tensor self, Tensor other) -> Tensor
inline Tensor Tensor::bitwise_and(const Tensor & other) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::bitwise_and_Tensor(const_cast<Tensor&>(*this), other);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::bitwise_and", "Tensor");
    return op.callUnboxed<Tensor, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), other);
#endif
}

// aten::bitwise_and_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
inline Tensor & Tensor::bitwise_and_(Scalar other) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::bitwise_and__Scalar(const_cast<Tensor&>(*this), other);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::bitwise_and_", "Scalar");
    return op.callUnboxed<Tensor &, Tensor &, Scalar>(const_cast<Tensor&>(*this), other);
#endif
}

// aten::bitwise_and_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
inline Tensor & Tensor::bitwise_and_(const Tensor & other) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::bitwise_and__Tensor(const_cast<Tensor&>(*this), other);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::bitwise_and_", "Tensor");
    return op.callUnboxed<Tensor &, Tensor &, const Tensor &>(const_cast<Tensor&>(*this), other);
#endif
}

// aten::__and__.Scalar(Tensor self, Scalar other) -> Tensor
inline Tensor Tensor::__and__(Scalar other) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::__and___Scalar(const_cast<Tensor&>(*this), other);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::__and__", "Scalar");
    return op.callUnboxed<Tensor, const Tensor &, Scalar>(const_cast<Tensor&>(*this), other);
#endif
}

// aten::__and__.Tensor(Tensor self, Tensor other) -> Tensor
inline Tensor Tensor::__and__(const Tensor & other) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::__and___Tensor(const_cast<Tensor&>(*this), other);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::__and__", "Tensor");
    return op.callUnboxed<Tensor, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), other);
#endif
}

// aten::__iand__.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
inline Tensor & Tensor::__iand__(Scalar other) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::__iand___Scalar(const_cast<Tensor&>(*this), other);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::__iand__", "Scalar");
    return op.callUnboxed<Tensor &, Tensor &, Scalar>(const_cast<Tensor&>(*this), other);
#endif
}

// aten::__iand__.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
inline Tensor & Tensor::__iand__(const Tensor & other) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::__iand___Tensor(const_cast<Tensor&>(*this), other);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::__iand__", "Tensor");
    return op.callUnboxed<Tensor &, Tensor &, const Tensor &>(const_cast<Tensor&>(*this), other);
#endif
}

// aten::bitwise_or.Scalar(Tensor self, Scalar other) -> Tensor
inline Tensor Tensor::bitwise_or(Scalar other) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::bitwise_or_Scalar(const_cast<Tensor&>(*this), other);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::bitwise_or", "Scalar");
    return op.callUnboxed<Tensor, const Tensor &, Scalar>(const_cast<Tensor&>(*this), other);
#endif
}

// aten::bitwise_or.Tensor(Tensor self, Tensor other) -> Tensor
inline Tensor Tensor::bitwise_or(const Tensor & other) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::bitwise_or_Tensor(const_cast<Tensor&>(*this), other);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::bitwise_or", "Tensor");
    return op.callUnboxed<Tensor, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), other);
#endif
}

// aten::bitwise_or_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
inline Tensor & Tensor::bitwise_or_(Scalar other) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::bitwise_or__Scalar(const_cast<Tensor&>(*this), other);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::bitwise_or_", "Scalar");
    return op.callUnboxed<Tensor &, Tensor &, Scalar>(const_cast<Tensor&>(*this), other);
#endif
}

// aten::bitwise_or_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
inline Tensor & Tensor::bitwise_or_(const Tensor & other) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::bitwise_or__Tensor(const_cast<Tensor&>(*this), other);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::bitwise_or_", "Tensor");
    return op.callUnboxed<Tensor &, Tensor &, const Tensor &>(const_cast<Tensor&>(*this), other);
#endif
}

// aten::__or__.Scalar(Tensor self, Scalar other) -> Tensor
inline Tensor Tensor::__or__(Scalar other) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::__or___Scalar(const_cast<Tensor&>(*this), other);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::__or__", "Scalar");
    return op.callUnboxed<Tensor, const Tensor &, Scalar>(const_cast<Tensor&>(*this), other);
#endif
}

// aten::__or__.Tensor(Tensor self, Tensor other) -> Tensor
inline Tensor Tensor::__or__(const Tensor & other) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::__or___Tensor(const_cast<Tensor&>(*this), other);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::__or__", "Tensor");
    return op.callUnboxed<Tensor, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), other);
#endif
}

// aten::__ior__.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
inline Tensor & Tensor::__ior__(Scalar other) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::__ior___Scalar(const_cast<Tensor&>(*this), other);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::__ior__", "Scalar");
    return op.callUnboxed<Tensor &, Tensor &, Scalar>(const_cast<Tensor&>(*this), other);
#endif
}

// aten::__ior__.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
inline Tensor & Tensor::__ior__(const Tensor & other) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::__ior___Tensor(const_cast<Tensor&>(*this), other);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::__ior__", "Tensor");
    return op.callUnboxed<Tensor &, Tensor &, const Tensor &>(const_cast<Tensor&>(*this), other);
#endif
}

// aten::bitwise_xor.Scalar(Tensor self, Scalar other) -> Tensor
inline Tensor Tensor::bitwise_xor(Scalar other) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::bitwise_xor_Scalar(const_cast<Tensor&>(*this), other);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::bitwise_xor", "Scalar");
    return op.callUnboxed<Tensor, const Tensor &, Scalar>(const_cast<Tensor&>(*this), other);
#endif
}

// aten::bitwise_xor.Tensor(Tensor self, Tensor other) -> Tensor
inline Tensor Tensor::bitwise_xor(const Tensor & other) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::bitwise_xor_Tensor(const_cast<Tensor&>(*this), other);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::bitwise_xor", "Tensor");
    return op.callUnboxed<Tensor, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), other);
#endif
}

// aten::bitwise_xor_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
inline Tensor & Tensor::bitwise_xor_(Scalar other) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::bitwise_xor__Scalar(const_cast<Tensor&>(*this), other);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::bitwise_xor_", "Scalar");
    return op.callUnboxed<Tensor &, Tensor &, Scalar>(const_cast<Tensor&>(*this), other);
#endif
}

// aten::bitwise_xor_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
inline Tensor & Tensor::bitwise_xor_(const Tensor & other) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::bitwise_xor__Tensor(const_cast<Tensor&>(*this), other);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::bitwise_xor_", "Tensor");
    return op.callUnboxed<Tensor &, Tensor &, const Tensor &>(const_cast<Tensor&>(*this), other);
#endif
}

// aten::__xor__.Scalar(Tensor self, Scalar other) -> Tensor
inline Tensor Tensor::__xor__(Scalar other) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::__xor___Scalar(const_cast<Tensor&>(*this), other);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::__xor__", "Scalar");
    return op.callUnboxed<Tensor, const Tensor &, Scalar>(const_cast<Tensor&>(*this), other);
#endif
}

// aten::__xor__.Tensor(Tensor self, Tensor other) -> Tensor
inline Tensor Tensor::__xor__(const Tensor & other) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::__xor___Tensor(const_cast<Tensor&>(*this), other);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::__xor__", "Tensor");
    return op.callUnboxed<Tensor, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), other);
#endif
}

// aten::__ixor__.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
inline Tensor & Tensor::__ixor__(Scalar other) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::__ixor___Scalar(const_cast<Tensor&>(*this), other);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::__ixor__", "Scalar");
    return op.callUnboxed<Tensor &, Tensor &, Scalar>(const_cast<Tensor&>(*this), other);
#endif
}

// aten::__ixor__.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
inline Tensor & Tensor::__ixor__(const Tensor & other) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::__ixor___Tensor(const_cast<Tensor&>(*this), other);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::__ixor__", "Tensor");
    return op.callUnboxed<Tensor &, Tensor &, const Tensor &>(const_cast<Tensor&>(*this), other);
#endif
}

// aten::__lshift__.Scalar(Tensor self, Scalar other) -> Tensor
inline Tensor Tensor::__lshift__(Scalar other) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(*this);
    DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
    DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);
    switch (dispatchKeyToBackend(_dk)) {
        case Backend::CPU:
            return CPUType::__lshift___Scalar(const_cast<Tensor&>(*this), other);
            break;
        default:
            AT_ERROR("__lshift__ not implemented for ", at::toString(_dk));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::__lshift__", "Scalar");
    return op.callUnboxed<Tensor, const Tensor &, Scalar>(const_cast<Tensor&>(*this), other);
#endif
}

// aten::__lshift__.Tensor(Tensor self, Tensor other) -> Tensor
inline Tensor Tensor::__lshift__(const Tensor & other) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(*this, other);
    DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
    DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);
    switch (dispatchKeyToBackend(_dk)) {
        case Backend::CPU:
            return CPUType::__lshift___Tensor(const_cast<Tensor&>(*this), other);
            break;
        default:
            AT_ERROR("__lshift__ not implemented for ", at::toString(_dk));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::__lshift__", "Tensor");
    return op.callUnboxed<Tensor, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), other);
#endif
}

// aten::__ilshift__.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
inline Tensor & Tensor::__ilshift__(Scalar other) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(*this);
    DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
    DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);
    switch (dispatchKeyToBackend(_dk)) {
        case Backend::CPU:
            return CPUType::__ilshift___Scalar(const_cast<Tensor&>(*this), other);
            break;
        default:
            AT_ERROR("__ilshift__ not implemented for ", at::toString(_dk));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::__ilshift__", "Scalar");
    return op.callUnboxed<Tensor &, Tensor &, Scalar>(const_cast<Tensor&>(*this), other);
#endif
}

// aten::__ilshift__.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
inline Tensor & Tensor::__ilshift__(const Tensor & other) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(*this, other);
    DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
    DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);
    switch (dispatchKeyToBackend(_dk)) {
        case Backend::CPU:
            return CPUType::__ilshift___Tensor(const_cast<Tensor&>(*this), other);
            break;
        default:
            AT_ERROR("__ilshift__ not implemented for ", at::toString(_dk));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::__ilshift__", "Tensor");
    return op.callUnboxed<Tensor &, Tensor &, const Tensor &>(const_cast<Tensor&>(*this), other);
#endif
}

// aten::__rshift__.Scalar(Tensor self, Scalar other) -> Tensor
inline Tensor Tensor::__rshift__(Scalar other) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(*this);
    DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
    DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);
    switch (dispatchKeyToBackend(_dk)) {
        case Backend::CPU:
            return CPUType::__rshift___Scalar(const_cast<Tensor&>(*this), other);
            break;
        default:
            AT_ERROR("__rshift__ not implemented for ", at::toString(_dk));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::__rshift__", "Scalar");
    return op.callUnboxed<Tensor, const Tensor &, Scalar>(const_cast<Tensor&>(*this), other);
#endif
}

// aten::__rshift__.Tensor(Tensor self, Tensor other) -> Tensor
inline Tensor Tensor::__rshift__(const Tensor & other) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(*this, other);
    DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
    DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);
    switch (dispatchKeyToBackend(_dk)) {
        case Backend::CPU:
            return CPUType::__rshift___Tensor(const_cast<Tensor&>(*this), other);
            break;
        default:
            AT_ERROR("__rshift__ not implemented for ", at::toString(_dk));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::__rshift__", "Tensor");
    return op.callUnboxed<Tensor, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), other);
#endif
}

// aten::__irshift__.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
inline Tensor & Tensor::__irshift__(Scalar other) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(*this);
    DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
    DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);
    switch (dispatchKeyToBackend(_dk)) {
        case Backend::CPU:
            return CPUType::__irshift___Scalar(const_cast<Tensor&>(*this), other);
            break;
        default:
            AT_ERROR("__irshift__ not implemented for ", at::toString(_dk));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::__irshift__", "Scalar");
    return op.callUnboxed<Tensor &, Tensor &, Scalar>(const_cast<Tensor&>(*this), other);
#endif
}

// aten::__irshift__.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
inline Tensor & Tensor::__irshift__(const Tensor & other) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(*this, other);
    DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
    DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);
    switch (dispatchKeyToBackend(_dk)) {
        case Backend::CPU:
            return CPUType::__irshift___Tensor(const_cast<Tensor&>(*this), other);
            break;
        default:
            AT_ERROR("__irshift__ not implemented for ", at::toString(_dk));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::__irshift__", "Tensor");
    return op.callUnboxed<Tensor &, Tensor &, const Tensor &>(const_cast<Tensor&>(*this), other);
#endif
}

// aten::lgamma_(Tensor(a!) self) -> Tensor(a!)
inline Tensor & Tensor::lgamma_() const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(*this);
    DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
    DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);
    switch (dispatchKeyToBackend(_dk)) {
        case Backend::CPU:
            return CPUType::lgamma_(const_cast<Tensor&>(*this));
            break;
        default:
            AT_ERROR("lgamma_ not implemented for ", at::toString(_dk));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::lgamma_", "");
    return op.callUnboxed<Tensor &, Tensor &>(const_cast<Tensor&>(*this));
#endif
}

// aten::atan2_(Tensor(a!) self, Tensor other) -> Tensor(a!)
inline Tensor & Tensor::atan2_(const Tensor & other) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::atan2_(const_cast<Tensor&>(*this), other);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::atan2_", "");
    return op.callUnboxed<Tensor &, Tensor &, const Tensor &>(const_cast<Tensor&>(*this), other);
#endif
}

// aten::tril_(Tensor(a!) self, int diagonal=0) -> Tensor(a!)
inline Tensor & Tensor::tril_(int64_t diagonal) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(*this);
    DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
    DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);
    switch (dispatchKeyToBackend(_dk)) {
        case Backend::CPU:
            return CPUType::tril_(const_cast<Tensor&>(*this), diagonal);
            break;
        default:
            AT_ERROR("tril_ not implemented for ", at::toString(_dk));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::tril_", "");
    return op.callUnboxed<Tensor &, Tensor &, int64_t>(const_cast<Tensor&>(*this), diagonal);
#endif
}

// aten::triu_(Tensor(a!) self, int diagonal=0) -> Tensor(a!)
inline Tensor & Tensor::triu_(int64_t diagonal) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(*this);
    DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
    DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);
    switch (dispatchKeyToBackend(_dk)) {
        case Backend::CPU:
            return CPUType::triu_(const_cast<Tensor&>(*this), diagonal);
            break;
        default:
            AT_ERROR("triu_ not implemented for ", at::toString(_dk));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::triu_", "");
    return op.callUnboxed<Tensor &, Tensor &, int64_t>(const_cast<Tensor&>(*this), diagonal);
#endif
}

// aten::digamma_(Tensor(a!) self) -> Tensor(a!)
inline Tensor & Tensor::digamma_() const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::digamma_(const_cast<Tensor&>(*this));
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::digamma_", "");
    return op.callUnboxed<Tensor &, Tensor &>(const_cast<Tensor&>(*this));
#endif
}

// aten::polygamma_(Tensor(a!) self, int n) -> Tensor(a!)
inline Tensor & Tensor::polygamma_(int64_t n) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::polygamma_(const_cast<Tensor&>(*this), n);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::polygamma_", "");
    return op.callUnboxed<Tensor &, Tensor &, int64_t>(const_cast<Tensor&>(*this), n);
#endif
}

// aten::renorm_(Tensor(a!) self, Scalar p, int dim, Scalar maxnorm) -> Tensor(a!)
inline Tensor & Tensor::renorm_(Scalar p, int64_t dim, Scalar maxnorm) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(*this);
    DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
    DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);
    switch (dispatchKeyToBackend(_dk)) {
        case Backend::CPU:
            return CPUType::renorm_(const_cast<Tensor&>(*this), p, dim, maxnorm);
            break;
        default:
            AT_ERROR("renorm_ not implemented for ", at::toString(_dk));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::renorm_", "");
    return op.callUnboxed<Tensor &, Tensor &, Scalar, int64_t, Scalar>(const_cast<Tensor&>(*this), p, dim, maxnorm);
#endif
}

// aten::pow_.Scalar(Tensor(a!) self, Scalar exponent) -> Tensor(a!)
inline Tensor & Tensor::pow_(Scalar exponent) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(*this);
    DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
    DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);
    switch (dispatchKeyToBackend(_dk)) {
        case Backend::CPU:
            return CPUType::pow__Scalar(const_cast<Tensor&>(*this), exponent);
            break;
        default:
            AT_ERROR("pow_ not implemented for ", at::toString(_dk));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::pow_", "Scalar");
    return op.callUnboxed<Tensor &, Tensor &, Scalar>(const_cast<Tensor&>(*this), exponent);
#endif
}

// aten::pow_.Tensor(Tensor(a!) self, Tensor exponent) -> Tensor(a!)
inline Tensor & Tensor::pow_(const Tensor & exponent) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(*this, exponent);
    DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
    DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);
    switch (dispatchKeyToBackend(_dk)) {
        case Backend::CPU:
            return CPUType::pow__Tensor(const_cast<Tensor&>(*this), exponent);
            break;
        default:
            AT_ERROR("pow_ not implemented for ", at::toString(_dk));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::pow_", "Tensor");
    return op.callUnboxed<Tensor &, Tensor &, const Tensor &>(const_cast<Tensor&>(*this), exponent);
#endif
}

// aten::lerp_.Scalar(Tensor(a!) self, Tensor end, Scalar weight) -> Tensor(a!)
inline Tensor & Tensor::lerp_(const Tensor & end, Scalar weight) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(*this, end);
    DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
    DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);
    switch (dispatchKeyToBackend(_dk)) {
        case Backend::CPU:
            return CPUType::lerp__Scalar(const_cast<Tensor&>(*this), end, weight);
            break;
        default:
            AT_ERROR("lerp_ not implemented for ", at::toString(_dk));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::lerp_", "Scalar");
    return op.callUnboxed<Tensor &, Tensor &, const Tensor &, Scalar>(const_cast<Tensor&>(*this), end, weight);
#endif
}

// aten::lerp_.Tensor(Tensor(a!) self, Tensor end, Tensor weight) -> Tensor(a!)
inline Tensor & Tensor::lerp_(const Tensor & end, const Tensor & weight) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(*this, end, weight);
    DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
    DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);
    switch (dispatchKeyToBackend(_dk)) {
        case Backend::CPU:
            return CPUType::lerp__Tensor(const_cast<Tensor&>(*this), end, weight);
            break;
        default:
            AT_ERROR("lerp_ not implemented for ", at::toString(_dk));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::lerp_", "Tensor");
    return op.callUnboxed<Tensor &, Tensor &, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), end, weight);
#endif
}

// aten::fmod_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
inline Tensor & Tensor::fmod_(Scalar other) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(*this);
    DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
    DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);
    switch (dispatchKeyToBackend(_dk)) {
        case Backend::CPU:
            return CPUType::fmod__Scalar(const_cast<Tensor&>(*this), other);
            break;
        default:
            AT_ERROR("fmod_ not implemented for ", at::toString(_dk));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::fmod_", "Scalar");
    return op.callUnboxed<Tensor &, Tensor &, Scalar>(const_cast<Tensor&>(*this), other);
#endif
}

// aten::fmod_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
inline Tensor & Tensor::fmod_(const Tensor & other) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(*this, other);
    DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
    DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);
    switch (dispatchKeyToBackend(_dk)) {
        case Backend::CPU:
            return CPUType::fmod__Tensor(const_cast<Tensor&>(*this), other);
            break;
        default:
            AT_ERROR("fmod_ not implemented for ", at::toString(_dk));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::fmod_", "Tensor");
    return op.callUnboxed<Tensor &, Tensor &, const Tensor &>(const_cast<Tensor&>(*this), other);
#endif
}

// aten::remainder_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
inline Tensor & Tensor::remainder_(Scalar other) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(*this);
    DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
    DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);
    switch (dispatchKeyToBackend(_dk)) {
        case Backend::CPU:
            return CPUType::remainder__Scalar(const_cast<Tensor&>(*this), other);
            break;
        default:
            AT_ERROR("remainder_ not implemented for ", at::toString(_dk));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::remainder_", "Scalar");
    return op.callUnboxed<Tensor &, Tensor &, Scalar>(const_cast<Tensor&>(*this), other);
#endif
}

// aten::remainder_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
inline Tensor & Tensor::remainder_(const Tensor & other) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(*this, other);
    DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
    DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);
    switch (dispatchKeyToBackend(_dk)) {
        case Backend::CPU:
            return CPUType::remainder__Tensor(const_cast<Tensor&>(*this), other);
            break;
        default:
            AT_ERROR("remainder_ not implemented for ", at::toString(_dk));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::remainder_", "Tensor");
    return op.callUnboxed<Tensor &, Tensor &, const Tensor &>(const_cast<Tensor&>(*this), other);
#endif
}

// aten::addbmm_(Tensor(a!) self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1) -> Tensor(a!)
inline Tensor & Tensor::addbmm_(const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(*this, batch1, batch2);
    DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
    DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);
    switch (dispatchKeyToBackend(_dk)) {
        case Backend::CPU:
            return CPUType::addbmm_(const_cast<Tensor&>(*this), batch1, batch2, beta, alpha);
            break;
        default:
            AT_ERROR("addbmm_ not implemented for ", at::toString(_dk));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::addbmm_", "");
    return op.callUnboxed<Tensor &, Tensor &, const Tensor &, const Tensor &, Scalar, Scalar>(const_cast<Tensor&>(*this), batch1, batch2, beta, alpha);
#endif
}

// aten::addbmm(Tensor self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1) -> Tensor
inline Tensor Tensor::addbmm(const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(*this, batch1, batch2);
    DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
    DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);
    switch (dispatchKeyToBackend(_dk)) {
        case Backend::CPU:
            return CPUType::addbmm(const_cast<Tensor&>(*this), batch1, batch2, beta, alpha);
            break;
        default:
            AT_ERROR("addbmm not implemented for ", at::toString(_dk));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::addbmm", "");
    return op.callUnboxed<Tensor, const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar>(const_cast<Tensor&>(*this), batch1, batch2, beta, alpha);
#endif
}

// aten::addcdiv_(Tensor(a!) self, Tensor tensor1, Tensor tensor2, *, Scalar value=1) -> Tensor(a!)
inline Tensor & Tensor::addcdiv_(const Tensor & tensor1, const Tensor & tensor2, Scalar value) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::addcdiv_(const_cast<Tensor&>(*this), tensor1, tensor2, value);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::addcdiv_", "");
    return op.callUnboxed<Tensor &, Tensor &, const Tensor &, const Tensor &, Scalar>(const_cast<Tensor&>(*this), tensor1, tensor2, value);
#endif
}

// aten::random_.from(Tensor(a!) self, int from, int? to, *, Generator? generator=None) -> Tensor(a!)
inline Tensor & Tensor::random_(int64_t from, c10::optional<int64_t> to, c10::optional<Generator> generator) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::random__from(const_cast<Tensor&>(*this), from, to, generator);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::random_", "from");
    return op.callUnboxed<Tensor &, Tensor &, int64_t, c10::optional<int64_t>, c10::optional<Generator>>(const_cast<Tensor&>(*this), from, to, generator);
#endif
}

// aten::random_.to(Tensor(a!) self, int to, *, Generator? generator=None) -> Tensor(a!)
inline Tensor & Tensor::random_(int64_t to, c10::optional<Generator> generator) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::random__to(const_cast<Tensor&>(*this), to, generator);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::random_", "to");
    return op.callUnboxed<Tensor &, Tensor &, int64_t, c10::optional<Generator>>(const_cast<Tensor&>(*this), to, generator);
#endif
}

// aten::random_(Tensor(a!) self, *, Generator? generator=None) -> Tensor(a!)
inline Tensor & Tensor::random_(c10::optional<Generator> generator) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::random_(const_cast<Tensor&>(*this), generator);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::random_", "");
    return op.callUnboxed<Tensor &, Tensor &, c10::optional<Generator>>(const_cast<Tensor&>(*this), generator);
#endif
}

// aten::uniform_(Tensor(a!) self, float from=0, float to=1, *, Generator? generator=None) -> Tensor(a!)
inline Tensor & Tensor::uniform_(double from, double to, c10::optional<Generator> generator) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::uniform_(const_cast<Tensor&>(*this), from, to, generator);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::uniform_", "");
    return op.callUnboxed<Tensor &, Tensor &, double, double, c10::optional<Generator>>(const_cast<Tensor&>(*this), from, to, generator);
#endif
}

// aten::cauchy_(Tensor(a!) self, float median=0, float sigma=1, *, Generator? generator=None) -> Tensor(a!)
inline Tensor & Tensor::cauchy_(double median, double sigma, c10::optional<Generator> generator) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::cauchy_(const_cast<Tensor&>(*this), median, sigma, generator);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::cauchy_", "");
    return op.callUnboxed<Tensor &, Tensor &, double, double, c10::optional<Generator>>(const_cast<Tensor&>(*this), median, sigma, generator);
#endif
}

// aten::log_normal_(Tensor(a!) self, float mean=1, float std=2, *, Generator? generator=None) -> Tensor(a!)
inline Tensor & Tensor::log_normal_(double mean, double std, c10::optional<Generator> generator) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::log_normal_(const_cast<Tensor&>(*this), mean, std, generator);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::log_normal_", "");
    return op.callUnboxed<Tensor &, Tensor &, double, double, c10::optional<Generator>>(const_cast<Tensor&>(*this), mean, std, generator);
#endif
}

// aten::exponential_(Tensor(a!) self, float lambd=1, *, Generator? generator=None) -> Tensor(a!)
inline Tensor & Tensor::exponential_(double lambd, c10::optional<Generator> generator) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::exponential_(const_cast<Tensor&>(*this), lambd, generator);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::exponential_", "");
    return op.callUnboxed<Tensor &, Tensor &, double, c10::optional<Generator>>(const_cast<Tensor&>(*this), lambd, generator);
#endif
}

// aten::geometric_(Tensor(a!) self, float p, *, Generator? generator=None) -> Tensor(a!)
inline Tensor & Tensor::geometric_(double p, c10::optional<Generator> generator) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::geometric_(const_cast<Tensor&>(*this), p, generator);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::geometric_", "");
    return op.callUnboxed<Tensor &, Tensor &, double, c10::optional<Generator>>(const_cast<Tensor&>(*this), p, generator);
#endif
}

// aten::diag(Tensor self, int diagonal=0) -> Tensor
inline Tensor Tensor::diag(int64_t diagonal) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(*this);
    DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
    DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);
    switch (dispatchKeyToBackend(_dk)) {
        case Backend::CPU:
            return CPUType::diag(const_cast<Tensor&>(*this), diagonal);
            break;
        default:
            AT_ERROR("diag not implemented for ", at::toString(_dk));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::diag", "");
    return op.callUnboxed<Tensor, const Tensor &, int64_t>(const_cast<Tensor&>(*this), diagonal);
#endif
}

// aten::cross(Tensor self, Tensor other, int? dim=None) -> Tensor
inline Tensor Tensor::cross(const Tensor & other, c10::optional<int64_t> dim) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::cross(const_cast<Tensor&>(*this), other, dim);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::cross", "");
    return op.callUnboxed<Tensor, const Tensor &, const Tensor &, c10::optional<int64_t>>(const_cast<Tensor&>(*this), other, dim);
#endif
}

// aten::triu(Tensor self, int diagonal=0) -> Tensor
inline Tensor Tensor::triu(int64_t diagonal) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::triu(const_cast<Tensor&>(*this), diagonal);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::triu", "");
    return op.callUnboxed<Tensor, const Tensor &, int64_t>(const_cast<Tensor&>(*this), diagonal);
#endif
}

// aten::tril(Tensor self, int diagonal=0) -> Tensor
inline Tensor Tensor::tril(int64_t diagonal) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::tril(const_cast<Tensor&>(*this), diagonal);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::tril", "");
    return op.callUnboxed<Tensor, const Tensor &, int64_t>(const_cast<Tensor&>(*this), diagonal);
#endif
}

// aten::trace(Tensor self) -> Tensor
inline Tensor Tensor::trace() const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(*this);
    DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
    DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);
    switch (dispatchKeyToBackend(_dk)) {
        case Backend::CPU:
            return CPUType::trace(const_cast<Tensor&>(*this));
            break;
        default:
            AT_ERROR("trace not implemented for ", at::toString(_dk));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::trace", "");
    return op.callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}

// aten::ne.Scalar(Tensor self, Scalar other) -> Tensor
inline Tensor Tensor::ne(Scalar other) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(*this);
    DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
    DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);
    switch (dispatchKeyToBackend(_dk)) {
        case Backend::CPU:
            return CPUType::ne_Scalar(const_cast<Tensor&>(*this), other);
            break;
        case Backend::QuantizedCPU:
            return QuantizedCPUType::ne_Scalar(const_cast<Tensor&>(*this), other);
            break;
        default:
            AT_ERROR("ne not implemented for ", at::toString(_dk));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::ne", "Scalar");
    return op.callUnboxed<Tensor, const Tensor &, Scalar>(const_cast<Tensor&>(*this), other);
#endif
}

// aten::ne.Tensor(Tensor self, Tensor other) -> Tensor
inline Tensor Tensor::ne(const Tensor & other) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(*this, other);
    DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
    DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);
    switch (dispatchKeyToBackend(_dk)) {
        case Backend::CPU:
            return CPUType::ne_Tensor(const_cast<Tensor&>(*this), other);
            break;
        case Backend::QuantizedCPU:
            return QuantizedCPUType::ne_Tensor(const_cast<Tensor&>(*this), other);
            break;
        default:
            AT_ERROR("ne not implemented for ", at::toString(_dk));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::ne", "Tensor");
    return op.callUnboxed<Tensor, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), other);
#endif
}

// aten::eq.Scalar(Tensor self, Scalar other) -> Tensor
inline Tensor Tensor::eq(Scalar other) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(*this);
    DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
    DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);
    switch (dispatchKeyToBackend(_dk)) {
        case Backend::CPU:
            return CPUType::eq_Scalar(const_cast<Tensor&>(*this), other);
            break;
        case Backend::QuantizedCPU:
            return QuantizedCPUType::eq_Scalar(const_cast<Tensor&>(*this), other);
            break;
        default:
            AT_ERROR("eq not implemented for ", at::toString(_dk));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::eq", "Scalar");
    return op.callUnboxed<Tensor, const Tensor &, Scalar>(const_cast<Tensor&>(*this), other);
#endif
}

// aten::eq.Tensor(Tensor self, Tensor other) -> Tensor
inline Tensor Tensor::eq(const Tensor & other) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(*this, other);
    DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
    DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);
    switch (dispatchKeyToBackend(_dk)) {
        case Backend::CPU:
            return CPUType::eq_Tensor(const_cast<Tensor&>(*this), other);
            break;
        case Backend::QuantizedCPU:
            return QuantizedCPUType::eq_Tensor(const_cast<Tensor&>(*this), other);
            break;
        default:
            AT_ERROR("eq not implemented for ", at::toString(_dk));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::eq", "Tensor");
    return op.callUnboxed<Tensor, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), other);
#endif
}

// aten::ge.Scalar(Tensor self, Scalar other) -> Tensor
inline Tensor Tensor::ge(Scalar other) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(*this);
    DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
    DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);
    switch (dispatchKeyToBackend(_dk)) {
        case Backend::CPU:
            return CPUType::ge_Scalar(const_cast<Tensor&>(*this), other);
            break;
        case Backend::QuantizedCPU:
            return QuantizedCPUType::ge_Scalar(const_cast<Tensor&>(*this), other);
            break;
        default:
            AT_ERROR("ge not implemented for ", at::toString(_dk));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::ge", "Scalar");
    return op.callUnboxed<Tensor, const Tensor &, Scalar>(const_cast<Tensor&>(*this), other);
#endif
}

// aten::ge.Tensor(Tensor self, Tensor other) -> Tensor
inline Tensor Tensor::ge(const Tensor & other) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(*this, other);
    DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
    DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);
    switch (dispatchKeyToBackend(_dk)) {
        case Backend::CPU:
            return CPUType::ge_Tensor(const_cast<Tensor&>(*this), other);
            break;
        case Backend::QuantizedCPU:
            return QuantizedCPUType::ge_Tensor(const_cast<Tensor&>(*this), other);
            break;
        default:
            AT_ERROR("ge not implemented for ", at::toString(_dk));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::ge", "Tensor");
    return op.callUnboxed<Tensor, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), other);
#endif
}

// aten::le.Scalar(Tensor self, Scalar other) -> Tensor
inline Tensor Tensor::le(Scalar other) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(*this);
    DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
    DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);
    switch (dispatchKeyToBackend(_dk)) {
        case Backend::CPU:
            return CPUType::le_Scalar(const_cast<Tensor&>(*this), other);
            break;
        case Backend::QuantizedCPU:
            return QuantizedCPUType::le_Scalar(const_cast<Tensor&>(*this), other);
            break;
        default:
            AT_ERROR("le not implemented for ", at::toString(_dk));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::le", "Scalar");
    return op.callUnboxed<Tensor, const Tensor &, Scalar>(const_cast<Tensor&>(*this), other);
#endif
}

// aten::le.Tensor(Tensor self, Tensor other) -> Tensor
inline Tensor Tensor::le(const Tensor & other) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(*this, other);
    DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
    DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);
    switch (dispatchKeyToBackend(_dk)) {
        case Backend::CPU:
            return CPUType::le_Tensor(const_cast<Tensor&>(*this), other);
            break;
        case Backend::QuantizedCPU:
            return QuantizedCPUType::le_Tensor(const_cast<Tensor&>(*this), other);
            break;
        default:
            AT_ERROR("le not implemented for ", at::toString(_dk));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::le", "Tensor");
    return op.callUnboxed<Tensor, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), other);
#endif
}

// aten::gt.Scalar(Tensor self, Scalar other) -> Tensor
inline Tensor Tensor::gt(Scalar other) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(*this);
    DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
    DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);
    switch (dispatchKeyToBackend(_dk)) {
        case Backend::CPU:
            return CPUType::gt_Scalar(const_cast<Tensor&>(*this), other);
            break;
        case Backend::QuantizedCPU:
            return QuantizedCPUType::gt_Scalar(const_cast<Tensor&>(*this), other);
            break;
        default:
            AT_ERROR("gt not implemented for ", at::toString(_dk));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::gt", "Scalar");
    return op.callUnboxed<Tensor, const Tensor &, Scalar>(const_cast<Tensor&>(*this), other);
#endif
}

// aten::gt.Tensor(Tensor self, Tensor other) -> Tensor
inline Tensor Tensor::gt(const Tensor & other) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(*this, other);
    DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
    DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);
    switch (dispatchKeyToBackend(_dk)) {
        case Backend::CPU:
            return CPUType::gt_Tensor(const_cast<Tensor&>(*this), other);
            break;
        case Backend::QuantizedCPU:
            return QuantizedCPUType::gt_Tensor(const_cast<Tensor&>(*this), other);
            break;
        default:
            AT_ERROR("gt not implemented for ", at::toString(_dk));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::gt", "Tensor");
    return op.callUnboxed<Tensor, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), other);
#endif
}

// aten::lt.Scalar(Tensor self, Scalar other) -> Tensor
inline Tensor Tensor::lt(Scalar other) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(*this);
    DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
    DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);
    switch (dispatchKeyToBackend(_dk)) {
        case Backend::CPU:
            return CPUType::lt_Scalar(const_cast<Tensor&>(*this), other);
            break;
        case Backend::QuantizedCPU:
            return QuantizedCPUType::lt_Scalar(const_cast<Tensor&>(*this), other);
            break;
        default:
            AT_ERROR("lt not implemented for ", at::toString(_dk));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::lt", "Scalar");
    return op.callUnboxed<Tensor, const Tensor &, Scalar>(const_cast<Tensor&>(*this), other);
#endif
}

// aten::lt.Tensor(Tensor self, Tensor other) -> Tensor
inline Tensor Tensor::lt(const Tensor & other) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(*this, other);
    DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
    DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);
    switch (dispatchKeyToBackend(_dk)) {
        case Backend::CPU:
            return CPUType::lt_Tensor(const_cast<Tensor&>(*this), other);
            break;
        case Backend::QuantizedCPU:
            return QuantizedCPUType::lt_Tensor(const_cast<Tensor&>(*this), other);
            break;
        default:
            AT_ERROR("lt not implemented for ", at::toString(_dk));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::lt", "Tensor");
    return op.callUnboxed<Tensor, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), other);
#endif
}

// aten::take(Tensor self, Tensor index) -> Tensor
inline Tensor Tensor::take(const Tensor & index) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(*this, index);
    DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
    DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);
    switch (dispatchKeyToBackend(_dk)) {
        case Backend::CPU:
            return CPUType::take(const_cast<Tensor&>(*this), index);
            break;
        default:
            AT_ERROR("take not implemented for ", at::toString(_dk));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::take", "");
    return op.callUnboxed<Tensor, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), index);
#endif
}

// aten::index_select(Tensor self, int dim, Tensor index) -> Tensor
inline Tensor Tensor::index_select(int64_t dim, const Tensor & index) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(*this, index);
    DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
    DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);
    switch (dispatchKeyToBackend(_dk)) {
        case Backend::CPU:
            return CPUType::index_select(const_cast<Tensor&>(*this), dim, index);
            break;
        default:
            AT_ERROR("index_select not implemented for ", at::toString(_dk));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::index_select", "");
    return op.callUnboxed<Tensor, const Tensor &, int64_t, const Tensor &>(const_cast<Tensor&>(*this), dim, index);
#endif
}

// aten::index_select.dimname(Tensor self, Dimname dim, Tensor index) -> Tensor
inline Tensor Tensor::index_select(Dimname dim, const Tensor & index) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::index_select_dimname(const_cast<Tensor&>(*this), dim, index);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::index_select", "dimname");
    return op.callUnboxed<Tensor, const Tensor &, Dimname, const Tensor &>(const_cast<Tensor&>(*this), dim, index);
#endif
}

// aten::masked_select(Tensor self, Tensor mask) -> Tensor
inline Tensor Tensor::masked_select(const Tensor & mask) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(*this, mask);
    DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
    DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);
    switch (dispatchKeyToBackend(_dk)) {
        case Backend::CPU:
            return CPUType::masked_select(const_cast<Tensor&>(*this), mask);
            break;
        default:
            AT_ERROR("masked_select not implemented for ", at::toString(_dk));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::masked_select", "");
    return op.callUnboxed<Tensor, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), mask);
#endif
}

// aten::nonzero(Tensor self) -> Tensor
inline Tensor Tensor::nonzero() const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(*this);
    DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
    DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);
    switch (dispatchKeyToBackend(_dk)) {
        case Backend::CPU:
            return CPUType::nonzero(const_cast<Tensor&>(*this));
            break;
        default:
            AT_ERROR("nonzero not implemented for ", at::toString(_dk));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::nonzero", "");
    return op.callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}

// aten::nonzero_numpy(Tensor self) -> Tensor[]
inline std::vector<Tensor> Tensor::nonzero_numpy() const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::nonzero_numpy(const_cast<Tensor&>(*this));
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::nonzero_numpy", "");
    return op.callUnboxed<std::vector<Tensor>, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}

// aten::gather(Tensor self, int dim, Tensor index, *, bool sparse_grad=False) -> Tensor
inline Tensor Tensor::gather(int64_t dim, const Tensor & index, bool sparse_grad) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(*this, index);
    DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
    DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);
    switch (dispatchKeyToBackend(_dk)) {
        case Backend::CPU:
            return CPUType::gather(const_cast<Tensor&>(*this), dim, index, sparse_grad);
            break;
        default:
            AT_ERROR("gather not implemented for ", at::toString(_dk));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::gather", "");
    return op.callUnboxed<Tensor, const Tensor &, int64_t, const Tensor &, bool>(const_cast<Tensor&>(*this), dim, index, sparse_grad);
#endif
}

// aten::gather.dimname(Tensor self, Dimname dim, Tensor index, *, bool sparse_grad=False) -> Tensor
inline Tensor Tensor::gather(Dimname dim, const Tensor & index, bool sparse_grad) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::gather_dimname(const_cast<Tensor&>(*this), dim, index, sparse_grad);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::gather", "dimname");
    return op.callUnboxed<Tensor, const Tensor &, Dimname, const Tensor &, bool>(const_cast<Tensor&>(*this), dim, index, sparse_grad);
#endif
}

// aten::addcmul(Tensor self, Tensor tensor1, Tensor tensor2, *, Scalar value=1) -> Tensor
inline Tensor Tensor::addcmul(const Tensor & tensor1, const Tensor & tensor2, Scalar value) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::addcmul(const_cast<Tensor&>(*this), tensor1, tensor2, value);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::addcmul", "");
    return op.callUnboxed<Tensor, const Tensor &, const Tensor &, const Tensor &, Scalar>(const_cast<Tensor&>(*this), tensor1, tensor2, value);
#endif
}

// aten::addcmul_(Tensor(a!) self, Tensor tensor1, Tensor tensor2, *, Scalar value=1) -> Tensor(a!)
inline Tensor & Tensor::addcmul_(const Tensor & tensor1, const Tensor & tensor2, Scalar value) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::addcmul_(const_cast<Tensor&>(*this), tensor1, tensor2, value);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::addcmul_", "");
    return op.callUnboxed<Tensor &, Tensor &, const Tensor &, const Tensor &, Scalar>(const_cast<Tensor&>(*this), tensor1, tensor2, value);
#endif
}

// aten::addcdiv(Tensor self, Tensor tensor1, Tensor tensor2, *, Scalar value=1) -> Tensor
inline Tensor Tensor::addcdiv(const Tensor & tensor1, const Tensor & tensor2, Scalar value) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::addcdiv(const_cast<Tensor&>(*this), tensor1, tensor2, value);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::addcdiv", "");
    return op.callUnboxed<Tensor, const Tensor &, const Tensor &, const Tensor &, Scalar>(const_cast<Tensor&>(*this), tensor1, tensor2, value);
#endif
}

// aten::lstsq(Tensor self, Tensor A) -> (Tensor solution, Tensor QR)
inline std::tuple<Tensor,Tensor> Tensor::lstsq(const Tensor & A) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(*this, A);
    DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
    DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);
    switch (dispatchKeyToBackend(_dk)) {
        case Backend::CPU:
            return CPUType::lstsq(const_cast<Tensor&>(*this), A);
            break;
        default:
            AT_ERROR("lstsq not implemented for ", at::toString(_dk));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::lstsq", "");
    return op.callUnboxed<std::tuple<Tensor,Tensor>, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), A);
#endif
}

// aten::triangular_solve(Tensor self, Tensor A, bool upper=True, bool transpose=False, bool unitriangular=False) -> (Tensor solution, Tensor cloned_coefficient)
inline std::tuple<Tensor,Tensor> Tensor::triangular_solve(const Tensor & A, bool upper, bool transpose, bool unitriangular) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::triangular_solve(const_cast<Tensor&>(*this), A, upper, transpose, unitriangular);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::triangular_solve", "");
    return op.callUnboxed<std::tuple<Tensor,Tensor>, const Tensor &, const Tensor &, bool, bool, bool>(const_cast<Tensor&>(*this), A, upper, transpose, unitriangular);
#endif
}

// aten::symeig(Tensor self, bool eigenvectors=False, bool upper=True) -> (Tensor eigenvalues, Tensor eigenvectors)
inline std::tuple<Tensor,Tensor> Tensor::symeig(bool eigenvectors, bool upper) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::symeig(const_cast<Tensor&>(*this), eigenvectors, upper);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::symeig", "");
    return op.callUnboxed<std::tuple<Tensor,Tensor>, const Tensor &, bool, bool>(const_cast<Tensor&>(*this), eigenvectors, upper);
#endif
}

// aten::eig(Tensor self, bool eigenvectors=False) -> (Tensor eigenvalues, Tensor eigenvectors)
inline std::tuple<Tensor,Tensor> Tensor::eig(bool eigenvectors) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(*this);
    DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
    DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);
    switch (dispatchKeyToBackend(_dk)) {
        case Backend::CPU:
            return CPUType::eig(const_cast<Tensor&>(*this), eigenvectors);
            break;
        default:
            AT_ERROR("eig not implemented for ", at::toString(_dk));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::eig", "");
    return op.callUnboxed<std::tuple<Tensor,Tensor>, const Tensor &, bool>(const_cast<Tensor&>(*this), eigenvectors);
#endif
}

// aten::svd(Tensor self, bool some=True, bool compute_uv=True) -> (Tensor U, Tensor S, Tensor V)
inline std::tuple<Tensor,Tensor,Tensor> Tensor::svd(bool some, bool compute_uv) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::svd(const_cast<Tensor&>(*this), some, compute_uv);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::svd", "");
    return op.callUnboxed<std::tuple<Tensor,Tensor,Tensor>, const Tensor &, bool, bool>(const_cast<Tensor&>(*this), some, compute_uv);
#endif
}

// aten::cholesky(Tensor self, bool upper=False) -> Tensor
inline Tensor Tensor::cholesky(bool upper) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::cholesky(const_cast<Tensor&>(*this), upper);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::cholesky", "");
    return op.callUnboxed<Tensor, const Tensor &, bool>(const_cast<Tensor&>(*this), upper);
#endif
}

// aten::cholesky_solve(Tensor self, Tensor input2, bool upper=False) -> Tensor
inline Tensor Tensor::cholesky_solve(const Tensor & input2, bool upper) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::cholesky_solve(const_cast<Tensor&>(*this), input2, upper);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::cholesky_solve", "");
    return op.callUnboxed<Tensor, const Tensor &, const Tensor &, bool>(const_cast<Tensor&>(*this), input2, upper);
#endif
}

// aten::solve(Tensor self, Tensor A) -> (Tensor solution, Tensor LU)
inline std::tuple<Tensor,Tensor> Tensor::solve(const Tensor & A) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::solve(const_cast<Tensor&>(*this), A);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::solve", "");
    return op.callUnboxed<std::tuple<Tensor,Tensor>, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), A);
#endif
}

// aten::cholesky_inverse(Tensor self, bool upper=False) -> Tensor
inline Tensor Tensor::cholesky_inverse(bool upper) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(*this);
    DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
    DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);
    switch (dispatchKeyToBackend(_dk)) {
        case Backend::CPU:
            return CPUType::cholesky_inverse(const_cast<Tensor&>(*this), upper);
            break;
        default:
            AT_ERROR("cholesky_inverse not implemented for ", at::toString(_dk));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::cholesky_inverse", "");
    return op.callUnboxed<Tensor, const Tensor &, bool>(const_cast<Tensor&>(*this), upper);
#endif
}

// aten::qr(Tensor self, bool some=True) -> (Tensor Q, Tensor R)
inline std::tuple<Tensor,Tensor> Tensor::qr(bool some) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::qr(const_cast<Tensor&>(*this), some);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::qr", "");
    return op.callUnboxed<std::tuple<Tensor,Tensor>, const Tensor &, bool>(const_cast<Tensor&>(*this), some);
#endif
}

// aten::geqrf(Tensor self) -> (Tensor a, Tensor tau)
inline std::tuple<Tensor,Tensor> Tensor::geqrf() const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(*this);
    DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
    DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);
    switch (dispatchKeyToBackend(_dk)) {
        case Backend::CPU:
            return CPUType::geqrf(const_cast<Tensor&>(*this));
            break;
        default:
            AT_ERROR("geqrf not implemented for ", at::toString(_dk));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::geqrf", "");
    return op.callUnboxed<std::tuple<Tensor,Tensor>, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}

// aten::orgqr(Tensor self, Tensor input2) -> Tensor
inline Tensor Tensor::orgqr(const Tensor & input2) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(*this, input2);
    DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
    DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);
    switch (dispatchKeyToBackend(_dk)) {
        case Backend::CPU:
            return CPUType::orgqr(const_cast<Tensor&>(*this), input2);
            break;
        default:
            AT_ERROR("orgqr not implemented for ", at::toString(_dk));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::orgqr", "");
    return op.callUnboxed<Tensor, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), input2);
#endif
}

// aten::ormqr(Tensor self, Tensor input2, Tensor input3, bool left=True, bool transpose=False) -> Tensor
inline Tensor Tensor::ormqr(const Tensor & input2, const Tensor & input3, bool left, bool transpose) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(*this, input2, input3);
    DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
    DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);
    switch (dispatchKeyToBackend(_dk)) {
        case Backend::CPU:
            return CPUType::ormqr(const_cast<Tensor&>(*this), input2, input3, left, transpose);
            break;
        default:
            AT_ERROR("ormqr not implemented for ", at::toString(_dk));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::ormqr", "");
    return op.callUnboxed<Tensor, const Tensor &, const Tensor &, const Tensor &, bool, bool>(const_cast<Tensor&>(*this), input2, input3, left, transpose);
#endif
}

// aten::lu_solve(Tensor self, Tensor LU_data, Tensor LU_pivots) -> Tensor
inline Tensor Tensor::lu_solve(const Tensor & LU_data, const Tensor & LU_pivots) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::lu_solve(const_cast<Tensor&>(*this), LU_data, LU_pivots);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::lu_solve", "");
    return op.callUnboxed<Tensor, const Tensor &, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), LU_data, LU_pivots);
#endif
}

// aten::multinomial(Tensor self, int num_samples, bool replacement=False, *, Generator? generator=None) -> Tensor
inline Tensor Tensor::multinomial(int64_t num_samples, bool replacement, c10::optional<Generator> generator) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(*this);
    DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
    DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);
    switch (dispatchKeyToBackend(_dk)) {
        case Backend::CPU:
            return CPUType::multinomial(const_cast<Tensor&>(*this), num_samples, replacement, generator);
            break;
        default:
            AT_ERROR("multinomial not implemented for ", at::toString(_dk));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::multinomial", "");
    return op.callUnboxed<Tensor, const Tensor &, int64_t, bool, c10::optional<Generator>>(const_cast<Tensor&>(*this), num_samples, replacement, generator);
#endif
}

// aten::lgamma(Tensor self) -> Tensor
inline Tensor Tensor::lgamma() const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(*this);
    DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
    DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);
    switch (dispatchKeyToBackend(_dk)) {
        case Backend::CPU:
            return CPUType::lgamma(const_cast<Tensor&>(*this));
            break;
        default:
            AT_ERROR("lgamma not implemented for ", at::toString(_dk));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::lgamma", "");
    return op.callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}

// aten::digamma(Tensor self) -> Tensor
inline Tensor Tensor::digamma() const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::digamma(const_cast<Tensor&>(*this));
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::digamma", "");
    return op.callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}

// aten::polygamma(int n, Tensor self) -> Tensor
inline Tensor Tensor::polygamma(int64_t n) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::polygamma(n, const_cast<Tensor&>(*this));
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::polygamma", "");
    return op.callUnboxed<Tensor, int64_t, const Tensor &>(n, const_cast<Tensor&>(*this));
#endif
}

// aten::erfinv(Tensor self) -> Tensor
inline Tensor Tensor::erfinv() const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(*this);
    DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
    DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);
    switch (dispatchKeyToBackend(_dk)) {
        case Backend::CPU:
            return CPUType::erfinv(const_cast<Tensor&>(*this));
            break;
        default:
            AT_ERROR("erfinv not implemented for ", at::toString(_dk));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::erfinv", "");
    return op.callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}

// aten::erfinv_(Tensor(a!) self) -> Tensor(a!)
inline Tensor & Tensor::erfinv_() const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(*this);
    DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
    DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);
    switch (dispatchKeyToBackend(_dk)) {
        case Backend::CPU:
            return CPUType::erfinv_(const_cast<Tensor&>(*this));
            break;
        default:
            AT_ERROR("erfinv_ not implemented for ", at::toString(_dk));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::erfinv_", "");
    return op.callUnboxed<Tensor &, Tensor &>(const_cast<Tensor&>(*this));
#endif
}

// aten::sign(Tensor self) -> Tensor
inline Tensor Tensor::sign() const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::sign(const_cast<Tensor&>(*this));
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::sign", "");
    return op.callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}

// aten::sign_(Tensor(a!) self) -> Tensor(a!)
inline Tensor & Tensor::sign_() const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::sign_(const_cast<Tensor&>(*this));
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::sign_", "");
    return op.callUnboxed<Tensor &, Tensor &>(const_cast<Tensor&>(*this));
#endif
}

// aten::dist(Tensor self, Tensor other, Scalar p=2) -> Tensor
inline Tensor Tensor::dist(const Tensor & other, Scalar p) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::dist(const_cast<Tensor&>(*this), other, p);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::dist", "");
    return op.callUnboxed<Tensor, const Tensor &, const Tensor &, Scalar>(const_cast<Tensor&>(*this), other, p);
#endif
}

// aten::atan2(Tensor self, Tensor other) -> Tensor
inline Tensor Tensor::atan2(const Tensor & other) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::atan2(const_cast<Tensor&>(*this), other);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::atan2", "");
    return op.callUnboxed<Tensor, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), other);
#endif
}

// aten::lerp.Scalar(Tensor self, Tensor end, Scalar weight) -> Tensor
inline Tensor Tensor::lerp(const Tensor & end, Scalar weight) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(*this, end);
    DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
    DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);
    switch (dispatchKeyToBackend(_dk)) {
        case Backend::CPU:
            return CPUType::lerp_Scalar(const_cast<Tensor&>(*this), end, weight);
            break;
        default:
            AT_ERROR("lerp not implemented for ", at::toString(_dk));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::lerp", "Scalar");
    return op.callUnboxed<Tensor, const Tensor &, const Tensor &, Scalar>(const_cast<Tensor&>(*this), end, weight);
#endif
}

// aten::lerp.Tensor(Tensor self, Tensor end, Tensor weight) -> Tensor
inline Tensor Tensor::lerp(const Tensor & end, const Tensor & weight) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(*this, end, weight);
    DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
    DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);
    switch (dispatchKeyToBackend(_dk)) {
        case Backend::CPU:
            return CPUType::lerp_Tensor(const_cast<Tensor&>(*this), end, weight);
            break;
        default:
            AT_ERROR("lerp not implemented for ", at::toString(_dk));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::lerp", "Tensor");
    return op.callUnboxed<Tensor, const Tensor &, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), end, weight);
#endif
}

// aten::histc(Tensor self, int bins=100, Scalar min=0, Scalar max=0) -> Tensor
inline Tensor Tensor::histc(int64_t bins, Scalar min, Scalar max) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(*this);
    DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
    DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);
    switch (dispatchKeyToBackend(_dk)) {
        case Backend::CPU:
            return CPUType::histc(const_cast<Tensor&>(*this), bins, min, max);
            break;
        default:
            AT_ERROR("histc not implemented for ", at::toString(_dk));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::histc", "");
    return op.callUnboxed<Tensor, const Tensor &, int64_t, Scalar, Scalar>(const_cast<Tensor&>(*this), bins, min, max);
#endif
}

// aten::fmod.Scalar(Tensor self, Scalar other) -> Tensor
inline Tensor Tensor::fmod(Scalar other) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(*this);
    DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
    DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);
    switch (dispatchKeyToBackend(_dk)) {
        case Backend::CPU:
            return CPUType::fmod_Scalar(const_cast<Tensor&>(*this), other);
            break;
        default:
            AT_ERROR("fmod not implemented for ", at::toString(_dk));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::fmod", "Scalar");
    return op.callUnboxed<Tensor, const Tensor &, Scalar>(const_cast<Tensor&>(*this), other);
#endif
}

// aten::fmod.Tensor(Tensor self, Tensor other) -> Tensor
inline Tensor Tensor::fmod(const Tensor & other) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(*this, other);
    DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
    DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);
    switch (dispatchKeyToBackend(_dk)) {
        case Backend::CPU:
            return CPUType::fmod_Tensor(const_cast<Tensor&>(*this), other);
            break;
        default:
            AT_ERROR("fmod not implemented for ", at::toString(_dk));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::fmod", "Tensor");
    return op.callUnboxed<Tensor, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), other);
#endif
}

// aten::remainder.Scalar(Tensor self, Scalar other) -> Tensor
inline Tensor Tensor::remainder(Scalar other) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(*this);
    DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
    DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);
    switch (dispatchKeyToBackend(_dk)) {
        case Backend::CPU:
            return CPUType::remainder_Scalar(const_cast<Tensor&>(*this), other);
            break;
        default:
            AT_ERROR("remainder not implemented for ", at::toString(_dk));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::remainder", "Scalar");
    return op.callUnboxed<Tensor, const Tensor &, Scalar>(const_cast<Tensor&>(*this), other);
#endif
}

// aten::remainder.Tensor(Tensor self, Tensor other) -> Tensor
inline Tensor Tensor::remainder(const Tensor & other) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(*this, other);
    DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
    DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);
    switch (dispatchKeyToBackend(_dk)) {
        case Backend::CPU:
            return CPUType::remainder_Tensor(const_cast<Tensor&>(*this), other);
            break;
        default:
            AT_ERROR("remainder not implemented for ", at::toString(_dk));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::remainder", "Tensor");
    return op.callUnboxed<Tensor, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), other);
#endif
}

// aten::min.other(Tensor self, Tensor other) -> Tensor
inline Tensor Tensor::min(const Tensor & other) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::min_other(const_cast<Tensor&>(*this), other);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::min", "other");
    return op.callUnboxed<Tensor, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), other);
#endif
}

// aten::min(Tensor self) -> Tensor
inline Tensor Tensor::min() const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(*this);
    DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
    DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);
    switch (dispatchKeyToBackend(_dk)) {
        case Backend::CPU:
            return CPUType::min(const_cast<Tensor&>(*this));
            break;
        case Backend::QuantizedCPU:
            return QuantizedCPUType::min(const_cast<Tensor&>(*this));
            break;
        default:
            AT_ERROR("min not implemented for ", at::toString(_dk));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::min", "");
    return op.callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}

// aten::max.other(Tensor self, Tensor other) -> Tensor
inline Tensor Tensor::max(const Tensor & other) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::max_other(const_cast<Tensor&>(*this), other);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::max", "other");
    return op.callUnboxed<Tensor, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), other);
#endif
}

// aten::max(Tensor self) -> Tensor
inline Tensor Tensor::max() const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(*this);
    DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
    DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);
    switch (dispatchKeyToBackend(_dk)) {
        case Backend::CPU:
            return CPUType::max(const_cast<Tensor&>(*this));
            break;
        case Backend::QuantizedCPU:
            return QuantizedCPUType::max(const_cast<Tensor&>(*this));
            break;
        default:
            AT_ERROR("max not implemented for ", at::toString(_dk));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::max", "");
    return op.callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}

// aten::median(Tensor self) -> Tensor
inline Tensor Tensor::median() const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(*this);
    DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
    DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);
    switch (dispatchKeyToBackend(_dk)) {
        case Backend::CPU:
            return CPUType::median(const_cast<Tensor&>(*this));
            break;
        default:
            AT_ERROR("median not implemented for ", at::toString(_dk));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::median", "");
    return op.callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}

// aten::sort(Tensor self, int dim=-1, bool descending=False) -> (Tensor values, Tensor indices)
inline std::tuple<Tensor,Tensor> Tensor::sort(int64_t dim, bool descending) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(*this);
    DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
    DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);
    switch (dispatchKeyToBackend(_dk)) {
        case Backend::CPU:
            return CPUType::sort(const_cast<Tensor&>(*this), dim, descending);
            break;
        case Backend::QuantizedCPU:
            return QuantizedCPUType::sort(const_cast<Tensor&>(*this), dim, descending);
            break;
        default:
            AT_ERROR("sort not implemented for ", at::toString(_dk));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::sort", "");
    return op.callUnboxed<std::tuple<Tensor,Tensor>, const Tensor &, int64_t, bool>(const_cast<Tensor&>(*this), dim, descending);
#endif
}

// aten::sort.dimname(Tensor self, Dimname dim, bool descending=False) -> (Tensor values, Tensor indices)
inline std::tuple<Tensor,Tensor> Tensor::sort(Dimname dim, bool descending) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::sort_dimname(const_cast<Tensor&>(*this), dim, descending);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::sort", "dimname");
    return op.callUnboxed<std::tuple<Tensor,Tensor>, const Tensor &, Dimname, bool>(const_cast<Tensor&>(*this), dim, descending);
#endif
}

// aten::argsort(Tensor self, int dim=-1, bool descending=False) -> Tensor
inline Tensor Tensor::argsort(int64_t dim, bool descending) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::argsort(const_cast<Tensor&>(*this), dim, descending);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::argsort", "");
    return op.callUnboxed<Tensor, const Tensor &, int64_t, bool>(const_cast<Tensor&>(*this), dim, descending);
#endif
}

// aten::argsort.dimname(Tensor self, Dimname dim, bool descending=False) -> Tensor
inline Tensor Tensor::argsort(Dimname dim, bool descending) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::argsort_dimname(const_cast<Tensor&>(*this), dim, descending);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::argsort", "dimname");
    return op.callUnboxed<Tensor, const Tensor &, Dimname, bool>(const_cast<Tensor&>(*this), dim, descending);
#endif
}

// aten::topk(Tensor self, int k, int dim=-1, bool largest=True, bool sorted=True) -> (Tensor values, Tensor indices)
inline std::tuple<Tensor,Tensor> Tensor::topk(int64_t k, int64_t dim, bool largest, bool sorted) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(*this);
    DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
    DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);
    switch (dispatchKeyToBackend(_dk)) {
        case Backend::CPU:
            return CPUType::topk(const_cast<Tensor&>(*this), k, dim, largest, sorted);
            break;
        case Backend::QuantizedCPU:
            return QuantizedCPUType::topk(const_cast<Tensor&>(*this), k, dim, largest, sorted);
            break;
        default:
            AT_ERROR("topk not implemented for ", at::toString(_dk));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::topk", "");
    return op.callUnboxed<std::tuple<Tensor,Tensor>, const Tensor &, int64_t, int64_t, bool, bool>(const_cast<Tensor&>(*this), k, dim, largest, sorted);
#endif
}

// aten::all(Tensor self) -> Tensor
inline Tensor Tensor::all() const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::all(const_cast<Tensor&>(*this));
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::all", "");
    return op.callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}

// aten::any(Tensor self) -> Tensor
inline Tensor Tensor::any() const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(*this);
    DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
    DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);
    switch (dispatchKeyToBackend(_dk)) {
        case Backend::CPU:
            return CPUType::any(const_cast<Tensor&>(*this));
            break;
        default:
            AT_ERROR("any not implemented for ", at::toString(_dk));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::any", "");
    return op.callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}

// aten::renorm(Tensor self, Scalar p, int dim, Scalar maxnorm) -> Tensor
inline Tensor Tensor::renorm(Scalar p, int64_t dim, Scalar maxnorm) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(*this);
    DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
    DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);
    switch (dispatchKeyToBackend(_dk)) {
        case Backend::CPU:
            return CPUType::renorm(const_cast<Tensor&>(*this), p, dim, maxnorm);
            break;
        default:
            AT_ERROR("renorm not implemented for ", at::toString(_dk));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::renorm", "");
    return op.callUnboxed<Tensor, const Tensor &, Scalar, int64_t, Scalar>(const_cast<Tensor&>(*this), p, dim, maxnorm);
#endif
}

// aten::unfold(Tensor(a) self, int dimension, int size, int step) -> Tensor(a)
inline Tensor Tensor::unfold(int64_t dimension, int64_t size, int64_t step) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(*this);
    DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
    DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);
    switch (dispatchKeyToBackend(_dk)) {
        case Backend::CPU:
            return CPUType::unfold(const_cast<Tensor&>(*this), dimension, size, step);
            break;
        default:
            AT_ERROR("unfold not implemented for ", at::toString(_dk));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::unfold", "");
    return op.callUnboxed<Tensor, const Tensor &, int64_t, int64_t, int64_t>(const_cast<Tensor&>(*this), dimension, size, step);
#endif
}

// aten::equal(Tensor self, Tensor other) -> bool
inline bool Tensor::equal(const Tensor & other) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(*this, other);
    DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
    DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);
    switch (dispatchKeyToBackend(_dk)) {
        case Backend::CPU:
            return CPUType::equal(const_cast<Tensor&>(*this), other);
            break;
        case Backend::QuantizedCPU:
            return QuantizedCPUType::equal(const_cast<Tensor&>(*this), other);
            break;
        default:
            AT_ERROR("equal not implemented for ", at::toString(_dk));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::equal", "");
    return op.callUnboxed<bool, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), other);
#endif
}

// aten::pow.Tensor_Tensor(Tensor self, Tensor exponent) -> Tensor
inline Tensor Tensor::pow(const Tensor & exponent) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(*this, exponent);
    DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
    DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);
    switch (dispatchKeyToBackend(_dk)) {
        case Backend::CPU:
            return CPUType::pow_Tensor_Tensor(const_cast<Tensor&>(*this), exponent);
            break;
        default:
            AT_ERROR("pow not implemented for ", at::toString(_dk));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::pow", "Tensor_Tensor");
    return op.callUnboxed<Tensor, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), exponent);
#endif
}

// aten::normal_(Tensor(a!) self, float mean=0, float std=1, *, Generator? generator=None) -> Tensor(a!)
inline Tensor & Tensor::normal_(double mean, double std, c10::optional<Generator> generator) const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::normal_(const_cast<Tensor&>(*this), mean, std, generator);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::normal_", "");
    return op.callUnboxed<Tensor &, Tensor &, double, double, c10::optional<Generator>>(const_cast<Tensor&>(*this), mean, std, generator);
#endif
}

// aten::alias(Tensor(a) self) -> Tensor(a)
inline Tensor Tensor::alias() const {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::alias(const_cast<Tensor&>(*this));
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::alias", "");
    return op.callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}

inline caffe2::TypeMeta Tensor::dtype() const noexcept {
  return impl_->dtype();
}

inline Layout Tensor::layout() const noexcept {
  return impl_->layout();
}

inline Device Tensor::device() const {
  return impl_->device();
}

inline int64_t Tensor::get_device() const {
  // NB: this is not a native function to avoid dispatching overhead.
  return impl_->get_device();
}

inline int64_t get_device(Tensor self) {
  return self.get_device();
}

inline bool Tensor::is_cuda() const {
  // NB: this is not a native function to avoid dispatching overhead.
  return impl_->is_cuda();
}

inline NamedTensorMeta* Tensor::get_named_tensor_meta() {
  return static_cast<NamedTensorMeta*>(impl_->named_tensor_meta());
}

inline const NamedTensorMeta* Tensor::get_named_tensor_meta() const {
  return static_cast<NamedTensorMeta*>(impl_->named_tensor_meta());
}

inline bool Tensor::has_names() const {
  // If a user is using unnamed tensors, then we can short-circuit right here.
  // Otherwise, impl::has_names attempts to retrieve names.
  if (!impl_->has_named_tensor_meta()) {
    return false;
  }
  return impl::has_names(unsafeGetTensorImpl());
}

inline bool is_cuda(Tensor self) {
  return self.is_cuda();
}

inline bool Tensor::is_hip() const {
  // NB: this is not a native function to avoid dispatching overhead.
  return impl_->is_hip();
}

inline bool is_hip(Tensor self) {
  return self.is_hip();
}

inline bool Tensor::is_sparse() const {
  // NB: this is not a native function to avoid dispatching overhead.
  return impl_->is_sparse();
}

inline bool is_sparse(Tensor self) {
  return self.is_sparse();
}

inline bool Tensor::is_mkldnn() const {
  // NB: this is not a native function to avoid dispatching overhead.
  return impl_->is_mkldnn();
}

inline bool is_mkldnn(Tensor self) {
  return self.is_mkldnn();
}

inline bool Tensor::is_quantized() const {
  // NB: this is not a native function to avoid dispatching overhead.
  return impl_->is_quantized();
}

inline bool is_quantized(Tensor self) {
  return self.is_quantized();
}

#define DEFINE_CAST(T, name)                     \
  template <>                                    \
  inline T* Tensor::data_ptr() const {           \
    TORCH_CHECK(                                 \
        scalar_type() == ScalarType::name,       \
        "expected scalar type ",                 \
        #name,                                   \
        " but found ",                           \
        c10::toString(scalar_type()));           \
    return static_cast<T*>(this->unsafeGetTensorImpl()->data());    \
  }

AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_EXCEPT_COMPLEX_HALF(DEFINE_CAST)
AT_FORALL_QINT_TYPES(DEFINE_CAST)
#undef DEFINE_CAST

// TODO(@zasdfgbnm): Remove this!
// This is needed only when the migration of std::complex to c10::complex
// is not done. This should be removed once the migration is done.
template <>
inline std::complex<float>* Tensor::data_ptr() const {
  TORCH_CHECK(scalar_type() == ScalarType::ComplexFloat,
    "expected scalar type ComplexFloat but found ",
    c10::toString(scalar_type()));
  return static_cast<std::complex<float>*>(this->unsafeGetTensorImpl()->data());
}
template <>
inline std::complex<double>* Tensor::data_ptr() const {
  TORCH_CHECK(scalar_type() == ScalarType::ComplexDouble,
    "expected scalar type ComplexDouble but found ",
    c10::toString(scalar_type()));
  return static_cast<std::complex<double>*>(this->unsafeGetTensorImpl()->data());
}
// end TODO

#define DEFINE_ITEM(T, name)      \
  template <>                     \
  inline T Tensor::item() const { \
    return item().to##name();     \
  }

AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_EXCEPT_COMPLEX_HALF(DEFINE_ITEM)
#undef DEFINE_ITEM

// TODO(@zasdfgbnm): Remove this!
// This is needed only when the migration of std::complex to c10::complex
// is not done. This should be removed once the migration is done.
template <>
inline std::complex<float> Tensor::item() const {
  // casting from c10::complex<float> to std::complex<float>
  return static_cast<std::complex<float>>(item().toComplexFloat());
}
template <>
inline std::complex<double> Tensor::item() const {
  // casting from c10::complex<double> to std::complex<double>
  return static_cast<std::complex<double>>(item().toComplexFloat()); 
}
// end TODO

// Gradient Node and Edges
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

template <typename T>
auto Tensor::register_hook(T&& hook) const -> Tensor::hook_return_void_t<T> {
  // Return the grad argument in case of a hook with void return type to have an
  // std::function with Tensor return type
  std::function<void(Tensor)> fn(hook);
  return _register_hook([fn](const Tensor& grad) {
    fn(grad);
    return Tensor();
  });
}

template <typename T>
auto Tensor::register_hook(T&& hook) const -> Tensor::hook_return_var_t<T> {
  return _register_hook(hook);
}


} //namespace at
