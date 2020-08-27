// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: face_dataset.proto

#ifndef GOOGLE_PROTOBUF_INCLUDED_face_5fdataset_2eproto
#define GOOGLE_PROTOBUF_INCLUDED_face_5fdataset_2eproto

#include <limits>
#include <string>

#include <google/protobuf/port_def.inc>
#if PROTOBUF_VERSION < 3012000
#error This file was generated by a newer version of protoc which is
#error incompatible with your Protocol Buffer headers. Please update
#error your headers.
#endif
#if 3012004 < PROTOBUF_MIN_PROTOC_VERSION
#error This file was generated by an older version of protoc which is
#error incompatible with your Protocol Buffer headers. Please
#error regenerate this file with a newer version of protoc.
#endif

#include <google/protobuf/port_undef.inc>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/arena.h>
#include <google/protobuf/arenastring.h>
#include <google/protobuf/generated_message_table_driven.h>
#include <google/protobuf/generated_message_util.h>
#include <google/protobuf/inlined_string_field.h>
#include <google/protobuf/metadata_lite.h>
#include <google/protobuf/generated_message_reflection.h>
#include <google/protobuf/message.h>
#include <google/protobuf/repeated_field.h>  // IWYU pragma: export
#include <google/protobuf/extension_set.h>  // IWYU pragma: export
#include <google/protobuf/unknown_field_set.h>
// @@protoc_insertion_point(includes)
#include <google/protobuf/port_def.inc>
#define PROTOBUF_INTERNAL_EXPORT_face_5fdataset_2eproto
PROTOBUF_NAMESPACE_OPEN
namespace internal {
class AnyMetadata;
}  // namespace internal
PROTOBUF_NAMESPACE_CLOSE

// Internal implementation detail -- do not use these members.
struct TableStruct_face_5fdataset_2eproto {
  static const ::PROTOBUF_NAMESPACE_ID::internal::ParseTableField entries[]
    PROTOBUF_SECTION_VARIABLE(protodesc_cold);
  static const ::PROTOBUF_NAMESPACE_ID::internal::AuxillaryParseTableField aux[]
    PROTOBUF_SECTION_VARIABLE(protodesc_cold);
  static const ::PROTOBUF_NAMESPACE_ID::internal::ParseTable schema[2]
    PROTOBUF_SECTION_VARIABLE(protodesc_cold);
  static const ::PROTOBUF_NAMESPACE_ID::internal::FieldMetadata field_metadata[];
  static const ::PROTOBUF_NAMESPACE_ID::internal::SerializationTable serialization_table[];
  static const ::PROTOBUF_NAMESPACE_ID::uint32 offsets[];
};
extern const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable descriptor_table_face_5fdataset_2eproto;
namespace dataset_faces {
class DatasetObject;
class DatasetObjectDefaultTypeInternal;
extern DatasetObjectDefaultTypeInternal _DatasetObject_default_instance_;
class FaceObject;
class FaceObjectDefaultTypeInternal;
extern FaceObjectDefaultTypeInternal _FaceObject_default_instance_;
}  // namespace dataset_faces
PROTOBUF_NAMESPACE_OPEN
template<> ::dataset_faces::DatasetObject* Arena::CreateMaybeMessage<::dataset_faces::DatasetObject>(Arena*);
template<> ::dataset_faces::FaceObject* Arena::CreateMaybeMessage<::dataset_faces::FaceObject>(Arena*);
PROTOBUF_NAMESPACE_CLOSE
namespace dataset_faces {

// ===================================================================

class FaceObject PROTOBUF_FINAL :
    public ::PROTOBUF_NAMESPACE_ID::Message /* @@protoc_insertion_point(class_definition:dataset_faces.FaceObject) */ {
 public:
  inline FaceObject() : FaceObject(nullptr) {};
  virtual ~FaceObject();

  FaceObject(const FaceObject& from);
  FaceObject(FaceObject&& from) noexcept
    : FaceObject() {
    *this = ::std::move(from);
  }

  inline FaceObject& operator=(const FaceObject& from) {
    CopyFrom(from);
    return *this;
  }
  inline FaceObject& operator=(FaceObject&& from) noexcept {
    if (GetArena() == from.GetArena()) {
      if (this != &from) InternalSwap(&from);
    } else {
      CopyFrom(from);
    }
    return *this;
  }

  inline const ::PROTOBUF_NAMESPACE_ID::UnknownFieldSet& unknown_fields() const {
    return _internal_metadata_.unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(::PROTOBUF_NAMESPACE_ID::UnknownFieldSet::default_instance);
  }
  inline ::PROTOBUF_NAMESPACE_ID::UnknownFieldSet* mutable_unknown_fields() {
    return _internal_metadata_.mutable_unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
  }

  static const ::PROTOBUF_NAMESPACE_ID::Descriptor* descriptor() {
    return GetDescriptor();
  }
  static const ::PROTOBUF_NAMESPACE_ID::Descriptor* GetDescriptor() {
    return GetMetadataStatic().descriptor;
  }
  static const ::PROTOBUF_NAMESPACE_ID::Reflection* GetReflection() {
    return GetMetadataStatic().reflection;
  }
  static const FaceObject& default_instance();

  static void InitAsDefaultInstance();  // FOR INTERNAL USE ONLY
  static inline const FaceObject* internal_default_instance() {
    return reinterpret_cast<const FaceObject*>(
               &_FaceObject_default_instance_);
  }
  static constexpr int kIndexInFileMessages =
    0;

  friend void swap(FaceObject& a, FaceObject& b) {
    a.Swap(&b);
  }
  inline void Swap(FaceObject* other) {
    if (other == this) return;
    if (GetArena() == other->GetArena()) {
      InternalSwap(other);
    } else {
      ::PROTOBUF_NAMESPACE_ID::internal::GenericSwap(this, other);
    }
  }
  void UnsafeArenaSwap(FaceObject* other) {
    if (other == this) return;
    GOOGLE_DCHECK(GetArena() == other->GetArena());
    InternalSwap(other);
  }

  // implements Message ----------------------------------------------

  inline FaceObject* New() const final {
    return CreateMaybeMessage<FaceObject>(nullptr);
  }

  FaceObject* New(::PROTOBUF_NAMESPACE_ID::Arena* arena) const final {
    return CreateMaybeMessage<FaceObject>(arena);
  }
  void CopyFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) final;
  void MergeFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) final;
  void CopyFrom(const FaceObject& from);
  void MergeFrom(const FaceObject& from);
  PROTOBUF_ATTRIBUTE_REINITIALIZES void Clear() final;
  bool IsInitialized() const final;

  size_t ByteSizeLong() const final;
  const char* _InternalParse(const char* ptr, ::PROTOBUF_NAMESPACE_ID::internal::ParseContext* ctx) final;
  ::PROTOBUF_NAMESPACE_ID::uint8* _InternalSerialize(
      ::PROTOBUF_NAMESPACE_ID::uint8* target, ::PROTOBUF_NAMESPACE_ID::io::EpsCopyOutputStream* stream) const final;
  int GetCachedSize() const final { return _cached_size_.Get(); }

  private:
  inline void SharedCtor();
  inline void SharedDtor();
  void SetCachedSize(int size) const final;
  void InternalSwap(FaceObject* other);
  friend class ::PROTOBUF_NAMESPACE_ID::internal::AnyMetadata;
  static ::PROTOBUF_NAMESPACE_ID::StringPiece FullMessageName() {
    return "dataset_faces.FaceObject";
  }
  protected:
  explicit FaceObject(::PROTOBUF_NAMESPACE_ID::Arena* arena);
  private:
  static void ArenaDtor(void* object);
  inline void RegisterArenaDtor(::PROTOBUF_NAMESPACE_ID::Arena* arena);
  public:

  ::PROTOBUF_NAMESPACE_ID::Metadata GetMetadata() const final;
  private:
  static ::PROTOBUF_NAMESPACE_ID::Metadata GetMetadataStatic() {
    ::PROTOBUF_NAMESPACE_ID::internal::AssignDescriptors(&::descriptor_table_face_5fdataset_2eproto);
    return ::descriptor_table_face_5fdataset_2eproto.file_level_metadata[kIndexInFileMessages];
  }

  public:

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  enum : int {
    kEmbeddingsFieldNumber = 2,
    kNameFieldNumber = 1,
  };
  // repeated double embeddings = 2 [packed = true];
  int embeddings_size() const;
  private:
  int _internal_embeddings_size() const;
  public:
  void clear_embeddings();
  private:
  double _internal_embeddings(int index) const;
  const ::PROTOBUF_NAMESPACE_ID::RepeatedField< double >&
      _internal_embeddings() const;
  void _internal_add_embeddings(double value);
  ::PROTOBUF_NAMESPACE_ID::RepeatedField< double >*
      _internal_mutable_embeddings();
  public:
  double embeddings(int index) const;
  void set_embeddings(int index, double value);
  void add_embeddings(double value);
  const ::PROTOBUF_NAMESPACE_ID::RepeatedField< double >&
      embeddings() const;
  ::PROTOBUF_NAMESPACE_ID::RepeatedField< double >*
      mutable_embeddings();

  // required string name = 1;
  bool has_name() const;
  private:
  bool _internal_has_name() const;
  public:
  void clear_name();
  const std::string& name() const;
  void set_name(const std::string& value);
  void set_name(std::string&& value);
  void set_name(const char* value);
  void set_name(const char* value, size_t size);
  std::string* mutable_name();
  std::string* release_name();
  void set_allocated_name(std::string* name);
  GOOGLE_PROTOBUF_RUNTIME_DEPRECATED("The unsafe_arena_ accessors for"
  "    string fields are deprecated and will be removed in a"
  "    future release.")
  std::string* unsafe_arena_release_name();
  GOOGLE_PROTOBUF_RUNTIME_DEPRECATED("The unsafe_arena_ accessors for"
  "    string fields are deprecated and will be removed in a"
  "    future release.")
  void unsafe_arena_set_allocated_name(
      std::string* name);
  private:
  const std::string& _internal_name() const;
  void _internal_set_name(const std::string& value);
  std::string* _internal_mutable_name();
  public:

  // @@protoc_insertion_point(class_scope:dataset_faces.FaceObject)
 private:
  class _Internal;

  template <typename T> friend class ::PROTOBUF_NAMESPACE_ID::Arena::InternalHelper;
  typedef void InternalArenaConstructable_;
  typedef void DestructorSkippable_;
  ::PROTOBUF_NAMESPACE_ID::internal::HasBits<1> _has_bits_;
  mutable ::PROTOBUF_NAMESPACE_ID::internal::CachedSize _cached_size_;
  ::PROTOBUF_NAMESPACE_ID::RepeatedField< double > embeddings_;
  mutable std::atomic<int> _embeddings_cached_byte_size_;
  ::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr name_;
  friend struct ::TableStruct_face_5fdataset_2eproto;
};
// -------------------------------------------------------------------

class DatasetObject PROTOBUF_FINAL :
    public ::PROTOBUF_NAMESPACE_ID::Message /* @@protoc_insertion_point(class_definition:dataset_faces.DatasetObject) */ {
 public:
  inline DatasetObject() : DatasetObject(nullptr) {};
  virtual ~DatasetObject();

  DatasetObject(const DatasetObject& from);
  DatasetObject(DatasetObject&& from) noexcept
    : DatasetObject() {
    *this = ::std::move(from);
  }

  inline DatasetObject& operator=(const DatasetObject& from) {
    CopyFrom(from);
    return *this;
  }
  inline DatasetObject& operator=(DatasetObject&& from) noexcept {
    if (GetArena() == from.GetArena()) {
      if (this != &from) InternalSwap(&from);
    } else {
      CopyFrom(from);
    }
    return *this;
  }

  inline const ::PROTOBUF_NAMESPACE_ID::UnknownFieldSet& unknown_fields() const {
    return _internal_metadata_.unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(::PROTOBUF_NAMESPACE_ID::UnknownFieldSet::default_instance);
  }
  inline ::PROTOBUF_NAMESPACE_ID::UnknownFieldSet* mutable_unknown_fields() {
    return _internal_metadata_.mutable_unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
  }

  static const ::PROTOBUF_NAMESPACE_ID::Descriptor* descriptor() {
    return GetDescriptor();
  }
  static const ::PROTOBUF_NAMESPACE_ID::Descriptor* GetDescriptor() {
    return GetMetadataStatic().descriptor;
  }
  static const ::PROTOBUF_NAMESPACE_ID::Reflection* GetReflection() {
    return GetMetadataStatic().reflection;
  }
  static const DatasetObject& default_instance();

  static void InitAsDefaultInstance();  // FOR INTERNAL USE ONLY
  static inline const DatasetObject* internal_default_instance() {
    return reinterpret_cast<const DatasetObject*>(
               &_DatasetObject_default_instance_);
  }
  static constexpr int kIndexInFileMessages =
    1;

  friend void swap(DatasetObject& a, DatasetObject& b) {
    a.Swap(&b);
  }
  inline void Swap(DatasetObject* other) {
    if (other == this) return;
    if (GetArena() == other->GetArena()) {
      InternalSwap(other);
    } else {
      ::PROTOBUF_NAMESPACE_ID::internal::GenericSwap(this, other);
    }
  }
  void UnsafeArenaSwap(DatasetObject* other) {
    if (other == this) return;
    GOOGLE_DCHECK(GetArena() == other->GetArena());
    InternalSwap(other);
  }

  // implements Message ----------------------------------------------

  inline DatasetObject* New() const final {
    return CreateMaybeMessage<DatasetObject>(nullptr);
  }

  DatasetObject* New(::PROTOBUF_NAMESPACE_ID::Arena* arena) const final {
    return CreateMaybeMessage<DatasetObject>(arena);
  }
  void CopyFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) final;
  void MergeFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) final;
  void CopyFrom(const DatasetObject& from);
  void MergeFrom(const DatasetObject& from);
  PROTOBUF_ATTRIBUTE_REINITIALIZES void Clear() final;
  bool IsInitialized() const final;

  size_t ByteSizeLong() const final;
  const char* _InternalParse(const char* ptr, ::PROTOBUF_NAMESPACE_ID::internal::ParseContext* ctx) final;
  ::PROTOBUF_NAMESPACE_ID::uint8* _InternalSerialize(
      ::PROTOBUF_NAMESPACE_ID::uint8* target, ::PROTOBUF_NAMESPACE_ID::io::EpsCopyOutputStream* stream) const final;
  int GetCachedSize() const final { return _cached_size_.Get(); }

  private:
  inline void SharedCtor();
  inline void SharedDtor();
  void SetCachedSize(int size) const final;
  void InternalSwap(DatasetObject* other);
  friend class ::PROTOBUF_NAMESPACE_ID::internal::AnyMetadata;
  static ::PROTOBUF_NAMESPACE_ID::StringPiece FullMessageName() {
    return "dataset_faces.DatasetObject";
  }
  protected:
  explicit DatasetObject(::PROTOBUF_NAMESPACE_ID::Arena* arena);
  private:
  static void ArenaDtor(void* object);
  inline void RegisterArenaDtor(::PROTOBUF_NAMESPACE_ID::Arena* arena);
  public:

  ::PROTOBUF_NAMESPACE_ID::Metadata GetMetadata() const final;
  private:
  static ::PROTOBUF_NAMESPACE_ID::Metadata GetMetadataStatic() {
    ::PROTOBUF_NAMESPACE_ID::internal::AssignDescriptors(&::descriptor_table_face_5fdataset_2eproto);
    return ::descriptor_table_face_5fdataset_2eproto.file_level_metadata[kIndexInFileMessages];
  }

  public:

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  enum : int {
    kFaceobjectsFieldNumber = 1,
  };
  // repeated .dataset_faces.FaceObject faceobjects = 1;
  int faceobjects_size() const;
  private:
  int _internal_faceobjects_size() const;
  public:
  void clear_faceobjects();
  ::dataset_faces::FaceObject* mutable_faceobjects(int index);
  ::PROTOBUF_NAMESPACE_ID::RepeatedPtrField< ::dataset_faces::FaceObject >*
      mutable_faceobjects();
  private:
  const ::dataset_faces::FaceObject& _internal_faceobjects(int index) const;
  ::dataset_faces::FaceObject* _internal_add_faceobjects();
  public:
  const ::dataset_faces::FaceObject& faceobjects(int index) const;
  ::dataset_faces::FaceObject* add_faceobjects();
  const ::PROTOBUF_NAMESPACE_ID::RepeatedPtrField< ::dataset_faces::FaceObject >&
      faceobjects() const;

  // @@protoc_insertion_point(class_scope:dataset_faces.DatasetObject)
 private:
  class _Internal;

  template <typename T> friend class ::PROTOBUF_NAMESPACE_ID::Arena::InternalHelper;
  typedef void InternalArenaConstructable_;
  typedef void DestructorSkippable_;
  ::PROTOBUF_NAMESPACE_ID::RepeatedPtrField< ::dataset_faces::FaceObject > faceobjects_;
  mutable ::PROTOBUF_NAMESPACE_ID::internal::CachedSize _cached_size_;
  friend struct ::TableStruct_face_5fdataset_2eproto;
};
// ===================================================================


// ===================================================================

#ifdef __GNUC__
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif  // __GNUC__
// FaceObject

// required string name = 1;
inline bool FaceObject::_internal_has_name() const {
  bool value = (_has_bits_[0] & 0x00000001u) != 0;
  return value;
}
inline bool FaceObject::has_name() const {
  return _internal_has_name();
}
inline void FaceObject::clear_name() {
  name_.ClearToEmpty(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), GetArena());
  _has_bits_[0] &= ~0x00000001u;
}
inline const std::string& FaceObject::name() const {
  // @@protoc_insertion_point(field_get:dataset_faces.FaceObject.name)
  return _internal_name();
}
inline void FaceObject::set_name(const std::string& value) {
  _internal_set_name(value);
  // @@protoc_insertion_point(field_set:dataset_faces.FaceObject.name)
}
inline std::string* FaceObject::mutable_name() {
  // @@protoc_insertion_point(field_mutable:dataset_faces.FaceObject.name)
  return _internal_mutable_name();
}
inline const std::string& FaceObject::_internal_name() const {
  return name_.Get();
}
inline void FaceObject::_internal_set_name(const std::string& value) {
  _has_bits_[0] |= 0x00000001u;
  name_.Set(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), value, GetArena());
}
inline void FaceObject::set_name(std::string&& value) {
  _has_bits_[0] |= 0x00000001u;
  name_.Set(
    &::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), ::std::move(value), GetArena());
  // @@protoc_insertion_point(field_set_rvalue:dataset_faces.FaceObject.name)
}
inline void FaceObject::set_name(const char* value) {
  GOOGLE_DCHECK(value != nullptr);
  _has_bits_[0] |= 0x00000001u;
  name_.Set(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), ::std::string(value),
              GetArena());
  // @@protoc_insertion_point(field_set_char:dataset_faces.FaceObject.name)
}
inline void FaceObject::set_name(const char* value,
    size_t size) {
  _has_bits_[0] |= 0x00000001u;
  name_.Set(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), ::std::string(
      reinterpret_cast<const char*>(value), size), GetArena());
  // @@protoc_insertion_point(field_set_pointer:dataset_faces.FaceObject.name)
}
inline std::string* FaceObject::_internal_mutable_name() {
  _has_bits_[0] |= 0x00000001u;
  return name_.Mutable(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), GetArena());
}
inline std::string* FaceObject::release_name() {
  // @@protoc_insertion_point(field_release:dataset_faces.FaceObject.name)
  if (!_internal_has_name()) {
    return nullptr;
  }
  _has_bits_[0] &= ~0x00000001u;
  return name_.ReleaseNonDefault(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), GetArena());
}
inline void FaceObject::set_allocated_name(std::string* name) {
  if (name != nullptr) {
    _has_bits_[0] |= 0x00000001u;
  } else {
    _has_bits_[0] &= ~0x00000001u;
  }
  name_.SetAllocated(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), name,
      GetArena());
  // @@protoc_insertion_point(field_set_allocated:dataset_faces.FaceObject.name)
}
inline std::string* FaceObject::unsafe_arena_release_name() {
  // @@protoc_insertion_point(field_unsafe_arena_release:dataset_faces.FaceObject.name)
  GOOGLE_DCHECK(GetArena() != nullptr);
  _has_bits_[0] &= ~0x00000001u;
  return name_.UnsafeArenaRelease(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(),
      GetArena());
}
inline void FaceObject::unsafe_arena_set_allocated_name(
    std::string* name) {
  GOOGLE_DCHECK(GetArena() != nullptr);
  if (name != nullptr) {
    _has_bits_[0] |= 0x00000001u;
  } else {
    _has_bits_[0] &= ~0x00000001u;
  }
  name_.UnsafeArenaSetAllocated(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(),
      name, GetArena());
  // @@protoc_insertion_point(field_unsafe_arena_set_allocated:dataset_faces.FaceObject.name)
}

// repeated double embeddings = 2 [packed = true];
inline int FaceObject::_internal_embeddings_size() const {
  return embeddings_.size();
}
inline int FaceObject::embeddings_size() const {
  return _internal_embeddings_size();
}
inline void FaceObject::clear_embeddings() {
  embeddings_.Clear();
}
inline double FaceObject::_internal_embeddings(int index) const {
  return embeddings_.Get(index);
}
inline double FaceObject::embeddings(int index) const {
  // @@protoc_insertion_point(field_get:dataset_faces.FaceObject.embeddings)
  return _internal_embeddings(index);
}
inline void FaceObject::set_embeddings(int index, double value) {
  embeddings_.Set(index, value);
  // @@protoc_insertion_point(field_set:dataset_faces.FaceObject.embeddings)
}
inline void FaceObject::_internal_add_embeddings(double value) {
  embeddings_.Add(value);
}
inline void FaceObject::add_embeddings(double value) {
  _internal_add_embeddings(value);
  // @@protoc_insertion_point(field_add:dataset_faces.FaceObject.embeddings)
}
inline const ::PROTOBUF_NAMESPACE_ID::RepeatedField< double >&
FaceObject::_internal_embeddings() const {
  return embeddings_;
}
inline const ::PROTOBUF_NAMESPACE_ID::RepeatedField< double >&
FaceObject::embeddings() const {
  // @@protoc_insertion_point(field_list:dataset_faces.FaceObject.embeddings)
  return _internal_embeddings();
}
inline ::PROTOBUF_NAMESPACE_ID::RepeatedField< double >*
FaceObject::_internal_mutable_embeddings() {
  return &embeddings_;
}
inline ::PROTOBUF_NAMESPACE_ID::RepeatedField< double >*
FaceObject::mutable_embeddings() {
  // @@protoc_insertion_point(field_mutable_list:dataset_faces.FaceObject.embeddings)
  return _internal_mutable_embeddings();
}

// -------------------------------------------------------------------

// DatasetObject

// repeated .dataset_faces.FaceObject faceobjects = 1;
inline int DatasetObject::_internal_faceobjects_size() const {
  return faceobjects_.size();
}
inline int DatasetObject::faceobjects_size() const {
  return _internal_faceobjects_size();
}
inline void DatasetObject::clear_faceobjects() {
  faceobjects_.Clear();
}
inline ::dataset_faces::FaceObject* DatasetObject::mutable_faceobjects(int index) {
  // @@protoc_insertion_point(field_mutable:dataset_faces.DatasetObject.faceobjects)
  return faceobjects_.Mutable(index);
}
inline ::PROTOBUF_NAMESPACE_ID::RepeatedPtrField< ::dataset_faces::FaceObject >*
DatasetObject::mutable_faceobjects() {
  // @@protoc_insertion_point(field_mutable_list:dataset_faces.DatasetObject.faceobjects)
  return &faceobjects_;
}
inline const ::dataset_faces::FaceObject& DatasetObject::_internal_faceobjects(int index) const {
  return faceobjects_.Get(index);
}
inline const ::dataset_faces::FaceObject& DatasetObject::faceobjects(int index) const {
  // @@protoc_insertion_point(field_get:dataset_faces.DatasetObject.faceobjects)
  return _internal_faceobjects(index);
}
inline ::dataset_faces::FaceObject* DatasetObject::_internal_add_faceobjects() {
  return faceobjects_.Add();
}
inline ::dataset_faces::FaceObject* DatasetObject::add_faceobjects() {
  // @@protoc_insertion_point(field_add:dataset_faces.DatasetObject.faceobjects)
  return _internal_add_faceobjects();
}
inline const ::PROTOBUF_NAMESPACE_ID::RepeatedPtrField< ::dataset_faces::FaceObject >&
DatasetObject::faceobjects() const {
  // @@protoc_insertion_point(field_list:dataset_faces.DatasetObject.faceobjects)
  return faceobjects_;
}

#ifdef __GNUC__
  #pragma GCC diagnostic pop
#endif  // __GNUC__
// -------------------------------------------------------------------


// @@protoc_insertion_point(namespace_scope)

}  // namespace dataset_faces

// @@protoc_insertion_point(global_scope)

#include <google/protobuf/port_undef.inc>
#endif  // GOOGLE_PROTOBUF_INCLUDED_GOOGLE_PROTOBUF_INCLUDED_face_5fdataset_2eproto