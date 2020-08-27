// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: face_dataset.proto

#include "face_dataset.pb.h"

#include <algorithm>

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/extension_set.h>
#include <google/protobuf/wire_format_lite.h>
#include <google/protobuf/descriptor.h>
#include <google/protobuf/generated_message_reflection.h>
#include <google/protobuf/reflection_ops.h>
#include <google/protobuf/wire_format.h>
// @@protoc_insertion_point(includes)
#include <google/protobuf/port_def.inc>
extern PROTOBUF_INTERNAL_EXPORT_face_5fdataset_2eproto ::PROTOBUF_NAMESPACE_ID::internal::SCCInfo<0> scc_info_FaceObject_face_5fdataset_2eproto;
namespace dataset_faces {
class FaceObjectDefaultTypeInternal {
 public:
  ::PROTOBUF_NAMESPACE_ID::internal::ExplicitlyConstructed<FaceObject> _instance;
} _FaceObject_default_instance_;
class DatasetObjectDefaultTypeInternal {
 public:
  ::PROTOBUF_NAMESPACE_ID::internal::ExplicitlyConstructed<DatasetObject> _instance;
} _DatasetObject_default_instance_;
}  // namespace dataset_faces
static void InitDefaultsscc_info_DatasetObject_face_5fdataset_2eproto() {
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  {
    void* ptr = &::dataset_faces::_DatasetObject_default_instance_;
    new (ptr) ::dataset_faces::DatasetObject();
    ::PROTOBUF_NAMESPACE_ID::internal::OnShutdownDestroyMessage(ptr);
  }
  ::dataset_faces::DatasetObject::InitAsDefaultInstance();
}

::PROTOBUF_NAMESPACE_ID::internal::SCCInfo<1> scc_info_DatasetObject_face_5fdataset_2eproto =
    {{ATOMIC_VAR_INIT(::PROTOBUF_NAMESPACE_ID::internal::SCCInfoBase::kUninitialized), 1, 0, InitDefaultsscc_info_DatasetObject_face_5fdataset_2eproto}, {
      &scc_info_FaceObject_face_5fdataset_2eproto.base,}};

static void InitDefaultsscc_info_FaceObject_face_5fdataset_2eproto() {
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  {
    void* ptr = &::dataset_faces::_FaceObject_default_instance_;
    new (ptr) ::dataset_faces::FaceObject();
    ::PROTOBUF_NAMESPACE_ID::internal::OnShutdownDestroyMessage(ptr);
  }
  ::dataset_faces::FaceObject::InitAsDefaultInstance();
}

::PROTOBUF_NAMESPACE_ID::internal::SCCInfo<0> scc_info_FaceObject_face_5fdataset_2eproto =
    {{ATOMIC_VAR_INIT(::PROTOBUF_NAMESPACE_ID::internal::SCCInfoBase::kUninitialized), 0, 0, InitDefaultsscc_info_FaceObject_face_5fdataset_2eproto}, {}};

static ::PROTOBUF_NAMESPACE_ID::Metadata file_level_metadata_face_5fdataset_2eproto[2];
static constexpr ::PROTOBUF_NAMESPACE_ID::EnumDescriptor const** file_level_enum_descriptors_face_5fdataset_2eproto = nullptr;
static constexpr ::PROTOBUF_NAMESPACE_ID::ServiceDescriptor const** file_level_service_descriptors_face_5fdataset_2eproto = nullptr;

const ::PROTOBUF_NAMESPACE_ID::uint32 TableStruct_face_5fdataset_2eproto::offsets[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) = {
  PROTOBUF_FIELD_OFFSET(::dataset_faces::FaceObject, _has_bits_),
  PROTOBUF_FIELD_OFFSET(::dataset_faces::FaceObject, _internal_metadata_),
  ~0u,  // no _extensions_
  ~0u,  // no _oneof_case_
  ~0u,  // no _weak_field_map_
  PROTOBUF_FIELD_OFFSET(::dataset_faces::FaceObject, name_),
  PROTOBUF_FIELD_OFFSET(::dataset_faces::FaceObject, embeddings_),
  0,
  ~0u,
  ~0u,  // no _has_bits_
  PROTOBUF_FIELD_OFFSET(::dataset_faces::DatasetObject, _internal_metadata_),
  ~0u,  // no _extensions_
  ~0u,  // no _oneof_case_
  ~0u,  // no _weak_field_map_
  PROTOBUF_FIELD_OFFSET(::dataset_faces::DatasetObject, faceobjects_),
};
static const ::PROTOBUF_NAMESPACE_ID::internal::MigrationSchema schemas[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) = {
  { 0, 7, sizeof(::dataset_faces::FaceObject)},
  { 9, -1, sizeof(::dataset_faces::DatasetObject)},
};

static ::PROTOBUF_NAMESPACE_ID::Message const * const file_default_instances[] = {
  reinterpret_cast<const ::PROTOBUF_NAMESPACE_ID::Message*>(&::dataset_faces::_FaceObject_default_instance_),
  reinterpret_cast<const ::PROTOBUF_NAMESPACE_ID::Message*>(&::dataset_faces::_DatasetObject_default_instance_),
};

const char descriptor_table_protodef_face_5fdataset_2eproto[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) =
  "\n\022face_dataset.proto\022\rdataset_faces\"2\n\nF"
  "aceObject\022\014\n\004name\030\001 \002(\t\022\026\n\nembeddings\030\002 "
  "\003(\001B\002\020\001\"\?\n\rDatasetObject\022.\n\013faceobjects\030"
  "\001 \003(\0132\031.dataset_faces.FaceObject"
  ;
static const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable*const descriptor_table_face_5fdataset_2eproto_deps[1] = {
};
static ::PROTOBUF_NAMESPACE_ID::internal::SCCInfoBase*const descriptor_table_face_5fdataset_2eproto_sccs[2] = {
  &scc_info_DatasetObject_face_5fdataset_2eproto.base,
  &scc_info_FaceObject_face_5fdataset_2eproto.base,
};
static ::PROTOBUF_NAMESPACE_ID::internal::once_flag descriptor_table_face_5fdataset_2eproto_once;
const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable descriptor_table_face_5fdataset_2eproto = {
  false, false, descriptor_table_protodef_face_5fdataset_2eproto, "face_dataset.proto", 152,
  &descriptor_table_face_5fdataset_2eproto_once, descriptor_table_face_5fdataset_2eproto_sccs, descriptor_table_face_5fdataset_2eproto_deps, 2, 0,
  schemas, file_default_instances, TableStruct_face_5fdataset_2eproto::offsets,
  file_level_metadata_face_5fdataset_2eproto, 2, file_level_enum_descriptors_face_5fdataset_2eproto, file_level_service_descriptors_face_5fdataset_2eproto,
};

// Force running AddDescriptors() at dynamic initialization time.
static bool dynamic_init_dummy_face_5fdataset_2eproto = (static_cast<void>(::PROTOBUF_NAMESPACE_ID::internal::AddDescriptors(&descriptor_table_face_5fdataset_2eproto)), true);
namespace dataset_faces {

// ===================================================================

void FaceObject::InitAsDefaultInstance() {
}
class FaceObject::_Internal {
 public:
  using HasBits = decltype(std::declval<FaceObject>()._has_bits_);
  static void set_has_name(HasBits* has_bits) {
    (*has_bits)[0] |= 1u;
  }
  static bool MissingRequiredFields(const HasBits& has_bits) {
    return ((has_bits[0] & 0x00000001) ^ 0x00000001) != 0;
  }
};

FaceObject::FaceObject(::PROTOBUF_NAMESPACE_ID::Arena* arena)
  : ::PROTOBUF_NAMESPACE_ID::Message(arena),
  embeddings_(arena) {
  SharedCtor();
  RegisterArenaDtor(arena);
  // @@protoc_insertion_point(arena_constructor:dataset_faces.FaceObject)
}
FaceObject::FaceObject(const FaceObject& from)
  : ::PROTOBUF_NAMESPACE_ID::Message(),
      _has_bits_(from._has_bits_),
      embeddings_(from.embeddings_) {
  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
  name_.UnsafeSetDefault(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
  if (from._internal_has_name()) {
    name_.Set(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), from._internal_name(),
      GetArena());
  }
  // @@protoc_insertion_point(copy_constructor:dataset_faces.FaceObject)
}

void FaceObject::SharedCtor() {
  ::PROTOBUF_NAMESPACE_ID::internal::InitSCC(&scc_info_FaceObject_face_5fdataset_2eproto.base);
  name_.UnsafeSetDefault(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
}

FaceObject::~FaceObject() {
  // @@protoc_insertion_point(destructor:dataset_faces.FaceObject)
  SharedDtor();
  _internal_metadata_.Delete<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
}

void FaceObject::SharedDtor() {
  GOOGLE_DCHECK(GetArena() == nullptr);
  name_.DestroyNoArena(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
}

void FaceObject::ArenaDtor(void* object) {
  FaceObject* _this = reinterpret_cast< FaceObject* >(object);
  (void)_this;
}
void FaceObject::RegisterArenaDtor(::PROTOBUF_NAMESPACE_ID::Arena*) {
}
void FaceObject::SetCachedSize(int size) const {
  _cached_size_.Set(size);
}
const FaceObject& FaceObject::default_instance() {
  ::PROTOBUF_NAMESPACE_ID::internal::InitSCC(&::scc_info_FaceObject_face_5fdataset_2eproto.base);
  return *internal_default_instance();
}


void FaceObject::Clear() {
// @@protoc_insertion_point(message_clear_start:dataset_faces.FaceObject)
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  embeddings_.Clear();
  cached_has_bits = _has_bits_[0];
  if (cached_has_bits & 0x00000001u) {
    name_.ClearNonDefaultToEmpty();
  }
  _has_bits_.Clear();
  _internal_metadata_.Clear<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
}

const char* FaceObject::_InternalParse(const char* ptr, ::PROTOBUF_NAMESPACE_ID::internal::ParseContext* ctx) {
#define CHK_(x) if (PROTOBUF_PREDICT_FALSE(!(x))) goto failure
  _Internal::HasBits has_bits{};
  ::PROTOBUF_NAMESPACE_ID::Arena* arena = GetArena(); (void)arena;
  while (!ctx->Done(&ptr)) {
    ::PROTOBUF_NAMESPACE_ID::uint32 tag;
    ptr = ::PROTOBUF_NAMESPACE_ID::internal::ReadTag(ptr, &tag);
    CHK_(ptr);
    switch (tag >> 3) {
      // required string name = 1;
      case 1:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 10)) {
          auto str = _internal_mutable_name();
          ptr = ::PROTOBUF_NAMESPACE_ID::internal::InlineGreedyStringParser(str, ptr, ctx);
          #ifndef NDEBUG
          ::PROTOBUF_NAMESPACE_ID::internal::VerifyUTF8(str, "dataset_faces.FaceObject.name");
          #endif  // !NDEBUG
          CHK_(ptr);
        } else goto handle_unusual;
        continue;
      // repeated double embeddings = 2 [packed = true];
      case 2:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 18)) {
          ptr = ::PROTOBUF_NAMESPACE_ID::internal::PackedDoubleParser(_internal_mutable_embeddings(), ptr, ctx);
          CHK_(ptr);
        } else if (static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 17) {
          _internal_add_embeddings(::PROTOBUF_NAMESPACE_ID::internal::UnalignedLoad<double>(ptr));
          ptr += sizeof(double);
        } else goto handle_unusual;
        continue;
      default: {
      handle_unusual:
        if ((tag & 7) == 4 || tag == 0) {
          ctx->SetLastTag(tag);
          goto success;
        }
        ptr = UnknownFieldParse(tag,
            _internal_metadata_.mutable_unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(),
            ptr, ctx);
        CHK_(ptr != nullptr);
        continue;
      }
    }  // switch
  }  // while
success:
  _has_bits_.Or(has_bits);
  return ptr;
failure:
  ptr = nullptr;
  goto success;
#undef CHK_
}

::PROTOBUF_NAMESPACE_ID::uint8* FaceObject::_InternalSerialize(
    ::PROTOBUF_NAMESPACE_ID::uint8* target, ::PROTOBUF_NAMESPACE_ID::io::EpsCopyOutputStream* stream) const {
  // @@protoc_insertion_point(serialize_to_array_start:dataset_faces.FaceObject)
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  cached_has_bits = _has_bits_[0];
  // required string name = 1;
  if (cached_has_bits & 0x00000001u) {
    ::PROTOBUF_NAMESPACE_ID::internal::WireFormat::VerifyUTF8StringNamedField(
      this->_internal_name().data(), static_cast<int>(this->_internal_name().length()),
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormat::SERIALIZE,
      "dataset_faces.FaceObject.name");
    target = stream->WriteStringMaybeAliased(
        1, this->_internal_name(), target);
  }

  // repeated double embeddings = 2 [packed = true];
  if (this->_internal_embeddings_size() > 0) {
    target = stream->WriteFixedPacked(2, _internal_embeddings(), target);
  }

  if (PROTOBUF_PREDICT_FALSE(_internal_metadata_.have_unknown_fields())) {
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormat::InternalSerializeUnknownFieldsToArray(
        _internal_metadata_.unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(::PROTOBUF_NAMESPACE_ID::UnknownFieldSet::default_instance), target, stream);
  }
  // @@protoc_insertion_point(serialize_to_array_end:dataset_faces.FaceObject)
  return target;
}

size_t FaceObject::ByteSizeLong() const {
// @@protoc_insertion_point(message_byte_size_start:dataset_faces.FaceObject)
  size_t total_size = 0;

  // required string name = 1;
  if (_internal_has_name()) {
    total_size += 1 +
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::StringSize(
        this->_internal_name());
  }
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  // repeated double embeddings = 2 [packed = true];
  {
    unsigned int count = static_cast<unsigned int>(this->_internal_embeddings_size());
    size_t data_size = 8UL * count;
    if (data_size > 0) {
      total_size += 1 +
        ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::Int32Size(
            static_cast<::PROTOBUF_NAMESPACE_ID::int32>(data_size));
    }
    int cached_size = ::PROTOBUF_NAMESPACE_ID::internal::ToCachedSize(data_size);
    _embeddings_cached_byte_size_.store(cached_size,
                                    std::memory_order_relaxed);
    total_size += data_size;
  }

  if (PROTOBUF_PREDICT_FALSE(_internal_metadata_.have_unknown_fields())) {
    return ::PROTOBUF_NAMESPACE_ID::internal::ComputeUnknownFieldsSize(
        _internal_metadata_, total_size, &_cached_size_);
  }
  int cached_size = ::PROTOBUF_NAMESPACE_ID::internal::ToCachedSize(total_size);
  SetCachedSize(cached_size);
  return total_size;
}

void FaceObject::MergeFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) {
// @@protoc_insertion_point(generalized_merge_from_start:dataset_faces.FaceObject)
  GOOGLE_DCHECK_NE(&from, this);
  const FaceObject* source =
      ::PROTOBUF_NAMESPACE_ID::DynamicCastToGenerated<FaceObject>(
          &from);
  if (source == nullptr) {
  // @@protoc_insertion_point(generalized_merge_from_cast_fail:dataset_faces.FaceObject)
    ::PROTOBUF_NAMESPACE_ID::internal::ReflectionOps::Merge(from, this);
  } else {
  // @@protoc_insertion_point(generalized_merge_from_cast_success:dataset_faces.FaceObject)
    MergeFrom(*source);
  }
}

void FaceObject::MergeFrom(const FaceObject& from) {
// @@protoc_insertion_point(class_specific_merge_from_start:dataset_faces.FaceObject)
  GOOGLE_DCHECK_NE(&from, this);
  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  embeddings_.MergeFrom(from.embeddings_);
  if (from._internal_has_name()) {
    _internal_set_name(from._internal_name());
  }
}

void FaceObject::CopyFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) {
// @@protoc_insertion_point(generalized_copy_from_start:dataset_faces.FaceObject)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

void FaceObject::CopyFrom(const FaceObject& from) {
// @@protoc_insertion_point(class_specific_copy_from_start:dataset_faces.FaceObject)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool FaceObject::IsInitialized() const {
  if (_Internal::MissingRequiredFields(_has_bits_)) return false;
  return true;
}

void FaceObject::InternalSwap(FaceObject* other) {
  using std::swap;
  _internal_metadata_.Swap<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(&other->_internal_metadata_);
  swap(_has_bits_[0], other->_has_bits_[0]);
  embeddings_.InternalSwap(&other->embeddings_);
  name_.Swap(&other->name_, &::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), GetArena());
}

::PROTOBUF_NAMESPACE_ID::Metadata FaceObject::GetMetadata() const {
  return GetMetadataStatic();
}


// ===================================================================

void DatasetObject::InitAsDefaultInstance() {
}
class DatasetObject::_Internal {
 public:
};

DatasetObject::DatasetObject(::PROTOBUF_NAMESPACE_ID::Arena* arena)
  : ::PROTOBUF_NAMESPACE_ID::Message(arena),
  faceobjects_(arena) {
  SharedCtor();
  RegisterArenaDtor(arena);
  // @@protoc_insertion_point(arena_constructor:dataset_faces.DatasetObject)
}
DatasetObject::DatasetObject(const DatasetObject& from)
  : ::PROTOBUF_NAMESPACE_ID::Message(),
      faceobjects_(from.faceobjects_) {
  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
  // @@protoc_insertion_point(copy_constructor:dataset_faces.DatasetObject)
}

void DatasetObject::SharedCtor() {
  ::PROTOBUF_NAMESPACE_ID::internal::InitSCC(&scc_info_DatasetObject_face_5fdataset_2eproto.base);
}

DatasetObject::~DatasetObject() {
  // @@protoc_insertion_point(destructor:dataset_faces.DatasetObject)
  SharedDtor();
  _internal_metadata_.Delete<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
}

void DatasetObject::SharedDtor() {
  GOOGLE_DCHECK(GetArena() == nullptr);
}

void DatasetObject::ArenaDtor(void* object) {
  DatasetObject* _this = reinterpret_cast< DatasetObject* >(object);
  (void)_this;
}
void DatasetObject::RegisterArenaDtor(::PROTOBUF_NAMESPACE_ID::Arena*) {
}
void DatasetObject::SetCachedSize(int size) const {
  _cached_size_.Set(size);
}
const DatasetObject& DatasetObject::default_instance() {
  ::PROTOBUF_NAMESPACE_ID::internal::InitSCC(&::scc_info_DatasetObject_face_5fdataset_2eproto.base);
  return *internal_default_instance();
}


void DatasetObject::Clear() {
// @@protoc_insertion_point(message_clear_start:dataset_faces.DatasetObject)
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  faceobjects_.Clear();
  _internal_metadata_.Clear<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
}

const char* DatasetObject::_InternalParse(const char* ptr, ::PROTOBUF_NAMESPACE_ID::internal::ParseContext* ctx) {
#define CHK_(x) if (PROTOBUF_PREDICT_FALSE(!(x))) goto failure
  ::PROTOBUF_NAMESPACE_ID::Arena* arena = GetArena(); (void)arena;
  while (!ctx->Done(&ptr)) {
    ::PROTOBUF_NAMESPACE_ID::uint32 tag;
    ptr = ::PROTOBUF_NAMESPACE_ID::internal::ReadTag(ptr, &tag);
    CHK_(ptr);
    switch (tag >> 3) {
      // repeated .dataset_faces.FaceObject faceobjects = 1;
      case 1:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 10)) {
          ptr -= 1;
          do {
            ptr += 1;
            ptr = ctx->ParseMessage(_internal_add_faceobjects(), ptr);
            CHK_(ptr);
            if (!ctx->DataAvailable(ptr)) break;
          } while (::PROTOBUF_NAMESPACE_ID::internal::ExpectTag<10>(ptr));
        } else goto handle_unusual;
        continue;
      default: {
      handle_unusual:
        if ((tag & 7) == 4 || tag == 0) {
          ctx->SetLastTag(tag);
          goto success;
        }
        ptr = UnknownFieldParse(tag,
            _internal_metadata_.mutable_unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(),
            ptr, ctx);
        CHK_(ptr != nullptr);
        continue;
      }
    }  // switch
  }  // while
success:
  return ptr;
failure:
  ptr = nullptr;
  goto success;
#undef CHK_
}

::PROTOBUF_NAMESPACE_ID::uint8* DatasetObject::_InternalSerialize(
    ::PROTOBUF_NAMESPACE_ID::uint8* target, ::PROTOBUF_NAMESPACE_ID::io::EpsCopyOutputStream* stream) const {
  // @@protoc_insertion_point(serialize_to_array_start:dataset_faces.DatasetObject)
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  // repeated .dataset_faces.FaceObject faceobjects = 1;
  for (unsigned int i = 0,
      n = static_cast<unsigned int>(this->_internal_faceobjects_size()); i < n; i++) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::
      InternalWriteMessage(1, this->_internal_faceobjects(i), target, stream);
  }

  if (PROTOBUF_PREDICT_FALSE(_internal_metadata_.have_unknown_fields())) {
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormat::InternalSerializeUnknownFieldsToArray(
        _internal_metadata_.unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(::PROTOBUF_NAMESPACE_ID::UnknownFieldSet::default_instance), target, stream);
  }
  // @@protoc_insertion_point(serialize_to_array_end:dataset_faces.DatasetObject)
  return target;
}

size_t DatasetObject::ByteSizeLong() const {
// @@protoc_insertion_point(message_byte_size_start:dataset_faces.DatasetObject)
  size_t total_size = 0;

  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  // repeated .dataset_faces.FaceObject faceobjects = 1;
  total_size += 1UL * this->_internal_faceobjects_size();
  for (const auto& msg : this->faceobjects_) {
    total_size +=
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::MessageSize(msg);
  }

  if (PROTOBUF_PREDICT_FALSE(_internal_metadata_.have_unknown_fields())) {
    return ::PROTOBUF_NAMESPACE_ID::internal::ComputeUnknownFieldsSize(
        _internal_metadata_, total_size, &_cached_size_);
  }
  int cached_size = ::PROTOBUF_NAMESPACE_ID::internal::ToCachedSize(total_size);
  SetCachedSize(cached_size);
  return total_size;
}

void DatasetObject::MergeFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) {
// @@protoc_insertion_point(generalized_merge_from_start:dataset_faces.DatasetObject)
  GOOGLE_DCHECK_NE(&from, this);
  const DatasetObject* source =
      ::PROTOBUF_NAMESPACE_ID::DynamicCastToGenerated<DatasetObject>(
          &from);
  if (source == nullptr) {
  // @@protoc_insertion_point(generalized_merge_from_cast_fail:dataset_faces.DatasetObject)
    ::PROTOBUF_NAMESPACE_ID::internal::ReflectionOps::Merge(from, this);
  } else {
  // @@protoc_insertion_point(generalized_merge_from_cast_success:dataset_faces.DatasetObject)
    MergeFrom(*source);
  }
}

void DatasetObject::MergeFrom(const DatasetObject& from) {
// @@protoc_insertion_point(class_specific_merge_from_start:dataset_faces.DatasetObject)
  GOOGLE_DCHECK_NE(&from, this);
  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  faceobjects_.MergeFrom(from.faceobjects_);
}

void DatasetObject::CopyFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) {
// @@protoc_insertion_point(generalized_copy_from_start:dataset_faces.DatasetObject)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

void DatasetObject::CopyFrom(const DatasetObject& from) {
// @@protoc_insertion_point(class_specific_copy_from_start:dataset_faces.DatasetObject)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool DatasetObject::IsInitialized() const {
  if (!::PROTOBUF_NAMESPACE_ID::internal::AllAreInitialized(faceobjects_)) return false;
  return true;
}

void DatasetObject::InternalSwap(DatasetObject* other) {
  using std::swap;
  _internal_metadata_.Swap<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(&other->_internal_metadata_);
  faceobjects_.InternalSwap(&other->faceobjects_);
}

::PROTOBUF_NAMESPACE_ID::Metadata DatasetObject::GetMetadata() const {
  return GetMetadataStatic();
}


// @@protoc_insertion_point(namespace_scope)
}  // namespace dataset_faces
PROTOBUF_NAMESPACE_OPEN
template<> PROTOBUF_NOINLINE ::dataset_faces::FaceObject* Arena::CreateMaybeMessage< ::dataset_faces::FaceObject >(Arena* arena) {
  return Arena::CreateMessageInternal< ::dataset_faces::FaceObject >(arena);
}
template<> PROTOBUF_NOINLINE ::dataset_faces::DatasetObject* Arena::CreateMaybeMessage< ::dataset_faces::DatasetObject >(Arena* arena) {
  return Arena::CreateMessageInternal< ::dataset_faces::DatasetObject >(arena);
}
PROTOBUF_NAMESPACE_CLOSE

// @@protoc_insertion_point(global_scope)
#include <google/protobuf/port_undef.inc>
