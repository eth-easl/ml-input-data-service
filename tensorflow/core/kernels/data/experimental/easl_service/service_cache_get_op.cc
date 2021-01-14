#include "tensorflow/core/kernels/data/experimental/easl_service/service_cache_get_op.h"

#include "tensorflow/core/platform/tstring.h"
#include "absl/memory/memory.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/data/name_utils.h"
#include "tensorflow/core/kernels/data/experimental/easl_service/service_cache_util.h"


namespace tensorflow {
namespace data {
namespace experimental {
namespace easl{

/* static */ constexpr const char* const ServiceCacheGetOp::kDatasetType;
/* static */ constexpr const char* const ServiceCacheGetOp::kPath;



class ServiceCacheGetOp::Dataset : public DatasetBase {
 public:
  Dataset(OpKernelContext* ctx, const std::string& path,
          const DataTypeVector& output_types,
          const std::vector<PartialTensorShape>& output_shapes);

  ~Dataset() override;

  std::unique_ptr<IteratorBase> MakeIteratorInternal(
      const string& prefix) const override;

  const DataTypeVector& output_dtypes() const override;

  const std::vector<PartialTensorShape>& output_shapes() const override;

  string DebugString() const override;

  Status CheckExternalState() const override;

 protected:

  Status AsGraphDefInternal(SerializationContext* ctx,
                            DatasetGraphDefBuilder* b,
                            Node** output) const override;

 private:
  class Iterator;

  const tstring path_;
  const DataTypeVector output_dtypes_;
  const std::vector<PartialTensorShape> output_shapes_;

};

class ServiceCacheGetOp::Dataset::Iterator : public DatasetIterator<Dataset> {
 public:
  explicit Iterator(const Params& params);

  Status Initialize(IteratorContext* ctx) override;

  Status GetNextInternal(IteratorContext* ctx, std::vector<Tensor>* out_tensors,
                         bool* end_of_sequence) override;

 protected:
  Status SaveInternal(SerializationContext* ctx,
                      IteratorStateWriter* writer) override;

  Status RestoreInternal(IteratorContext* ctx,
                         IteratorStateReader* reader) override;

 private:
  mutex mu_;
  std::unique_ptr<tensorflow::data::easl::service_cache_util::Reader> reader_
  TF_GUARDED_BY(mu_);
};

// -----------------------------------------------------------------------------
// DatasetOp
// -----------------------------------------------------------------------------

ServiceCacheGetOp::ServiceCacheGetOp(OpKernelConstruction* ctx)
    : DatasetOpKernel(ctx) {

  // (damien-aymon)This op does not have access to the original input dataset
  // it replaces. The dtypes and shapes must therefore be set as attributes
  // of this op.
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kOutputTypes, &output_dtypes_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kOutputShapes, &output_shapes_));
}

void ServiceCacheGetOp::MakeDataset(OpKernelContext* ctx,
                                    DatasetBase** output) {
  tstring path;
  OP_REQUIRES_OK(ctx, ParseScalarArgument(ctx, kPath, &path));

  *output = new ServiceCacheGetOp::Dataset(
      ctx, path, output_dtypes_, output_shapes_);
}

// -----------------------------------------------------------------------------
// Dataset
// -----------------------------------------------------------------------------

ServiceCacheGetOp::Dataset::Dataset(
    OpKernelContext* ctx,
    const std::string& path,
    const DataTypeVector& output_dtypes,
    const std::vector<PartialTensorShape>& output_shapes)
    : DatasetBase(DatasetContext(ctx)),
    path_(path),
    output_dtypes_(output_dtypes),
    output_shapes_(output_shapes) {}

ServiceCacheGetOp::Dataset::~Dataset() {}

std::unique_ptr<IteratorBase>
ServiceCacheGetOp::Dataset::MakeIteratorInternal(const string& prefix) const {
  VLOG(0) << "EASL - prefix to get op: " << prefix;
  return absl::make_unique<Iterator>(
      Iterator::Params{this, absl::StrCat(prefix, "::ServiceCacheGet")});
}

const DataTypeVector& ServiceCacheGetOp::Dataset::output_dtypes() const {
  return output_dtypes_;
}

const std::vector<PartialTensorShape>&
ServiceCacheGetOp::Dataset::output_shapes() const {
  return output_shapes_;
}

string ServiceCacheGetOp::Dataset::DebugString() const {
  return name_utils::DatasetDebugString(kDatasetType);
}

Status ServiceCacheGetOp::Dataset::CheckExternalState() const {
  return Status::OK();
}

Status ServiceCacheGetOp::Dataset::AsGraphDefInternal(
    SerializationContext* ctx, DatasetGraphDefBuilder* b, Node** output) const {

  Node* path = nullptr;
  TF_RETURN_IF_ERROR(b->AddScalar(path_, &path));

  return b->AddDataset(this, /*inputs=*/ {path}, output);
}


// -----------------------------------------------------------------------------
// Iterator
// -----------------------------------------------------------------------------

ServiceCacheGetOp::Dataset::Iterator::Iterator(const Params& params)
    : DatasetIterator<Dataset>(params) {};

Status ServiceCacheGetOp::Dataset::Iterator::Initialize(
    IteratorContext* ctx) {
  for(auto dt: dataset()->output_dtypes_){
    VLOG(0) << DataTypeString(dt);
  }
  reader_ =
      std::make_unique<tensorflow::data::easl::service_cache_util::Reader>(
          ctx->env(), dataset()->path_, dataset()->output_dtypes_);

  return reader_->Initialize();
}

Status ServiceCacheGetOp::Dataset::Iterator::SaveInternal(
    SerializationContext* ctx, IteratorStateWriter* writer) {
  return errors::Unimplemented("Checkpointing is currently not supported.");
}

Status ServiceCacheGetOp::Dataset::Iterator::RestoreInternal(
    IteratorContext* ctx, IteratorStateReader* reader) {
  return errors::Unimplemented("Checkpointing is currently not supported.");
}

Status ServiceCacheGetOp::Dataset::Iterator::GetNextInternal(
    IteratorContext* ctx, std::vector<Tensor>* out_tensors,
    bool* end_of_sequence) {
  mutex_lock l(mu_);
  VLOG(0) << "EASL - entered cache get GetNextInternal";
  return reader_->Read(out_tensors, end_of_sequence);
}

namespace {
REGISTER_KERNEL_BUILDER(Name("ServiceCacheGetDataset").Device(DEVICE_CPU),
                        ServiceCacheGetOp);
}  // namespace


} // namespace easl
} // namespace experimental
} // namespace data
} // namespace tensorflow