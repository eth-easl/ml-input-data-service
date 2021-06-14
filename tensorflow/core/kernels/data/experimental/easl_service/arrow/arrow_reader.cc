//
// Created by simon on 30.03.21.
//

#include "tensorflow/core/kernels/data/experimental/easl_service/arrow/arrow_reader.h"

#include <utility>
#include "arrow/ipc/feather.h"
#include "arrow/io/file.h"

namespace tensorflow {
namespace data {
namespace easl{

ArrowReader::ArrowReader(std::vector<int> col_selection) {
  this->col_selection_ = col_selection;
}

Status ArrowReader::Initialize(Env *env, const std::string &filename, const string &compression_type,
                               const DataTypeVector &dtypes, const std::vector<PartialTensorShape> &shapes,
                               std::shared_ptr<ArrowUtil::ArrowMetadata> metadata) {

  // read metadata
  this->metadata_ = metadata;
  this->experimental_ = metadata_->IsExperimental();  // TODO: use metadata to choose experimental
  if(metadata_->IsPartialBatching()) {
    TF_RETURN_IF_ERROR(metadata_->GetPartialBatches(filename, &partial_shapes));
  }
  TF_RETURN_IF_ERROR(metadata_->GetRowShape(&shapes_));
  shapes_initialized_ = !shapes_.empty();

  // initialize internal data structures
  this->env_ = env;
  this->filename_ = filename;
  this->compression_type_ = compression_type;
  this->dtypes_ = dtypes;
  this->current_batch_idx_ = 0; // gets increased upon every invocation of read_tensors
  this->current_row_idx_ = 0;


  // open file and read table
  file_ = arrow::io::MemoryMappedFile::Open(filename_, arrow::io::FileMode::READ).ValueOrDie();
  std::shared_ptr<arrow::ipc::feather::Reader> reader;
  ARROW_ASSIGN_CHECKED(reader, arrow::ipc::feather::Reader::Open(file_));
  std::shared_ptr<::arrow::Table> table;
  CHECK_ARROW(reader->Read(col_selection_, &table));
  total_rows_ = table->num_rows();
  // read individual record batches and append to class internal datastructure (size of record batches
  // given by configuration of writer)
  arrow::TableBatchReader tr(*table);
  std::shared_ptr<arrow::RecordBatch> batch;
  CHECK_ARROW(tr.ReadNext(&batch));
  while(batch != nullptr) {
    record_batches_.push_back(batch);
    CHECK_ARROW(tr.ReadNext(&batch));
  }

  return Status::OK();
}


Status ArrowReader::InitShapesAndTypes() {
  for(int i = 0; i < current_batch_->num_columns(); i++) {
    std::shared_ptr<arrow::Array> arr = current_batch_->column(i);

    // get the TensorShape for the current column entry:
    TensorShape output_shape = TensorShape({});
    DataType output_type;

    TF_RETURN_IF_ERROR(ArrowUtil::AssignSpec(arr, 0, 0, &output_type, &output_shape));  //batch_size = 0

    // add to internal data structures
    this->dtypes_.push_back(output_type);
    this->shapes_.push_back(output_shape);

  }
  shapes_initialized_ = true;
  return Status::OK();
}


Status ArrowReader::ReadTensors(std::vector<Tensor> *read_tensors) {

  // increments current_batch_idx_ by 1 (initialized to -1). If no more batches,
  // return status with OUT_OF_RANGE error.
  TF_RETURN_IF_ERROR(NextBatch());
  // Invariant: current_batch_ != nullptr

  #ifdef DEBUGGING
  VLOG(0) << "[ArrowReader] successfully read next RecordBatch. Nullptr? : " << (current_batch_ == nullptr);
  #endif

  if(!shapes_initialized_) {  // if no metadata --> fall back to implicitly extracting shape / type
    TF_RETURN_IF_ERROR(InitShapesAndTypes());
    #ifdef DEBUGGING
    VLOG(0) << "[ArrowReader] Initialized tensor shapes and dtypes implicitly from data.";
    #endif
  }

  #ifdef DEBUGGING
  VLOG(0) << "[ArrowReader] Converting RecordBatch with " << current_batch_->num_rows() << ""
                  " rows and " << current_batch_->num_columns() << " columns";
  #endif

  // go over all rows of record batch
  for(int i = 0; i < current_batch_->num_rows(); i++) {
    for(int j = 0; j < current_batch_->num_columns(); j++) {
      std::shared_ptr<arrow::Array> arr = current_batch_->column(j);  // don't need redirection here -> already filtered

      #ifdef DEBUGGING
      VLOG(0) << "[ArrowReader] extracted array from RecordBatch, array null? : " << (arr == nullptr);
      #endif

      DataType output_type = this->dtypes_[col_selection_[j]];  // metadata containts all shapes of all columns
      TensorShape output_shape;
      bool partial_batch = !partial_shapes.empty() && current_row_idx_ == total_rows_ - 1;

      if(partial_batch) {  // if partial batch in last row
        output_shape = this->partial_shapes[col_selection_[j]];
      } else {
        output_shape = this->shapes_[col_selection_[j]];
      }

      // Allocate a new tensor and assign Arrow data to it
      Tensor tensor(output_type, output_shape); // this constructor will use the default_cpu_allocator.
      #ifdef DEBUGGING
      VLOG(0) << "Allocated tensor with dtype: " << DataTypeString(output_type) << "  shape: " << output_shape.DebugString();
      #endif

      // String arrays and normal arrays have different shapes in experimental.
      if(output_type == DataType::DT_STRING || !experimental_) {
        #ifdef DEBUGGING
        VLOG(0) << "[ArrowReader] Assign Tensor Standard";
        #endif

        TF_RETURN_IF_ERROR(ArrowUtil::AssignTensor(arr, i, &tensor));
      } else {
        #ifdef DEBUGGING
        VLOG(0) << "[ArrowReader] Assign Tensor Experimental";
        #endif

        TF_RETURN_IF_ERROR(ArrowUtil::AssignTensorExperimental(arr, i, &tensor));
      }

      if(partial_batch) {
        metadata_->AddLastRowBatch(tensor);
      } else {
        read_tensors->emplace_back(std::move(tensor));
      }

      #ifdef DEBUGGING
      VLOG(0) << "[ArrowReader] Produced one tensor.";
      #endif

    }
  }

  return Status::OK();
}


Status ArrowReader::NextBatch() {
  #ifdef DEBUGGING
  VLOG(0) << "[ArrowReader] retrieving RecordBatch (idx): " << current_batch_idx_ << " of tot: " << record_batches_.size();
  #endif

  if (current_batch_idx_ < record_batches_.size()) {
    current_batch_ = record_batches_[current_batch_idx_];
    current_batch_idx_++;
  } else  {
    file_->Close();  // close mmapped file
    return Status(error::OUT_OF_RANGE, "finished reading all record batches");
  }
  return Status::OK();
}


} // namespace easl
} // namespace data
} // namespace tensorflow