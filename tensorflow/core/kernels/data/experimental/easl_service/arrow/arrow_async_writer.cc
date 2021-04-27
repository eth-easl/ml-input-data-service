//
// Created by simon on 24.04.21.
//

#include "arrow_async_writer.h"
#include "tensorflow/core/kernels/data/experimental/easl_service/arrow/arrow_writer.h"
#include "tensorflow/core/platform/stringprintf.h"

namespace tensorflow {
namespace data {
namespace easl{
namespace service_cache_util {
namespace arrow_async_writer{

namespace {
  std::string GetFileName(const std::string& shard_directory,
                          uint64 file_id, uint64 split_id = 0) {
    return io::JoinPath(shard_directory, strings::Printf("%07llu_%llu.easl",
                                                         static_cast<unsigned long long>(file_id),
                                                         static_cast<unsigned long long>(split_id)));
  }
}






ArrowAsyncWriter::ArrowAsyncWriter(const int writer_count) : MultiThreadedAsyncWriter(writer_count) {
  metadata_ = std::make_shared<ArrowUtil::ArrowMetadata>();
  VLOG(0) << "ArrowAsyncWriter created";
  metadata_->SetExperimental(experimental_);
  VLOG(0) << "ArrowAsyncWriter setting experimental";

}

Status ArrowAsyncWriter::WriterThread(Env* env, const std::string& shard_directory,
                    uint64 writer_id, const std::string& compression,
                    int64 version, DataTypeVector output_types) {

  uint64_t storageEstimate = 0; // estimated storage space on disk in bytes
  uint64_t rowStorage = 0; // storage size of a single dataset row (single be.value). Assume all have the same size.
  uint64 split_id = 0; // name all produced arrow files by this thread

  TF_RETURN_IF_ERROR(env->RecursivelyCreateDir(shard_directory));
  LOG(INFO) << "(Writer_" << writer_id << ") Created Dir ";

  // register thread for concurrently writing to arrowMetadata file
  metadata_->RegisterWorker();

  std::unique_ptr<ArrowWriter> arrowWriter;
  arrowWriter = absl::make_unique<ArrowWriter>();
  TF_RETURN_IF_ERROR(arrowWriter->Create(env, GetFileName(shard_directory, writer_id),
                                         compression, output_types, metadata_));

  int count = 0;
  LOG(INFO) << "(Writer_" << writer_id << ") Starting to write ";

  while (true) {
    snapshot_util::ElementOrEOF be;
    Consume(&be);

    LOG(INFO) << "(Writer_" << writer_id << ") Read - "
              << be.end_of_sequence << " - Total: " << ++count;
    if (be.end_of_sequence) {
      TF_RETURN_IF_ERROR(arrowWriter->Close());
      LOG(INFO) << "(Writer_" << writer_id << ") Closed w/ total read "
                << count;
      break;
    }

    // update memory estimate:
    if(rowStorage == 0) {
      std::vector<Tensor> &tensors = be.value;
      for(Tensor t : tensors) {
        rowStorage += t.TotalBytes();
      }
    } else {
      storageEstimate += rowStorage;
    }

    // create new reader if memoryThreshold exceeded
    if(storageEstimate > memoryThreshold) {
      TF_RETURN_IF_ERROR(arrowWriter->Close());
      storageEstimate = rowStorage;
      // create new writer for remaining tensors:
      arrowWriter = absl::make_unique<ArrowWriter>();
      TF_RETURN_IF_ERROR(arrowWriter->Create(env, GetFileName(shard_directory, writer_id, ++split_id),
                                             compression, output_types, metadata_));
      LOG(INFO) << "(Writer_" << writer_id << ") Exceeded memory threshold, created new file (split_id = "
                                              "" << split_id <<")...";
    }

    TF_RETURN_IF_ERROR(arrowWriter->WriteTensors(be.value));
  }

  // Write accumulated metadata before closing --> if last thread writes to file
  metadata_->WriteMetadataToFile(shard_directory);
  return Status::OK();
}

} // namespace arrow_async_wirter
} // namespace service_cache_util
} // namespace easl
} // namespace data
} // namespace tensorflow