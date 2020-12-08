#ifndef TENSORFLOW_CORE_KERNELS_DATA_EXPERIMENTAL_EASL_SERVICE_SERVICE_CACHE_UTIL_
#define TENSORFLOW_CORE_KERNELS_DATA_EXPERIMENTAL_EASL_SERVICE_SERVICE_CACHE_UTIL_


#include "tensorflow/core/kernels/data/experimental/snapshot_util.h"
#include "tensorflow/core/framework/types.h"

namespace tensorflow {
namespace data {
namespace easl{
namespace service_cache_util {

// (damien-aymon)
// Top-level writer that handles writes to the service cache.
// In the future, this class should handle more complex logic on how and where
// to distribute individual elements to cache files.
// For now, this class simply wraps a single snapshot_util::async writer using
// a TFRecordWriter.
class Writer {
 public:
  Writer(const std::string& target_dir, Env* env);

  Status Write(const std::vector<Tensor>& tensors);

  ~Writer();

 private:
  const std::string target_dir_;

  std::unique_ptr<snapshot_util::AsyncWriter> async_writer_;
};

class Reader {
 public:
  Reader(const std::string &target_dir, const DataTypeVector& dtypes, Env *env);

  Status Initialize();

  Status Read(std::vector<Tensor>* &read_tensors);

  ~Reader();

 private:
  const std::string target_dir_;
  const DataTypeVector dtypes_;
  Env* env_;

  std::unique_ptr<snapshot_util::Reader> reader_;
};

} // namespace service_cache_util
} // namespace easl
} // namespace data
} // namespace tensorflow

#endif // TENSORFLOW_CORE_KERNELS_DATA_EXPERIMENTAL_EASL_SERVICE_SERVICE_CACHE_UTIL_
