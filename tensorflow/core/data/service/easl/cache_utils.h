#ifndef TENSORFLOW_CORE_DATA_SERVICE_EASL_CACHE_UTILS_H_
#define TENSORFLOW_CORE_DATA_SERVICE_EASL_CACHE_UTILS_H_

#include <string>
#include "tensorflow/core/platform/default/integral_types.h"
#include "tensorflow/core/data/service/easl/dispatcher_cache_state.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/data/service/common.pb.h"
#include "tensorflow/core/protobuf/service_config.pb.h"


namespace tensorflow {
namespace data {
namespace service {
namespace easl{
namespace cache_utils {

std::string DatasetPutKey(const int64 id, const uint64 fingerprint);

std::string DatasetGetKey(const int64 id, const uint64 fingerprint);

// Sets the dataset_key to either ".._put", ".._get" or standard depending on
// the cache state.
// If the dataset is cached, the ".._get" version is set.
// Otherwise, if this task is responsible for caching the dataset at this
// worker, the ".._put" version is set.
// Finally, if none of the above match, the standard (plain compute) version
// is set.
// WARNING, a lock must be protecting cache_state, which is not thread safe!!!
Status DatasetKey(const ::tensorflow::data::easl::CacheState& cache_state,
                  const int64 dataset_id,
                  const uint64 fingerprint,
                  const std::string& worker_address,
                  const int64 task_id,
                  std::string& dataset_key);

Status AddPutOperator(const DatasetDef& dataset, DatasetDef& updated_dataset,
                      const experimental::DispatcherConfig& dispatcher_config);

Status AddGetOperator(const DatasetDef& dataset, DatasetDef& updated_dataset,
                      const experimental::DispatcherConfig& dispatcher_config);


} // namespace cache_utils
} // namespace easl
} // namespace service
} // namespace data
} // namespace tensorflow

#endif //TENSORFLOW_CORE_DATA_SERVICE_EASL_CACHE_UTILS_H_
