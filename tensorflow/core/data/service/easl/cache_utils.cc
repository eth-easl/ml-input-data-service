#include <queue>

#include "tensorflow/core/data/service/easl/cache_utils.h"

#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/grappler/mutable_graph_view.h"
#include "tensorflow/core/grappler/optimizers/data/easl_optimizers/add_put_op.h"
#include "tensorflow/core/grappler/optimizers/data/easl_optimizers/add_get_op.h"
#include "tensorflow/core/grappler/optimizers/data/graph_utils.h"
#include "tensorflow/core/grappler/utils/graph_view.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/data/service/easl/cache_model.h"

namespace tensorflow {
namespace data {
namespace service {
namespace easl {
namespace cache_utils {

Status DoBFS(NodeDef* sink_node, GraphDef& graph_def, string prefix) {
  absl::flat_hash_set<std::string> visited;
  std::queue<NodeDef*> bfs_queue;
  bfs_queue.push(sink_node);

  VLOG(1) << "(" << prefix << ") BFS @ current_node: "
          << "Root --> " << sink_node->op();

  while (!bfs_queue.empty()) {
    NodeDef* current_node = bfs_queue.front();
    bfs_queue.pop();
    visited.insert(current_node->name());

    // Iterate throught the neighbors
    for (int i = 0; i < current_node->input_size(); ++i) {
      if (!visited.contains(current_node->input(i))) {
        int idx = tensorflow::grappler::graph_utils::FindGraphNodeWithName(
            current_node->input(i), graph_def);
        NodeDef* neighbor_node = graph_def.mutable_node(idx);
        bfs_queue.push(neighbor_node);

        VLOG(1) << "(" << prefix << ") BFS @ current_node: "
                << SummarizeNodeDef(*current_node) << " --> "
                << SummarizeNodeDef(*neighbor_node);
      }
    }
  }

  return Status::OK();
}

std::string DatasetPutKey(const int64 id, const uint64 fingerprint) {
  return absl::StrCat("id_", id, "_fp_", fingerprint, "_put");
}

std::string DatasetGetKey(const int64 id, const uint64 fingerprint) {
  return absl::StrCat("id_", id, "_fp_", fingerprint, "_get");
}

std::string DatasetKey(
    const int64 id, const uint64 fingerprint, const std::string& job_type){
  if(job_type=="COMPUTE"){
    return absl::StrCat("id_", id, "_fp_", fingerprint);
  } else if (job_type=="GET"){
    return DatasetGetKey(id, fingerprint);
  } else if (job_type=="PUT"){
    return DatasetPutKey(id, fingerprint);
  }
  return "";
}

// TODO (damien-aymon) deprecated, left here for reference.
/*
Status DatasetKey(const ::tensorflow::data::easl::CacheState& cache_state,
                  const int64 dataset_id,
                  const uint64 fingerprint,
                  const std::string& worker_address,
                  const int64 task_id,
                  std::string& dataset_key){
  if(cache_state.IsDatasetCached(fingerprint, worker_address)){
    dataset_key =
        absl::StrCat("id_", dataset_id, "_fp_", fingerprint, "_get");
    VLOG(0) << "Use get dataset for fingerprint " << fingerprint
            << " at worker " << worker_address;
    return Status::OK();
  }

  int64 caching_task;
  TF_RETURN_IF_ERROR(cache_state.GetCachingTaskId(
      fingerprint, worker_address, caching_task));
  if(caching_task == task_id) {
    dataset_key =
        absl::StrCat("id_", dataset_id, "_fp_", fingerprint, "_put");
    VLOG(0) << "Use put dataset for fingerprint " << fingerprint
            << " at worker " << worker_address;
    return Status::OK();
  }

  dataset_key =
      absl::StrCat("id_", dataset_id, "_fp_", fingerprint);
  VLOG(0) << "Use standard dataset for fingerprint " << fingerprint
          << " at worker " << worker_address;
  return Status::OK();
}*/

Status DetermineJobType(const experimental::DispatcherConfig& dispatcher_config,
                     ::tensorflow::data::CacheState& cache_state,
                     const ::tensorflow::data::easl::MetadataStore& metadata_store,
                     const uint64 fingerprint,
                     const std::string& dataset_key,
                     const int64 job_id,
                     std::string& job_type) {
  // First check if we should use a "fixed" cache policy:
  // 2==compute, 3==cache(put, then get from 2nd epoch)
  // ---------------------------------------------------------------------------
  if(dispatcher_config.cache_policy()==2){
    job_type = "COMPUTE";
    return Status::OK();
  } else if(dispatcher_config.cache_policy()==3){
    if(cache_state.IsDatasetCached(fingerprint)){
      job_type = "GET";
    } else {
      job_type = "PUT";
    }
    return Status::OK();
  }
  // ---------------------------------------------------------------------------

  // Cache policy = EASL (cache_policy==1)
  // ---------------------------------------------------------------------------

  // If dataset was previously cached, assume it was faster than compute
  // and decide to read.
  if(cache_state.IsDatasetCached(fingerprint)){
    job_type = "GET";
    return Status::OK();
  }
  std::shared_ptr<::tensorflow::data::easl::InputPipelineMetrics> job_metrics;
  Status s = metadata_store.GetInputPipelineMetricsByDatasetKey(dataset_key, job_metrics);

  // We do not yet have the metrics for this dataset
  if(errors::IsNotFound(s)){
    job_type = "COMPUTE";
    return Status::OK();
  } else if (!s.ok()){
    return s;
  }

  // Pipeline stats
  using NodeMetrics = ::tensorflow::data::easl::NodeMetrics;
  std::shared_ptr<NodeMetrics> node_metrics;
  TF_RETURN_IF_ERROR(metadata_store.GetLastNodeMetricsByDatasetKey(dataset_key, node_metrics));

  uint64 row_size = 0;
  double compute_time_per_row_ms = 0;

  size_t num_workers = (node_metrics->metrics_).size();
  DCHECK(num_workers > 0);

  for(std::pair<std::string, std::shared_ptr<NodeMetrics::Metrics>> e : node_metrics->metrics_){
    std::shared_ptr<NodeMetrics::Metrics> worker_metrics = e.second;
    // TODO average out row size here for datasets with varying row size?
    row_size = worker_metrics->bytes_produced() / worker_metrics->num_elements();
    compute_time_per_row_ms += worker_metrics->in_prefix_time_ms();
  }

  compute_time_per_row_ms = compute_time_per_row_ms / num_workers;

  VLOG(0) << "row size " << row_size;
  VLOG(0) << "compute time " << compute_time_per_row_ms;

  // Caching model
  double cache_read_time_per_row_ms = ::tensorflow::data::cache_model::GetTimePerRow(row_size);

  VLOG(0) << "cache time " << cache_read_time_per_row_ms;

  // Simplest possible caching decision:
  if(cache_read_time_per_row_ms < compute_time_per_row_ms){
    job_type = "PUT"; // Job should be put, otherwise cache will never fill up.
    VLOG(0) << "dedide put";
    cache_state.RegisterCachingJob(fingerprint, job_id);
  } else {
    VLOG(0) << "decide compute";
    job_type = "COMPUTE";
  }

  return Status::OK();
}

// Status DetermineElasticity(
//   const int64 job_id,
//   const DispatcherState& dispatcher_state,
//   const std::string& job_type,
//   const experimental::DispatcherConfig& dispatcher_config,
//   const ::tensorflow::data::easl::MetadataStore& metadata_store,
//   const uint64 fingerprint,
//   const std::string& dataset_key
//   ) {
  
//   // ReserveWorkers()
//   // [TODO]: Finish implementation
//   return Status::OK();
// }

Status AddPutOperator(const DatasetDef& dataset,
                      const uint64 fingerprint,
                      const experimental::DispatcherConfig& dispatcher_config,
                      DatasetDef& updated_dataset) {
  // TODO remove this.
  //updated_dataset = dataset;
  //return Status::OK();
  VLOG(1) << "(AddPutOperator) At the start of the method";
  // Copy over the original dataset
  updated_dataset = dataset;

  // Initialize the optimizer
  tensorflow::grappler::easl::AddPutOp optimizer;
  // Transfer arguments from dispatcher config to optimizer config.
  tensorflow::RewriterConfig_CustomGraphOptimizer config;

  // TODO - set path where to store graph.
  (*(config.mutable_parameter_map()))["path"].set_placeholder(
      absl::StrCat(dispatcher_config.cache_path(), "/", fingerprint));
  (*(config.mutable_parameter_map()))["cache_format"].set_i(
      dispatcher_config.cache_format());
  (*(config.mutable_parameter_map()))["cache_compression"].set_i(
      dispatcher_config.cache_compression());
  (*(config.mutable_parameter_map()))["cache_ops_parallelism"].set_i(
      dispatcher_config.cache_ops_parallelism());

  optimizer.Init(&config);

  // Get the graph def and wrap it in a GrapplerItem
  GraphDef* graph_def = updated_dataset.mutable_graph();
  std::string output_node;

  // Find the output node; the one before '_Retval'
  for (const auto& node : graph_def->node()) {
    if (node.op() == "_Retval") {
      output_node = node.input(0);
    }
  }

  // Create a 'Sink' node and attatch it to the real output
  NodeDef* sink = graph_def->mutable_node()->Add();
  tensorflow::grappler::graph_utils::SetUniqueGraphNodeName("Sink", graph_def,
                                                            sink);
  sink->set_op("Identity");
  sink->add_input(output_node);
  (*sink->mutable_attr())["T"].set_type(DT_VARIANT);

  // Do BFS
  DoBFS(sink, *graph_def, "AddPutOperator");

  // Create the MuttableGraphView
  tensorflow::grappler::MutableGraphView graph(graph_def);
  optimizer.ApplyOptimization(graph, sink, graph_def);

  // Do BFS
  DoBFS(sink, *graph_def, "AfterAddPutOperator");

  // Disconnect the 'Sink' node
  // sink->mutable_input()->Clear();
  VLOG(1) << "(AddPutOperator) At the end of the method";

  return Status::OK();
}


Status AddGetOperator(const DatasetDef& dataset,
                      const uint64 fingerprint,
                      const experimental::DispatcherConfig& dispatcher_config,
                      DatasetDef& updated_dataset){
  // TODO remove this.
  //updated_dataset = dataset;
  //return Status::OK();

  VLOG(1) << "(AddGetOperator) At the start of the method";
  // Copy over the original dataset
  updated_dataset = dataset;

  // Initialize the optimizer  
  tensorflow::grappler::easl::AddGetOp optimizer;
  // Transfer arguments from dispatcher config to optimizer config.
  tensorflow::RewriterConfig_CustomGraphOptimizer config;

  // TODO - set path where to store graph.
  (*(config.mutable_parameter_map()))["path"].set_placeholder(
      absl::StrCat(dispatcher_config.cache_path(), "/", fingerprint));
  (*(config.mutable_parameter_map()))["cache_format"].set_i(
      dispatcher_config.cache_format());
  (*(config.mutable_parameter_map()))["cache_compression"].set_i(
      dispatcher_config.cache_compression());
  (*(config.mutable_parameter_map()))["cache_ops_parallelism"].set_i(
      dispatcher_config.cache_ops_parallelism());

  optimizer.Init(&config);

  // Get the graph def and wrap it in a GrapplerItem
  GraphDef* graph_def = updated_dataset.mutable_graph();
  std::string output_node;

  // Find the output node; the one before '_Retval'
  for (const auto& node : graph_def->node()) {
    if (node.op() == "_Retval") {
      output_node = node.input(0);
    }
  }

  // Create a 'Sink' node and attatch it to the real output
  NodeDef* sink = graph_def->mutable_node()->Add();
  tensorflow::grappler::graph_utils::SetUniqueGraphNodeName("Sink", graph_def,
                                                            sink);
  sink->set_op("Identity");
  sink->add_input(output_node);
  (*sink->mutable_attr())["T"].set_type(DT_VARIANT);

  // Do BFS
  DoBFS(sink, *graph_def, "AddGetOperator");

  // Create the MuttableGraphView
  tensorflow::grappler::MutableGraphView graph(graph_def);
  optimizer.ApplyOptimization(graph, sink, graph_def);

  // Do BFS
  DoBFS(sink, *graph_def, "AfterAddGetOperator");

  // Disconnect the 'Sink' node
  // sink->mutable_input()->Clear();
  VLOG(1) << "(AddGetOperator) At the end of the method";

  return Status::OK();
}



} // namespace cache_utils
} // namespace easl
} // namespace service
} // namespace data
} // namespace tensorflow
