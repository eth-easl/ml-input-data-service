#include "tensorflow/core/data/service/easl/metadata_store.h"

#include <memory>
#include <sstream>
#include <ostream>
#include <fstream>

#include "tensorflow/core/platform/errors.h"

namespace tensorflow {
namespace data {
namespace easl {


// Metrics from the Client
ModelMetrics::Metrics::Metrics(double get_next_time_ms, 
  double inter_arrival_time_ms) 
  : get_next_time_ms_(get_next_time_ms),
    inter_arrival_time_ms_(inter_arrival_time_ms) {}

ModelMetrics::Metrics::Metrics(ModelMetrics::Metrics& other) 
  : get_next_time_ms_(other.get_next_time_ms()),
    inter_arrival_time_ms_(other.inter_arrival_time_ms()) {}

void ModelMetrics::Metrics::Update(ModelMetrics::Metrics& other) {
  get_next_time_ms_ = other.get_next_time_ms_;
  inter_arrival_time_ms_ = other.inter_arrival_time_ms_;
}

Status ModelMetrics::UpdateClientMetrics(int64 client_id, 
  ModelMetrics::Metrics& metrics) {
  auto it = metrics_.find(client_id);
  if (it == metrics_.end()) {
    auto entry = std::make_shared<Metrics>(metrics);
    metrics_.insert({client_id, entry});
    VLOG(2) << "Created model metrics for client " << client_id;
  } else {
    it->second->Update(metrics);
    VLOG(2) << "Updated model metrics for client " << client_id;
  }
  return Status::OK();
}

Status ModelMetrics::GetClientMetrics(int64 client_id, 
  std::shared_ptr<Metrics>& metrics) {
  auto it = metrics_.find(client_id);
  if (it != metrics_.end()) {
    metrics = it->second;
    return Status::OK();
  }
  return errors::NotFound("No metrics under the client with id ", client_id);
}

void ModelMetrics:: DumpToStream(std::stringstream& ss){
  ss << "{ " << std::endl;
  bool first = true;
  for(auto pair : metrics_){
    if(!first){ ss << "," << std::endl; first = false; } // Add comma before element, not for first though.
    ss << "\"" << std::to_string(pair.first) << "\" : ";
    ss << "{ \"get_next_time_ms\" : " << std::to_string(pair.second->get_next_time_ms()) << " , ";
    ss << "\"inter_arrival_time_ms\" : " << std::to_string(pair.second->inter_arrival_time_ms()) << " }";
    ss << std::endl;
  }
  ss << "}";
}


// Metrics from the Worker Nodes
NodeMetrics::Metrics::Metrics(NodeMetrics::Metrics& other) 
  : bytes_consumed_(other.bytes_consumed()),
    bytes_produced_(other.bytes_produced()),
    num_elements_(other.num_elements()),
    // computation_time_(other.computation_time()),
    in_node_time_ms_(other.in_node_time_ms()),
    in_prefix_time_ms_(other.in_prefix_time_ms()) {}

NodeMetrics::Metrics::Metrics(int64 bytes_consumed, int64 bytes_produced, 
  int64 num_elements, 
  // int64 computation_time, 
  double in_node_time_ms, double in_prefix_time_ms) 
  : bytes_consumed_(bytes_consumed),
    bytes_produced_(bytes_produced),
    num_elements_(num_elements),
    // computation_time_(computation_time),
    in_node_time_ms_(in_node_time_ms),
    in_prefix_time_ms_(in_prefix_time_ms) {}

void NodeMetrics::Metrics::Update(NodeMetrics::Metrics& other) {
  bytes_consumed_ = other.bytes_consumed_;
  bytes_produced_ = other.bytes_produced_;
  num_elements_ = other.num_elements_;
  // computation_time_ = other.computation_time_;
  in_node_time_ms_ = other.in_node_time_ms_;
  in_prefix_time_ms_ = other.in_prefix_time_ms_; 
}

Status NodeMetrics::UpdateWorkerMetrics(string worker_address, 
  NodeMetrics::Metrics& metrics) {
  auto it = metrics_.find(worker_address);
  if (it != metrics_.end()) {
    it->second->Update(metrics);
  } else {
    auto entry = std::make_shared<NodeMetrics::Metrics>(metrics);
    metrics_.insert({worker_address, entry});
  }
  return Status::OK();
}

Status NodeMetrics::GetWorkerMetrics(string worker_address, 
  std::shared_ptr<Metrics>& metrics) {
  auto it = metrics_.find(worker_address);
  if (it != metrics_.end()) {
    metrics = it->second;
    return Status::OK();
  }
  return errors::NotFound("No metrics under the worker with address ", 
    worker_address);
}

void NodeMetrics::DumpToStream(std::stringstream &ss) {
  ss << "{ " << std::endl;
  bool first = true;
  for(auto pair : metrics_){
    // Add comma before element, not for first though.
    if(!first){ ss << "," << std::endl; }
    else { first = false; }
    ss << "\"" << pair.first << "\" : ";
    ss << "{ \"bytes_consumed\" : " << std::to_string(pair.second->bytes_consumed()) << " , ";
    ss << "\"bytes_produced\" : " << std::to_string(pair.second->bytes_produced()) << " , ";
    ss << "\"num_elements\" : " << std::to_string(pair.second->num_elements()) << " , ";
    // ss << "\"computation_time\" : " << std::to_string(pair.second->computation_time()) << " , ";
    ss << "\"in_node_time_ms\" : " << std::to_string(pair.second->in_node_time_ms()) << " , ";
    ss << "\"in_prefix_time_ms\" : " << std::to_string(pair.second->in_prefix_time_ms()) << " }";
    ss << std::endl;
  }
  ss << "}";
}

// Input pipeline metrics
Status InputPipelineMetrics::GetNodeMetrics(string long_name, 
  std::shared_ptr<NodeMetrics>& metrics) {
  auto it = metrics_.find(long_name);
  if (it != metrics_.end()) {
    metrics = it->second;
    return Status::OK();
  }
  return errors::NotFound("No metrics for node ", long_name); 
}

Status InputPipelineMetrics::GetLastNodeMetrics(
  std::shared_ptr<NodeMetrics>& metrics) {
  if (last_node_name_ == "") {
    return errors::NotFound("Last node was not given a name");
  }
  GetNodeMetrics(last_node_name_, metrics);
  return Status::OK();
}

Status InputPipelineMetrics::GetLastTFNodeMetrics(
  std::shared_ptr<NodeMetrics>& metrics) {
  if (last_tf_node_name_ == "") {
    return errors::NotFound("Last TF node was not given a name");
  }
  GetNodeMetrics(last_tf_node_name_, metrics);
  return Status::OK();
}

Status InputPipelineMetrics::GetWorkerMetrics(string worker_address, 
  NodeMetrics::MetricsCollection& metrics) {
  for (auto& entry : metrics_) {
    std::shared_ptr<NodeMetrics::Metrics> node_metrics;
    Status s = entry.second->GetWorkerMetrics(worker_address, node_metrics);
    if (s.ok()) {
      metrics.insert({entry.first, node_metrics});
    }
  }
  return Status::OK();
}

Status InputPipelineMetrics::UpdateNodeMetrics(string long_name,
  string worker_address, NodeMetrics::Metrics& metrics) {
  auto it = metrics_.find(long_name); 
  if (it == metrics_.end()) {
    auto node_metrics = std::make_shared<NodeMetrics>();
    node_metrics->UpdateWorkerMetrics(worker_address, metrics);
    metrics_.insert({long_name, node_metrics});
    VLOG(2) << "Created node " << long_name << "'s metrics for worker " 
            << worker_address;
  } else {
    it->second->UpdateWorkerMetrics(worker_address, metrics);
    VLOG(2) << "Updated node " << long_name << "'s metrics for worker " 
            << worker_address;
  }
  return Status::OK();
}

std::string InputPipelineMetrics::GetLastNodeName() { return last_node_name_; }
void InputPipelineMetrics::SetLastNodeName(std::string last_node_name) {
  last_node_name_ = last_node_name;
}

std::string InputPipelineMetrics::GetLastTFNodeName() { return last_tf_node_name_; }
void InputPipelineMetrics::SetLastTFNodeName(std::string last_tf_node_name) {
  last_tf_node_name_ = last_tf_node_name;
}

void InputPipelineMetrics::DumpToStream(std::stringstream& ss){
  ss << "{ " << std::endl;
  ss << "\"last_node_name\" : \"" << last_node_name_ << "\"";

  bool first = true;
  for(auto pair : metrics_){
    ss << ", " << std::endl;
    ss << "\"" << pair.first << "\" : ";
    pair.second->DumpToStream(ss);
  }
  ss << std::endl << "}";
}


// Job metrics
JobMetrics::JobMetrics(int64 job_id,
                       int64 dataset_id,
                       int64 dataset_fingerprint,
                       std::string& dataset_key,
                       int64 worker_count)
      : job_id_(job_id),
        dataset_id_(dataset_id),
        dataset_fingerprint_(dataset_fingerprint),
        dataset_key_(dataset_key),
        model_metrics_(), 
        input_pipeline_metrics_(),
        worker_count_(worker_count) {
          model_metrics_ = std::make_shared<ModelMetrics>();
          input_pipeline_metrics_ = std::make_shared<InputPipelineMetrics>();
        }


// JobMetrics
void JobMetrics::DumpToFile(const std::string& path){
  // Start constructing json string:
  std::stringstream ss;
  DumpToStream(ss);

  std::ofstream fstream;
  fstream.open(
      path + "/metrics_job_" + std::to_string(job_id_) + "_ds_key_" + dataset_key_  + ".json",
      std::ofstream::out | std::ofstream::app);
  fstream << ss.rdbuf();
  fstream.close();
}

void JobMetrics::DumpToStream(std::stringstream& ss){
  ss << "{" << std::endl;
  ss << "\"ModelMetrics\" : ";
  model_metrics_->DumpToStream(ss);
  ss << "," << std::endl;
  ss << "\"PipelineMetrics\" : ";
  input_pipeline_metrics_->DumpToStream(ss);
  ss << "}" << std::endl;
}


// Metadata store 
MetadataStore::MetadataStore() 
  : job_metadata_(),
    dataset_key_metadata_() {}

Status MetadataStore::CreateJob(int64 job_id, int64 dataset_id, 
  int64 dataset_fingerprint, std::string& dataset_key, int64 worker_count) {
  std::string ds_key = dataset_key;
  auto job_metrics = std::make_shared<JobMetrics>(
      job_id, dataset_id, dataset_fingerprint, ds_key, worker_count);
  job_metadata_.insert_or_assign(job_id, job_metrics);

  return Status::OK();
}

//Find a the job metric, delete it and add it to the dataset_key keyed metrics for persistence
Status MetadataStore::RemoveJob(int64 job_id) {
  // Update datasetKey indexed store with new JobMetrics.
  auto it = job_metadata_.find(job_id);
  if (it == job_metadata_.end()) {
    return errors::NotFound("Job with id ", job_id, " does not have metrics");
  }
  auto job_metrics = it->second;
  //dataset_key_metadata_.insert_or_assign(job_metrics->dataset_key_, job_metrics);

  // Properly erase job.
  job_metadata_.erase(job_id);
  return Status::OK();
}

Status MetadataStore::GetJobMetrics(int64 job_id, 
  std::shared_ptr<JobMetrics>& metrics) const {
  auto it = job_metadata_.find(job_id);
  if (it == job_metadata_.end()) {
    return errors::NotFound("Job with id ", job_id, " does not have metrics");
  }
  metrics = it->second;
  return Status::OK();
}

Status MetadataStore::GetModelMetrics(int64 job_id, 
  std::shared_ptr<ModelMetrics>& metrics) const {
  std::shared_ptr<JobMetrics> job_metrics;
  Status s = GetJobMetrics(job_id, job_metrics); 
  if (s.ok()) {
    metrics = job_metrics->model_metrics_;
  }
  return s;
}

Status MetadataStore::GetInputPipelineMetrics(int64 job_id, 
  std::shared_ptr<InputPipelineMetrics>& metrics) const {
  std::shared_ptr<JobMetrics> job_metrics;
  Status s = GetJobMetrics(job_id, job_metrics);
  if (s.ok()) {
    metrics = job_metrics->input_pipeline_metrics_;
  }
  return s;
}

Status MetadataStore::GetLastNodeMetrics(int64 job_id, 
  std::shared_ptr<NodeMetrics>& metrics) const {
  std::shared_ptr<JobMetrics> job_metrics;
  Status s = GetJobMetrics(job_id, job_metrics);
  if (s.ok()) {
    return job_metrics->input_pipeline_metrics_->GetLastNodeMetrics(metrics);
  }
  return s;
}

Status MetadataStore::GetLastTFNodeMetrics(int64 job_id, 
  std::shared_ptr<NodeMetrics>& metrics) const {
  std::shared_ptr<JobMetrics> job_metrics;
  Status s = GetJobMetrics(job_id, job_metrics);
  if (s.ok()) {
    return job_metrics->input_pipeline_metrics_->GetLastTFNodeMetrics(metrics);
  }
  return s;
}

Status MetadataStore::GetLastNodeMetricsByDatasetKey(
  const std::string& dataset_key, std::shared_ptr<NodeMetrics>& metrics) const {
  std::shared_ptr<JobMetrics> job_metrics;
  Status s = GetJobMetricsByDatasetKey(dataset_key, job_metrics);
  if (s.ok()) {
    return job_metrics->input_pipeline_metrics_->GetLastNodeMetrics(metrics);
  }
  return s;
}

Status MetadataStore::GetLastTFNodeMetricsByDatasetKey(
  const std::string& dataset_key, std::shared_ptr<NodeMetrics>& metrics) const {
  std::shared_ptr<JobMetrics> job_metrics;
  Status s = GetJobMetricsByDatasetKey(dataset_key, job_metrics);
  if (s.ok()) {
    return job_metrics->input_pipeline_metrics_->GetLastTFNodeMetrics(metrics);
  }
  return s;
}

Status MetadataStore::GetJobMetricsByDatasetKey(
    const std::string& dataset_key, std::shared_ptr<JobMetrics>& metrics) const {
  auto it = dataset_key_metadata_.find(dataset_key);
  if (it == dataset_key_metadata_.end()) {
    return errors::NotFound("Dataset ", dataset_key, " does not (yet) have metrics");
  }
  metrics = it->second;
  return Status::OK();
}

Status MetadataStore::GetModelMetricsByDatasetKey(
    const std::string& dataset_key, std::shared_ptr<ModelMetrics>& metrics) const {
  std::shared_ptr<JobMetrics> job_metrics;
  Status s = GetJobMetricsByDatasetKey(dataset_key, job_metrics);
  if (s.ok()) {
    metrics = job_metrics->model_metrics_;
  }
  return s;
}

Status MetadataStore::GetInputPipelineMetricsByDatasetKey(
    const std::string& dataset_key, std::shared_ptr<InputPipelineMetrics>& metrics) const {
  std::shared_ptr<JobMetrics> job_metrics;
  Status s = GetJobMetricsByDatasetKey(dataset_key, job_metrics);
  if (s.ok()) {
    metrics = job_metrics->input_pipeline_metrics_;
  }
  return s;
}

Status MetadataStore::UpdateModelMetrics(int64 job_id, int64 client_id, 
  ModelMetrics::Metrics& metrics) {
  std::shared_ptr<ModelMetrics> model_metrics;
  Status s = GetModelMetrics(job_id, model_metrics);
  if (s.ok()) {
    model_metrics->UpdateClientMetrics(client_id, metrics);
  } 
  // if s != ok --> no such job exists
  return s;
}

Status MetadataStore::UpdateInputPipelineMetrics(int64 job_id, 
  string node_long_name, string worker_address, NodeMetrics::Metrics& metrics) {
  std::shared_ptr<InputPipelineMetrics> pipeline_metrics;
  Status s = GetInputPipelineMetrics(job_id, pipeline_metrics);
  if (s.ok()) {
    pipeline_metrics->UpdateNodeMetrics(node_long_name, worker_address, 
      metrics);
  } 
  // if s != ok --> no such job exists
  return s;
}

Status MetadataStore::UpdateDatasetKeyJobMetrics(int64 job_id, 
  const std::string& dataset_key){
  auto it = job_metadata_.find(job_id);
  if (it == job_metadata_.end()) {
    return errors::NotFound("Job with id ", job_id, " does not have metrics");
  }
  auto job_metrics = it->second;
  dataset_key_metadata_.insert_or_assign(job_metrics->dataset_key_, job_metrics);

  return Status::OK();
}

Status MetadataStore::UpdateLastNodes(int64 job_id, string last_node_name, 
  string last_tf_node_name) {
  std::shared_ptr<InputPipelineMetrics> pipeline_metrics;
  Status s = GetInputPipelineMetrics(job_id, pipeline_metrics);
  if (s.ok()) {
    pipeline_metrics->SetLastNodeName(last_node_name);
    pipeline_metrics->SetLastTFNodeName(last_tf_node_name);
  } 
  // if s != ok --> no such job exists
  return s;
}


Status MetadataStore::DumpJobMetricsToFile(int64 job_id, const std::string& path){
  std::shared_ptr<JobMetrics> jobMetrics;
  TF_RETURN_IF_ERROR(GetJobMetrics(job_id, jobMetrics));
  jobMetrics->DumpToFile(path);
}

Status MetadataStore::AppendJobMetricsDumps(Env* env, const std::string& path) {
  for(auto pair: job_metadata_ ){
    std::string fname = path + "/metrics_updates_job_" + std::to_string(pair.first) + ".json";
    std::stringstream ss;

    Status s = env->FileExists(fname);
    if(!s.ok()){ // Initialize file if not present.
      if (!errors::IsNotFound(s)) {
        return s;
      }
      ss << "[" << std::endl;
    } else {
      ss << ", " << std::endl;
    }

    pair.second->DumpToStream(ss);
    // Append to file
    std::ofstream fstream;
    fstream.open(
        fname,
        std::ofstream::out | std::ofstream::app);
    fstream << ss.rdbuf();
    fstream.close();
  }
}

void TerminateJobMetricsAppendDumps(int64 job_id, const std::string& path){
  std::string fname = path + "metrics_updates_job_" + std::to_string(job_id) + ".json";
  std::stringstream ss;

  ss << " ]" << std::endl;

  std::ofstream fstream;
  fstream.open(
      fname,
      std::ofstream::out | std::ofstream::app);
  fstream << ss.rdbuf();
  fstream.close();
}



} // namespace easl
} // namespace data
} // namespace tensorflow
