/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#ifndef TENSORFLOW_CORE_DATA_SERVICE_EASL_METADATA_STORE_H_
#define TENSORFLOW_CORE_DATA_SERVICE_EASL_METADATA_STORE_H_

#include "absl/container/flat_hash_map.h"
#include "tensorflow/core/data/service/common.pb.h"
#include "tensorflow/core/data/service/dispatcher.pb.h"
#include "tensorflow/core/data/service/data_service.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/env.h"


namespace tensorflow {
namespace data {
namespace easl {

class ModelMetrics {
  public:
    class Metrics {
      public:
        Metrics(Metrics& other);
        Metrics(double get_next_time_ms, double inter_arrival_time_ms);

        void Update(Metrics& other);

        void set_get_next_time_ms(double x) { get_next_time_ms_ = x; }
        void set_inter_arrival_time_ms(double x) { inter_arrival_time_ms_ = x; }

        double get_next_time_ms() { return get_next_time_ms_; }
        double inter_arrival_time_ms() { return inter_arrival_time_ms_; }

      private:
        double get_next_time_ms_; 
        double inter_arrival_time_ms_;
    };

    using MetricsCollection = 
      absl::flat_hash_map<int64, std::shared_ptr<ModelMetrics::Metrics>>;

    ModelMetrics() {}

    // Update the values for a client
    Status UpdateClientMetrics(int64 client_id, Metrics& metrics);
    Status GetClientMetrics(int64 client_id, std::shared_ptr<Metrics>& metrics);

    // Dump metrics to a string stream
    void DumpToStream(std::stringstream& ss);

    // The keys are the client id
    MetricsCollection metrics_;
};

class NodeMetrics {
  public: 
    class Metrics {
      public:
        explicit Metrics(Metrics& other);
        explicit Metrics(int64 bytes_consumed, int64 bytes_produced, 
                          int64 num_elements, 
                          // int64 computation_time, 
                          double in_node_time_ms, double in_prefix_time_ms);
        
        void Update(Metrics& other);
        
        void set_bytes_consumed(int64 x)   { bytes_consumed_ = x; }
        void set_bytes_produced(int64 x)   { bytes_produced_ = x; }
        void set_num_elements(int64 x)     { num_elements_ = x; }
        // void set_computation_time(int64 x) { computation_time_ = x; }
        void set_in_node_time_ms(double x)    { in_node_time_ms_ = x; }
        void set_in_prefix_time_ms(double x)  { in_prefix_time_ms_ = x; }

        int64 bytes_consumed()   { return bytes_consumed_; }
        int64 bytes_produced()   { return bytes_produced_; }
        int64 num_elements()     { return num_elements_; }
        // int64 computation_time() { return computation_time_; }
        double in_node_time_ms()     { return in_node_time_ms_; }
        double in_prefix_time_ms()   { return in_prefix_time_ms_; }

      private:
        int64 bytes_consumed_;
        int64 bytes_produced_;
        int64 num_elements_;
        // int64 computation_time_;
        double in_node_time_ms_;
        double in_prefix_time_ms_; 
    };

    using MetricsCollection =
      absl::flat_hash_map<string, std::shared_ptr<NodeMetrics::Metrics>>;

    NodeMetrics() {}

    // Get or update the metrics of a worker
    Status UpdateWorkerMetrics(string worker_address, Metrics& metrics);
    Status GetWorkerMetrics(string worker_address, 
      std::shared_ptr<Metrics>& metrics);

    // Dump metrics to string stream
    void DumpToStream(std::stringstream& ss);

  // The key here is the worker address
    MetricsCollection metrics_;
};


class InputPipelineMetrics {
  public:
    using MetricsCollection = 
      absl::flat_hash_map<string, std::shared_ptr<NodeMetrics>>;

    InputPipelineMetrics() 
      : last_node_name_(""),
        last_tf_node_name_("") {}
    InputPipelineMetrics(std::string last_node_name, 
      std::string last_tf_node_name) 
      : last_node_name_(last_node_name), 
        last_tf_node_name_(last_tf_node_name) {}

    // Get the metrics for a single node
    Status GetNodeMetrics(string long_name, 
      std::shared_ptr<NodeMetrics>& metrics);
    Status GetLastNodeMetrics(std::shared_ptr<NodeMetrics>& metrics);
    Status GetLastTFNodeMetrics(std::shared_ptr<NodeMetrics>& metrics);

    // Get the metrics from the same worker for each node in the graph 
    Status GetWorkerMetrics(string worker_address, 
      NodeMetrics::MetricsCollection& metrics);

    // Methods for setting data
    Status UpdateNodeMetrics(string long_name, string worker_address, 
      NodeMetrics::Metrics& metrics);

    // Methods for managing the last node name
    std::string GetLastNodeName();
    void SetLastNodeName(std::string last_node_name);
    std::string GetLastTFNodeName();
    void SetLastTFNodeName(std::string last_tf_node_name);

    void DumpToStream(std::stringstream& ss);

    // Last user node name
    std::string last_node_name_;
    std::string last_tf_node_name_;
    // The keys are the long name of the node
    MetricsCollection metrics_;
};

class JobMetrics {
  public:
    JobMetrics(int64 job_id,
               int64 dataset_id,
               int64 dataset_fingerprint,
               std::string& dataset_key);

    void DumpToFile(const std::string& path);
    void DumpToStream(std::stringstream& ss);

    int64 job_id_;
    int64 dataset_id_;
    int64 dataset_fingerprint_;
    std::string dataset_key_;
    std::shared_ptr<ModelMetrics> model_metrics_;
    std::shared_ptr<InputPipelineMetrics> input_pipeline_metrics_;
};

class MetadataStore {
 public:
  MetadataStore();
  MetadataStore(const MetadataStore &) = delete;
  MetadataStore &operator=(const MetadataStore &) = delete;

  // Create a job entry
  Status CreateJob(int64 job_id,
                   int64 dataset_id,
                   int64 dataset_fingerprint,
                   std::string& dataset_key);

  // Remove job
  Status RemoveJob(int64 job_id);

  // Get metrics
  Status GetJobMetrics(int64 job_id, std::shared_ptr<JobMetrics>& metrics) const;

  Status GetModelMetrics(int64 job_id, 
    std::shared_ptr<ModelMetrics>& metrics) const;

  Status GetInputPipelineMetrics(int64 job_id, 
    std::shared_ptr<InputPipelineMetrics>& metrics) const;

  Status GetJobMetricsByDatasetKey(const std::string& dataset_key, 
    std::shared_ptr<JobMetrics>& metrics) const;

  Status GetModelMetricsByDatasetKey(const std::string& dataset_key, 
    std::shared_ptr<ModelMetrics>& metrics) const;

  Status GetInputPipelineMetricsByDatasetKey(const std::string& dataset_key, 
    std::shared_ptr<InputPipelineMetrics>& metrics) const;
  
  Status GetLastNodeMetrics(int64 job_id, 
    std::shared_ptr<NodeMetrics>& metrics) const;
  Status GetLastTFNodeMetrics(int64 job_id, 
    std::shared_ptr<NodeMetrics>& metrics) const;

  Status GetLastNodeMetricsByDatasetKey(const std::string& dataset_key,
    std::shared_ptr<NodeMetrics>& metrics) const;
  Status GetLastTFNodeMetricsByDatasetKey(const std::string& dataset_key,
    std::shared_ptr<NodeMetrics>& metrics) const;

  // Update or create the metrics for a client
  Status UpdateModelMetrics(int64 job_id, int64 client_id, 
    ModelMetrics::Metrics& metrics);

  // Update or create the metrics for a client
  Status UpdateInputPipelineMetrics(int64 job_id, string node_long_name, 
    string worker_address, NodeMetrics::Metrics& metrics);
  
  Status UpdateLastNodes(int64 job_id, string last_node_name, 
    string last_tf_node_name);

  // Update or create the metrics for the dataset key from the given job.
  Status UpdateDatasetKeyJobMetrics(int64 job_id, const std::string& dataset_key);

  // Dumps the job metrics in a file named after its id at the given path.
  Status DumpJobMetricsToFile(int64 job_id, const std::string& path);

  // Appends json representation of current status to a separate file for each job
  Status AppendJobMetricsDumps(Env* env, const std::string& path);


 private:
  // Key is job id
  absl::flat_hash_map<int64, std::shared_ptr<JobMetrics>> job_metadata_;
  // Key is dataset_key
  absl::flat_hash_map<std::string, std::shared_ptr<JobMetrics>> dataset_key_metadata_;
};

// Utils function to append the missing trailing "]"
// at the end of a metrics update dump file.
void TerminateJobMetricsAppendDumps(int64 job_id, const std::string& path);
} // namespace easl
} // namespace data
} // namespace tensorflow

#endif  // TENSORFLOW_CORE_DATA_SERVICE_EASL_METADATA_STORE_H_
