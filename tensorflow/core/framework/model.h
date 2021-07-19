/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_FRAMEWORK_MODEL_H_
#define TENSORFLOW_CORE_FRAMEWORK_MODEL_H_

#include <list>
#include <memory>
#include <string>
// TODO(b/114492873): Move this include into core/platform.
#include <thread>  // NOLINT
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "tensorflow/core/framework/cancellation.h"
#include "tensorflow/core/framework/metrics.h"
#include "tensorflow/core/framework/model.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/lib/histogram/histogram.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/platform/cpu_info.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/path.h"

namespace tensorflow {
namespace data {
namespace model {

// A constant that can be used to enable auto-tuning.
constexpr int64 kAutotune = -1;
constexpr char kParallelism[] = "parallelism";
constexpr char kBufferSize[] = "buffer_size";

// A key used to identify the input time of the model.
constexpr char kModelInputTimeKey[] = "model_input_time";

enum class TraversalOrder {
  BFS = 0,
  REVERSE_BFS = 1,
};

// Represents thread-safe state that can be shared between an input pipeline and
// the performance model.
struct SharedState {
 public:
  SharedState(int64 value, std::shared_ptr<mutex> mu,
              std::shared_ptr<condition_variable> cond_var)
      : value(value),
        mu(std::move(mu)),
        cond_var(std::move(cond_var)),
        tunable(value == kAutotune) {}

  double value;
  const std::shared_ptr<mutex> mu;
  const std::shared_ptr<condition_variable> cond_var;
  const bool tunable;
};

// Represents a parameter.
struct Parameter {
  Parameter(const string& name, std::shared_ptr<SharedState> state, double min,
            double max)
      : name(name),
        // Sometimes non-autotune nodes (with `autotune_=false`) may contain
        // parameters (for example inputs of parallel interleave dataset which
        // are not in the current cycle). To avoid unrealistic situation
        // (say `buffer_size=-1` or `parallelism=-1`) in the optimization
        // computation, if the state value is `kAutotune=-1` (just to indicate
        // the `SharedState` is tunable), we initialize the parameter value to
        // be the minimal value of the state.
        value(state->value == kAutotune ? min : state->value),
        min(min),
        max(max),
        state(std::move(state)) {}

  // Human-readable name of the parameter.
  const string name;

  // Identifies the model value of the parameter. This can be different from
  // the actual value (e.g. during optimization search).
  double value;

  // Identifies the minimum value of the parameter.
  const double min;

  // Identifies the maximum value of the parameter.
  const double max;

  // Shared state of the parameter.
  std::shared_ptr<SharedState> state;
};

std::shared_ptr<Parameter> MakeParameter(const string& name,
                                         std::shared_ptr<SharedState> state,
                                         double min, double max);

// Abstract representation of a TensorFlow input pipeline node. It collects
// information about inputs to this node, processing time spent executing the
// node logic, number of elements produced by the node, various other
// information (e.g. batch size or execution parallelism).
//
// Developers of tf.data transformations are not expected to interact with
// this class directly. Boiler plate code for creating the abstract
// representation of the input pipeline and collecting common information has
// been added to the implementation of `DatasetBase` and `DatasetBaseIterator`
// respectively.
//
// In addition, `DatasetBaseIterator` provides wrappers that can be used for
// transformation-specific information collection. The `SetMetadata` wrapper
// can be used to pass arbitrary metadata to the modeling framework, while the
// `StartWork` and `StopWork` wrappers should be used to correctly account for
// processing time of multi-threaded transformation that yield the CPU; such
// transformations should invoke `StartWork()` when a transformation thread
// starts executing (e.g. when created or woken up) and `StopWork()` when a
// transformation thread stops executing (e.g. when returning or waiting).
class Node {
 public:
  // Arguments for `Node` constructor.
  struct Args {
    int64 id;
    string name;
    std::shared_ptr<Node> output;
  };

  using Factory = std::function<std::shared_ptr<Node>(Args)>;
  using NodeVector = std::vector<std::shared_ptr<Node>>;
  using NodePairList =
      std::list<std::pair<std::shared_ptr<Node>, std::shared_ptr<Node>>>;
  using ModelParameters =
      std::vector<std::pair<string, std::shared_ptr<Parameter>>>;
  using NodeValues = absl::flat_hash_map<string, double>;
  using ParameterGradients =
      absl::flat_hash_map<std::pair<string, string>, double>;

  explicit Node(Args args)
      : id_(args.id),
        name_(std::move(args.name)),
        autotune_(true),
        buffered_bytes_(0),
        buffered_elements_(0),
        bytes_consumed_(0),
        bytes_produced_(0),
        num_elements_(0),
        processing_time_(0),
        record_metrics_(true),
        metrics_(name_),
        output_(args.output.get()) {}

  virtual ~Node() {
    // Clear the sub-nodes instead of relying on implicit shared pointer
    // destructor to avoid potential stack overflow when the tree is deep.
    std::deque<std::shared_ptr<Node>> queue;
    {
      mutex_lock l(mu_);
      while (inputs_.size() > 0) {
        queue.push_back(inputs_.front());
        inputs_.pop_front();
      }
    }
    while (!queue.empty()) {
      auto node = queue.back();
      queue.pop_back();
      {
        mutex_lock l(node->mu_);
        while (node->inputs_.size() > 0) {
          queue.push_back(node->inputs_.front());
          node->inputs_.pop_front();
        }
      }
    }

    FlushMetrics();
  }

  // Adds an input.
  void add_input(std::shared_ptr<Node> node) TF_LOCKS_EXCLUDED(mu_) {
    mutex_lock l(mu_);
    inputs_.push_back(node);
  }

  // Increments the aggregate processing time by the given delta.
  void add_processing_time(int64 delta) TF_LOCKS_EXCLUDED(mu_) {
    processing_time_ += delta;
  }

  // Returns an indication whether autotuning is enabled for this node.
  bool autotune() const TF_LOCKS_EXCLUDED(mu_) {
    return autotune_;
  }

  // Returns the number of bytes stored in this node's buffer.
  int64 buffered_bytes() const TF_LOCKS_EXCLUDED(mu_) {
    return buffered_bytes_;
  }

  // Returns the number of elements stored in this node's buffer.
  int64 buffered_elements() const TF_LOCKS_EXCLUDED(mu_) {
    return buffered_elements_;
  }

  // Returns the number of bytes consumed by the node.
  int64 bytes_consumed() const TF_LOCKS_EXCLUDED(mu_) {
    return bytes_consumed_;
  }

  // Returns the number of bytes produced by the node.
  int64 bytes_produced() const TF_LOCKS_EXCLUDED(mu_) {
    return bytes_produced_;
  }

  // Indicates whether the node has tunable parameters.
  bool has_tunable_parameters() const TF_LOCKS_EXCLUDED(mu_) {
    tf_shared_lock l(mu_);
    for (const auto& pair : parameters_) {
      if (pair.second->state->tunable) return true;
    }
    return false;
  }

  // Returns the unique node ID.
  int64 id() const TF_LOCKS_EXCLUDED(mu_) { return id_; }

  // Returns the node inputs.
  std::list<std::shared_ptr<Node>> inputs() const TF_LOCKS_EXCLUDED(mu_) {
    tf_shared_lock l(mu_);
    return inputs_;
  }

  // Returns a longer node name that is guaranteed to be unique.
  string long_name() const { return strings::StrCat(name_, "(id:", id_, ")"); }

  // Returns the node name.
  const string& name() const { return name_; }

  // Returns the number of elements produced by the node.
  int64 num_elements() const TF_LOCKS_EXCLUDED(mu_) {
    return num_elements_;
  }

  // Returns the node output.
  Node* output() const { return output_; }

  // Returns the parameter value.
  double parameter_value(const string& name) const TF_LOCKS_EXCLUDED(mu_) {
    tf_shared_lock l(mu_);
    return parameters_.at(name)->state->value;
  }

  // Returns the aggregate processing time.
  int64 processing_time() const TF_LOCKS_EXCLUDED(mu_) {
    return processing_time_;
  }

  // Records that the node consumed the given number of bytes.
  void record_bytes_consumed(int64 num_bytes) { bytes_consumed_ += num_bytes; }

  // Records that the node produced the given number of bytes.
  void record_bytes_produced(int64 num_bytes) { bytes_produced_ += num_bytes; }

  // Records the change in this node's buffer.
  void record_buffer_event(int64 bytes_delta, int64 elements_delta) {
    buffered_bytes_ += bytes_delta;
    buffered_elements_ += elements_delta;
  }

  // Records that the node produced an element.
  void record_element() TF_LOCKS_EXCLUDED(mu_) {
    num_elements_++;
  }

  // Records that a node thread has started executing.
  void record_start(int64 time_nanos) TF_LOCKS_EXCLUDED(mu_) {
    DCHECK_EQ(work_start_, 0);
    work_start_ = time_nanos;
  }

  // Records that a node thread has stopped executing.
  void record_stop(int64 time_nanos) TF_LOCKS_EXCLUDED(mu_) {
    // TODO(jsimsa): Use DCHECK_NE(work_start_, 0) here.
    if (work_start_ != 0) {
      processing_time_ += time_nanos - work_start_;
      work_start_ = 0;
    } else {
      VLOG(1) << "Encountered a stop event without a matching start event.";
    }
  }

  // Returns whether work is currently being recorded, i.e. whether we are
  // currently between a `record_start` and a `record_stop`.
  bool is_recording() TF_LOCKS_EXCLUDED(mu_) { return work_start_ > 0; }

  // Removes an input.
  void remove_input(std::shared_ptr<Node> input) TF_LOCKS_EXCLUDED(mu_) {
    mutex_lock l(mu_);
    inputs_.remove(input);
  }

  // Sets the value that determines whether autotuning is enabled for this node.
  void set_autotune(bool autotune) TF_LOCKS_EXCLUDED(mu_) {
    autotune_.store(autotune);
  }

  // Given the average time between events when the elements in the buffer are
  // produced (`producer_time`), the average time between events when elements
  // in the buffer are consumed (`consumer_time`) and the buffer size, the
  // method computes the expected time an consumer event will have to wait.
  //
  // The wait time is approximated as the product of the probability the buffer
  // will be empty and the time it takes to produce an element into the buffer.
  //
  // The formula used for computing the probability is derived by modeling the
  // problem as an M/M/1/K queue
  // (https://en.wikipedia.org/wiki/Birth%E2%80%93death_process#M/M/1/K_queue).
  //
  // Collects derivatives of `ComputeWaitTime` w.r.t `producer_time`,
  // `consumer_time' and `buffer_size` if the corresponding pointers are not
  // `nullptr`.
  static double ComputeWaitTime(const double& producer_time,
                                const double& consumer_time,
                                const double& buffer_size,
                                double* producer_time_derivative,
                                double* consumer_time_derivative,
                                double* buffer_size_derivative);

  // Collects tunable parameters in the subtree rooted in this node.
  ModelParameters CollectTunableParameters() const TF_LOCKS_EXCLUDED(mu_);

  // Returns a human-readable representation of this node.
  string DebugString() const TF_LOCKS_EXCLUDED(mu_);

  // Flushes the metrics recorded by this node.
  void FlushMetrics() TF_LOCKS_EXCLUDED(mu_);

  // Flushes the metrics recorded by this node.
  void PrintMetrics();

  // Returns the per-element output time for this node and if `gradients` is not
  // `nullptr`, collects the output time gradient w.r.t. tunable parameters of
  // the subtree rooted in this node.
  double OutputTime(NodeValues* input_times,
                    ParameterGradients* gradients) const TF_LOCKS_EXCLUDED(mu_);

  // Returns a copy of this node, making a deep copy of its inputs and a
  // shallow copy of its tunable parameters.
  //
  // The purpose for this method is to allow the model optimization logic to
  // operate over immutable state while allowing concurrent model updates.
  std::shared_ptr<Node> Snapshot() const TF_LOCKS_EXCLUDED(mu_);

  // Returns the per-element processing time spent in this node.
  double SelfProcessingTime() const TF_LOCKS_EXCLUDED(mu_);

  // Returns the total number of bytes buffered in all nodes in the subtree for
  // which autotuning is enabled.
  double TotalBufferedBytes() const TF_LOCKS_EXCLUDED(mu_);

  // Collects the total buffer limit of all nodes in the subtree for which
  // autotuning is enabled. This number represents the amount of memory that
  // would be used by the subtree nodes if all of their buffers were full.
  double TotalMaximumBufferedBytes() const TF_LOCKS_EXCLUDED(mu_);

  // Returns the per-element CPU time spent in the subtree rooted in this node.
  // If `processing_times` is not `nullptr`, collects the per-element CPU time
  // spent in each node of the subtree.
  double TotalProcessingTime(NodeValues* processing_times)
      TF_LOCKS_EXCLUDED(mu_);

  // Recursively produces a proto for this node and its subtree.
  virtual Status ToProto(ModelProto::Node* node_proto) const;

  // Recursively restores a node and its subtree from the proto.
  static Status FromProto(ModelProto::Node node_proto,
                          std::shared_ptr<Node> output,
                          std::shared_ptr<Node>* node);

 protected:
  // Used for (incrementally) recording metrics. The class is thread-safe.
  class Metrics {
   public:
    explicit Metrics(const string& name)
        : bytes_consumed_counter_(metrics::GetTFDataBytesConsumedCounter(name)),
          bytes_produced_counter_(metrics::GetTFDataBytesProducedCounter(name)),
          num_elements_counter_(metrics::GetTFDataElementsCounter(name)),
          computation_time_counter_(metrics::GetTFDataProcessingTimeCounter(
              name)),
          recorded_bytes_consumed_(0),
          recorded_bytes_produced_(0),
          recorded_num_elements_(0),
          recorded_computation_time_(0) {}

    // Expects the total number of bytes consumed and records the delta since
    // last invocation.
    void record_bytes_consumed(int64 total_bytes) {
      int64 delta =
          total_bytes - recorded_bytes_consumed_.exchange(total_bytes);
      bytes_consumed_counter_->IncrementBy(delta);
    }

    // Expects the total number of bytes produced and records the delta since
    // last invocation.
    void record_bytes_produced(int64 total_bytes) {
      int64 delta =
          total_bytes - recorded_bytes_produced_.exchange(total_bytes);
      bytes_produced_counter_->IncrementBy(delta);
    }

    // Expects the total number of elements produced and records the delta since
    // last invocation.
    void record_num_elements(int64 total_elements) {
      int64 delta =
          total_elements - recorded_num_elements_.exchange(total_elements);
      num_elements_counter_->IncrementBy(delta);
    }

    // Expects the total amount of time spent in computation by this node
    void record_computation_time(int64 total_computation_time) {
      int64 delta = total_computation_time -
          recorded_computation_time_.exchange(total_computation_time);
      computation_time_counter_->IncrementBy(delta);
    }

    // Default fields of the Metrics class
    monitoring::CounterCell* const bytes_consumed_counter_;
    monitoring::CounterCell* const bytes_produced_counter_;
    monitoring::CounterCell* const num_elements_counter_;
    std::atomic<int64> recorded_bytes_consumed_;
    std::atomic<int64> recorded_bytes_produced_;
    std::atomic<int64> recorded_num_elements_;

    // Custom added fields of the Metrics class
    monitoring::CounterCell* const computation_time_counter_;
    std::atomic<int64> recorded_computation_time_;
  };

 protected:
  // Returns the number of inputs.
  int64 num_inputs() const TF_SHARED_LOCKS_REQUIRED(mu_) {
    int64 num_inputs = 0;
    for (auto& input : inputs_) {
      // Inputs for which autotuning is disabled are excluded.
      if (input->autotune()) {
        ++num_inputs;
      }
    }
    return num_inputs;
  }

  // Creates a clone of this node.
  virtual std::shared_ptr<Node> Clone(std::shared_ptr<Node> output) const
      TF_SHARED_LOCKS_REQUIRED(mu_) = 0;

  // Returns the average size of an element buffered in this node.
  double AverageBufferedElementSize() const TF_SHARED_LOCKS_REQUIRED(mu_);

  // Returns the sum of per-element output time for the tunable inputs of this
  // node.
  double OutputTimeForInputs(const NodeValues& output_times) const
      TF_SHARED_LOCKS_REQUIRED(mu_);

  // Returns the sum of output time gradient w.r.t. input time for the tunable
  // inputs of this node.
  double OutputTimeGradientsForInputs(const NodeValues& output_time_gradients)
      const TF_SHARED_LOCKS_REQUIRED(mu_);

  // Computes the input time for this node and stores it in `input_times`.
  virtual void InputTimeLocked(NodeValues* input_times) const
      TF_SHARED_LOCKS_REQUIRED(mu_) = 0;

  // Computes the per-element output time for this node and stores it in
  // `output_times`. If `gradients` is not `nullptr`, computes the output time
  // gradient w.r.t. tunable parameters of the subtree rooted in this node and
  // stores it in `gradients`, also computes the output time gradient w.r.t.
  // input time and stores it in `output_time_gradients`.
  virtual void OutputTimeLocked(const NodeValues& input_times,
                                ParameterGradients* gradients,
                                NodeValues* output_times,
                                NodeValues* output_time_gradients) const
      TF_SHARED_LOCKS_REQUIRED(mu_) = 0;

  // Returns the sum of per-element processing time for the inputs of this node
  // by adding values for input nodes in `total_processing_times`. Processing
  // time for a given input is a weighted combination of a statistic based on
  // history of input processing time and the actual time. This is done to
  // improve accuracy of processing time estimation for newly created inputs.
  //
  // Uniform distribution of per-element processing times across different
  // inputs is assumed.
  double TotalProcessingTimeForInputs(const NodeValues& total_processing_times)
      TF_SHARED_LOCKS_REQUIRED(mu_);

  // Returns the per-element processing time spent in this node.
  double SelfProcessingTimeLocked() const TF_SHARED_LOCKS_REQUIRED(mu_);

  // Computes the per-element CPU time spent in the subtree rooted in this node
  // and stores it in `total_processing_times`. If `processing_times` is not
  // `nullptr`, collects the per-element CPU time spent in each node of the
  // subtree.
  virtual void TotalProcessingTimeLocked(NodeValues* processing_times,
                                         NodeValues* total_processing_times)
      TF_SHARED_LOCKS_REQUIRED(mu_) = 0;

  // Returns a vector of nodes of the subtree rooted in this node. The nodes are
  // either in breadth-first search or reverse breadth-first search order
  // depending on the `order` argument. The nodes are collected based on the
  // results of the `collect_node` predicate: if the predicate returns `false`
  // for a given node, then the subtree rooted in this node is excluded. The
  // root node itself is not collected.
  NodeVector CollectNodes(TraversalOrder order,
                          bool collect_node(const std::shared_ptr<Node>)) const
      TF_SHARED_LOCKS_REQUIRED(mu_);

  // Collects tunable parameters in the subtree rooted in this node assuming
  // mutex locked.
  ModelParameters CollectTunableParametersLocked() const
      TF_SHARED_LOCKS_REQUIRED(mu_);

  // Collect tunable parameters on the nodes which have recorded elements.
  void CollectTunableParametersHelper(ModelParameters* parameters) const
      TF_SHARED_LOCKS_REQUIRED(mu_);

  // Build up debug string for the node and store in the debug strings map.
  void DebugStringHelper(absl::flat_hash_map<string, string>* debug_strings)
      const TF_SHARED_LOCKS_REQUIRED(mu_);

  // Copy the node and add the (input, copy) pairs to the NodePairList.
  std::shared_ptr<Node> SnapshotHelper(std::shared_ptr<Node> cloned_output,
                                       NodePairList* node_pairs) const;

  // Compute total buffered bytes for the node and store in the total bytes map.
  void TotalBufferedBytesHelper(NodeValues* total_bytes) const
      TF_SHARED_LOCKS_REQUIRED(mu_);

  // Compute total maximum buffered bytes for the node and store in the total
  // bytes map.
  void TotalMaximumBufferedBytesHelper(NodeValues* total_bytes) const
      TF_SHARED_LOCKS_REQUIRED(mu_);

  // Compute and return the maximum buffered bytes on the node itself. By
  // default non-tunable nodes are assumed not to buffer any bytes, so the
  // tunable nodes as subclasses are expected to override this method to ensure
  // that the optimization algorithm respects the memory budget.
  virtual double MaximumBufferedBytes() const TF_SHARED_LOCKS_REQUIRED(mu_);

  // Restores node from the proto. Note that this is not done recursively, i.e.
  // input nodes are not restored.
  static Status FromProtoHelper(ModelProto::Node node_proto,
                                std::shared_ptr<Node> node);

  // Stores the time passed to the last call to `Node::record_start()` on the
  // current thread.
  //
  // NOTE: This thread-local variable is shared between all instances of `Node`
  // on which the same thread calls `record_start()` or `record_stop()`. It
  // relies on the invariant that at most one `Node` can be "active" on a
  // particular thread at any time. Therefore if `n->record_start()` is called
  // on thread `t`, then `n->record_stop()` must be called before another call
  // to `Node::record_start()` (for any node).
  static thread_local int64 work_start_;  // Will be initialized to zero.

  mutable mutex mu_;
  const int64 id_;
  const string name_;

  // Indicates whether the subtree rooted in this node should be included in
  // autotuning. In particular, if this is `false`, then the subtree is excluded
  // from computation of output time and processing time.
  std::atomic<bool> autotune_;
  std::atomic<int64> buffered_bytes_;
  std::atomic<int64> buffered_elements_;
  std::atomic<int64> bytes_consumed_;
  std::atomic<int64> bytes_produced_;
  std::atomic<int64> num_elements_;
  std::atomic<int64> processing_time_;
  std::atomic<bool> record_metrics_;
  Metrics metrics_;
  absl::flat_hash_map<string, std::shared_ptr<Parameter>> parameters_
      TF_GUARDED_BY(mu_);

  // Statistic of inputs processing time history.
  double input_processing_time_sum_ = 0.0L;
  int64 input_processing_time_count_ = 0;

  // Inputs of this node. These can represent an iterator created from the input
  // dataset but also other input iterators (e.g. created by the user-defined
  // functions of `flat_map` or `interleave`).
  std::list<std::shared_ptr<Node>> inputs_ TF_GUARDED_BY(mu_);

  // The reference to the output node is not owned so that deletion of a
  // node results in recursive deletion of the subtree rooted in the node.
  Node* const output_;

  public:

  // EASL - Used for statistics monitoring at the service workers.
  class MetricDump {
    const int64 bytes_consumed_;
    const int64 bytes_produced_;
    const int64 num_elements_;
    const int64 computation_time_;
    
    // Indicates time in node and in prefix rooted at node 
    double in_node_time_;
    double in_prefix_time_;
    std::string last_node_name_; 
    std::string last_tf_node_name_;
    
    public:
      explicit MetricDump(const Node::Metrics& metrics)
          : bytes_consumed_(metrics.recorded_bytes_consumed_), 
            bytes_produced_(metrics.recorded_bytes_produced_), 
            num_elements_(metrics.recorded_num_elements_), 
            computation_time_(metrics.recorded_computation_time_),
            in_node_time_(0.0),
            in_prefix_time_(0.0),
            last_node_name_(""),
            last_tf_node_name_("") {}

      const int64 bytes_consumed() const { return bytes_consumed_; }
      const int64 bytes_produced() const { return bytes_produced_; }
      const int64 num_elements() const { return num_elements_; }
      const int64 computation_time() const { return computation_time_; }

      // Methods for getting and setting the time metrics
      void set_in_node_time(double x) { in_node_time_ = x; }
      void set_in_prefix_time(double x) { in_prefix_time_ = x; }
      double in_node_time() { return in_node_time_; }
      double in_prefix_time() { return in_prefix_time_; }

      // Methods for getting and setting the last node name
      void set_last_node_name(std::string x) { last_node_name_ = x; }
      std::string last_node_name() { return last_node_name_; }
      void set_last_tf_node_name(std::string x) { last_tf_node_name_ = x; }
      std::string last_tf_node_name() { return last_tf_node_name_; }

      // Method which logs the metrics of this object
      void log_metrics() const {
        VLOG(1) << " > bytes_consumed = " << bytes_consumed_ << "\n"
                << " > bytes_produced = " << bytes_produced_ << "\n"
                << " > num_elements = " << num_elements_ << "\n"
                << " > computation_time = " << computation_time_ << "\n"
                << " > in_node_time = " << in_node_time_ << "\n"
                << " > in_prefix_time = " << in_prefix_time_ << "\n"
                << " > last_node_name = " << last_node_name_ << "\n"
                << " > last_tf_node_name = " << last_tf_node_name_;
      }
  };

  // This creates a DumpMetrics object, which represents a snapshot of the metrics
  Node::MetricDump SnapshotCurrentMetrics();
};

// InterleaveMany is used to model datasets whose inputs are used to create
// datasets whose elements are then interleaved.
std::shared_ptr<Node> MakeInterleaveManyNode(Node::Args args);

// AsyncInterleaveMany nodes are the asynchronous version of InterleaveMany
// nodes.
std::shared_ptr<Node> MakeAsyncInterleaveManyNode(
    Node::Args args, std::vector<std::shared_ptr<Parameter>> parameters);

// KnownMany nodes model datasets that synchronously consume known number of
// input element per output element.
std::shared_ptr<Node> MakeKnownRatioNode(Node::Args args, double ratio);

// AsyncKnownRatio nodes are the asynchronous version of KnownRate nodes.
std::shared_ptr<Node> MakeAsyncKnownRatioNode(
    Node::Args args, double ratio, double memory_ratio,
    std::vector<std::shared_ptr<Parameter>> parameters);

std::shared_ptr<Node> MakeAsyncKnownRatioNode(
    Node::Args args, double ratio,
    std::vector<std::shared_ptr<Parameter>> parameters);

// Source nodes represent data sources.
std::shared_ptr<Node> MakeSourceNode(Node::Args args);

// UnknownMany nodes represent datasets that synchronously consume an
// unknown number of input elements per output.
//
// Unlike KnownRatio nodes which expect the ratio between inputs and outputs is
// specified as a parameter, UnknownRatio estimates the ratio empirically.
std::shared_ptr<Node> MakeUnknownRatioNode(Node::Args args);

// Unknown nodes represent datasets for which we do not have a model. It acts
// as pass-through between inputs and output.
std::shared_ptr<Node> MakeUnknownNode(Node::Args args);

// Abstract representation of a TensorFlow input pipeline that can be used
// for collecting runtime information and optimizing performance. It collects
// runtime information about execution of the input pipeline that is used to
// create a performance model, which is in turn used to identify optimal values
// of tunable parameters.
//
// Developers of tf.data transformations are not expected to interact with this
// class directly. Boiler plate code for creating the abstract representation of
// the input pipeline and collecting runtime information has been added to the
// implementation of `DatasetBase` and `DatasetBaseIterator` respectively.
class Model {
 public:
  using OptimizationParams = ModelProto::OptimizationParams;
  using ModelParameters = Node::ModelParameters;
  using NodeValues = Node::NodeValues;
  using ParameterGradients = Node::ParameterGradients;
  using ModelMetrics =
    std::shared_ptr<absl::flat_hash_map<string, Node::MetricDump>>;

  // Represents minimum necessary information to recreate an optimization run.
  struct OptimizationSnapshot {
    // Output node of the model being optimized.
    std::shared_ptr<Node> output;
    OptimizationParams params;

    // Indicates whether this snapshot has been saved by `SaveLoop`.
    bool saved;
  };

  // Buffer of snapshots and its mutex.
  struct SnapshotBuffer {
    std::shared_ptr<std::deque<OptimizationSnapshot>> snapshots;
    std::shared_ptr<mutex> mu;
  };

  // Creates a new model.
  Model()
      : collect_resource_usage_(false),
        optimization_period_ms_(kOptimizationPeriodMinMs),
        snapshot_buffer_mu_(std::make_shared<mutex>()),
        snapshot_buffer_(std::make_shared<std::deque<OptimizationSnapshot>>()) {
    const char* save_dir = std::getenv("TF_DATA_AUTOTUNE_DEBUG_DIR");
    if (save_dir) {
      save_dir_ = string(save_dir);
    }
    {
      mutex_lock l(*publish_mu());
      (*snapshot_buffers())[this] = SnapshotBuffer{
          std::shared_ptr<std::deque<OptimizationSnapshot>>(snapshot_buffer_),
          std::shared_ptr<mutex>(snapshot_buffer_mu_)};
    }
  }

  ~Model() {
    if (!save_dir_.empty()) {
      save_thread_cancelled_ = true;
      save_cond_var_.notify_all();
    }
    {
      mutex_lock l(*publish_mu());
      (*snapshot_buffers()).erase(this);
    }
  }

  // Indicates whether to collect resource usage.
  bool collect_resource_usage() const { return collect_resource_usage_; }

  // Returns a pointer to the model's output node.
  const std::shared_ptr<Node> output() {
    mutex_lock l(mu_);
    return output_;
  }

  // Indicates whether publishing mode is enabled.
  static bool publish() {
    tf_shared_lock l(*publish_mu());
    return publish_;
  }

  // Adds a node with the given name and given parent.
  void AddNode(Node::Factory factory, const string& name,
               std::shared_ptr<Node> parent, std::shared_ptr<Node>* out_node)
      TF_LOCKS_EXCLUDED(mu_);

  // Flushes metrics record by the model.
  void FlushMetrics() TF_LOCKS_EXCLUDED(mu_);

  // Logs metrics record by the model.
  void PrintMetrics() TF_LOCKS_EXCLUDED(mu_);

  ModelMetrics CollectMetrics();

  // Uses the given algorithm and resource budgets to periodically perform the
  // autotuning optimization.
  //
  // To terminate the execution of the optimization loop, the caller needs to
  // invoke `cancellation_mgr->StartCancel()`.
  Status OptimizeLoop(AutotuneAlgorithm algorithm, int64 cpu_budget,
                      int64 ram_budget,
                      CancellationManager* cancellation_manager);

  // Uses the given algorithm and resource budgets to perform the autotuning
  // optimization.
  void Optimize(AutotuneAlgorithm algorithm, int64 cpu_budget, int64 ram_budget,
                double model_input_time,
                CancellationManager* cancellation_manager);

  // Collects the output time and if `gradients` is not `nullptr`, the output
  // time gradient w.r.t. tunable parameters of the subtree rooted in the given
  // node.
  double OutputTime(std::shared_ptr<Node> node, double model_input_time,
                    ParameterGradients* gradients);

  // Removes the given node.
  void RemoveNode(std::shared_ptr<Node> node) TF_LOCKS_EXCLUDED(mu_);

  // Produces a proto for this model.
  Status ToProto(ModelProto* model_proto);

  // Restores a model from the proto.
  static Status FromProto(ModelProto model_proto,
                          std::unique_ptr<Model>* model);

  // Saves this model with a given snapshot and its optimization parameters to a
  // file. Note that the file directory must already exist.
  Status Save(const string& fname, std::shared_ptr<Node> snapshot,
              const OptimizationParams& optimization_params);

  // Loads a model and its optimization parameters from a file with the given
  // name.
  static Status Load(const string& fname, std::unique_ptr<Model>* model,
                     OptimizationParams* optimization_params);

  // Enables publishing mode in which each existing model keeps a number of the
  // latest optimization snapshots in a buffer. The snapshots can be accessed
  // using `PublishLatest`.
  static void EnablePublishing() {
    mutex_lock l(*publish_mu());
    publish_ = true;
  }

  // If publishing is enabled, collects the latest optimization snapshot of each
  // existing model and appends its proto to the given string.
  static Status PublishLatest(absl::Cord* model);

 private:
  static constexpr int64 kOptimizationPeriodMinMs = 10;
  static constexpr int64 kOptimizationPeriodMaxMs =
      60 * EnvTime::kSecondsToMillis;

  // Maximum number of optimization snapshots kept in a buffer for saving or
  // publishing.
  static constexpr int64 kMaxNumBufferedSnapshots = 1;

  // Indicates whether publishing mode is enabled.
  static bool publish_ TF_GUARDED_BY(*publish_mu());

  // Used to coordinate (de)registering of optimization snapshot buffers for
  // publishing.
  static mutex* publish_mu() {
    static mutex lock(LINKER_INITIALIZED);
    return &lock;
  }

  // Mapping from all existing model pointers to their optimization snapshot
  // buffers and locks required to access the buffers.
  static absl::flat_hash_map<Model*, SnapshotBuffer>* snapshot_buffers()
      TF_SHARED_LOCKS_REQUIRED(*publish_mu()) {
    static absl::flat_hash_map<Model*, SnapshotBuffer>* const snapshot_buffers =
        new absl::flat_hash_map<Model*, SnapshotBuffer>();
    return snapshot_buffers;
  }

  // Collects tunable parameters in the tree rooted in the given node, returning
  // a vector which contains pairs of node names and tunable parameters.
  ModelParameters CollectTunableParameters(std::shared_ptr<Node> node);

  // Flushes metrics recorded by the model.
  // TODO (damien-aymon) should this method be public or private?
  //void FlushMetrics() TF_LOCKS_EXCLUDED(mu_);

  // This optimization algorithm starts by setting all tunable parallelism
  // parameters to the minimum value. It then repeatedly identifies the
  // parameter whose increase in parallelism decreases the output time the most.
  // This process is repeated until all parameters reach their maximum values or
  // the projected output time is less than or equal to the processing time
  // needed to produce an element divided by CPU budget.
  void OptimizeHillClimb(std::shared_ptr<Node> snapshot,
                         const OptimizationParams& optimization_params,
                         CancellationManager* cancellation_manager);

  // This optimization algorithm starts by setting all tunable parallelism
  // parameters to the minimum value. It then improves current parameters by
  // making a step in the direction opposite to the gradient of `OutputTime` and
  // projecting resulting values on the feasible intervals. Improvement step is
  // repeated until either the output time improvement is smaller than threshold
  // value or the output time is less than the processing time needed to produce
  // an element divided by CPU budget.
  void OptimizeGradientDescent(std::shared_ptr<Node> snapshot,
                               const OptimizationParams& optimization_params,
                               CancellationManager* cancellation_manager);

  // Determines if we should stop the gradient descent optimization iterations
  // based on number of increasable parameters, CPU budget, RAM budget and
  // current resource usage.
  bool ShouldStop(int64 cpu_budget, int64 ram_budget,
                  const ModelParameters& parameters,
                  const ModelParameters& parallelism_parameters,
                  const ModelParameters& buffer_size_parameters,
                  std::shared_ptr<Node> snapshot, bool* cpu_budget_reached);

  // Collects the processing time for the given node.
  double TotalProcessingTime(std::shared_ptr<Node> node);

  // Collects the total number of bytes buffered in all nodes in the subtree
  // rooted in the given node for which autotuning is enabled.
  double TotalBufferedBytes(std::shared_ptr<Node> node);

  // Collects the total buffer limit of all nodes in the subtree rooted in the
  // given node for which autotuning is enabled. This number represents the
  // amount of memory that would be used by the subtree nodes if all of their
  // buffers were full.
  double TotalMaximumBufferedBytes(std::shared_ptr<Node> node);

  // Starts a model saving thread if it hasn't started yet.
  Status EnsureSaveLoopThreadStarted();

  // Periodically saves the state of optimization that is kept in
  // `snapshot_buffer_`.
  //
  // The saving loop is terminated when the model is destroyed.
  Status SaveLoop();

  // Used for coordination between different input pipeline threads. Exclusive
  // access is required only when adding or removing nodes. Concurrent access to
  // existing nodes is protected by a node mutex.
  mutex mu_;
  // Used for coordinating the optimization loop and model modifications.
  condition_variable optimize_cond_var_;
  int64 id_counter_ TF_GUARDED_BY(mu_) = 1;
  std::shared_ptr<Node> output_ TF_GUARDED_BY(mu_);

  // Indicates whether the modeling framework should collect resource usage
  // (e.g. CPU, memory). The logic for collecting this information assumes that
  // the collection is not repeatedly disabled and enabled. As a consequence,
  // the implementation starts collecting resource usage when it encounters a
  // tunable parameter (because the information is used for tuning the value of
  // the parameter) and never stops.
  std::atomic<bool> collect_resource_usage_;

  // Determines the time the optimization loop should wait between
  // running optimizations.
  int64 optimization_period_ms_ TF_GUARDED_BY(mu_);

  // Thread that runs the model saving loop.
  std::unique_ptr<Thread> save_thread_ TF_GUARDED_BY(snapshot_buffer_mu_);

  // Used for coordinating the saving loop and model optimization.
  condition_variable save_cond_var_;

  // Indicates whether the save thread is cancelled.
  bool save_thread_cancelled_ = false;

  // Contains path to the model saving directory if saving is enabled, empty
  // otherwise.
  string save_dir_;

  // Used for coordination of the optimization snapshot buffer access and
  // updates.
  std::shared_ptr<mutex> snapshot_buffer_mu_;

  // Contains pairs of model snapshots and optimization parameters to be saved
  // or published if the corresponding mode is enabled, empty otherwise. Buffer
  // elements are pushed and popped by `OptimizeLoop`, and read by `SaveLoop`
  // and `PublishLatest`.
  std::shared_ptr<std::deque<OptimizationSnapshot>> snapshot_buffer_
      TF_GUARDED_BY(snapshot_buffer_mu_);
};

}  // namespace model
}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_FRAMEWORK_MODEL_H_
