// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

#include <cuda_runtime_api.h>

#include <transformer_engine/normalization.h>
#include <transformer_engine/transformer_engine.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr float kEpsilon = 1.0e-5F;
constexpr double kRelativeScaleFloor = 0.0078125;
constexpr int kWarmupInvocations = 10;
constexpr int kMeasuredInvocations = 50;

void check_cuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
  }
}

enum class Boundary { kForward, kBackwardRecompute, kComposed };
enum class Backend { kTransformerEngine, kCudnn };

struct Config {
  Boundary boundary;
  Backend forward_backend;
  Backend backward_backend;
  std::size_t rows;
  std::size_t features;
  std::filesystem::path case_directory;
  std::filesystem::path output;
  std::string counterbalance_id;
  std::size_t counterbalance_position;
};

std::string boundary_name(Boundary boundary) {
  switch (boundary) {
    case Boundary::kForward:
      return "forward";
    case Boundary::kBackwardRecompute:
      return "backward_recompute";
    case Boundary::kComposed:
      return "composed";
  }
  throw std::logic_error("unknown boundary");
}

std::string backend_name(Backend backend) {
  return backend == Backend::kCudnn ? "cudnn" : "transformer_engine";
}

Boundary parse_boundary(const std::string& value) {
  if (value == "forward") {
    return Boundary::kForward;
  }
  if (value == "backward_recompute") {
    return Boundary::kBackwardRecompute;
  }
  if (value == "composed") {
    return Boundary::kComposed;
  }
  throw std::invalid_argument("boundary must be forward, backward_recompute, or composed");
}

Backend parse_backend(const std::string& value) {
  if (value == "transformer_engine") {
    return Backend::kTransformerEngine;
  }
  if (value == "cudnn") {
    return Backend::kCudnn;
  }
  throw std::invalid_argument("backend must be transformer_engine or cudnn");
}

std::map<std::string, std::string> parse_flags(int argc, char** argv) {
  std::map<std::string, std::string> flags;
  for (int index = 1; index < argc; index += 2) {
    if (index + 1 >= argc || std::string(argv[index]).rfind("--", 0) != 0) {
      throw std::invalid_argument("arguments must be --name value pairs");
    }
    const std::string name = std::string(argv[index]).substr(2);
    if (!flags.emplace(name, argv[index + 1]).second) {
      throw std::invalid_argument("duplicate argument: --" + name);
    }
  }
  return flags;
}

std::string take_flag(std::map<std::string, std::string>* flags, const std::string& name) {
  const auto found = flags->find(name);
  if (found == flags->end()) {
    throw std::invalid_argument("missing argument: --" + name);
  }
  std::string value = found->second;
  flags->erase(found);
  return value;
}

std::size_t parse_dimension(const std::string& value, const char* name) {
  std::size_t consumed = 0;
  const unsigned long long parsed = std::stoull(value, &consumed);
  if (consumed != value.size() || parsed > std::numeric_limits<std::size_t>::max()) {
    throw std::invalid_argument(std::string(name) + " is not a size_t");
  }
  return static_cast<std::size_t>(parsed);
}

Config parse_config(int argc, char** argv) {
  auto flags = parse_flags(argc, argv);
  Config config{
      .boundary = parse_boundary(take_flag(&flags, "boundary")),
      .forward_backend = parse_backend(take_flag(&flags, "forward-backend")),
      .backward_backend = parse_backend(take_flag(&flags, "backward-backend")),
      .rows = parse_dimension(take_flag(&flags, "rows"), "rows"),
      .features = parse_dimension(take_flag(&flags, "features"), "features"),
      .case_directory = take_flag(&flags, "case-directory"),
      .output = take_flag(&flags, "output"),
      .counterbalance_id = take_flag(&flags, "counterbalance-id"),
      .counterbalance_position =
          parse_dimension(take_flag(&flags, "counterbalance-position"),
                          "counterbalance-position"),
  };
  if (!flags.empty()) {
    throw std::invalid_argument("unknown argument: --" + flags.begin()->first);
  }
  if (!((config.rows == 2048 && config.features == 4096) ||
        (config.rows == 7 && config.features == 13))) {
    throw std::invalid_argument("shape must be 2048x4096 or 7x13");
  }
  if (config.counterbalance_id != "shape_boundary_backend_alternating_v1" ||
      config.counterbalance_position >= 24) {
    throw std::invalid_argument("counterbalance metadata is outside the closed 24-run plan");
  }
  return config;
}

std::size_t checked_product(std::size_t left, std::size_t right) {
  if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left) {
    throw std::overflow_error("tensor element count overflow");
  }
  return left * right;
}

std::vector<std::uint16_t> read_bfloat16(const std::filesystem::path& path,
                                         std::size_t elements) {
  const std::size_t bytes = checked_product(elements, sizeof(std::uint16_t));
  if (std::filesystem::file_size(path) != bytes) {
    throw std::runtime_error(path.string() + " has the wrong byte size");
  }
  std::vector<std::uint16_t> result(elements);
  std::ifstream stream(path, std::ios::binary);
  stream.read(reinterpret_cast<char*>(result.data()), static_cast<std::streamsize>(bytes));
  if (!stream) {
    throw std::runtime_error("failed to read " + path.string());
  }
  return result;
}

class DeviceAllocation {
 public:
  explicit DeviceAllocation(std::size_t bytes) : bytes_(bytes) {
    if (bytes != 0) {
      check_cuda(cudaMalloc(&pointer_, bytes), "cudaMalloc");
    }
  }

  ~DeviceAllocation() {
    if (pointer_ != nullptr) {
      cudaFree(pointer_);
    }
  }

  DeviceAllocation(const DeviceAllocation&) = delete;
  DeviceAllocation& operator=(const DeviceAllocation&) = delete;

  void* data() const { return pointer_; }
  std::size_t bytes() const { return bytes_; }

 private:
  void* pointer_ = nullptr;
  std::size_t bytes_ = 0;
};

class Tensor {
 public:
  Tensor() : tensor_(nvte_create_tensor(NVTE_DELAYED_TENSOR_SCALING)) {}

  Tensor(void* data, NVTEDType dtype, const std::vector<std::size_t>& shape) : Tensor() {
    NVTEBasicTensor basic{
        .data_ptr = data,
        .dtype = dtype,
        .shape = nvte_make_shape(shape.data(), shape.size()),
    };
    nvte_set_tensor_param_v2(tensor_, kNVTERowwiseData, &basic, sizeof(basic));
  }

  ~Tensor() { nvte_destroy_tensor(tensor_); }

  Tensor(const Tensor&) = delete;
  Tensor& operator=(const Tensor&) = delete;

  NVTETensor get() const { return tensor_; }

 private:
  NVTETensor tensor_;
};

class Stream {
 public:
  Stream() { check_cuda(cudaStreamCreate(&stream_), "cudaStreamCreate"); }
  ~Stream() { cudaStreamDestroy(stream_); }
  Stream(const Stream&) = delete;
  Stream& operator=(const Stream&) = delete;
  cudaStream_t get() const { return stream_; }

 private:
  cudaStream_t stream_ = nullptr;
};

class Event {
 public:
  Event() { check_cuda(cudaEventCreate(&event_), "cudaEventCreate"); }
  ~Event() { cudaEventDestroy(event_); }
  Event(const Event&) = delete;
  Event& operator=(const Event&) = delete;
  cudaEvent_t get() const { return event_; }

 private:
  cudaEvent_t event_ = nullptr;
};

struct WorkspaceMetadata {
  std::vector<std::size_t> shape;
  NVTEDType dtype;
  std::size_t byte_count;
};

struct Workspace {
  WorkspaceMetadata metadata;
  std::unique_ptr<DeviceAllocation> allocation;
  std::unique_ptr<Tensor> tensor;
};

Workspace materialize_workspace(const Tensor& query_tensor) {
  const NVTEShape queried_shape = nvte_tensor_shape(query_tensor.get());
  std::vector<std::size_t> shape(queried_shape.data, queried_shape.data + queried_shape.ndim);
  const NVTEDType dtype = nvte_tensor_type(query_tensor.get());
  if (dtype != kNVTEByte) {
    throw std::runtime_error("workspace query returned a non-byte dtype");
  }
  const std::size_t bytes = nvte_tensor_size_bytes(query_tensor.get());
  auto allocation = std::make_unique<DeviceAllocation>(bytes);
  auto tensor = std::make_unique<Tensor>(allocation->data(), dtype, shape);
  return Workspace{
      .metadata = WorkspaceMetadata{.shape = std::move(shape), .dtype = dtype, .byte_count = bytes},
      .allocation = std::move(allocation),
      .tensor = std::move(tensor),
  };
}

void copy_to_device(const std::vector<std::uint16_t>& source, DeviceAllocation* destination,
                    cudaStream_t stream) {
  check_cuda(cudaMemcpyAsync(destination->data(), source.data(), destination->bytes(),
                             cudaMemcpyHostToDevice, stream),
             "cudaMemcpyAsync host to device");
}

std::vector<std::uint16_t> copy_from_device(const DeviceAllocation& source,
                                            std::size_t elements, cudaStream_t stream) {
  std::vector<std::uint16_t> result(elements);
  check_cuda(cudaMemcpyAsync(result.data(), source.data(), source.bytes(), cudaMemcpyDeviceToHost,
                             stream),
             "cudaMemcpyAsync device to host");
  check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize after output copy");
  return result;
}

float bfloat16_to_float(std::uint16_t value) {
  const std::uint32_t bits = static_cast<std::uint32_t>(value) << 16;
  float result = 0;
  static_assert(sizeof(result) == sizeof(bits));
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

std::int32_t ordered_bfloat16(std::uint16_t value) {
  if ((value & 0x8000U) != 0) {
    return 0x7FFF - static_cast<std::int32_t>(value & 0x7FFFU);
  }
  return 0x8000 + static_cast<std::int32_t>(value);
}

struct ErrorMetrics {
  double max_absolute_error = 0;
  double mean_absolute_error = 0;
  double relative_linf_error = 0;
  std::int32_t max_bfloat16_ulp_error = 0;
};

ErrorMetrics compare(const std::vector<std::uint16_t>& actual,
                     const std::vector<std::uint16_t>& reference) {
  if (actual.size() != reference.size() || actual.empty()) {
    throw std::invalid_argument("comparison arrays must have equal nonzero sizes");
  }
  ErrorMetrics metrics;
  double absolute_sum = 0;
  double reference_linf = 0;
  for (std::size_t index = 0; index < actual.size(); ++index) {
    const double actual_value = bfloat16_to_float(actual[index]);
    const double reference_value = bfloat16_to_float(reference[index]);
    if (!std::isfinite(actual_value) || !std::isfinite(reference_value)) {
      throw std::runtime_error("comparison inputs must be finite bfloat16 values");
    }
    const double absolute = std::abs(actual_value - reference_value);
    metrics.max_absolute_error = std::max(metrics.max_absolute_error, absolute);
    absolute_sum += absolute;
    reference_linf = std::max(reference_linf, std::abs(reference_value));
    metrics.max_bfloat16_ulp_error =
        std::max(metrics.max_bfloat16_ulp_error,
                 std::abs(ordered_bfloat16(actual[index]) - ordered_bfloat16(reference[index])));
  }
  metrics.mean_absolute_error = absolute_sum / static_cast<double>(actual.size());
  metrics.relative_linf_error =
      metrics.max_absolute_error / std::max(reference_linf, kRelativeScaleFloor);
  return metrics;
}

struct OutputMetric {
  std::string role;
  ErrorMetrics metrics;
};

struct Tensors {
  DeviceAllocation x_data;
  DeviceAllocation gamma_data;
  DeviceAllocation dy_data;
  DeviceAllocation y_data;
  DeviceAllocation dx_data;
  DeviceAllocation dgamma_data;
  DeviceAllocation rsigma_data;
  DeviceAllocation throwaway_data;
  Tensor x;
  Tensor gamma;
  Tensor dy;
  Tensor y;
  Tensor dx;
  Tensor dgamma;
  Tensor rsigma;
  Tensor throwaway;

  Tensors(std::size_t rows, std::size_t features)
      : x_data(checked_product(checked_product(rows, features), 2)),
        gamma_data(checked_product(features, 2)),
        dy_data(checked_product(checked_product(rows, features), 2)),
        y_data(checked_product(checked_product(rows, features), 2)),
        dx_data(checked_product(checked_product(rows, features), 2)),
        dgamma_data(checked_product(features, 2)),
        rsigma_data(checked_product(rows, 4)),
        throwaway_data(checked_product(checked_product(rows, features), 2)),
        x(x_data.data(), kNVTEBFloat16, {rows, features}),
        gamma(gamma_data.data(), kNVTEBFloat16, {features}),
        dy(dy_data.data(), kNVTEBFloat16, {rows, features}),
        y(y_data.data(), kNVTEBFloat16, {rows, features}),
        dx(dx_data.data(), kNVTEBFloat16, {rows, features}),
        dgamma(dgamma_data.data(), kNVTEBFloat16, {features}),
        rsigma(rsigma_data.data(), kNVTEFloat32, {rows}),
        throwaway(throwaway_data.data(), kNVTEBFloat16, {rows, features}) {}
};

Workspace query_forward_workspace(Tensors* tensors, int multiprocessor_count,
                                  cudaStream_t stream) {
  Tensor empty_workspace;
  nvte_rmsnorm_fwd(tensors->x.get(), tensors->gamma.get(), kEpsilon, tensors->throwaway.get(),
                   tensors->rsigma.get(), empty_workspace.get(), multiprocessor_count, false,
                   stream);
  return materialize_workspace(empty_workspace);
}

Workspace query_backward_workspace(Tensors* tensors, int multiprocessor_count,
                                   cudaStream_t stream) {
  Tensor empty_workspace;
  nvte_rmsnorm_bwd(tensors->dy.get(), tensors->x.get(), tensors->rsigma.get(),
                   tensors->gamma.get(), tensors->dx.get(), tensors->dgamma.get(),
                   empty_workspace.get(), multiprocessor_count, false, stream);
  return materialize_workspace(empty_workspace);
}

void invoke_boundary(Boundary boundary, Tensors* tensors, Workspace* forward_workspace,
                     Workspace* backward_workspace, int multiprocessor_count,
                     cudaStream_t stream) {
  Tensor* forward_output =
      boundary == Boundary::kBackwardRecompute ? &tensors->throwaway : &tensors->y;
  nvte_rmsnorm_fwd(tensors->x.get(), tensors->gamma.get(), kEpsilon, forward_output->get(),
                   tensors->rsigma.get(), forward_workspace->tensor->get(),
                   multiprocessor_count, false, stream);
  if (boundary != Boundary::kForward) {
    nvte_rmsnorm_bwd(tensors->dy.get(), tensors->x.get(), tensors->rsigma.get(),
                     tensors->gamma.get(), tensors->dx.get(), tensors->dgamma.get(),
                     backward_workspace->tensor->get(), multiprocessor_count, false, stream);
  }
}

std::vector<double> measure(Boundary boundary, Tensors* tensors, Workspace* forward_workspace,
                            Workspace* backward_workspace, int multiprocessor_count,
                            cudaStream_t stream) {
  for (int iteration = 0; iteration < kWarmupInvocations; ++iteration) {
    invoke_boundary(boundary, tensors, forward_workspace, backward_workspace,
                    multiprocessor_count, stream);
  }
  check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize after warmup");

  Event start;
  Event stop;
  std::vector<double> samples;
  samples.reserve(kMeasuredInvocations);
  for (int iteration = 0; iteration < kMeasuredInvocations; ++iteration) {
    check_cuda(cudaEventRecord(start.get(), stream), "cudaEventRecord start");
    invoke_boundary(boundary, tensors, forward_workspace, backward_workspace,
                    multiprocessor_count, stream);
    check_cuda(cudaEventRecord(stop.get(), stream), "cudaEventRecord stop");
    check_cuda(cudaEventSynchronize(stop.get()), "cudaEventSynchronize stop");
    float milliseconds = 0;
    check_cuda(cudaEventElapsedTime(&milliseconds, start.get(), stop.get()),
               "cudaEventElapsedTime");
    samples.push_back(milliseconds);
  }
  return samples;
}

double median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  const std::size_t middle = values.size() / 2;
  return (values[middle - 1] + values[middle]) / 2;
}

void write_shape(std::ostream& stream, const std::vector<std::size_t>& shape) {
  stream << '[';
  for (std::size_t index = 0; index < shape.size(); ++index) {
    if (index != 0) {
      stream << ',';
    }
    stream << shape[index];
  }
  stream << ']';
}

void write_workspace(std::ostream& stream, const WorkspaceMetadata* metadata) {
  if (metadata == nullptr) {
    stream << "null";
    return;
  }
  stream << "{\"shape\":";
  write_shape(stream, metadata->shape);
  stream << ",\"dtype\":\"byte\",\"byte_count\":" << metadata->byte_count << '}';
}

void write_metrics(std::ostream& stream, const std::vector<OutputMetric>& outputs) {
  stream << '[';
  for (std::size_t index = 0; index < outputs.size(); ++index) {
    if (index != 0) {
      stream << ',';
    }
    const auto& output = outputs[index];
    stream << "{\"role\":\"" << output.role << "\",\"metrics\":{"
           << "\"max_absolute_error\":" << output.metrics.max_absolute_error << ','
           << "\"mean_absolute_error\":" << output.metrics.mean_absolute_error << ','
           << "\"relative_linf_error\":" << output.metrics.relative_linf_error << ','
           << "\"max_bfloat16_ulp_error\":" << output.metrics.max_bfloat16_ulp_error
           << "}}";
  }
  stream << ']';
}

void write_result(const Config& config, int device, int multiprocessor_count,
                  const WorkspaceMetadata& forward_workspace,
                  const WorkspaceMetadata* backward_workspace,
                  const std::vector<double>& samples, const std::vector<OutputMetric>& outputs,
                  int driver_version, int runtime_version) {
  std::ofstream stream(config.output);
  if (!stream) {
    throw std::runtime_error("failed to open output file");
  }
  stream << std::setprecision(17);
  stream << "{\n  \"schema_version\": 1,\n"
         << "  \"status\": \"unsealed_hardware_observation\",\n"
         << "  \"boundary\": \"" << boundary_name(config.boundary) << "\",\n"
         << "  \"shape\": [" << config.rows << ',' << config.features << "],\n"
         << "  \"tensor_contract\": {\"dtype\":\"bfloat16\","
            "\"layout\":\"row_major_contiguous\",\"matrix_strides_elements\":["
         << config.features << ",1],\"vector_strides_elements\":[1],"
            "\"rsigma_dtype\":\"float32\"},\n"
         << "  \"backends\": {\"forward\":\"" << backend_name(config.forward_backend)
         << "\",\"backward\":\"" << backend_name(config.backward_backend) << "\"},\n"
         << "  \"counterbalance\": {\"plan_id\":\"" << config.counterbalance_id
         << "\",\"position\":" << config.counterbalance_position
         << ",\"execution_unit\":\"single_backend_pair\"},\n"
         << "  \"workspace_queries\": {\"forward\":";
  write_workspace(stream, &forward_workspace);
  stream << ",\"backward\":";
  write_workspace(stream, backward_workspace);
  stream << "},\n  \"timing\": {\"warmup_invocations\":" << kWarmupInvocations
         << ",\"measured_invocations\":" << kMeasuredInvocations
         << ",\"synchronization\":\"cudaEventSynchronize(stop) per sample\","
            "\"raw_cuda_event_milliseconds\":[";
  for (std::size_t index = 0; index < samples.size(); ++index) {
    if (index != 0) {
      stream << ',';
    }
    stream << samples[index];
  }
  stream << "],\"median_cuda_event_milliseconds\":" << median(samples) << "},\n"
         << "  \"comparison\": {\"reference\":"
            "\"independent_numpy_binary64_closed_form_then_bfloat16_outputs\","
            "\"relative_scale_floor\":0.0078125,\"outputs\":";
  write_metrics(stream, outputs);
  stream << ",\"oracle_relative_thresholds\":null,"
            "\"acceptance_status\":\"blocked_until_reviewed_hardware_artifact\"},\n"
         << "  \"provenance\": {"
            "\"marin_revision\":null,\"adapter_sha256\":null,"
            "\"transformer_engine\":{\"version\":\"2.17.0\","
            "\"source_tag\":\"v2.17\","
            "\"source_commit\":\"2e559f062497bef768dfbe9d7e45548fadeca80a\","
            "\"resolved_library_path\":null,\"library_sha256\":null,"
            "\"elf_build_id\":null,\"resolved_shared_library_dependencies\":null},"
            "\"toolchain\":{\"compiler\":null,\"build_flags\":null,"
            "\"target_architectures\":null},\"cuda\":{\"toolkit_version\":null,"
            "\"nvcc_version\":null,\"driver_version\":"
         << driver_version << ",\"runtime_version\":" << runtime_version
         << "},\"device\":{\"ordinal\":" << device
         << ",\"model\":null,\"uuid\":null,\"compute_capability\":null,"
            "\"physical_sm_count\":"
         << multiprocessor_count << "}}\n}\n";
}

std::vector<OutputMetric> collect_comparisons(const Config& config, Tensors* tensors,
                                              cudaStream_t stream) {
  const std::size_t matrix_elements = checked_product(config.rows, config.features);
  std::vector<OutputMetric> outputs;
  if (config.boundary != Boundary::kBackwardRecompute) {
    outputs.push_back(OutputMetric{
        .role = "y",
        .metrics = compare(copy_from_device(tensors->y_data, matrix_elements, stream),
                           read_bfloat16(config.case_directory / "reference_y.bf16",
                                         matrix_elements)),
    });
  }
  if (config.boundary != Boundary::kForward) {
    outputs.push_back(OutputMetric{
        .role = "dx",
        .metrics = compare(copy_from_device(tensors->dx_data, matrix_elements, stream),
                           read_bfloat16(config.case_directory / "reference_dx.bf16",
                                         matrix_elements)),
    });
    outputs.push_back(OutputMetric{
        .role = "dgamma",
        .metrics = compare(copy_from_device(tensors->dgamma_data, config.features, stream),
                           read_bfloat16(config.case_directory / "reference_dgamma.bf16",
                                         config.features)),
    });
  }
  return outputs;
}

int run(const Config& config) {
  const std::size_t matrix_elements = checked_product(config.rows, config.features);
  const auto host_x = read_bfloat16(config.case_directory / "x.bf16", matrix_elements);
  const auto host_gamma = read_bfloat16(config.case_directory / "gamma.bf16", config.features);
  std::vector<std::uint16_t> host_dy;
  if (config.boundary != Boundary::kForward) {
    host_dy = read_bfloat16(config.case_directory / "dy.bf16", matrix_elements);
  }

  Stream stream;
  Tensors tensors(config.rows, config.features);
  copy_to_device(host_x, &tensors.x_data, stream.get());
  copy_to_device(host_gamma, &tensors.gamma_data, stream.get());
  if (config.boundary != Boundary::kForward) {
    copy_to_device(host_dy, &tensors.dy_data, stream.get());
  }
  check_cuda(cudaStreamSynchronize(stream.get()), "cudaStreamSynchronize after input copies");

  nvte_enable_cudnn_norm_fwd(config.forward_backend == Backend::kCudnn);
  nvte_enable_cudnn_norm_bwd(config.backward_backend == Backend::kCudnn);

  int device = 0;
  check_cuda(cudaGetDevice(&device), "cudaGetDevice");
  int multiprocessor_count = 0;
  check_cuda(cudaDeviceGetAttribute(&multiprocessor_count, cudaDevAttrMultiProcessorCount, device),
             "cudaDeviceGetAttribute multiprocessor count");
  if (multiprocessor_count <= 0) {
    throw std::runtime_error("physical device multiprocessor count must be positive");
  }

  Workspace forward_workspace =
      query_forward_workspace(&tensors, multiprocessor_count, stream.get());
  std::unique_ptr<Workspace> backward_workspace;
  if (config.boundary != Boundary::kForward) {
    backward_workspace = std::make_unique<Workspace>(
        query_backward_workspace(&tensors, multiprocessor_count, stream.get()));
  }

  const auto samples = measure(config.boundary, &tensors, &forward_workspace,
                               backward_workspace.get(), multiprocessor_count, stream.get());
  const auto outputs = collect_comparisons(config, &tensors, stream.get());
  int driver_version = 0;
  int runtime_version = 0;
  check_cuda(cudaDriverGetVersion(&driver_version), "cudaDriverGetVersion");
  check_cuda(cudaRuntimeGetVersion(&runtime_version), "cudaRuntimeGetVersion");
  write_result(config, device, multiprocessor_count, forward_workspace.metadata,
               backward_workspace == nullptr ? nullptr : &backward_workspace->metadata, samples,
               outputs, driver_version, runtime_version);
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    return run(parse_config(argc, argv));
  } catch (const std::exception& error) {
    std::cerr << "target1_te_oracle_runner: " << error.what() << '\n';
    return 2;
  }
}
