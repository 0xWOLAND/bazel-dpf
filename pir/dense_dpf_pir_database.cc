// Copyright 2023 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "pir/dense_dpf_pir_database.h"

#include <stddef.h>

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/numeric/int128.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "dpf/status_macros.h"
#include "pir/internal/inner_product_hwy.h"

namespace distributed_point_functions {

namespace {

// Returns the number of bytes occupied by a value of `value_size_in_bytes`
// when aligned according to the block type.
size_t AlignBytes(size_t value_size_in_bytes) {
  constexpr size_t kAlignmentSize =
      sizeof(typename DenseDpfPirDatabase::BlockType);
  constexpr size_t kAlignmentMask = ~(kAlignmentSize - 1);
  // The number of aligned bytes is the least multiple of kAlignmentSize larger
  // than `value_size_in_bytes`, i.e. it is
  //   ceil(value_size_in_bytes / kAlignmentSize) * kAlignmentSize.
  // The division and subsequent multiplication is saved by simply masking off
  // the lowest `kAlignmentSize` bits.
  size_t round_up_value_size_in_bytes =
      (value_size_in_bytes + kAlignmentSize - 1);
  return round_up_value_size_in_bytes & kAlignmentMask;
}


inline constexpr int64_t NumBytesToNumBlocks(int64_t num_bytes) {
  return (num_bytes + (sizeof(DenseDpfPirDatabase::BlockType) - 1)) /
         sizeof(DenseDpfPirDatabase::BlockType);
}

}  // namespace

DenseDpfPirDatabase::Builder::Builder()
    : total_database_bytes_(0) {}

std::unique_ptr<DenseDpfPirDatabase::Interface::Builder>
DenseDpfPirDatabase::Builder::Clone() const {
  auto result = std::make_unique<Builder>();
  result->total_database_bytes_ = total_database_bytes_;
  result->values_ = values_;
  return result;
}

DenseDpfPirDatabase::Builder& DenseDpfPirDatabase::Builder::Insert(
    std::string value) {
  total_database_bytes_ += AlignBytes(value.size());
  values_.push_back(std::move(value));
  return *this;
}

DenseDpfPirDatabase::Builder& DenseDpfPirDatabase::Builder::Clear() {
  values_.clear();
  total_database_bytes_ = 0;
  return *this;
}

absl::StatusOr<std::unique_ptr<DenseDpfPirDatabase::Interface>>
DenseDpfPirDatabase::Builder::Build() {
  auto database = absl::WrapUnique(
      new DenseDpfPirDatabase(values_.size(), total_database_bytes_));
  std::vector<std::string> values =
      std::move(values_);  // Ensures values are freed after returning.
  for (std::string& value : values) {
    DPF_RETURN_IF_ERROR(database->Append(std::move(value)));
  }
  return database;
}

DenseDpfPirDatabase::DenseDpfPirDatabase(int64_t num_values,
                                         int64_t total_database_bytes)
    : max_value_size_(0) {
  // Reserve space for storing the desired number of bytes
  buffer_.reserve(NumBytesToNumBlocks(total_database_bytes));
  // Reserve space for storing the database values
  value_offsets_.reserve(num_values);
  content_views_.reserve(num_values);
}

// Appends a record `value` at the current end of the database.
absl::Status DenseDpfPirDatabase::Append(std::string value) {
  // The new value will be stored at the end of the current buffer space.
  const size_t offset = buffer_.size();
  const size_t value_size = value.size();
  if (value_size == 0) {
    // We have an empty value, so we store its offset and return.
    value_offsets_.push_back({offset, 0});
    content_views_.push_back(absl::string_view());
    return absl::OkStatus();
  }

  // Number of buffer elements needed to store the aligned value
  const size_t value_size_aligned = AlignBytes(value_size);
  const size_t num_additional_blocks = value_size_aligned / sizeof(BlockType);
  const size_t num_existing_blocks = buffer_.capacity();
  // Save the old buffer head pointer to ensure it is not reallocated, which
  // would invalidate existing content_views.
  const BlockType* const buffer_head_old = buffer_.data();
  if (offset + num_additional_blocks > num_existing_blocks) {
    // We don't have enough space in the buffer for this element. This signals
    // an implementation error in Buider::Build().
    return absl::InternalError(
        "Not enough buffer space available. This should not happen.");
  }
  buffer_.resize(buffer_.size() + num_additional_blocks);
  if (buffer_head_old != &buffer_.at(0)) {
    return absl::InternalError(
        "Buffer was reallocated unexpectedly. This should not happen.");
  }

  // Append the value to the buffer
  char* const buffer_at_offset = reinterpret_cast<char*>(&buffer_.at(offset));
  value.copy(buffer_at_offset, value_size);
  if (value_size > max_value_size_) {
    max_value_size_ = value_size;
  }

  // Store the position and the view of the value in `buffer_`.
  value_offsets_.push_back({offset, value_size});
  content_views_.push_back(absl::string_view(buffer_at_offset, value_size));
  return absl::OkStatus();
}

absl::Status DenseDpfPirDatabase::UpdateEntry(size_t index, std::string new_value) {
  return BatchUpdateEntry({index}, {new_value});
}

absl::Status DenseDpfPirDatabase::BatchUpdateEntry(
    const std::vector<size_t>& indices,
    const std::vector<std::string>& new_values) {
  if (indices.size() != new_values.size())
    return absl::InvalidArgumentError("Indices and values size mismatch");

  // Validate indices and calculate space requirements
  struct Update { size_t idx, new_sz, cur_off, cur_sz, cur_align, new_align; };
  std::vector<Update> updates(indices.size());
  for (size_t i = 0; i < indices.size(); ++i) {
    const size_t idx = indices[i];
    if (idx >= size()) return absl::OutOfRangeError("Index out of bounds");
    const auto& [off, sz] = value_offsets_[idx];
    updates[i] = {idx, new_values[i].size(), off, sz, AlignBytes(sz), AlignBytes(new_values[i].size())};
  }

  // Calculate net space needs
  size_t add_bytes = 0, reclaim_bytes = 0;
  for (const auto& u : updates) {
    if (u.new_align > u.cur_align) add_bytes += u.new_align - u.cur_align;
    else if (u.new_align < u.cur_align) reclaim_bytes += u.cur_align - u.new_align;
  }
  size_t net_bytes = add_bytes > reclaim_bytes ? add_bytes - reclaim_bytes : 0;

  // Grow buffer if necessary
  size_t cur_off = buffer_.size();
  if (net_bytes > 0) {
    const size_t blocks = (net_bytes + sizeof(BlockType) - 1) / sizeof(BlockType);
    buffer_.resize(buffer_.size() + blocks);
  }

  // Track free space for reuse
  struct Free { size_t off, sz; };
  std::vector<Free> free_spaces;
  for (size_t i = 0; i < updates.size(); ++i) {
    const auto& u = updates[i];
    const std::string& val = new_values[i];

    // Find target offset
    size_t tgt_off;
    if (u.new_align <= u.cur_align) {
      tgt_off = u.cur_off;
    } else {
      auto it = std::find_if(free_spaces.begin(), free_spaces.end(),
                             [u](const Free& f) { return f.sz >= u.new_align; });
      if (it != free_spaces.end()) {
        tgt_off = it->off;
        it->sz -= u.new_align;
        if (it->sz == 0) free_spaces.erase(it);
      } else {
        tgt_off = cur_off;
        cur_off += u.new_align / sizeof(BlockType);
      }
    }

    // Mark old space as free if abandoned
    if (tgt_off != u.cur_off && u.cur_align > 0)
      free_spaces.push_back({u.cur_off, u.cur_align});

    // Update buffer and metadata
    char* buf = reinterpret_cast<char*>(&buffer_[tgt_off]);
    if (u.new_sz < u.cur_sz && tgt_off == u.cur_off)
      std::memset(buf + u.new_sz, 0, u.cur_sz - u.new_sz);
    if (u.new_sz > 0) std::memcpy(buf, val.data(), u.new_sz);
    value_offsets_[u.idx] = {tgt_off, u.new_sz};
    content_views_[u.idx] = absl::string_view(buf, u.new_sz);
    max_value_size_ = std::max(max_value_size_, u.new_sz);
  }

  return absl::OkStatus();
}

// Returns the inner product between the database values and a bit vector
// (packed in blocks).
absl::StatusOr<std::vector<std::string>> DenseDpfPirDatabase::InnerProductWith(
    absl::Span<const std::vector<BlockType>> selections) const {
  return pir_internal::InnerProduct(content_views_, selections,
                                    max_value_size_);
}

}  // namespace distributed_point_functions
