/*
 * Copyright 2023 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef DISTRIBUTED_POINT_FUNCTIONS_PIR_TESTING_FAKE_HYBRID_DECRYPT_H_
#define DISTRIBUTED_POINT_FUNCTIONS_PIR_TESTING_FAKE_HYBRID_DECRYPT_H_

#include <memory>

#include "absl/status/statusor.h"
#include "tink/hybrid_decrypt.h"
#include "tink/hybrid_encrypt.h"

namespace distributed_point_functions {
namespace pir_testing {

// Creates a HybridDecrypt with a fixed key for testing.
absl::StatusOr<std::unique_ptr<crypto::tink::HybridDecrypt>>
CreateFakeHybridDecrypt();

// Creates a HybridEncrypt with a fixed key for testing.
absl::StatusOr<std::unique_ptr<crypto::tink::HybridEncrypt>>
CreateFakeHybridEncrypt();

}  // namespace pir_testing
}  // namespace distributed_point_functions

#endif  // DISTRIBUTED_POINT_FUNCTIONS_PIR_TESTING_FAKE_HYBRID_DECRYPT_H_
