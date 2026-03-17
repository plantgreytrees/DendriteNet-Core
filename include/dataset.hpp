#pragma once
#include "tensor.hpp"
#include <vector>
#include <utility>

namespace dendrite {

/// A single labelled input sample.
struct Sample {
    Tensor input;
    Tensor target;  // one-hot encoded
    int    label;
};

/// Abstract base class for dataset providers.
/// Subclass this (e.g. in examples/datasets/) to plug in a custom task.
/// Call load() once; it returns a pre-split (train, test) pair.
struct DatasetProvider {
    virtual ~DatasetProvider() = default;

    /// Return train and test splits.
    /// Implementations should respect train_split (fraction of data for training).
    virtual std::pair<std::vector<Sample>, std::vector<Sample>>
    load(float train_split = 0.8f) = 0;

    /// Dimensionality of each input tensor.
    virtual size_t input_dim()  const = 0;

    /// Number of output classes.
    virtual size_t output_dim() const = 0;
};

} // namespace dendrite
