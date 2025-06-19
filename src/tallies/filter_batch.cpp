#include "openmc/tallies/filter_batch.h"

#include <algorithm> // for is_sorted
#include <stdexcept> // for runtime_error

#include <fmt/core.h>

#include "openmc/search.h"
#include "openmc/xml_interface.h"
#include "openmc/simulation.h"

namespace openmc {

//==============================================================================
// WeightFilter implementation
//==============================================================================

void BatchFilter::from_xml(pugi::xml_node node)
{
  auto bins = get_node_array<int32_t>(node, "bins");
  this->set_bins(bins);
}

void BatchFilter::set_bins(span<const int32_t> bins)
{
    if (!std::is_sorted(bins.begin(), bins.end())) {
    throw std::runtime_error {"Batch bins must be monotonically increasing."};
  }

  // Clear existing bins
  bins_.clear();
  bins_.reserve(bins.size());

  // Copy bins
  bins_.insert(bins_.end(), bins.begin(), bins.end());
  n_bins_ = bins_.size();

}

void BatchFilter::get_all_bins(
    const Particle& p, TallyEstimator estimator, FilterMatch& match) const
{
    int current_batch = simulation::current_batch;
    auto it = std::find(bins_.begin(), bins_.end(), current_batch);

    if (it != bins_.end()) {
        // Calculate the bin index (distance from beginning)
        int bin_index = std::distance(bins_.begin(), it);
        
        // Push the correct bin index with weight 1.0
        match.bins_.push_back(bin_index);
        match.weights_.push_back(1.0);
    }
}

std::string BatchFilter::text_label(int bin) const
{
  return fmt::format("Batch={}", bins_[bin]);
}

void BatchFilter::to_statepoint(hid_t filter_group) const
{
  Filter::to_statepoint(filter_group);
  write_dataset(filter_group, "bins", bins_);
}

} // namespace openmc