#include "openmc/tallies/filter_score.h"

#include <fmt/core.h>

#include "openmc/capi.h"
#include "openmc/constants.h" // For C_NONE
#include "openmc/mgxs_interface.h"
#include "openmc/search.h"
#include "openmc/settings.h"
#include "openmc/xml_interface.h"
#include "openmc/tallies/tally.h"

namespace openmc {

//==============================================================================
// Score Filter implementation
//==============================================================================

void ScoreFilter::from_xml(pugi::xml_node node)
{
  auto bins = get_node_array<double>(node, "bins");
  this->set_bins(bins);
}

void ScoreFilter::set_bins(gsl::span<const double> bins)
{
  // Clear existing bins
  bins_.clear();
  bins_.reserve(bins.size());
  map_.clear();

  // Copy bins
  for (gsl::index i = 0; i < bins.size(); ++i) {
    bins_.push_back(bins[i]);
    map_[bins[i]] = i;
  }

  n_bins_ = bins_.size();
}

  void ScoreFilter::get_all_bins(const Particle& p, TallyEstimator estimator, FilterMatch& match) const
{
  // Get the score for the particle
  double score = p.score(); 

  // Bin the score according to the filter bins by finding the matching bin
  if (score >= bins_.front() && score <= bins_.back()) {
    auto bin = lower_bound_index(bins_.begin(), bins_.end(), score);
    match.bins_.push_back(bin);
    match.weights_.push_back(1.0);
  }

  if (p.score())
}

void ScoreFilter::to_statepoint(hid_t filter_group) const
{
  Filter::to_statepoint(filter_group);
  write_dataset(filter_group, "bins", bins_);
}

std::string ScoreFilter::text_label(int bin) const
{
  return fmt::format("Scores [{}, {})", bins_[bin], bins_[bin + 1]);
}


} // namespace openmc
