#ifndef OPENMC_TALLIES_FILTER_BATCH_H
#define OPENMC_TALLIES_FILTER_BATCH_H

#include <string>

#include "openmc/span.h"
#include "openmc/tallies/filter.h"
#include "openmc/vector.h"

namespace openmc {

//==============================================================================
//! Bins the weights of the particles.
//==============================================================================

class BatchFilter : public Filter {
public:
  //----------------------------------------------------------------------------
  // Constructors, destructors

  ~BatchFilter() = default;

  //----------------------------------------------------------------------------
  // Methods

  std::string type_str() const override { return "batch"; }
  FilterType type() const override { return FilterType::BATCH; }

  void from_xml(pugi::xml_node node) override;
  
  void get_all_bins(const Particle& p, TallyEstimator estimator,
    FilterMatch& match) const override;

  void to_statepoint(hid_t filter_group) const override;

  std::string text_label(int bin) const override;

  //----------------------------------------------------------------------------
  // Accessors

  const vector<int32_t>& bins() const { return bins_; }
  void set_bins(span<const int32_t> bins);

protected:
  //----------------------------------------------------------------------------
  // Data members
  vector<int32_t> bins_;
};

} // namespace openmc
#endif // OPENMC_TALLIES_FILTER_BATCH_H
