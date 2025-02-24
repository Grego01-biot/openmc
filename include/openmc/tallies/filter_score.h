#ifndef OPENMC_TALLIES_FILTER_SCORE_H
#define OPENMC_TALLIES_FILTER_SCORE_H

#include <gsl/gsl-lite.hpp>

#include "openmc/particle.h"
#include "openmc/tallies/filter.h"
#include "openmc/vector.h"

namespace openmc {

//==============================================================================
//! Bins tally events based on the score of the event.
//==============================================================================

class ScoreFilter : public Filter {
public:
  //----------------------------------------------------------------------------
  // Constructors, destructors

  ~ScoreFilter() = default;

  //----------------------------------------------------------------------------
  // Methods

  std::string type_str() const override { return "score"; }
  FilterType type() const override { return FilterType::SCORE; }

  void from_xml(pugi::xml_node node) override;

  void get_all_bins(const Particle& p, TallyEstimator estimator,
    FilterMatch& match) const override;

  void to_statepoint(hid_t filter_group) const override;

  std::string text_label(int bin) const override;

  //----------------------------------------------------------------------------
  // Accessors

  const vector<double>& bins() const { return bins_; }
  void set_bins(gsl::span<const double> bins);

protected:
  //----------------------------------------------------------------------------
  // Data members

  vector<double> bins_;
  std::unordered_map<double, int> map_;
};

} // namespace openmc
#endif // OPENMC_TALLIES_FILTER_SCORE_H
