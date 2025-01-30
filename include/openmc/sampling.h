#ifndef OPENMC_SAMPLING_H
#define OPENMC_SAMPLING_H

#include "xtensor/xtensor.hpp"

#include "xsdata.h"
#include "openmc/cross_sections.h"
#include "openmc/hdf5_interface.h"
#include "openmc/memory.h"
#include "openmc/scattdata.h"
#include "openmc/vector.h"

namespace openmc {

// Type of sampling 
enum class SamplingType {
 NORMAL,
 LOG_NORMAL,
 LATIN_HYPERCUBE, 
 ORTHOGONAL
};

class Sampling {

private: 
    xt::xtensor<double, 1> _mean;
    xt::xtensor<double, 2> _covar;
    xt::xtensor<double, 2> _transform;
    std::mt19937 randN;
    
public: 
    // sampling function
    void setMean(const xt::xtensor<double, 1>& mean);
    void setCovar(const xt::xtensor<double, 2>& covar);

    void log_normal_sampling(xt::xtensor<double, 2> mean, xt::xtensor<double, 2> covariance);
    void latin_hypercube_sampling(xt::xtensor<double, 2> mean, xt::xtensor<double, 2> covariance);

    bool is_symmetric(const xt::xtensor<double, 2>& matrix);
    bool is_positive_definite(const xt::xtensor<double, 2>& matrix);


    std::pair<xt::xtensor<std::complex<double>, 1>, xt::xtensor<std::complex<double>, 2>> eigenvalue_solver(const xt::xtensor<double, 2>& matrix);
    xt::xtensor<double, 2> hessenberg_decomposition(const xt::xtensor<double, 2>& matrix);
    std::pair<xt::xtensor<double, 2>, xt::xtensor<double, 2>> schur_decomposition(const xt::xtensor<double, 2>& H);
    std::pair<xt::xtensor<double, 2>, xt::xtensor<double, 2>> qr_decomposition(const xt::xtensor<double, 2>& matrix);
    xt::xtensor<std::complex<double>, 2> compute_eigenvectors(const xt::xtensor<double, 2>& T, const xt::xtensor<double, 2>& U);
    xt::tensor<double, 2> cholesky_decomposition(xt::tensor<double, 2> covariance);

};

} // namespace openmc
#endif // OPENMC_SAMPLING_H


