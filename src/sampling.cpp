#include "openmc/sampling.h"

#include "xtensor/xbuilder.hpp"
#include "xtensor/xmath.hpp"
#include "xtensor/xtensor.hpp"
#include "xtensor/xview.hpp"

#include "xsdata.h"
#include "openmc/cross_sections.h"
#include "openmc/hdf5_interface.h"
#include "openmc/memory.h"
#include "openmc/scattdata.h"
#include "openmc/vector.h"

#include <algorithm> // for min
#include <cmath>     // for sqrt, abs, pow
#include <iterator>  // for back_inserter
#include <limits>    //for infinity
#include <string>

namespace openmc {


void setMean(const xt::xtensor<double, 1>& mean) { _mean = mean; }

void setCovar(const xt::xtensor<double, 2>& covar)
{
  _covar = covar;

  /*
  While the Cholesky decomposition is particularly useful to solve selfadjoint 
  problems like D^*D x = b, for that purpose, we recommend the Cholesky decomposition 
  without square root which is more stable and even faster. 
  Nevertheless, this standard Cholesky decomposition remains useful in many other situations 
  like generalised eigen problems with hermitian matrices*/

  cov_symmetric = is_symmetric(_covar);
  cov_pos_definite = is_positive_definite(_covar);

  if (cov_symmetric && cov_pos_definite)
  {
    //Use Cholesky decomposition
    xt::xtensor<double, 2> L = cholesky_decomposition(_covar);
    _transform = L;

  } else
    {
       auto [eigenvalues, eigenvectors] = eigenvalue_solver(_covar);
        xt::xtensor<double, 2> D = xt::zeros<double>({eigenvalues.size(), eigenvalues.size()});
        for (size_t i = 0; i < eigenvalues.size(); ++i) {
            D(i, i) = std::sqrt(std::max(eigenvalues(i).real(), 0.0));
        }
        _transform = xt::linalg::dot(eigenvectors, D);
    }
}

xt::xtensor<double, 2> normal_sampling (int nn)
{
    std::normal_distribution<double> dist(0.0, 1.0);
    xt::xtensor<double, 2> Z = xt::zeros<double>({_covar.shape()[0], nn});
    for (size_t i = 0; i < Z.shape()[0]; ++i) {
        for (size_t j = 0; j < Z.shape()[1]; ++j) {
            Z(i, j) = dist(randN);
        }
    }
    xt::xtensor<double, 2> samples = xt::linalg::dot(_transform, Z);
    for (size_t i = 0; i < samples.shape()[1]; ++i) {
        xt::view(samples, xt::all(), i) += _mean;
    }
    return samples;
}

bool is_symmetric(const xt::xtensor<double, 2>& matrix)
{
  for (size_t i = 0; i < matrix.shape()[0]; ++i) {
    for (size_t j = 0; j < matrix.shape()[1]; ++j) {
      if (matrix(i, j) != matrix(j, i)) {
        return false;
      }
    }
  }
  return true;
}

bool is_positive_definite(const xt::xtensor<double, 2>& matrix)
{
  // Simple check for positive definiteness
  for (size_t i = 0; i < matrix.shape()[0]; ++i) {
    if (matrix(i, i) <= 0) {
      return false;
    }
  }
  return true;
}

std::pair<xt::xtensor<std::complex<double>, 1>, xt::xtensor<std::complex<double>, 2>> eigenvalue_solver(const xt::xtensor<double, 2>& matrix)
{
  size_t n = matrix.shape()[0];
  xt::xtensor<std::complex<double>, 1> eigenvalues = xt::zeros<std::complex<double>>({n});
  xt::xtensor<std::complex<double>, 2> eigenvectors = xt::eye<std::complex<double>>(n);

  // Step 1: Reduce to Hessenberg form
  xt::xtensor<double, 2> H = hessenberg_decomposition(matrix);

  // Step 2: Reduce to Schur form
  xt::xtensor<double, 2> T;
  xt::xtensor<double, 2> U;
  std::tie(T, U) = schur_decomposition(H);

  // Step 3: Extract eigenvalues from Schur form
  for (size_t i = 0; i < n; ++i) {
    if (i == n - 1 || T(i + 1, i) == 0) {
      eigenvalues(i) = T(i, i);
    } else {
      double p = 0.5 * (T(i, i) - T(i + 1, i + 1));
      double z = std::sqrt(std::abs(p * p + T(i + 1, i) * T(i, i + 1)));
      eigenvalues(i) = std::complex<double>(T(i + 1, i + 1) + p, z);
      eigenvalues(i + 1) = std::complex<double>(T(i + 1, i + 1) + p, -z);
      ++i;
    }
  }

  // Step 4: Compute eigenvectors
  eigenvectors = compute_eigenvectors(T, U);

  return {eigenvalues, eigenvectors};
}

xt::xtensor<double, 2> hessenberg_decomposition(const xt::xtensor<double, 2>& matrix)
{
  size_t n = matrix.shape()[0];
  xt::xtensor<double, 2> H = matrix;

  for (size_t k = 0; k < n - 2; ++k) {
    xt::xtensor<double, 1> x = xt::view(H, xt::range(k + 1, n), k);
    double alpha = -std::copysign(xt::linalg::norm(x)(), x(0));
    xt::xtensor<double, 1> e1 = xt::zeros<double>({n - k - 1});
    e1(0) = 1.0;
    xt::xtensor<double, 1> u = x - alpha * e1;
    xt::xtensor<double, 1> v = u / xt::linalg::norm(u)();

    xt::xtensor<double, 2> P = xt::eye<double>(n);
    xt::view(P, xt::range(k + 1, n), xt::range(k + 1, n)) -= 2.0 * xt::linalg::outer(v, v);

    H = xt::linalg::dot(P, xt::linalg::dot(H, P));
  }

  return H;
}

std::pair<xt::xtensor<double, 2>, xt::xtensor<double, 2>> schur_decomposition(const xt::xtensor<double, 2>& H)
{
  size_t n = H.shape()[0];
  xt::xtensor<double, 2> T = H;
  xt::xtensor<double, 2> U = xt::eye<double>(n);

  for (size_t iter = 0; iter < 1000; ++iter) {
    // QR decomposition
    xt::xtensor<double, 2> Q, R;
    std::tie(Q, R) = qr_decomposition(T);

    // Update T and U
    T = xt::linalg::dot(R, Q);
    U = xt::linalg::dot(U, Q);
  }

  return {T, U};
}

std::pair<xt::xtensor<double, 2>, xt::xtensor<double, 2>> qr_decomposition(const xt::xtensor<double, 2>& matrix)
{
  size_t n = matrix.shape()[0];
  xt::xtensor<double, 2> Q = xt::eye<double>(n);
  xt::xtensor<double, 2> R = matrix;

  for (size_t k = 0; k < n - 1; ++k) {
    xt::xtensor<double, 1> x = xt::view(R, xt::range(k, n), k);
    double alpha = -std::copysign(xt::linalg::norm(x)(), x(0));
    xt::xtensor<double, 1> e1 = xt::zeros<double>({n - k});
    e1(0) = 1.0;
    xt::xtensor<double, 1> u = x - alpha * e1;
    xt::xtensor<double, 1> v = u / xt::linalg::norm(u)();

    xt::xtensor<double, 2> Q_k = xt::eye<double>(n);
    xt::view(Q_k, xt::range(k, n), xt::range(k, n)) -= 2.0 * xt::linalg::outer(v, v);

    R = xt::linalg::dot(Q_k, R);
    Q = xt::linalg::dot(Q, Q_k);
  }

  return {Q, R};
}

xt::xtensor<std::complex<double>, 2> compute_eigenvectors(const xt::xtensor<double, 2>& T, const xt::xtensor<double, 2>& U)
{
  size_t n = T.shape()[0];
  xt::xtensor<std::complex<double>, 2> eigenvectors = xt::eye<std::complex<double>>(n);

  // Back-substitution to find eigenvectors
  for (size_t i = n - 1; i < n; --i) {
    for (size_t j = i + 1; j < n; ++j) {
      eigenvectors(i, j) = -T(i, j) / (T(i, i) - T(j, j));
    }
  }

  // Transform eigenvectors to original basis
  eigenvectors = xt::linalg::dot(U, eigenvectors);

  return eigenvectors;
}

xt::tensor<double, 2> cholesky_decomposition(xt::tensor<double, 2> covariance)
{
  size_t n = covariance.shape()[0];
  xt::xtensor<double, 2> L = xt::zeros<double>({n, n});

  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j <= i; ++j) {
      double sum = 0.0;
      for (size_t k = 0; k < j; ++k) {
        sum += L(i, k) * L(j, k);
      }
      if (i == j) {
        L(i, j) = std::sqrt(covariance(i, i) - sum);
      } else {
        L(i, j) = (covariance(i, j) - sum) / L(j, j);
      }
    }
  }
  return L;
}


void log_normal_sampling(xt::xtensor<double, 2> mean, xt::xtensor<double, 2> covariance)
{
    return 0;
}

void latin_hypercube_sampling(xt::xtensor<double, 2> mean, xt::xtensor<double, 2> covariance)
{
    return 0;
}


}