#include "TestStructures.hh"
#include "casm/crystallography/BasicStructure.hh"
#include "casm/crystallography/BasicStructureTools.hh"
#include "casm/crystallography/CanonicalForm.hh"
#include "casm/crystallography/SymTools.hh"
#include "casm/external/MersenneTwister/MersenneTwister.h"
#include "gtest/gtest.h"

using namespace CASM;
using namespace CASM::xtal;

namespace CASM {
namespace xtal {

// Eigen::MatrixXd get_coordinate_frac(
//     xtal::BasicStructure const &prim) {
//   Eigen::MatrixXd coordinate_frac(3, prim.basis().size());
//   Index b = 0;
//   for (auto const &site : prim.basis()) {
//     coordinate_frac.col(b) = site.const_frac();
//     ++b;
//   }
//   return coordinate_frac;
// }
//
// // /// \brief Align lattice so 'a' and 'b' are in x-y plane
// // Lattice _make_aligned_lattice(double a, double b, double c, double alpha,
// double beta, double gamma, double tol) {
// //   auto _clean = [](double x) {
// //     if (almost_zero(x, 1e-12)) {
// //       return 0.0;
// //     }
// //     return x;
// //   };
// //   Eigen::Vector3d vec1(_clean(a), 0., 0.);
// //   Eigen::Vector3d vec2(_clean(b*cos(gamma)), _clean(b*sin(gamma)), 0.);
// //   double cx = c*cos(beta);
// //   double cy = c*(cos(alpha) - cos(beta)*cos(gamma))/(sin(gamma));
// //   double cz = sqrt(c*c - cx*cx - cy*cy);
// //   Eigen::Vector3d vec3(_clean(cx), _clean(cy), _clean(cz));
// //   return Lattice(vec1, vec2, vec3, tol);
// // }
// //
// // /// \brief Align lattice so 'a' and 'b' are in x-y plane
// // Lattice make_aligned_lattice(Lattice lattice) {
// //   lattice.make_right_handed();
// //   double a = lattice.length(0);
// //   double b = lattice.length(1);
// //   double c = lattice.length(2);
// //   double alpha = CASM::angle(lattice[1], lattice[2]);
// //   double beta = CASM::angle(lattice[2], lattice[0]);
// //   double gamma = CASM::angle(lattice[0], lattice[1]);
// //   return _make_aligned_lattice(a, b, c, alpha, beta, gamma,
// lattice.tol());
// // }
// //
// // /// \brief Align lattice so 'a' and 'b' are in x-y plane
// // Lattice make_aligned_lattice(double a, double b, double c, double
// alpha_deg, double beta_deg, double gamma_deg, double tol) {
// //   double alpha = alpha_deg * M_PI / 180.;
// //   double beta = beta_deg * M_PI / 180.;
// //   double gamma = gamma_deg * M_PI / 180.;
// //   return _make_aligned_lattice(a, b, c, alpha, beta, gamma, tol);
// // }
//
// /// \brief Find upper bound of tolerance range that returns the
// ///     same value of a function
// ///
// /// \brief init_tol Initial tolerance exponent, as in
// ///     `init_value=f(std::pow(10., init_tol))`
// /// \brief step Initial step size in tolerance exponent, as in
// ///     `next_value=f(std::pow(10., current_tol + step))`
// /// \brief step Minimum step size in tolerance exponent. When a
// ///     change in value is found, the step size is decreased by
// ///     a factor of ten in order to tighten the bounds. This is
// ///     the minimum allowed step size.
// /// \brief max Maximum tolerance exponent to search for a change
// ///     in function value.
// /// \param f A unary function of tolerance.
// ///
// /// \return {tol, value}, where tol is the tolerance exponent at which the
// ///     the value of `f` changes, or if no change found, then tol >= min,
// ///     and value is `f(std::pow(10., tol))`
// ///
// template <typename ValueType, typename UnaryFunctionOfTolerance>
// std::pair<double, ValueType> find_upper_tol(double init_tol, double step,
//                                             double min_step, double max,
//                                             UnaryFunctionOfTolerance &f) {
//   double base = 10.;
//   ValueType init_value = f(std::pow(base, init_tol));
//   ValueType value{init_value};
//   double tol = init_tol;
//   while (true) {
//     value = f(std::pow(base, tol + step));
//     if (value == init_value) {
//       tol += step;
//       if (tol >= max) {
//         break;
//       }
//     } else {
//       if (step / 10. <= min_step) {
//         tol += step;
//         break;
//       } else {
//         step /= 10.;
//       }
//     }
//   }
//   return std::make_pair(tol, value);
// }
//
// /// \brief Find lower bound of tolerance range that returns the
// ///     same value of a function
// ///
// /// \brief init_tol Initial tolerance exponent, as in
// ///     `init_value=f(std::pow(10., init_tol))`
// /// \brief step Initial step size in tolerance exponent, as in
// ///     `next_value=f(std::pow(10., current_tol - step))`
// /// \brief step Minimum step size in tolerance exponent. When a
// ///     change in value is found, the step size is decreased by
// ///     a factor of ten in order to tighten the bounds. This is
// ///     the minimum allowed step size.
// /// \brief min Minimum tolerance exponent to search for a change
// ///     in function value.
// /// \param f A unary function of tolerance.
// ///
// /// \return {tol, value}, where tol is the tolerance exponent at which the
// ///     the value of `f` changes, or if no change found, then tol <= min,
// ///     and value is `f(std::pow(10., tol))`
// ///
// template <typename ValueType, typename UnaryFunctionOfTolerance>
// std::pair<double, ValueType> find_lower_tol(double init_tol, double step,
//                                             double min_step, double min,
//                                             UnaryFunctionOfTolerance &f) {
//   double base = 10.;
//   ValueType init_value = f(std::pow(base, init_tol));
//   ValueType value{init_value};
//   double tol = init_tol;
//   while (true) {
//     value = f(std::pow(base, tol - step));
//     if (value == init_value) {
//       tol -= step;
//       if (tol <= min) {
//         break;
//       }
//     } else {
//       if (step / 10. <= min_step) {
//         tol -= step;
//         break;
//       } else {
//         step /= 10.;
//       }
//     }
//   }
//   return std::make_pair(tol, value);
// }
//
// struct CalcLatticePointGroupSize {
//   CalcLatticePointGroupSize(Lattice const &_lattice) : lattice(_lattice) {}
//
//   int operator()(double tol) {
//     point_group = make_point_group(lattice, tol);
//     return point_group.size();
//   }
//
//   Lattice const &lattice;
//   std::vector<SymOp> point_group;
// };
//
// struct CalcStandardNiggli {
//   CalcStandardNiggli(Lattice const &_lattice)
//       : lattice(_lattice), init_niggli_lat(canonical::equivalent(lattice)) {}
//
//   bool operator()(double tol) {
//     Lattice tmp_lattice{lattice.lat_column_mat(), tol};
//     Lattice niggli_lat = canonical::equivalent(tmp_lattice);
//     return almost_equal(init_niggli_lat.lat_column_mat(),
//                         niggli_lat.lat_column_mat(), tol);
//   }
//
//   Lattice const &lattice;
//   Lattice init_niggli_lat;
// };
//
// struct CalcFactorGroupSize {
//   CalcFactorGroupSize(BasicStructure const &_struc) : struc(_struc) {}
//
//   int operator()(double tol) {
//     BasicStructure tmp{struc};
//     tmp.set_lattice(Lattice{struc.lattice().lat_column_mat(), tol}, FRAC);
//     factor_group = make_factor_group(tmp);
//     return factor_group.size();
//   }
//
//   BasicStructure const &struc;
//   std::vector<SymOp> factor_group;
// };
//
// /// \brief Return the next largest lattice point group
// ///     resulting from a loosening tolerance
// ///
// /// Search for the next loosest tolerance which results in an
// /// increased lattice point group size and return that lattice
// /// point group. If none found in the given range of
// /// tolerances, return an empty vector.
// ///
// /// \brief lattice Lattice to calculate the point group of.
// /// \brief init_tol Initial tolerance exponent, as in
// ///     `init_point_group=f(std::pow(10., init_tol))`, where
// ///     `f` is a function that calculates the lattice point
// ///     group given a tolerance.
// /// \brief upper Maximum tolerance exponent to search for a change
// ///     in point group size.
// /// \brief step Initial step size in tolerance exponent, as in
// ///     `next_value=f(std::pow(10., current_tol + step))`
// /// \brief step Minimum step size in tolerance exponent. When a
// ///     change in point group size is found, the step size is
// ///     decreased by a factor of ten in order to tighten the
// ///     bounds. This is the minimum allowed step size.
// ///
// /// \return next_largest_lattice_point_group, The next largest
// ///     lattice point group, or the same lattice point group if
// ///     no larger point group is found in the given tolerance
// ///     range.
// std::vector<SymOp> make_next_largest_lattice_point_group(
//     xtal::Lattice const &lattice, double init_tol = -5., double upper = -3.,
//     double step = 1., double min_step = 0.09) {
//   CalcLatticePointGroupSize calc_pg(lattice);
//   find_upper_tol<int>(init_tol, step, min_step, upper, calc_pg);
//   return calc_pg.point_group;
// }
//
// /// \brief Return the next largest factor group
// ///     resulting from a loosening tolerance
// ///
// /// Search for the next loosest tolerance which results in an
// /// increased factor group size and return that factor group.
// /// If none found in the given range of tolerances, return an
// /// empty vector.
// ///
// /// \brief prim The prim to calculate the point group of.
// /// \brief init_tol Initial tolerance exponent, as in
// ///     `init_factor_group=f(std::pow(10., init_tol))`, where
// ///     `f` is a function that calculates the factor group
// ///     given a tolerance.
// /// \brief upper Maximum tolerance exponent to search for a change
// ///     in factor group size.
// /// \brief step Initial step size in tolerance exponent, as in
// ///     `next_value=f(std::pow(10., current_tol + step))`
// /// \brief step Minimum step size in tolerance exponent. When a
// ///     change in factor group size is found, the step size is
// ///     decreased by a factor of ten in order to tighten the
// ///     bounds. This is the minimum allowed step size.
// ///
// /// \return next_largest_lattice_factor_group, The next largest
// ///     factor group, or an empty vector if no larger
// ///     factor group is found in the given tolerance range.
// std::vector<SymOp> make_next_largest_factor_group(
//     xtal::BasicStructure const &prim, double init_tol = -5., double upper =
//     -3., double step = 1., double min_step = 0.09) {
//   CalcFactorGroupSize calc_fg(prim);
//   auto fg_upper = find_upper_tol<int>(init_tol, step, min_step, upper,
//   calc_fg); if (fg_upper.first < upper) {
//     return calc_fg.factor_group;
//   }
//   return std::vector<SymOp>();
// }
//
// /// \brief Return true if lattice point group size or niggli
// ///     lattice is sensitive to tolerance in a specified range
// ///
// /// Calculate the lattice point group and standard niggli
// /// lattice at the standard tolerance and upper and lower
// /// tolerances to check for changes.
// ///
// /// \brief lattice The lattice to check.
// /// \brief standard Standard tolerance exponent, as in
// ///     `init_value=f(std::pow(10., standard))`, where
// ///     `f` is a function that calculates the point group
// ///     or niggli lattice given a tolerance.
// /// \brief lower Minimum tolerance exponent to check.
// /// \brief upper Maximum tolerance exponent to check.
// ///
// /// \return True, if the lattice point group size or
// ///     standard niggli cell change based on choice
// ///     of tolerances in the range given.
// bool is_tolerance_sensitive_lattice(Lattice const &lattice,
//                                     double standard = -5., double lower =
//                                     -7., double upper = -3.) {
//   double base = 10.;
//   CalcLatticePointGroupSize calc_pg(lattice);
//   Index standard_pg_size = calc_pg(std::pow(base, standard));
//   Index value = calc_pg(std::pow(base, upper));
//   if (value != standard_pg_size) {
//     // std::cout << "A, value: " << value << " standard: " <<
//     standard_pg_size << std::endl; return true;
//   }
//   value = calc_pg(std::pow(base, lower));
//   if (value != standard_pg_size) {
//     // std::cout << "B, value: " << value << " standard: " <<
//     standard_pg_size << std::endl; return true;
//   }
//
//   // CalcStandardNiggli calc_standard_niggli(lattice);
//   // Index equal = calc_standard_niggli(std::pow(base, upper));
//   // if (!equal) {
//   //   std::cout << "C, equal?: " << equal << std::endl;
//   //   return true;
//   // }
//   // equal = calc_standard_niggli(std::pow(base, lower));
//   // if (!equal) {
//   //   std::cout << "D, equal?: " << equal << std::endl;
//   //   return true;
//   // }
//
//   return false;
// }
//
// /// \brief Return true if lattice point group size, niggli
// ///     lattice, or factor group is sensitive to tolerance
// ///     in a specified range
// ///
// /// Calculate the lattice point group, standard niggli
// /// lattice, and factor group at the standard tolerance
// /// and upper and lower tolerances to check for changes.
// ///
// /// \brief prim The prim to check.
// /// \brief standard Standard tolerance exponent, as in
// ///     `init_value=f(std::pow(10., standard))`, where
// ///     `f` is a function that calculates the lattice
// ///     point group, niggli lattice, or factor group
// ///     given a tolerance.
// /// \brief lower Minimum tolerance exponent to check.
// /// \brief upper Maximum tolerance exponent to check.
// ///
// /// \return True, if the lattice point group size,
// ///     standard niggli cell, or factor group change
// ///     based on choice of tolerances in the range given.
// bool is_tolerance_sensitive_prim(BasicStructure const &prim,
//                                  double standard = -5., double lower = -7.,
//                                  double upper = -3.) {
//   if (is_tolerance_sensitive_lattice(prim.lattice(), standard, lower, upper))
//   {
//     return true;
//   }
//
//   double base = 10.;
//   CalcFactorGroupSize calc_fg(prim);
//   Index standard_fg_size = calc_fg(std::pow(base, standard));
//   if (calc_fg(std::pow(base, upper)) != standard_fg_size) {
//     return true;
//   }
//   if (calc_fg(std::pow(base, lower)) != standard_fg_size) {
//     return true;
//   }
//   return false;
// }
//
// /// \brief Symmetrize a lattice so its point group size is not sensitive to
// /// tolerances in a particular range
// ///
// /// This method iteratively finds if, within a range of symmetry tolerances,
// /// there is a symmetry tolerance that increases the lattice point group
// size,
// /// and if there is, applies the larger point group and replaces the lattice
// /// with the average lattice. This continues until the lattice point group
// size
// /// is constant in the entire range of symmetry tolerances, or the maximum
// /// number of iterations is reached.
// ///
// /// \param lattice The lattice to symmetrize
// /// \param standard The standard symmetry tolerance exponent, as in
// ///     `tol=std::pow(10., standard)`
// /// \param lower The lower bound symmetry tolerance exponent, as in
// ///     `lower_tol=std::pow(10., lower)`
// /// \param upper The upper bound symmetry tolerance exponent, as in
// ///     `upper_tol=std::pow(10., upper)`
// /// \param n_max Maximum number of symmetrization iterations
// ///
// /// \returns {lattice, success} If success==true, lattice is a
// ///     symmetrized lattice, such that the lattice point group
// ///     size and standard niggli cell generated using the
// ///     standard tolerance is the same as the lattice point
// ///     group size generated using the lower and upper bound
// ///     tolerances. The tolerance of the returned lattice is
// ///     set to `std::pow(10., standard)`.
// ///
// std::pair<Lattice, bool> symmetrize_lattice(Lattice const &lattice, double
// standard = -5.,
//                            double lower = -7., double upper = -3.,
//                            Index n_max = 100) {
//   double base = 10.;
//   double tol = std::pow(base, standard);
//   Index n = 0;
//
//   Lattice tmp {lattice.lat_column_mat(), tol};
//   bool is_sensitive = is_tolerance_sensitive_lattice(tmp, standard, lower,
//   upper);
//
//   // std::cout << std::endl;
//   // std::cout << "lattice: \n" << lattice.lat_column_mat() << std::endl;
//   // std::cout << "symmetrized_lattice: \n" << tmp.lat_column_mat() <<
//   std::endl;
//   // std::cout << "n: " << n << std::endl;
//   // std::cout << "is_sensitive: " << is_sensitive << std::endl;
//
//   while (n < n_max && is_sensitive) {
//     double init_tol = standard;
//     double step = 1.;
//     double min_step = 0.09;
//
//     // make the lattice point group to be enforced (empty if no changes
//     found) std::vector<SymOp> lattice_point_group =
//         make_next_largest_lattice_point_group(tmp, init_tol, upper, step,
//                                               min_step);
//
//     // nothing beside identity_op? break
//     if (lattice_point_group.size() == 1) {
//       break;
//     }
//
//     // symmetrize lattice
//     tmp = Lattice{symmetrize(tmp, lattice_point_group).lat_column_mat(),
//     tol};
//     ++n;
//     is_sensitive = is_tolerance_sensitive_lattice(tmp, standard, lower,
//     upper);
//
//     // std::cout << std::endl;
//     // std::cout << "symmetrized_lattice: \n" << tmp.lat_column_mat() <<
//     std::endl;
//     // std::cout << "n: " << n << std::endl;
//     // std::cout << "is_sensitive: " << is_sensitive << std::endl;
//   }
//
//   // Lattice tmp_loose{lattice.lat_column_mat(), std::pow(base, upper)};
//   // auto pg_loose = make_point_group(tmp_loose);
//   //
//   // Lattice tmp_symmetrized{tmp.lat_column_mat(), std::pow(base, standard)};
//   // auto pg_symmetrized = make_point_group(tmp_symmetrized);
//   //
//   // Lattice tmp_tight{tmp.lat_column_mat(), std::pow(base, lower)};
//   // auto pg_tight = make_point_group(tmp_tight);
//   //
//   // std::cout << "loose: " << pg_loose.size() << "  symmetrized: " <<
//   pg_symmetrized.size() << "  tight: " << pg_tight.size() << std::endl;
//
//   return std::pair<Lattice, bool>(tmp, !is_sensitive);
// }
//
//
// /// \brief Symmetrize a prim so its factor group size is not sensitive to
// /// tolerances in a particular range
// ///
// /// This method iteratively finds if, within a range of symmetry tolerances,
// /// there is a symmetry tolerance that increases the factor group size, and
// if
// /// there is, applies the larger factor group and replaces the prim with the
// /// average lattice and basis coordinates. This continues until the factor
// group
// /// size is constant in the entire range of symmetry tolerances, or the
// maximum
// /// number of iterations is reached.
// ///
// /// \param prim The prim to symmetrize
// /// \param standard The standard symmetry tolerance exponent, as in
// ///     `tol=std::pow(10., standard)`
// /// \param lower The lower bound symmetry tolerance exponent, as in
// ///     `lower_tol=std::pow(10., lower)`
// /// \param upper The upper bound symmetry tolerance exponent, as in
// ///     `upper_tol=std::pow(10., upper)`
// /// \param n_max Maximum number of symmetrization iterations
// ///
// /// \returns {prim, success} If success==true, prim is a
// ///     symmetrized prim, such that the lattice point group
// ///     size, standard niggli cell, and factor group size
// ///     generated using the standard tolerance is the same
// ///     as the lattice point group size generated using the
// ///     lower and upper bound tolerances. The tolerance of
// ///     the returned prim's lattice is set to
// ///     `std::pow(10., standard)`.
// ///
// std::pair<BasicStructure, bool> symmetrize_prim(BasicStructure const &prim,
//                                double standard = -5., double lower = -7.,
//                                double upper = -3., Index n_max = 100) {
//
//   BasicStructure tmp = prim;
//   std::pair<Lattice, bool> symmetrized_lattice =
//       symmetrize_lattice(prim.lattice(), standard, lower, upper, n_max);
//   tmp.set_lattice(symmetrized_lattice.first, FRAC);
//
//   if (!symmetrized_lattice.second) {
//     return std::pair<BasicStructure, bool>(prim, false);
//   }
//
//   bool is_sensitive = is_tolerance_sensitive_prim(tmp, standard, lower,
//   upper); Index n = 0; while (n < n_max && is_sensitive) {
//     double init_tol = standard;
//     double step = 1.;
//     double min_step = 0.09;
//
//     // make the factor group to be enforced (empty if no changes found)
//     std::vector<SymOp> factor_group =
//         make_next_largest_factor_group(tmp, init_tol, upper, step, min_step);
//     if (factor_group.size()) {
//       // symmetrize the structure basis coordinates
//       tmp = symmetrize(tmp, factor_group);
//       ++n;
//       is_sensitive = is_tolerance_sensitive_prim(tmp, standard, lower,
//       upper); continue;
//     } else {
//       break;
//     }
//   }
//
//   return std::pair<BasicStructure, bool>(tmp, !is_sensitive);
// }

}  // namespace xtal
}  // namespace CASM

namespace test {

std::pair<Lattice, bool> simple_symmetrize_lattice(
    Lattice const &lattice, double point_group_tolerance) {
  std::stringstream ss;
  bool ss_init = false;

  Lattice tmp = lattice;
  bool success = false;
  Index n = 0;
  Index n_max = 10;

  do {
    auto pg_loose =
        make_point_group(Lattice{tmp.lat_column_mat(), point_group_tolerance});

    tmp = symmetrize(tmp, pg_loose);
    auto pg_symmetrized = make_point_group(tmp);
    success = (pg_symmetrized.size() >= pg_loose.size());

    // std::cout << std::endl;
    // std::cout << "lattice: \n" << lattice.lat_column_mat() << std::endl;
    // std::cout << "symmetrized_lattice: \n" << tmp.lat_column_mat() <<
    // std::endl; Lattice tmp_tight{tmp.lat_column_mat(), std::pow(base, -7.)};
    // auto pg_tight = make_point_group(tmp_tight);
    // std::cout << "loose: " << pg_loose.size() << "  symmetrized: " <<
    // pg_symmetrized.size() << "  tight: " << pg_tight.size() << std::endl;

    if (!success) {
      if (!ss_init) {
        ss_init = true;
        ss << "begin simple_symmetrize_lattice" << std::endl;
        ss << "lattice:\n" << lattice.lat_column_mat() << std::endl;
      }
      ss << "---" << std::endl;
      ss << "tmp_lattice:\n" << tmp.lat_column_mat() << std::endl;
      ss << "loose: " << pg_loose.size()
         << "  symmetrized: " << pg_symmetrized.size() << std::endl;
    }
    ++n;
  } while (!success && n < n_max);

  if (n == n_max) {
    std::cout << ss.str() << std::endl;
    throw std::runtime_error("Could not symmetrize lattice");
  }

  if (ss_init) {
    std::cout << ss.str() << std::endl;
  }
  return std::pair<Lattice, bool>(tmp, success);
}

std::pair<BasicStructure, bool> simple_symmetrize_prim(
    BasicStructure const &prim, double factor_group_tolerance) {
  Lattice symmetrized_lattice =
      symmetrize(prim.lattice(), factor_group_tolerance);
  Lattice symmetrized_lattice_loose_tol{symmetrized_lattice.lat_column_mat(),
                                        factor_group_tolerance};

  BasicStructure tmp{prim};
  tmp.set_lattice(symmetrized_lattice_loose_tol, FRAC);
  auto fg_loose = make_factor_group(tmp);

  tmp.set_lattice(symmetrized_lattice, FRAC);
  tmp = symmetrize(tmp, fg_loose);

  auto fg_symmetrized = make_factor_group(tmp);
  bool success = (fg_symmetrized.size() >= fg_loose.size());

  if (!success) {
    std::cout << std::endl;
    std::cout << "loose: " << fg_loose.size()
              << "  symmetrized: " << fg_symmetrized.size() << std::endl;
  }

  return std::pair<BasicStructure, bool>(tmp, success);
}

Lattice randomize_lattice(Lattice const &init_lattice, double amplitude,
                          MTRand &random) {
  auto _rand = [&]() { return random.rand() * 2. * amplitude - amplitude; };
  Eigen::Matrix3d L = init_lattice.lat_column_mat();
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      L(i, j) += _rand();
    }
  }
  return Lattice(L, init_lattice.tol());
}

std::vector<Site> randomize_basis(std::vector<Site> const &init_basis,
                                  double amplitude, MTRand &random) {
  auto _rand = [&]() { return random.rand() * 2. * amplitude - amplitude; };
  auto basis = init_basis;
  for (int i = 0; i < basis.size(); ++i) {
    for (int j = 0; j < 3; ++j) {
      basis[i].cart(j) += _rand();
    }
  }
  return basis;
}

xtal::BasicStructure randomize_prim(xtal::BasicStructure const &init_struc,
                                    double amplitude, MTRand &random) {
  xtal::BasicStructure struc(init_struc);
  struc.set_lattice(randomize_lattice(struc.lattice(), amplitude, random),
                    FRAC);
  struc.set_basis(randomize_basis(struc.basis(), amplitude, random));
  return struc;
}

Lattice make_test_lattice_cubic(double a_param, double lattice_tol) {
  Eigen::Matrix3d L = Eigen::Matrix3d::Identity() * a_param;
  return Lattice(L, lattice_tol);
}

Lattice make_test_lattice_hexagonal(double a_param, double c_param,
                                    double lattice_tol) {
  Eigen::Matrix3d L;
  L << a_param, 0., 0.,                            //
      -a_param / 2., a_param * sqrt(3.) / 2., 0.,  //
      0., 0., c_param;                             //
  return Lattice(L.transpose(), lattice_tol);
}

/// conventional fcc with tetrahedral interstitial sites
BasicStructure make_test_prim1() {
  Lattice lattice = make_test_lattice_cubic(4.0, 1e-5);
  Eigen::MatrixXd basis(12, 3);
  basis << 0., 0., 0.,   //
      0., 0.5, 0.5,      //
      0.5, 0., 0.5,      //
      0.5, 0.5, 0.,      //
      0.25, 0.25, 0.25,  //
      0.25, 0.75, 0.25,  //
      0.75, 0.25, 0.25,  //
      0.75, 0.75, 0.25,  //
      0.25, 0.25, 0.75,  //
      0.25, 0.75, 0.75,  //
      0.75, 0.25, 0.75,  //
      0.75, 0.75, 0.75;  //
  Molecule A("A");
  Molecule B("B");
  Molecule C("C");
  Molecule D("D");

  BasicStructure prim(lattice);
  for (Index i = 0; i < 4; ++i) {
    Eigen::VectorXd coord_frac = basis.row(i);
    prim.push_back(Site(Coordinate(coord_frac, prim.lattice(), FRAC),
                        std::vector<Molecule>{A, B}));
  }
  for (Index i = 4; i < 12; ++i) {
    Eigen::VectorXd coord_frac = basis.row(i);
    prim.push_back(Site(Coordinate(coord_frac, prim.lattice(), FRAC),
                        std::vector<Molecule>{C, D}));
  }

  return prim;
}

void check_symmetrize_lattice(xtal::Lattice const &lattice,
                              int expected_point_group_size, double min,
                              double max, double step, int seed) {
  auto pg = xtal::make_point_group(lattice);
  EXPECT_EQ(pg.size(), expected_point_group_size);

  double max_delta = -100.;
  double min_delta = 100.;

  for (int seed = 0; seed < 10; ++seed) {
    for (double ampl = min; ampl < max; ampl += step) {
      double amplitude = std::pow(10., ampl);
      MTRand random(seed);
      xtal::Lattice test_lattice =
          randomize_lattice(lattice, amplitude, random);
      for (double upper = min; upper < max; upper += step) {
        // auto result_simple =
        //     simple_symmetrize_lattice(test_lattice, std::pow(10., upper));
        // EXPECT_TRUE(result_simple.second);
        //
        // auto pg = xtal::make_point_group(result_simple.first);
        // double delta = std::pow(10., upper) / 2. - std::pow(10., ampl);

        auto result = symmetrize(test_lattice, std::pow(10., upper));
        auto pg = xtal::make_point_group(result);
        double delta = std::pow(10., upper) / 2. - std::pow(10., ampl);

        if (delta > 0.) {
          EXPECT_EQ(pg.size(), expected_point_group_size)
              << "pg.size(): " << pg.size() << " | "
              << "ampl: " << ampl << " upper: " << upper << std::endl;
        }

        if (pg.size() != expected_point_group_size) {
          if (delta > max_delta) {
            max_delta = delta;
          }
          if (delta < min_delta) {
            min_delta = delta;
          }
        }
      }
    }
  }

  std::cout << "min_delta: " << min_delta << std::endl;
  std::cout << "max_delta: " << max_delta << std::endl;
}

void check_symmetrize_prim(xtal::BasicStructure const &prim,
                           int expected_factor_group_size, double min,
                           double max, double step, int seed) {
  auto fg = xtal::make_factor_group(prim);
  EXPECT_EQ(fg.size(), expected_factor_group_size);

  double max_delta = -100.;
  double min_delta = 100.;

  for (int seed = 0; seed < 10; ++seed) {
    for (double ampl = min; ampl < max; ampl += step) {
      double amplitude = std::pow(10., ampl);
      MTRand random(seed);
      xtal::BasicStructure test_prim = randomize_prim(prim, amplitude, random);
      for (double upper = min; upper < max; upper += step) {
        auto result_simple =
            simple_symmetrize_prim(test_prim, std::pow(10., upper));
        auto fg = xtal::make_factor_group(result_simple.first);

        EXPECT_TRUE(result_simple.second)
            << "fg.size(): " << fg.size() << " | "
            << "ampl: " << ampl << " upper: " << upper << std::endl;

        double delta = std::pow(10., upper) / 2. - std::pow(10., ampl);

        if (delta > 0.) {
          EXPECT_EQ(fg.size(), expected_factor_group_size)
              << "fg.size(): " << fg.size() << " | "
              << "ampl: " << ampl << " upper: " << upper << std::endl;
        }

        if (fg.size() != expected_factor_group_size) {
          if (delta > max_delta) {
            max_delta = delta;
          }
          if (delta < min_delta) {
            min_delta = delta;
          }
        }
      }
    }
  }

  std::cout << "min_delta: " << min_delta << std::endl;
  std::cout << "max_delta: " << max_delta << std::endl;
}

}  // namespace test

TEST(SymmetrizeLatticeTest, CubicTest1) {
  using namespace CASM::xtal;

  int seed = 6;
  Lattice lattice = test::make_test_lattice_cubic(4.0, 1e-5);
  int expected_point_group_size = 48;
  test::check_symmetrize_lattice(lattice, expected_point_group_size, -5., -1.,
                                 0.1, seed);
}

TEST(SymmetrizeLatticeTest, CubicTest2) {
  using namespace CASM::xtal;

  int seed = 6;
  Lattice lattice = test::make_test_lattice_cubic(40.0, 1e-5);
  int expected_point_group_size = 48;
  test::check_symmetrize_lattice(lattice, expected_point_group_size, -2., 0.,
                                 0.1, seed);
}

TEST(SymmetrizeLatticeTest, HexagonalTest1) {
  using namespace CASM::xtal;

  int seed = 6;
  Lattice lattice =
      test::make_test_lattice_hexagonal(3.0, 3.0 * sqrt(8. / 3.), 1e-5);
  int expected_point_group_size = 24;
  test::check_symmetrize_lattice(lattice, expected_point_group_size, -3., -1.,
                                 0.1, seed);
}

TEST(SymmetrizeLatticeTest, HexagonalTest2) {
  using namespace CASM::xtal;

  int seed = 6;
  Lattice lattice =
      test::make_test_lattice_hexagonal(30.0, 30.0 * sqrt(8. / 3.), 1e-5);
  int expected_point_group_size = 24;
  test::check_symmetrize_lattice(lattice, expected_point_group_size, -2., 0.,
                                 0.1, seed);
}

TEST(SymmetrizePrimTest, CubicTest1) {
  using namespace CASM::xtal;

  int seed = 6;
  // conventional fcc with tetrahedral interstitial sites
  xtal::BasicStructure prim = test::make_test_prim1();
  int expected_factor_group_size = 192;
  test::check_symmetrize_prim(prim, expected_factor_group_size, -3., -1., 0.1,
                              seed);
}

TEST(SymmetrizePrimTest, HexagonalTest1) {
  using namespace CASM::xtal;

  int seed = 6;
  xtal::BasicStructure prim = test::ZrO_prim();
  int expected_point_group_size = 24;
  test::check_symmetrize_prim(prim, expected_point_group_size, -3., -1., 0.1,
                              seed);
}
