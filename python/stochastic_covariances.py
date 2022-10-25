"""
Covariance matrix for stochastic integrals given by Minier, et al. 2003
"Weak first and second order numerical schemes for SDEs"

When run as an individual script (i.e. not called from another module),
some limit tests are evaluted using parameters from the Minier paper to 
verify all covariance coefficients are implemented correctly.
"""


if __name__ == "__main__":
  from sys import exit

  import numpy as np
  from numpy import exp
  from numpy import sqrt
  from numpy import square
else:
  from tensorflow.math import exp
  from tensorflow import sqrt
  from tensorflow import square
  import tensorflow.math as math
import argparse
from distutils.util import strtobool

def get_inputs():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "-t",
    "--test_limits",
    default=str(True),
    type=strtobool,
    help="Test limit cases",
  )
  parser.add_argument(
    "-d",
    "--debug",
    default=str(False),
    type=strtobool,
    help="Print debug checks",
  )
  return parser.parse_args()

def gammaLowerSqr(dt, BiSqr, theta, Ti, tau):

  # BiSqrTi = cmptMultiply(BiSqr, Ti)
  # gammaSqr = cmptMultiply(0.5*BiSqrTi, ones-cmptExp(cmptDivideHW(-2.0*dt,Ti)))
  gammaSqr = BiSqr * Ti * 0.5 * (1.0 - exp(-2.0 * dt / Ti))
  return gammaSqr


def GammaUpperSqr(dt, BiSqr, theta, Ti, tau):

  # 0.5 * Ti *(1-exp(-2dt/Ti))
  term1 = 0.5 * Ti * (1 - exp(-2 * dt / Ti))
  # cmptMultiply(Ti, 0.5*(ones-cmptExp(cmptDivideHW(-2.0*dt,Ti))))
  term2 = 2.0 * tau * Ti * (1.0 - exp(-dt / Ti) * exp(-dt / tau)) / (Ti + tau)
  # 0.5 * tau * (1 - exp(-2dt/tau) )
  term3 = 0.5 * tau * (1 - exp(-2 * dt / tau))
  # print("tau", tau) * Ti))"term1 {term1)"term2 {term2)"term3 {term3) * theta * BiSqr * (term1 - term2 + term3)
  # GammaUSqr = cmptMultiply(thetaSqrBiSqr, (term1 - term2 + term3))
  GammaUSqr = theta * theta * BiSqr * (term1 - term2 + term3)

  return GammaUSqr


def gammaGamma(dt, BiSqr, theta, Ti, tau):

  thetaBiSqrTi = BiSqr * theta * Ti

  # .5*(1-exp(-2dt/Ti))
  term1 = 0.5 * (1.0 - exp(-2.0 * dt / Ti))
  # [tau/(tau+Ti)]*[1-exp(-dt/Ti)*exp(-dt/tau)]
  term2 = tau * (1.0 - exp(-dt / Ti) * exp(-dt / tau)) / (tau + Ti)

  gG = thetaBiSqrTi * (term1 - term2)

  return gG


def gammaOmega(dt, BiSqr, theta, Ti, tau):

  # (Ti-tau)*(1-exp(-dt/Ti))
  term1 = (Ti - tau) * (1 - exp(-dt / Ti))
  # 0.5 Ti (1-exp(-2dt/Ti))
  term2 = 0.5 * Ti * (1 - exp(-2 * dt / Ti))
  # tau^2 / (Ti+tau) * (1-exp(-dt/Ti)exp(-dt/tau))
  term3 = tau * tau * (1 - (exp(-dt / Ti) * exp(-dt / tau))) / (Ti + tau)

  gO = BiSqr * theta * Ti
  gO = gO * (term1 - term2 + term3)

  return gO


def OmegaSqr(dt, BiSqr, theta, Ti, tau):

  # (Ti-tau)^2*dt
  BiSqr_thetaSqr = BiSqr * theta * theta
  TiMinTaup = Ti - tau
  term1 = dt * TiMinTaup ** 2.0
  # Ti^3 * 0.5 *[1-exp(-2dt/Ti)]
  term2 = 0.5 * Ti * Ti * Ti * (1.0 - exp(-2.0 * dt / Ti))
  # 0.5 * tau^3 * [1-exp(-2dt/tau)]
  term3 = tau * tau * tau * 0.5 * (1.0 - exp(-2.0 * dt / tau))
  # 2 Ti^2 * (Ti-tau) * [1-exp(-dt/Ti)]
  term4 = 2.0 * Ti * Ti * (Ti - tau) * (1.0 - exp(-dt / Ti))
  # 2 tau^2 * (Ti-tau) * [1-exp(-dt/tau)]
  term5 = 2.0 * tau * tau * (Ti - tau) * (1.0 - exp(-dt / tau))
  # 2 * tau^2 * [Ti^2/(Ti+tau)] * [1-exp(-dt/Ti)exp(-dt/tau)]
  term6 = (
    2.0
    * ((tau * tau * Ti * Ti) / (Ti + tau))
    * (1.0 - (exp(-dt / Ti) * exp(-dt / tau)))
  )

  OmegaSqr = term1 + term2 + term3 - term4 + term5 - term6

  return abs(OmegaSqr * BiSqr * theta * theta)


def GammaOmega(dt, BiSqr, theta, Ti, tau):

  TiMinTaup = Ti - tau
  # Ti [1-exp(-dt/Ti)]
  term1 = Ti * (1.0 - exp(-dt / Ti))
  # tau [ 1-exp(-dt/tau)]
  term2 = tau * (1.0 - exp(-dt / tau))
  # 0.5 Ti^2 [1-exp(-2dt/Ti)]
  term3 = 0.5 * Ti * Ti * (1.0 - exp(-2.0 * dt / Ti))
  # 0.5 tau^2 [1-exp(-2dt/tau)]
  term4 = 0.5 * tau * tau * (1.0 - exp(-2.0 * dt / tau))
  # tau Ti [(1-exp(-dt/Ti) exp(-dt/tau)]
  term5 = tau * Ti * (1.0 - (exp(-dt / Ti) * exp(-dt / tau)))

  GO = (TiMinTaup * (term1 - term2)) - term3 - term4 + term5
  # GO = TiMinTaup*(term1-term2 -term3 - term4 + term5)
  return GO * BiSqr * theta * theta


class IntegralCoeffs:
  """
  Helper class for centralising all calls to integral coeffs. 
  NOTE that due to some numerical/floating point issue in calculating 
  omega_sqr, P22 and P33, we currently take the absolute value...
  """
  def __init__(
    self,
    gamma_lwr_sqr,
    gamma_upr_sqr,
    omega_sqr,
    gamma_lwr_gamma_upr,
    gamma_lwr_omega,
    gamma_upr_omega,
  ):
    self.P11 = sqrt(gamma_lwr_sqr)
    self.P21 = gamma_lwr_omega / sqrt(gamma_lwr_sqr)
    # can get numerical issues for P22 sometimes in OpenFOAM - be careful
    self.P22 = sqrt(abs(omega_sqr - (square(gamma_lwr_omega) / gamma_lwr_sqr)))
    self.P31 = gamma_lwr_gamma_upr / sqrt(gamma_lwr_sqr)
    self.P32 = (1.0 / self.P22) * (gamma_upr_omega - (self.P21 * self.P31))
    self.P33 = sqrt(abs(gamma_upr_sqr - square(self.P31) - square(self.P32)))


if __name__ == "__main__":
  args = get_inputs()
  if args.test_limits:
    case_str_list = []
    tau_p_list = []
    Ti_list = []
    BiSqr_list = []
    theta_list = []
    limit_us_variance = []
    limit_up_variance = []

    dt = 1000.0  # set dt high to get limit of t-> \infty
    # dt = 3.e-3  # same for all cases
    # general case
    case_str = "general case"
    tau_p = 1.0e-1
    Ti = 2.0e-1
    Bi = 1.0e1
    BiSqr = Bi ** 2.0
    theta = Ti / (Ti - tau_p)
    case_str_list.append(case_str)
    tau_p_list.append(tau_p)
    Ti_list.append(Ti)
    BiSqr_list.append(BiSqr)
    theta_list.append(theta)

    # limit case (i): tau_p << dt << Ti
    case_str = "limit case (i): tau_p << dt << Ti"
    tau_p = 1.0e-5
    Ti = 1.0e-1
    Bi = 1.0e1
    BiSqr = Bi ** 2.0
    theta = Ti / (Ti - tau_p)
    case_str_list.append(case_str)
    tau_p_list.append(tau_p)
    Ti_list.append(Ti)
    BiSqr_list.append(BiSqr)
    theta_list.append(theta)

    # limit case (ii): Ti << dt << tau_p
    case_str = "limit case (ii): Ti << dt << tau_p"
    tau_p = 1.0e-1
    Ti = 1.0e-5
    Bi = 1.0e1
    BiSqr = Bi ** 2.0
    theta = Ti / (Ti - tau_p)
    case_str_list.append(case_str)
    tau_p_list.append(tau_p)
    Ti_list.append(Ti)
    BiSqr_list.append(BiSqr)
    theta_list.append(theta)

    # limit case (iii): Ti, tau_p << dt
    case_str = "limit case (iii): Ti, tau_p << dt"
    tau_p = 2.0e-5
    Ti = 1.0e-5
    Bi = 1.0e3
    BiSqr = Bi ** 2.0
    theta = Ti / (Ti - tau_p)
    case_str_list.append(case_str)
    tau_p_list.append(tau_p)
    Ti_list.append(Ti)
    BiSqr_list.append(BiSqr)
    theta_list.append(theta)

    # limit case (iv): Ti -> 0
    case_str = "limit case (iv): Ti -> 0"
    tau_p = 1.0e-1
    Ti = 1.0e-15
    Bi = 1.0e1
    BiSqr = Bi ** 2.0
    theta = Ti / (Ti - tau_p)
    case_str_list.append(case_str)
    tau_p_list.append(tau_p)
    Ti_list.append(Ti)
    BiSqr_list.append(BiSqr)
    theta_list.append(theta)

    for i, (case, tau_p, Ti, BiSqr, theta) in enumerate(
      zip(case_str_list, tau_p_list, Ti_list, BiSqr_list, theta_list)
    ):
      # analytical variances of u_p and u_s moments
      variance_us_analyt = BiSqr * Ti / 2
      variance_up_analyt = BiSqr * 0.5 * Ti * Ti / (Ti + tau_p)

      print(case)
      # calculate stochastic covariances
      gamma_lwr_sqr = gammaLowerSqr(dt, BiSqr, theta, Ti, tau_p)
      gamma_upr_sqr = GammaUpperSqr(dt, BiSqr, theta, Ti, tau_p)
      gamma_lwr_gamma_upr = gammaGamma(dt, BiSqr, theta, Ti, tau_p)
      gamma_lwr_omega = gammaOmega(dt, BiSqr, theta, Ti, tau_p)
      omega_sqr = OmegaSqr(dt, BiSqr, theta, Ti, tau_p)
      gamma_upr_omega = GammaOmega(dt, BiSqr, theta, Ti, tau_p)

      # integral coefficients
      coeffs = IntegralCoeffs(
        gamma_lwr_sqr=gamma_lwr_sqr,
        gamma_upr_sqr=gamma_upr_sqr,
        omega_sqr=omega_sqr,
        gamma_lwr_gamma_upr=gamma_lwr_gamma_upr,
        gamma_lwr_omega=gamma_lwr_omega,
        gamma_upr_omega=gamma_upr_omega,
      )
      P11 = coeffs.P11
      P21 = coeffs.P21
      P22 = coeffs.P22
      P31 = coeffs.P31
      P32 = coeffs.P32
      P33 = coeffs.P33
      variance_us = (P11) ** 2.0
      variance_up = (P31) ** 2 + (P32) ** 2 + (P33) ** 2.0
      if args.debug:
        print("tau", tau_p)
        print("Ti", Ti)
        print("BiSqr", BiSqr)
        print("theta", theta)
        print(f"variance_us {variance_us}")
        print(f"variance_up {variance_up}")

      # check that no nans exist and 2nd order moments agree with analytical vals
      try:
        assert ~np.isnan(variance_up), "u_p is nan"
      except AssertionError as error:
        print("tau", tau_p)
        print("Ti", Ti)
        print("BiSqr", BiSqr)
        print("theta", theta)
        print()
        print("next!")
        print("u_p is nan. Printing all terms")
        print("gamma_lwr_sqr", gamma_lwr_sqr)
        print("gamma_upr_sqr", gamma_upr_sqr)
        print("gamma_lwr_gamma_upr", gamma_lwr_gamma_upr)
        print("gamma_lwr_omega", gamma_lwr_omega)
        print("gamma_upr_omega", gamma_upr_omega)
        print("omega_sqr", omega_sqr)
        print()
        print("P11", P11)
        print("P21", P21)
        print(
          "P22",
          P22,
          f"= sqrt({omega_sqr} - ({square(gamma_lwr_omega)} / {gamma_lwr_sqr}))",
        )
        print("P31", P31)
        print("P32", P32, f"= 1/{P22} ({gamma_upr_omega} - {P21}*{P31})")
        print("P33", P33)
        exit()

      us_variances_equal = np.isclose(
        variance_us, variance_us_analyt, atol=10 ** -2
      )
      up_variances_equal = np.isclose(
        variance_up, variance_up_analyt, atol=10 ** -2
      )

      try:
        assert us_variances_equal, "u_s variances not equal"
      except AssertionError as error:
        print("variance_us", variance_us)
        print("analytical", variance_us_analyt)
      assert us_variances_equal, "u_s variances not equal"

      try:
        assert up_variances_equal, "u_p variances not equal"
        # assert np.isclose(variance_us**2., variance_us_analyt), "u_s variances not equal"
      except AssertionError as error:
        print("variance_up", variance_up)
        print("analytical", variance_up_analyt)
      assert up_variances_equal, "u_p variances not equal"
      print("\t passed!")

  else:
    case_str_list = []
    tau_p_list = []
    Ti_list = []
    BiSqr_list = []
    theta_list = []
    limit_us_variance = []
    limit_up_variance = []

    case_str = "JW debug keras"
    # Ti = 0.7965364
    # tau_p = 0.00010401
    # Bi = 0.6942309
    # dt = 1.8664752e-05


    Ti = 9.599103151503944e-08
    tau_p = 1.23685737e-06
    # Ti, tau_p = tau_p, Ti
    # Ti /= 10000
    # tau_p /= 10000

    Bi = 2191.0093780975717
    dt = 0.06018073
    BiSqr = Bi ** 2.0
    theta = Ti / (Ti - tau_p)
    case_str_list.append(case_str)
    tau_p_list.append(tau_p)
    Ti_list.append(Ti)
    BiSqr_list.append(BiSqr)
    theta_list.append(theta)
    for i, (case, tau_p, Ti, BiSqr, theta) in enumerate(
      zip(case_str_list, tau_p_list, Ti_list, BiSqr_list, theta_list)
    ):
      # analytical variances of u_p and u_s moments
      variance_us_analyt = BiSqr * Ti / 2
      variance_up_analyt = BiSqr * 0.5 * Ti * Ti / (Ti + tau_p)

      print(case)
      # calculate stochastic covariances
      gamma_lwr_sqr = gammaLowerSqr(dt, BiSqr, theta, Ti, tau_p)
      gamma_upr_sqr = GammaUpperSqr(dt, BiSqr, theta, Ti, tau_p)
      gamma_lwr_gamma_upr = gammaGamma(dt, BiSqr, theta, Ti, tau_p)
      gamma_lwr_omega = gammaOmega(dt, BiSqr, theta, Ti, tau_p)
      omega_sqr = OmegaSqr(dt, BiSqr, theta, Ti, tau_p)
      gamma_upr_omega = GammaOmega(dt, BiSqr, theta, Ti, tau_p)

      # integral coefficients
      coeffs = IntegralCoeffs(
        gamma_lwr_sqr=gamma_lwr_sqr,
        gamma_upr_sqr=gamma_upr_sqr,
        omega_sqr=omega_sqr,
        gamma_lwr_gamma_upr=gamma_lwr_gamma_upr,
        gamma_lwr_omega=gamma_lwr_omega,
        gamma_upr_omega=gamma_upr_omega,
      )
      P11 = coeffs.P11
      P21 = coeffs.P21
      P22 = coeffs.P22
      P31 = coeffs.P31
      P32 = coeffs.P32
      P33 = coeffs.P33
      variance_us = (P11) ** 2.0
      variance_up = ((P31) ** 2) + ((P32) ** 2) + ((P33) ** 2.0)

      print("P11", P11)
      # print("P21", P21)
      # print("P22", P22)
      print()
      print("P31", P31)
      print("P32", P32)
      print("P33", P33)
      print("timescale", Ti)
      print("tau_p", tau_p)
      print("theta", theta)
      print("std_dev u_s is ", P11)
      print("std_dev u_p is ", P31+ P32 + P33)
      print(f"variance_us {variance_us}")
      print(f"variance_up {variance_up}")
      # check that no nans exist and 2nd order moments agree with analytical vals
      try:
        assert ~np.isnan(variance_up), "u_p is nan"
      except AssertionError as error:
        print("tau", tau_p)
        print("Ti", Ti)
        print("BiSqr", BiSqr)
        print("theta", theta)
        print()
        print("next!")
        print("u_p is nan. Printing all terms")
        print("gamma_lwr_sqr", gamma_lwr_sqr)
        print("gamma_upr_sqr", gamma_upr_sqr)
        print("gamma_lwr_gamma_upr", gamma_lwr_gamma_upr)
        print("gamma_lwr_omega", gamma_lwr_omega)
        print("gamma_upr_omega", gamma_upr_omega)
        print("omega_sqr", omega_sqr)
        print()
        print("P11", P11)
        print("P21", P21)
        print(
          "P22",
          P22,
          f"= sqrt({omega_sqr} - ({square(gamma_lwr_omega)} / {gamma_lwr_sqr}))",
        )
        print("P31", P31)
        print("P32", P32, f"= 1/{P22} ({gamma_upr_omega} - {P21}*{P31})")
        print("P33", P33)
        exit()
