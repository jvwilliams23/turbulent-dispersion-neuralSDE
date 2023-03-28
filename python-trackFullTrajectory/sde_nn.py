import sys
from copy import copy
from sys import exit
import warnings

import numpy as np
import tensorflow as tf
import tensorflow.keras

from tensorflow.keras import layers
from tensorflow.keras import regularizers

import stochastic_covariances as sc

# For numeric stability, set the default floating-point dtype to float64
tf.keras.backend.set_floatx("float64")


class ModelBuilder:
  """
  Constructs neural network models with specified topology.
  """

  @staticmethod
  def activation_name_to_function(activation_name):
    if activation_name == "leaky_relu":
      activation = tf.nn.leaky_relu
    else:
      activation = activation_name
    return activation

  @staticmethod
  def set_regularization(*args):
    output_regularisations = []
    for i, arg in enumerate(args):
      reg_i = regularizers.l1_l2(l1=arg["l1_reg"], l2=arg["l2_reg"])
      output_regularisations.append(reg_i)
    return output_regularisations

  @staticmethod
  def set_init_weights(init_dict):
    """
    Extract initialiser function from string in dict w/ getattr(init_name)
    """
    init_name = init_dict["init_name"]
    kwargs = {}
    for key, item in init_dict.items():
      if key == "init_name":
        pass
      else:
        kwargs[key] = copy(item)
    # print(f"{**kwargs}")
    init_func = getattr(tf.keras.initializers, init_name)
    return init_func(**kwargs)

  @staticmethod
  def define_neural_network_architecture(
    n_input_dimensions,
    n_output_dimensions,
    n_layers,
    n_dim_per_layer,
    name,
    hidden_layer_weights_dict,
    output_layer_weights_G_dict,
    output_layer_weights_B_dict,
    reg_hidden_layer_drift_bias_kwargs={},
    reg_hidden_layer_diffusion_bias_kwargs={},
    activation_name_drift="relu",
    activation_name_diffusion="relu",
    output_activation_name_drift="softplus",
    output_activation_name_diffusion="softplus",
    dtype=tf.float64,
    debug=True,
    **kwargs,
  ):
    hidden_layer_drift_activation = ModelBuilder.activation_name_to_function(
      activation_name_drift
    )
    hidden_layer_diffusion_activation = (
      ModelBuilder.activation_name_to_function(activation_name_diffusion)
    )
    if "activation_name" in kwargs.keys():
      hidden_layer_drift_activation = ModelBuilder.activation_name_to_function(
        activation_name
      )
      hidden_layer_diffusion_activation = (
        ModelBuilder.activation_name_to_function(activation_name)
      )

    # initialize weights and biases
    initializer_weights = ModelBuilder.set_init_weights(
      hidden_layer_weights_dict
    )
    initializer_weights_out_mean_layer = ModelBuilder.set_init_weights(
      output_layer_weights_G_dict
    )
    initializer_weights_out_std_layer = ModelBuilder.set_init_weights(
      output_layer_weights_B_dict
    )
    initializer_biases = tf.keras.initializers.Zeros()
    # set hidden layer regularisation
    (
      reg_hidden_layer_drift_bias,
      reg_hidden_layer_diffusion_bias,
    ) = ModelBuilder.set_regularization(
      reg_hidden_layer_drift_bias_kwargs, reg_hidden_layer_diffusion_bias_kwargs
    )

    if debug:
      print("\n")
      print(initializer_weights)
      print(initializer_weights_out_mean_layer)
      print(initializer_weights_out_std_layer)
      print("\n")

    # initialize layers of architecture
    input_drift = layers.Input(
      (n_input_dimensions,),
      dtype=dtype,
      name=name + "_inputs",
    )
    input_diffusion = layers.Input(
      (n_input_dimensions,),
      dtype=dtype,
      name=name + "_inputs",
    )

    gp_x = input_drift
    for i in range(n_layers):
      gp_x = layers.Dense(
        n_dim_per_layer,
        activation=hidden_layer_drift_activation,
        kernel_initializer=initializer_weights,
        bias_initializer=initializer_biases,
        use_bias=True,
        # bias_regularizer=reg_hidden_layer_drift_bias,
        # activity_regularizer=regularizers.l1_l2(l1=l1_reg, l2=l2_reg),
        name=name + "_drift_hidden_{}".format(i),
        dtype=dtype,
      )(gp_x)
    gp_output_mean = layers.Dense(
      n_output_dimensions,
      activation=output_activation_name_drift,
      kernel_initializer=initializer_weights_out_mean_layer,
      use_bias=False,
      # activity_regularizer=reg_hidden_layer_drift_bias,
      name=name + "_drift_output",
      dtype=dtype,
    )(gp_x)
    gp_x = input_diffusion
    for i in range(n_layers):
      gp_x = layers.Dense(
        n_dim_per_layer,
        activation=hidden_layer_diffusion_activation,
        kernel_initializer=initializer_weights,
        bias_initializer=initializer_biases,
        use_bias=True,
        # kernel_regularizer=regularizers.l1_l2(l1=l1_reg, l2=l2_reg),
        # bias_regularizer=reg_hidden_layer_diffusion_bias,
        name=name + "_diffusion_hidden_{}".format(i),
        dtype=dtype,
      )(gp_x)
    gp_output_std = layers.Dense(
      n_output_dimensions,
      activation=output_activation_name_diffusion,
      kernel_initializer=initializer_weights_out_std_layer,
      # activity_regularizer=reg_hidden_layer_diffusion_bias,
      use_bias=False,
      name=name + "_diffusion_output",
      dtype=dtype,
    )(gp_x)
    if debug:
      print("ACTIVATIONS ARE")
      print("hidden layer diffusion", hidden_layer_diffusion_activation)
      print("hidden layer drift", hidden_layer_drift_activation)
      print("output layer diffusion", output_activation_name_drift)
      print("output layer drift", output_activation_name_diffusion)

    gp_drift = tf.keras.Model(
      input_drift, gp_output_mean, name=name + "_gaussian_process"
    )
    gp_diffusion = tf.keras.Model(
      input_diffusion, gp_output_std, name=name + "_gaussian_process"
    )
    return gp_drift, gp_diffusion


class neuralSDEModel(tf.keras.Model):
  def __init__(
    self,
    # encoder: tf.keras.Model,
    encoder_drift: tf.keras.Model,
    # encoder_diffusion: tf.keras.Model,
    debug=False,
    DRIFT_MULTIPLIER_CONST=None,
    DIFFUSION_MULTIPLIER_CONST=None,
    DIFFUSION_NORM_TIMESCALE_INDEX=1.0,
    DRIFT_NORM_TIMESCALE_INDEX=1.0,
    train_writer=None,
    number_of_cases=0,
    number_of_timesteps=0,
    **kwargs,
  ):
    super().__init__()
    self.encoder_drift = encoder_drift
    # self.encoder_diffusion = encoder_diffusion
    self.debug = debug
    # try to make keyword argument where user must give input (i.e. no default)
    assert type(DRIFT_MULTIPLIER_CONST) != type(None) and type(
      DIFFUSION_MULTIPLIER_CONST
    ) != type(None), (
      f"DRIFT_MULTIPLIER_CONST ({DRIFT_MULTIPLIER_CONST}) "
      f"or DIFFUSION_MULTIPLIER_CONST ({DIFFUSION_MULTIPLIER_CONST}) not given"
    )
    self.DRIFT_MULTIPLIER_CONST = DRIFT_MULTIPLIER_CONST
    self.DIFFUSION_MULTIPLIER_CONST = DIFFUSION_MULTIPLIER_CONST

    self.DIFFUSION_NORM_TIMESCALE_INDEX = DIFFUSION_NORM_TIMESCALE_INDEX
    self.DRIFT_NORM_TIMESCALE_INDEX = DRIFT_NORM_TIMESCALE_INDEX

    self.train_writer = train_writer
    self.number_of_timesteps = number_of_timesteps
    self.number_of_cases = number_of_cases

    print(
      f"DRIFT_MULTIPLIER_CONST ({DRIFT_MULTIPLIER_CONST}) "
      f"DIFFUSION_MULTIPLIER_CONST ({DIFFUSION_MULTIPLIER_CONST}) "
      f"DIFFUSION_NORM_TIMESCALE_INDEX ({self.DIFFUSION_NORM_TIMESCALE_INDEX}) "
      f"DRIFT_NORM_TIMESCALE_INDEX ({self.DRIFT_NORM_TIMESCALE_INDEX})"
    )
    print(
      f"looping for {self.number_of_timesteps} timesteps and "
      f"{self.number_of_cases} cases"
    )

  def train_step(self, data):
    inputs = data[0]

    step_size_all_cases = inputs[0][0]
    us_n_all_cases = inputs[0][1]
    up_n_all_cases = inputs[0][2]
    uc_all_cases = inputs[0][3]
    timescale_inp_all_cases = inputs[0][4]
    mesh_size_ratio_all_cases = inputs[0][5]
    us_np1_all_cases = inputs[0][6]
    up_np1_all_cases = inputs[0][7]
    viscosity_all_cases = inputs[1][0]
    epsilon_all_cases = inputs[1][1]
    tau_p = 1.0e-6
    offset = 1.0e-9

    loss = 0
    mean_tke_list = []
    out_dict = {}
    with tf.GradientTape() as tape:
      for c_i in range(0, self.number_of_cases):
        step_size_all = step_size_all_cases[:, c_i]
        us_n_all = us_n_all_cases[:, c_i]
        up_n_all = up_n_all_cases[:, c_i]
        uc_all = uc_all_cases[:, c_i]
        timescale_inp_all = timescale_inp_all_cases[:, c_i]
        mesh_size_ratio_all = mesh_size_ratio_all_cases[:, c_i]
        us_np1_all = us_np1_all_cases[:, c_i]
        up_np1_all = up_np1_all_cases[:, c_i]
        viscosity_all = viscosity_all_cases[:, c_i]
        epsilon_all = epsilon_all_cases[:, c_i]

        us_pred_full = us_n_all[:, 0]
        init_tke = tf.reduce_mean(
          tf.reduce_sum(0.5 * tf.multiply(us_n_all[:, 0], us_n_all[:, 0]), axis=1)
        )
        # forward pass NNs to get outputs
        for t_i, _ in enumerate(range(0, self.number_of_timesteps)):
          step_size = step_size_all[:, t_i][:, None]
          if self.debug:
            print(step_size)
            print(step_size_all.shape, step_size.shape)
            print("timescale_inp_all[:, t_i]", timescale_inp_all[:, t_i].shape)
            print("mesh_size_ratio_all[:, t_i]", mesh_size_ratio_all[:, t_i].shape)
          model_inputs = tf.stack(
            (
              timescale_inp_all[:, t_i],
              mesh_size_ratio_all[:, t_i],
            ),
            axis=1,
          )
          drift_nn_output = self.encoder_drift(model_inputs, training=True)
          diffusion_nn_output = 0.0

          # scale NN outputs
          drift = (
            (1 / drift_nn_output)
            * self.DRIFT_MULTIPLIER_CONST
            * (viscosity_all[:, t_i][:, None] ** 0.5)
            / (
              timescale_inp_all[:, t_i][:, None]
              ** self.DRIFT_NORM_TIMESCALE_INDEX
            )
          )
          diffusion = tf.math.sqrt(epsilon_all[:, t_i][:, None])
          # get drift and diffusion, and add checks to prevent NaNs
          drift = tf.where(
            tf.math.is_nan(drift),
            tf.zeros_like(drift) + 1.0e9,
            drift,
          )
          drift_time = 1.0 / drift
          diffusion = tf.where(
            tf.math.is_nan(diffusion),
            tf.zeros_like(diffusion) + 1.0e-6,
            diffusion,
          )
          # for HIT, pressure gradient and velocity laplacian are zero, so we just have
          # fluid velocity in drift term
          drift_times_timescale = copy(uc_all[:, t_i])

          diffusion_sqr = tf.square(diffusion)
          # stochastic covariance matrix coefficients (Minier 2003, table 2)
          theta = drift_time / (drift_time - tau_p)
          D = theta * (
            tf.math.exp(-step_size * drift) - tf.math.exp(-step_size / tau_p)
          )
          E = 1 - tf.math.exp(-step_size / tau_p)
          gamma_lwr_sqr = sc.gammaLowerSqr(
            step_size, diffusion_sqr, theta, drift_time, tau_p
          )
          # stochastic covariance coefficients (Minier 2003, table 3)
          gamma_upr_sqr = sc.GammaUpperSqr(
            step_size, diffusion_sqr, theta, drift_time, tau_p
          )
          gamma_lwr_gamma_upr = sc.gammaGamma(
            step_size, diffusion_sqr, theta, drift_time, tau_p
          )
          gamma_lwr_omega = sc.gammaOmega(
            step_size, diffusion_sqr, theta, drift_time, tau_p
          )
          omega_sqr = sc.OmegaSqr(
            step_size, diffusion_sqr, theta, drift_time, tau_p
          )
          gamma_upr_omega = sc.GammaOmega(
            step_size, diffusion_sqr, theta, drift_time, tau_p
          )
          # integral coefficients
          coeffs = sc.IntegralCoeffs(
            gamma_lwr_sqr=gamma_lwr_sqr,
            gamma_upr_sqr=gamma_upr_sqr,
            omega_sqr=omega_sqr,
            gamma_lwr_gamma_upr=gamma_lwr_gamma_upr,
            gamma_lwr_omega=gamma_lwr_omega,
            gamma_upr_omega=gamma_upr_omega,
          )
          # P11 is standard deviation of u_s (see Minier 2003 or Peirano 2006 scheme)
          P11 = coeffs.P11
          _variance_us = P11 ** 2.0

          # print("us shape", us_n_all[:, t_i].shape)
          # print("step_size shape", step_size.shape)
          # print("drift shape", drift.shape)
          # get deterministic part of u_s at next time step
          us_pred = us_pred_full * tf.math.exp(
            -step_size * drift
          ) + drift_times_timescale * (
            1.0 - tf.math.exp(-step_size * drift)
          )
          # compute mean average error over N=num_samples steps, to minimise random error
          # in stochastic term
          gauss_rand_vec = tf.random.normal(
            tf.shape(us_np1_all[:, t_i]), mean=0, dtype=tf.float64
          )

          # need to split up into each group first if flow configuration different.
          # e.g. If we average tke_us from Re=33 and Re=105, then flow we get some
          # weird average like Re=55 which is incorrect. Therefore, need an additional
          # loop over each `class' for us_truth and us_NN
          # mean_tke_groundtruth = tf.math.reduce_mean(0.5*us_np1_all[:, t_i]**2.0)
          # mean_tke_nn = tf.math.reduce_mean(0.5*(us_pred + rand_vec * P11)**2.0)
          # tmp_i = tf.tensordot(us_np1_all[:, t_i],us_np1_all[:, t_i], axes=-1)
          us_pred_full = us_pred + (P11 * gauss_rand_vec)
          tke_groundtruth = tf.reduce_sum(
            0.5 * tf.multiply(us_np1_all[:, t_i], us_np1_all[:, t_i]), axis=1
          )
          tke_nn = tf.reduce_sum(
            0.5 * tf.multiply(us_pred_full, us_pred_full), axis=1
          )
          mean_tke_groundtruth = tf.math.reduce_mean(tke_groundtruth)
          mean_tke_nn = tf.math.reduce_mean(tke_nn)
          tke_loss = tf.math.abs(mean_tke_groundtruth - mean_tke_nn) / init_tke
          loss += tke_loss
        out_dict[f"mean_tke_{c_i}"] = mean_tke_nn / init_tke
        # out_dict[f"mean_tke_truth_{c_i}"] = mean_tke_groundtruth / init_tke
        if self.debug:
          print("\nmean tke in time", mean_tke_groundtruth, mean_tke_nn)
          print("tke loss", tke_loss, "total loss", loss)


      trainable_vars = self.trainable_variables
      gradients = tape.gradient(loss, trainable_vars)
      # Update weights
      self.optimizer.apply_gradients(zip(gradients, trainable_vars))
      mean_drift_time = tf.reduce_mean(drift_time)
      with self.train_writer.as_default(step=self._train_counter):
        tf.summary.scalar("batch_loss", loss)
        tf.summary.scalar("batch_tke", mean_tke_nn / init_tke)
      # monte carlo loss (minimises large diffusion outliers)
      # when timescale is really high, this can go to zero (we don't want this)

    if self.debug:
      print("ALL LOSS", loss)
    out_dict["loss"] = loss

    if self.debug:
      try:
        assert ~(
          tf.math.reduce_any(tf.math.is_nan(us_pred))
          or tf.math.reduce_any(tf.math.is_nan(_variance_us))
          or tf.math.reduce_any(tf.math.is_inf(us_pred))
          or tf.math.reduce_any(tf.math.is_inf(_variance_us))
        ), "u_pred is nan"
      except AssertionError:
        print("\n\n\n\niterating")
        print("us_pred", us_pred)
        print("us_n", us_n)
        print("us_np1", us_np1)
        print("uc", uc)
        print("step_size", step_size)
        print("drift", drift)
        print("drift_time", drift_time)
        print("diffusion", diffusion)
        print("variance_us", _variance_us)
        exit()

    return out_dict

  def call(self, inputs):
    step_size_all_cases = inputs[0][0]
    us_n_all_cases = inputs[0][1]
    up_n_all_cases = inputs[0][2]
    uc_all_cases = inputs[0][3]
    timescale_inp_all_cases = inputs[0][4]
    mesh_size_ratio_all_cases = inputs[0][5]
    us_np1_all_cases = inputs[0][6]
    up_np1_all_cases = inputs[0][7]
    viscosity_all_cases = inputs[1][0]
    epsilon_all_cases = inputs[1][1]
    tau_p = 1.0e-6
    offset = 1.0e-9

    loss = 0
    for c_i in range(0, self.number_of_cases):
      step_size_all = step_size_all_cases[:, c_i]
      us_n_all = us_n_all_cases[:, c_i]
      up_n_all = up_n_all_cases[:, c_i]
      uc_all = uc_all_cases[:, c_i]
      timescale_inp_all = timescale_inp_all_cases[:, c_i]
      mesh_size_ratio_all = mesh_size_ratio_all_cases[:, c_i]
      us_np1_all = us_np1_all_cases[:, c_i]
      up_np1_all = up_np1_all_cases[:, c_i]
      viscosity_all = viscosity_all_cases[:, c_i]
      epsilon_all = epsilon_all_cases[:, c_i]

      us_pred_full = us_n_all[:, 0]
      init_tke = tf.reduce_mean(
        tf.reduce_sum(0.5 * tf.multiply(us_n_all[:, 0], us_n_all[:, 0]), axis=1)
      )
      # forward pass NNs to get outputs
      for t_i, _ in enumerate(range(0, self.number_of_timesteps)):
        step_size = step_size_all[:, t_i][:, None]
        if self.debug:
          print(step_size)
          print(step_size_all.shape, step_size.shape)
          print("timescale_inp_all[:, t_i]", timescale_inp_all[:, t_i].shape)
          print("mesh_size_ratio_all[:, t_i]", mesh_size_ratio_all[:, t_i].shape)
        model_inputs = tf.stack(
          (
            timescale_inp_all[:, t_i],
            mesh_size_ratio_all[:, t_i],
          ),
          axis=1,
        )
        drift_nn_output = self.encoder_drift(model_inputs, training=True)
        diffusion_nn_output = 0.0

        # scale NN outputs
        drift = (
          (1 / drift_nn_output)
          * self.DRIFT_MULTIPLIER_CONST
          * (viscosity_all[:, t_i][:, None] ** 0.5)
          / (
            timescale_inp_all[:, t_i][:, None]
            ** self.DRIFT_NORM_TIMESCALE_INDEX
          )
        )
        diffusion = tf.math.sqrt(epsilon_all[:, t_i][:, None])
        # get drift and diffusion, and add checks to prevent NaNs
        drift = tf.where(
          tf.math.is_nan(drift),
          tf.zeros_like(drift) + 1.0e9,
          drift,
        )
        drift_time = 1.0 / drift
        diffusion = tf.where(
          tf.math.is_nan(diffusion),
          tf.zeros_like(diffusion) + 1.0e-6,
          diffusion,
        )
        # for HIT, pressure gradient and velocity laplacian are zero, so we just have
        # fluid velocity in drift term
        drift_times_timescale = copy(uc_all[:, t_i])

        diffusion_sqr = tf.square(diffusion)
        # stochastic covariance matrix coefficients (Minier 2003, table 2)
        theta = drift_time / (drift_time - tau_p)
        D = theta * (
          tf.math.exp(-step_size * drift) - tf.math.exp(-step_size / tau_p)
        )
        E = 1 - tf.math.exp(-step_size / tau_p)
        gamma_lwr_sqr = sc.gammaLowerSqr(
          step_size, diffusion_sqr, theta, drift_time, tau_p
        )
        # stochastic covariance coefficients (Minier 2003, table 3)
        gamma_upr_sqr = sc.GammaUpperSqr(
          step_size, diffusion_sqr, theta, drift_time, tau_p
        )
        gamma_lwr_gamma_upr = sc.gammaGamma(
          step_size, diffusion_sqr, theta, drift_time, tau_p
        )
        gamma_lwr_omega = sc.gammaOmega(
          step_size, diffusion_sqr, theta, drift_time, tau_p
        )
        omega_sqr = sc.OmegaSqr(
          step_size, diffusion_sqr, theta, drift_time, tau_p
        )
        gamma_upr_omega = sc.GammaOmega(
          step_size, diffusion_sqr, theta, drift_time, tau_p
        )
        # integral coefficients
        coeffs = sc.IntegralCoeffs(
          gamma_lwr_sqr=gamma_lwr_sqr,
          gamma_upr_sqr=gamma_upr_sqr,
          omega_sqr=omega_sqr,
          gamma_lwr_gamma_upr=gamma_lwr_gamma_upr,
          gamma_lwr_omega=gamma_lwr_omega,
          gamma_upr_omega=gamma_upr_omega,
        )
        # P11 is standard deviation of u_s (see Minier 2003 or Peirano 2006 scheme)
        P11 = coeffs.P11
        _variance_us = P11 ** 2.0

        # print("us shape", us_n_all[:, t_i].shape)
        # print("step_size shape", step_size.shape)
        # print("drift shape", drift.shape)
        # get deterministic part of u_s at next time step
        us_pred = us_pred_full * tf.math.exp(
          -step_size * drift
        ) + drift_times_timescale * (
          1.0 - tf.math.exp(-step_size * drift)
        )
        # compute mean average error over N=num_samples steps, to minimise random error
        # in stochastic term
        gauss_rand_vec = tf.random.normal(
          tf.shape(us_np1_all[:, t_i]), mean=0, dtype=tf.float64
        )

        # need to split up into each group first if flow configuration different.
        # e.g. If we average tke_us from Re=33 and Re=105, then flow we get some
        # weird average like Re=55 which is incorrect. Therefore, need an additional
        # loop over each `class' for us_truth and us_NN
        # mean_tke_groundtruth = tf.math.reduce_mean(0.5*us_np1_all[:, t_i]**2.0)
        # mean_tke_nn = tf.math.reduce_mean(0.5*(us_pred + rand_vec * P11)**2.0)
        # tmp_i = tf.tensordot(us_np1_all[:, t_i],us_np1_all[:, t_i], axes=-1)
        us_pred_full = us_pred + (P11 * gauss_rand_vec)
        tke_groundtruth = tf.reduce_sum(
          0.5 * tf.multiply(us_np1_all[:, t_i], us_np1_all[:, t_i]), axis=1
        )
        tke_nn = tf.reduce_sum(
          0.5 * tf.multiply(us_pred_full, us_pred_full), axis=1
        )
        mean_tke_groundtruth = tf.math.reduce_mean(tke_groundtruth)
        mean_tke_nn = tf.math.reduce_mean(tke_nn)
        tke_loss = tf.math.abs(mean_tke_groundtruth - mean_tke_nn) / init_tke
        loss += tke_loss


    if self.debug:
      print("ALL LOSS", loss)

    self.add_loss(loss)

    if self.debug:
      try:
        assert ~(
          tf.math.reduce_any(tf.math.is_nan(us_pred))
          or tf.math.reduce_any(tf.math.is_nan(_variance_us))
          or tf.math.reduce_any(tf.math.is_inf(us_pred))
          or tf.math.reduce_any(tf.math.is_inf(_variance_us))
        ), "u_pred is nan"
      except AssertionError:
        print("\n\n\n\niterating")
        print("us_pred", us_pred)
        print("us_n", us_n)
        print("us_np1", us_np1)
        print("uc", uc)
        print("step_size", step_size)
        print("drift", drift)
        print("drift_time", drift_time)
        print("diffusion", diffusion)
        print("variance_us", _variance_us)
        exit()

    return drift_nn_output


class SDEIntegrator:
  """
  Template class used for integrating SDE for validation
  """

  def __init__(
    self,
    model,
    us_n_inds,
    timescale_inp_inds,
    uc_inds,
    up_n_inds,
    mesh_size_ind,
    viscosity=1.57e-5,
    DRIFT_MULTIPLIER_CONST=None,
    DIFFUSION_MULTIPLIER_CONST=None,
    DIFFUSION_NORM_TIMESCALE_INDEX=1.0,
    DRIFT_NORM_TIMESCALE_INDEX=1.0,
  ):
    self.model = copy(model)
    self.timescale_inp_inds = timescale_inp_inds
    self.mesh_size_ind = mesh_size_ind
    self.us_n_inds = us_n_inds
    self.up_n_inds = up_n_inds
    self.uc_inds = uc_inds
    # try to make keyword argument where user must give input (i.e. no default)
    assert type(DRIFT_MULTIPLIER_CONST) != type(None) and type(
      DIFFUSION_MULTIPLIER_CONST
    ) != type(None), (
      f"DRIFT_MULTIPLIER_CONST ({DRIFT_MULTIPLIER_CONST}) "
      f"or DIFFUSION_MULTIPLIER_CONST ({DIFFUSION_MULTIPLIER_CONST}) not given"
    )
    self.DRIFT_MULTIPLIER_CONST = DRIFT_MULTIPLIER_CONST
    self.DIFFUSION_MULTIPLIER_CONST = DIFFUSION_MULTIPLIER_CONST
    self.DIFFUSION_NORM_TIMESCALE_INDEX = DIFFUSION_NORM_TIMESCALE_INDEX
    self.DRIFT_NORM_TIMESCALE_INDEX = DRIFT_NORM_TIMESCALE_INDEX

    self.VISCOSITY = viscosity

    print(
      f"DRIFT_MULTIPLIER_CONST ({DRIFT_MULTIPLIER_CONST}) "
      f"DIFFUSION_MULTIPLIER_CONST ({DIFFUSION_MULTIPLIER_CONST}) "
      f"DIFFUSION_NORM_TIMESCALE_INDEX ({self.DIFFUSION_NORM_TIMESCALE_INDEX}) "
      f"DRIFT_NORM_TIMESCALE_INDEX ({self.DRIFT_NORM_TIMESCALE_INDEX})"
    )

  def sample_path(
    self,
    initial_velocity,
    step_size,
    number_of_timesteps,
    number_of_particles,
    mesh_size_all,
    timescale_all,
    fluid_vel_all,
    us_time_start,
    up_time_start,
    epsilon_all,
    pairs=False,
    tau_p_all=None,
    debug=False,
    no_model=False,
    return_nn_output=False,
    **kwargs,
  ):
    """
    Use the neural network to sample a path with the Euler Maruyama scheme.

    Parameters
    ----------
    *args (np.ndarray, n_time, n_particle, n_dim): additional fields given
    to model for training, such as epsilon, k, grad(U)
    """
    # can provide an initial condition for each particle
    # or some constant initial condition applied to all
    np.random.seed(20)
    paths = [initial_velocity.copy()]
    drift_time_list = []
    diffusion_list = []
    for it in range(number_of_timesteps):
      current_step_size = step_size[:, it][:, np.newaxis]
      # if pairs:
      #   us_n = us_time_start[:, it]
      #   up_n = up_time_start[:, it]
      # else:
      us_n = paths[-1][:, self.us_n_inds]
      up_n = paths[-1][:, self.up_n_inds]
      tau_p = copy(tau_p_all)
      timescale_inp = timescale_all[:, it]
      mesh_size = mesh_size_all[:, it]
      epsilon_t = epsilon_all[:, it]
      uc = fluid_vel_all[:, it]

      # get neural network inputs and do forward-pass
      model_inputs = tf.stack(
        (
          timescale_inp,
          mesh_size,
        ),
        axis=1,
      )
      drift_nn_output = self.model.encoder_drift(model_inputs)
      diffusion_nn_output = 0.0  # self.model.encoder_diffusion(model_inputs)

      # get model terms from NN outputs
      drift = (
        (1 / drift_nn_output)
        * self.DRIFT_MULTIPLIER_CONST
        * (self.VISCOSITY ** 0.5)
        / (timescale_inp[:, None] ** self.DRIFT_NORM_TIMESCALE_INDEX)
      )
      drift_time = 1.0 / (drift + 1.0e-20)

      # diffusion = (
      #   diffusion_nn_output
      #   * self.DIFFUSION_MULTIPLIER_CONST
      #   * mesh_size[:, None] ** self.DIFFUSION_NORM_TIMESCALE_INDEX
      #   * self.VISCOSITY
      # )
      diffusion = np.sqrt(epsilon_t[:, None])
      diffusion_sqr = tf.square(diffusion)

      # get parameters for time-integration scheme
      theta = (drift_time) / (drift_time - tau_p + 1.0e-7)
      theta = np.where(theta == np.inf, 1, theta)
      assert ~np.isinf(theta).any(), "theta is nan"
      # Stochastic Wiener process
      gauss_vec1 = np.random.normal(size=(number_of_particles, 3))
      gauss_vec2 = np.random.normal(size=(number_of_particles, 3))
      gauss_vec3 = np.random.normal(size=(number_of_particles, 3))
      if no_model:
        gauss_vec1 *= 0.0
        gauss_vec2 *= 0.0
        gauss_vec3 *= 0.0
      if debug:
        print(f"step {it}")
        print(
          f"\tdrift_nn_output is {np.mean(drift_nn_output)} {np.median(drift_nn_output)} {np.max(drift_nn_output)}"
        )
        print(
          f"\tdiffusion_norm is {np.mean(diffusion_nn_output)} {np.median(diffusion_nn_output)} {np.max(diffusion_nn_output)}"
        )
        print(f"\tdrift is {np.mean(drift)} {np.median(drift)}")
        print(
          f"\tdrift timescale is {np.mean(drift_time)} {np.median(drift_time)}"
        )
        print(f"\tdiffusion is {np.mean(diffusion)} {np.median(diffusion)}")
        print(f"\tmodel inputs are {np.mean(model_inputs, axis=0)}")
        print(f"\tfluid vel is {np.mean(abs(uc), axis=0)}")
        print(f"\tvel seen is {np.mean(abs(us_n), axis=0)}")
        print(f"\tparticle vel is {np.mean(abs(up_n), axis=0)}")
        print(f"\tdt = {np.mean(current_step_size, axis=0)} s")
        print(f"\tdt/Tf = {np.mean(current_step_size/drift_time, axis=0)}")
        print(f"\ttheta = {np.mean(theta, axis=0)}")
        print(
          f"\tterm 1 = {np.mean(abs(us_n) * np.exp(-current_step_size * drift))}"
        )
        print(
          f"\tterm 2 = {np.mean(abs(uc) * (1.0 - np.exp(-current_step_size * drift)))}"
        )

      # get integral terms
      D = theta * (
        tf.math.exp(-current_step_size * drift)
        - tf.math.exp(-current_step_size / tau_p)
      )
      E = 1 - tf.math.exp(-current_step_size / tau_p)
      # stochastic covariance matrix coefficients (Minier 2003, table 2)
      gamma_lwr_sqr = sc.gammaLowerSqr(
        current_step_size, diffusion_sqr, theta, drift_time, tau_p
      )
      gamma_upr_sqr = sc.GammaUpperSqr(
        current_step_size, diffusion_sqr, theta, drift_time, tau_p
      )
      gamma_lwr_gamma_upr = sc.gammaGamma(
        current_step_size, diffusion_sqr, theta, drift_time, tau_p
      )
      gamma_lwr_omega = sc.gammaOmega(
        current_step_size, diffusion_sqr, theta, drift_time, tau_p
      )
      omega_sqr = sc.OmegaSqr(
        current_step_size, diffusion_sqr, theta, drift_time, tau_p
      )
      gamma_upr_omega = sc.GammaOmega(
        current_step_size, diffusion_sqr, theta, drift_time, tau_p
      )

      # integral coefficients
      coeffs = sc.IntegralCoeffs(
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
      fluctuation_us = P11 * gauss_vec1
      fluctuation_up = P31 * gauss_vec1 + P33 * gauss_vec3 + P32 * gauss_vec2

      if debug:
        print(f"\tu_s variance = {np.mean(abs(fluctuation_us), axis=0)}")
        print(f"\tu_p variance = {np.mean(abs(fluctuation_up), axis=0)} s")

      # integrate u_s and u_p in time
      us_np1 = (
        us_n * np.exp(-current_step_size * drift)
        + ((uc) * (1.0 - np.exp(-current_step_size * drift)))
        + fluctuation_us
      )

      up_np1 = (
        (up_n * tf.math.exp(-current_step_size / tau_p))
        + (D * us_n)
        + ((uc) * (E - D))
        + fluctuation_up
      )

      if no_model:
        # test the effect of u_s = filtered velocity
        us_n = copy(uc)
        # equation 78 in Minier 2015 review Prog. Energy + Combustion Sci
        up_np1 = (up_n * tf.math.exp(-current_step_size / tau_p)) + uc * (
          1.0 - tf.math.exp(-current_step_size / tau_p)
        )
      if debug:
        print(f"\tvel seen next is {np.mean(abs(us_np1), axis=0)}")
        print(f"\tparticle vel next is {np.mean(abs(up_np1), axis=0)}")

      u_next = np.hstack((us_np1, up_np1))

      drift_time_list.append(drift_time)
      diffusion_list.append(diffusion)
      paths.append(tf.keras.backend.eval(u_next))

    if return_nn_output:
      return (
        [
          np.row_stack([paths[k][i] for k in range(len(paths))])
          for i in range(number_of_particles)
        ],
        drift_time_list,
        diffusion_list,
      )
    else:
      return [
        np.row_stack([paths[k][i] for k in range(len(paths))])
        for i in range(number_of_particles)
      ]
