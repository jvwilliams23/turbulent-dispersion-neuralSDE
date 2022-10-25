"""
Read OpenFOAM Lagrangian data and train stochastic NN to predict velocity
of fluid seen by particles.

"""
import argparse
import hjson
import os
from distutils.util import strtobool
from copy import copy
from datetime import datetime
from time import time
from sys import exit

from sklearn.utils import shuffle
import numpy as np
import pandas as pd
from shutil import rmtree

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # or any {'0', '1', '2'}

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model

import neuralSDE_utils as utils
from sde_nn import ModelBuilder, neuralSDEModel, SDEIntegrator

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
  "-q",
  "--quiet",
  default=False,
  action="store_true",
  help="Turn off print checks",
)
parser.add_argument(
  "-d",
  "--debug",
  default=False,
  action="store_true",
  help="Turn on all print checks",
)
parser.add_argument(
  "-rt",
  "--retrain",
  default=False,
  action="store_true",
  # type=strtobool,
  help="Retrain model. If false, uses previously saved weights",
)
parser.add_argument(
  "--restart",
  default=False,
  action="store_true",
  help="Restart training from last saved checkpoint.",
)
parser.add_argument(
  "--load_data_only",
  "-load",
  default=False,
  action="store_true",
  help="only load input data. Do not perform optimisation.",
)
parser.add_argument(
  "-e",
  "--epochs",
  default=100,
  type=int,
  help="Number of training iterations",
)
parser.add_argument(
  "-foam",
  "--save_openfoam_model",
  default=False,
  action="store_true",
  help="Save model weights and architecture for openfoam simulations",
)
parser.add_argument(
  "-wd",
  "--weight_dir",
  default="weights",
  type=str,
  help="Directory to write weights if re-training",
)
parser.add_argument(
  "-wbd",
  "--weight_base_dir",
  default="all_weights",
  type=str,
  help="Directory to save weights from various tests",
)
parser.add_argument(
  "-fd",
  "--fig_dir",
  default="",
  type=str,
  help="Directory to read or write figures to",
)
parser.add_argument(
  "-tc",
  "-tf",
  "--test_filt",
  default="7x",
  choices=["DNS", "3x", "5x", "7x", "9x", "11x", "48"],
  type=str,
  help="Test case to run",
)
parser.add_argument(
  "-tcs",
  "-st",
  "--test_case_stokes",
  default="0",
  choices=["0", "0.1", "0.5", "1", "1.0", "1.25", "5", "5.0"],
  type=str,
  help="Test case to run",
)
parser.add_argument(
  "-np",
  "--num_p",
  default=100,
  type=int,
  help=("Number of particles to train with in each case. "),
)
parser.add_argument(
  "-early",
  "--early_stopping",
  default=200,
  type=int,
  help=(
    "Patience value for early stopping. "
    "By default it is a really big value (same as turning it off)."
  ),
)
parser.add_argument(
  "-lr",
  "--learning_rate",
  default=0,
  type=float,
  help="Learning rate for training. If none given, reads from config file",
)
parser.add_argument(
  "-doff",
  "--diffusion_off",
  default=False,
  action="store_true",
  help="Set diffusion term off when True",
)
parser.add_argument(
  "-p",
  "--plots",
  default="True",
  type=strtobool,
  help="Save plots",
)
parser.add_argument(
  "-dp",
  "--debug_plots",
  default=False,
  action="store_true",
  help="Save input plots",
)
parser.add_argument(
  "-ext",
  "--extension",
  default="png",
  type=str,
  help="File extension for plots",
)
parser.add_argument(
  "-c",
  "--config",
  default="config.json",
  type=str,
  help="Config file to use",
)
parser.add_argument(
  "-t",
  "--test_dict_key",
  default="HIT_Re30",
  type=str,
  help=(
    "Key for config.json entry to use for testing -> "
    "config['simulations'][test_dict_key]"
  ),
)
parser.add_argument(
  "-pair",
  "--test_pairs",
  default="True",
  type=strtobool,  # "store_true",
  help="If False, test with write-time intervals. Else, test with pairs",
)

args = parser.parse_args()

with open(args.config) as f:
  config = hjson.load(f)

random_state = 84
tf.random.set_seed(random_state)

# assumes that data from "http://doi.org/10.34740/KAGGLE/DSV/3998403" 
# is downloaded to "dataset-filteredDNS"
data_dir_base = "dataset-filteredDNS/"
assert os.path.isdir(data_dir_base), f"is kaggle dataset downloaded to {data_dir_base}?"


assert (
  args.test_dict_key in config["simulations"].keys()
), f"test_dict_key = {args.test_dict_key} not found in config = {args.config}"


quiet = False

# Training parameters
if args.learning_rate != 0.0:
  LEARNING_RATE = args.learning_rate
else:
  LEARNING_RATE = config["network"]["train"]["learning_rate"]

N_EPOCHS = args.epochs

config["network"]["train"]["HIDDEN_LAYER_WEIGHTS_INIT_DICT"][
  "seed"
] = random_state
config["network"]["train"]["OUTPUT_LAYER_WEIGHTS_G_INIT_DICT"][
  "seed"
] = random_state
config["network"]["train"]["OUTPUT_LAYER_WEIGHTS_B_INIT_DICT"][
  "seed"
] = random_state

test_dict = config["simulations"][args.test_dict_key]

# find sub-dictionaries where "simulations" = True
if args.retrain:
  subdict_to_delete = []
  for key, subdict in config["simulations"].items():
    if subdict["train"] == False:
      subdict_to_delete.append(key)
  for key in subdict_to_delete:
    del config["simulations"][key]

# input normalisations
TIMESCALE_NORMALISATION = config["normalisation"]["TIMESCALE_NORMALISATION"]
MESH_SIZE_NORMALISATION = config["normalisation"]["MESH_SIZE_NORMALISATION"]
# network normalisations
DRIFT_MULTIPLIER_CONST = config["normalisation"]["DRIFT_MULTIPLIER"]
DIFFUSION_MULTIPLIER_CONST = config["normalisation"]["DIFFUSION_MULTIPLIER"]

NORMALISE_X_AXIS_IN_FIGURE = config["plotting"]["NORMALISE_X_AXIS_IN_FIGURE"]
if NORMALISE_X_AXIS_IN_FIGURE:
  PLOT_TIME_NORMALISATION = test_dict["TAU_L_LARGESCALE_256"]
else:
  PLOT_TIME_NORMALISATION = 1.0

if config["network"]["train"]["OPTIMIZER"] == "Adam":
  OPTIMIZER = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
elif config["network"]["train"]["OPTIMIZER"] == "SGD":
  OPTIMIZER = tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE)

# I/O info
WEIGHTS_DIR = f"{args.weight_base_dir}/{args.weight_dir}/"
WEIGHTS_PATH = f"{WEIGHTS_DIR}/weights3d.ckpt"
LOG_DIR = f"all_logs/{args.weight_dir.replace('weights', 'logs')}/"
if args.fig_dir == "":
  FIG_DIR = f"all_figures/{args.weight_dir.replace('weights', 'figures')}/"
else:
  FIG_DIR = f"all_figures/{args.fig_dir}/"

# make directory for figures if not existing
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(WEIGHTS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# make directory for figures if not existing
if not os.path.isdir(FIG_DIR):
  os.mkdir(FIG_DIR)

if args.diffusion_off:
  DIFFUSION_TERM_STR = "diffusionOff"
else:
  DIFFUSION_TERM_STR = "diffusionOn"

test_filt_int = int(args.test_filt[:-1])

DOMAIN_SIZE = np.array(test_dict["domain_dimensions"])
NUM_CELLS = np.array(test_dict["mesh_spacing"])

# test_inputs = test_inputs_list
mesh_size_test = np.product(
  (DOMAIN_SIZE / (NUM_CELLS / float(test_filt_int)))
) ** (1 / 3)

# extract OpenFOAM training data
print("get foam data")
us_inds = np.array([0, 1, 2])
up_inds = np.array([3, 4, 5])
epsilon_ind = 6
k_ind = 7
uc_inds = [8, 9, 10]

deltaT_ind = 11

mesh_size_ind = copy(epsilon_ind)
timescale_inds = copy(k_ind)


def read_filtered_data_csv(
  fname, max_num_p=10000000, use_dns_deltat=True, cutoff_time=100000
):
  """
  Read filtered DNS dataset from csv file with pandas
  Some formatting is done to arange the data into times, time-steps,
  features (x_data) and targets (y_data)
  """
  df = pd.read_csv(fname, index_col=0)
  num_p = np.unique(df["ID"]).size
  useen = np.array(df[["ufx DNS t", "ufy DNS t", "ufz DNS t"]])
  ufluid = np.array(df[["ufx filter t", "ufy filter t", "ufz filter t"]])
  uparticle = np.array(df[["upx DNS t", "upy DNS t", "upz DNS t"]])
  epsilon = np.array(df["epsilon_sgs t"])
  k = np.array(df["k_sgs t"])
  times_arr_base = np.array(df["time"])
  # if training:
  #   step_sizes = np.array(df["deltaT write"])
  # else:
  #   step_sizes = np.array(df["deltaT DNS"])
  if use_dns_deltat:
    step_sizes = np.array(df["deltaT DNS"])
  else:
    step_sizes = np.array(df["deltaT write"])

  x_data = np.hstack(
    (
      useen,
      uparticle,
      epsilon[:, None],
      k[:, None],
      ufluid,
      step_sizes[:, None],
      k[:, None],
      k[:, None],
    )
  )
  if use_dns_deltat:
    y_data = np.array(df)[:, -6:]
  # reshape data to num_steps, num_p, num_features
  times_arr_base = times_arr_base.reshape(-1, num_p)[:, :max_num_p]
  step_sizes = step_sizes.reshape(-1, num_p)[:, :max_num_p]
  x_data = x_data.reshape(-1, num_p, 14)[:, :max_num_p]
  # y_data = np.array(df)[:, -6:]

  if use_dns_deltat:
    y_data = y_data.reshape(-1, num_p, 6)[:, :max_num_p]
  else:
    times_arr_base = times_arr_base[:-1]
    step_sizes = step_sizes[:-1]
    # when using write-time as deltaT, we take initial data point for us and up
    # from next time step and use it as the target
    y_data = x_data[1:, :, :6]
    x_data = x_data[:-1]

  return (
    times_arr_base[:cutoff_time],
    step_sizes[:cutoff_time],
    x_data[:cutoff_time],
    y_data[:cutoff_time],
  )


def format_data(
  times_arr_base,
  step_sizes,
  x_data,
  y_data,
  viscosity,
  # viscosity=1.57e-5,
  mesh_size=6.28 / 256,
  train=True,
):
  """
    Format data arrays read from read_foam_data correct for training or testing.
    This includes normalising features, swapping particle-time axes and sorting
    into data at t=n (x_interval) and t=n+dt (y_interval)

    Parameters
    ----------
    times_arr_base (np.ndarray, shape=) : time step of each measurment
    step_sizes (np.ndarray, shape=) : time interval for each measurement
    x_data (np.ndarray, shape=(time, num_p, num_features)): training data for NN
    y_data (np.ndarray, shape=(time, num_p, num_out_feat)): data to fit NN to
    mesh_size (float): cell size (for HIT it is uniform in domain)

    Returns (all np.ndarray)
    -------
    time_interval : deltaT, time_{t+1}-time_t
    x_data : input training data
    x_interval : input training data for time_t
    y_data : output training data
    y_interval : output training data for time_{t+1}
    """

  # set subgrid properties to absolute value
  x_data[:, :, k_ind] = abs(x_data[:, :, k_ind])
  x_data[:, :, epsilon_ind] = abs(x_data[:, :, epsilon_ind])
  tke = copy(np.swapaxes(x_data[:, :, k_ind], 0, 1))
  eps = copy(np.swapaxes(x_data[:, :, epsilon_ind], 0, 1))
  assert np.all(eps > 0) & np.all(tke > 0), "tke or epsilon <= 0"

  timescale_sgs = np.swapaxes(
    x_data[:, :, k_ind] / x_data[:, :, epsilon_ind], 0, 1
  ).copy()

  # convert data at each time point to an interval
  # i.e. d u_s = drift (dx) dt * diffusion
  time_interval = []
  x_interval = []
  y_interval = []
  times_arr = []
  # get particle injection time assuming all particles enter at same time
  for time_step in range(0, len(x_data)):
    time_interval.append(x_data[time_step][:, deltaT_ind])
    x_interval.append(copy(x_data[time_step]))
    y_interval.append(copy(y_data[time_step]))
    times_arr.append(copy(times_arr_base[time_step]))

  # reshape intervals to correct format
  time_interval = np.swapaxes(time_interval, 0, 1)
  x_interval = np.swapaxes(x_interval, 0, 1).copy()
  y_interval = np.swapaxes(y_interval, 0, 1).copy()
  # reshape all arrays from (num_times, num_particles, num_axes)
  # to (num_particles, num_times, num_axes)
  times_arr = np.swapaxes(times_arr, 0, 1)
  step_sizes = np.swapaxes(step_sizes, 0, 1)
  x_data = np.swapaxes(x_data, 0, 1).copy()
  y_data = np.swapaxes(y_data, 0, 1)
  # epsilon from all times
  epsilon_all = x_data[:, :, epsilon_ind].copy()

  # get kolmogorov scales for non-dimensionalising
  timescale_kolmogorov_timeAll = (viscosity / epsilon_all) ** 0.5  # * 0 + 1
  # set normalisation to 0
  length_kolmogorov_timeAll = (viscosity**3 / epsilon_all) ** 0.25
  print(
    "x_data", x_data[0, 0, uc_inds], "x_interval", x_interval[0, 0, uc_inds]
  )

  timescale_sgs_norm = timescale_sgs / timescale_kolmogorov_timeAll
  length_scale_norm = mesh_size / length_kolmogorov_timeAll

  x_interval[:, :, timescale_inds] = (
    timescale_sgs_norm / TIMESCALE_NORMALISATION
  )
  x_data[:, :, timescale_inds] = timescale_sgs_norm / TIMESCALE_NORMALISATION

  # if args.debug:
  print("max timescale \t max timescale norm \t timescale norm with scaler")
  print(
    timescale_sgs.max(),
    timescale_sgs_norm.max(),
    timescale_sgs_norm.max() / TIMESCALE_NORMALISATION,
  )

  # overwrite epsilon col in input with delta/length scale
  x_interval[:, :, mesh_size_ind] = length_scale_norm / MESH_SIZE_NORMALISATION
  x_data[:, :, mesh_size_ind] = length_scale_norm / MESH_SIZE_NORMALISATION

  print("mesh size norm with scaler")
  print(x_data[:, :, mesh_size_ind].max())

  print("max vel is ", x_data[:, :, us_inds[1:]].max())

  return (
    time_interval,
    x_data,
    x_interval,
    y_data,
    y_interval,
  )

train_list_len = 0
num_cases_per_simulation = np.zeros(len(config["simulations"].values())).astype(
  int
)
for i, subdict in enumerate(config["simulations"].values()):
  for delta_filt in subdict["FILTER_TRAIN"]:
    train_list_len += 1
    num_cases_per_simulation[i] += 1

if args.retrain:
  times_arr_base = [None] * train_list_len
  step_sizes = [None] * train_list_len
  x_data = [None] * train_list_len
  y_data = [None] * train_list_len
  time_interval = [None] * train_list_len
  x_interval = [None] * train_list_len
  y_interval = [None] * train_list_len
  viscosity_list = [None] * train_list_len
  i = 0
  for subdict in config["simulations"].values():
    max_num_p_for_case_h = subdict["num_p"]
    for delta_filt in subdict["FILTER_TRAIN"]:
      t0 = time()

      DOMAIN_SIZE_i = np.array(subdict["domain_dimensions"])
      NUM_CELLS_i = np.array(subdict["mesh_spacing"])
      mesh_size_i = np.product(
        (DOMAIN_SIZE_i / (NUM_CELLS_i / float(delta_filt)))
      ) ** (1 / 3)
      if args.debug:
        print(
          "mesh size is ",
          mesh_size_i,
          f"based on domain {DOMAIN_SIZE_i} with {NUM_CELLS_i} cells",
          f"and {delta_filt} filter",
        )

      (
        times_arr_base[i],
        step_sizes[i],
        x_data[i],
        y_data[i],
      ) = read_filtered_data_csv(
        f"{data_dir_base}/{subdict['data_file'].format(filter_width=delta_filt)}",
        max_num_p=max_num_p_for_case_h,
        use_dns_deltat=False,
        cutoff_time=subdict["CUTOFF_TRAIN_TIME"],
      )

      # format data so columns match variables, and data is normalised
      (
        time_interval[i],
        x_data[i],
        x_interval[i],
        y_data[i],
        y_interval[i],
      ) = format_data(
        times_arr_base[i],
        step_sizes[i],
        x_data[i],
        y_data[i],
        mesh_size=mesh_size_i,
        viscosity=subdict["viscosity"],
      )
      if not args.quiet:
        print("x_data has shape", x_data[i].shape)
      # save max characteristic timescale for HIT
      viscosity_list[i] = (
        np.ones(x_data[i][:, :, -1].shape) * subdict["viscosity"]
      )
      i += 1

  time_interval_list = []
  x_interval_list = []
  y_interval_list = []
  viscosity_arr = []
  for j, (t, x, y) in enumerate(zip(time_interval, x_interval, y_interval)):
    time_interval_list.append(np.hstack(t))
    x_interval_list.append(np.row_stack(x))
    y_interval_list.append(np.row_stack(y))
    viscosity_arr.append(np.hstack(viscosity_list[j]))
    print(t.shape)
    print(time_interval_list[-1].shape)
    print()

  time_interval = np.hstack(time_interval_list)
  x_interval = np.vstack(x_interval_list)
  y_interval = np.vstack(y_interval_list)
  viscosity_arr = np.hstack(viscosity_arr)

  step_sizes_train = copy(time_interval)
  x_train = copy(x_interval)
  y_train = copy(y_interval)

  x_train = np.row_stack(x_train)
  y_train = np.row_stack(y_train)

  step_sizes_train = np.array(step_sizes_train)
  step_sizes_train = step_sizes_train.ravel()

  print(f"getting training data took {round(time()-t0, 4)} seconds.")
  t0 = time()

# base number of inputs
num_model_inputs = 2
print(f"training network with {num_model_inputs} inputs")

# define hyper-parameters
encoder_drift, encoder_diffusion = ModelBuilder.define_neural_network_architecture(
  n_input_dimensions=num_model_inputs,
  n_output_dimensions=1,
  n_layers=config["network"]["NUM_LAYERS"],
  n_dim_per_layer=config["network"]["NUM_NODES_PER_LAYER"],
  hidden_layer_weights_dict=config["network"]["train"][
    "HIDDEN_LAYER_WEIGHTS_INIT_DICT"
  ],
  output_layer_weights_G_dict=config["network"]["train"][
    "OUTPUT_LAYER_WEIGHTS_G_INIT_DICT"
  ],
  output_layer_weights_B_dict=config["network"]["train"][
    "OUTPUT_LAYER_WEIGHTS_B_INIT_DICT"
  ],
  reg_hidden_layer_drift_bias_kwargs=config["network"]["train"][
    "reg_hidden_layer_drift_bias"
  ],
  reg_hidden_layer_diffusion_bias_kwargs=config["network"]["train"][
    "reg_hidden_layer_diffusion_bias"
  ],
  name="GP",
  activation_name_drift=config["network"]["HIDDEN_LAYER_ACTIVATION_DRIFT"],
  activation_name_diffusion=config["network"][
    "HIDDEN_LAYER_ACTIVATION_DIFFUSION"
  ],
  output_activation_name_drift=config["network"][
    "OUTPUT_LAYER_ACTIVATION_DRIFT"
  ],
  output_activation_name_diffusion=config["network"][
    "OUTPUT_LAYER_ACTIVATION_DIFFUSION"
  ],
  seed=random_state,
)
# step size will be given during training
model = neuralSDEModel(
  debug=args.debug,
  encoder_drift=encoder_drift,
  encoder_diffusion=encoder_diffusion,
  DRIFT_MULTIPLIER_CONST=DRIFT_MULTIPLIER_CONST,
  DIFFUSION_MULTIPLIER_CONST=DIFFUSION_MULTIPLIER_CONST,
  DIFFUSION_NORM_TIMESCALE_INDEX=config["normalisation"][
    "DIFFUSION_NORM_TIMESCALE_INDEX"
  ],
  DRIFT_NORM_TIMESCALE_INDEX=config["normalisation"][
    "DRIFT_NORM_TIMESCALE_INDEX"
  ],
)
model.compile(
  optimizer=OPTIMIZER,
  run_eagerly=args.debug,
)

# save architecture to json for converting to cpp
if args.save_openfoam_model:
  with open("./cpp_models/encoder_drift_arch.json", "w") as fout:
    fout.write(encoder_drift.to_json())
  with open("./cpp_models/encoder_diffusion_arch.json", "w") as fout:
    fout.write(encoder_diffusion.to_json())

# initialise path to read weights from, or save weights to if re-training
checkpoint_path = WEIGHTS_PATH
checkpoint_dir = os.path.dirname(checkpoint_path)

if args.retrain:
  if not args.quiet:
    print("x_train has shape", x_train.shape)
  if not args.quiet:
    print("y_train has shape", y_train.shape)
  tau_p = 1.0e-6
  (
    step_sizes_train,
    x_train,
    y_train,
    viscosity_arr,
    # rearranged_indexes,
  ) = shuffle(
    step_sizes_train,
    x_train,
    y_train,
    viscosity_arr,
    # np.arange(0, len(x_train)),
    random_state=random_state,
  )
  # train_data[0] is model inputs and outputs for training
  # train_data[1] is some additional parameters for model normalisations
  train_data = [
    [
      step_sizes_train,
      x_train[:, us_inds],
      x_train[:, up_inds],
      x_train[:, uc_inds],
      x_train[:, timescale_inds],
      x_train[:, mesh_size_ind],
      y_train[:, us_inds],
      y_train[:, up_inds],
    ],
    [viscosity_arr],
  ]
  if args.load_data_only:
    exit()
  monitor = "val_loss"
  callbacks = []
  """
  Set verbosity level to 1, which shows progress bar per epoch
  verbosity 2 only shows output at end of epoch with loss
  """
  VERBOSE = 1
  # Create a callback that saves the model's weights
  callback_stopping = tf.keras.callbacks.EarlyStopping(
    monitor=monitor,
    patience=args.early_stopping,
    mode="min",
    restore_best_weights=True,
  )
  cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=True,
    monitor=monitor,
    save_best_only=True,
    save_freq="epoch",  # "epoch",
    verbose=VERBOSE,
  )
  callbacks.append(callback_stopping)
  callbacks.append(cp_callback)
  callbacks.append(tf.keras.callbacks.TerminateOnNaN())
  if args.restart:
    model = tf.keras.models.load_model("saved_model/chan_tracer_model")
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    print(checkpoint_dir)
    model.load_weights(latest).expect_partial()
  if args.retrain and not args.restart:
    # clean old log files
    if os.path.isdir(LOG_DIR):
      rmtree(LOG_DIR)
  if args.plots:
    log_dir_cb = f"{LOG_DIR}/fit/{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
      log_dir=log_dir_cb,
      write_graph=False,
      update_freq="batch",
      histogram_freq=1,  # histogram_freq=1
    )
    callbacks.append(tensorboard_callback)
  hist = model.fit(
    x=train_data,
    epochs=N_EPOCHS,
    batch_size=config["network"]["train"]["BATCH_SIZE"],
    verbose=VERBOSE,
    validation_split=config["network"]["train"]["VALIDATION_SPLIT"],
    callbacks=callbacks,
    shuffle=True,
  )
  model.save("saved_model/chan_tracer_model")

  if args.plots:
    utils.plot_loss(
      hist.history["loss"],
      hist.history["val_loss"],
      filename=f"{FIG_DIR}/loss.pdf",
    )
  TrainTime = round(time() - t0, 4)
  print(f"NN was trained during {TrainTime} seconds.")
  exit()
  # weights from model.fit() uses latest epoch.
  # We want to load weights from best time and use them (i.e. model.load_weights)

# model = tf.keras.models.load_model('saved_model/chan_tracer_model')
latest = tf.train.latest_checkpoint(checkpoint_dir)
print("checkpoint is", checkpoint_dir)
model.load_weights(latest).expect_partial()

# exit()
if args.save_openfoam_model:
  # save model weights to h5 for converting to cpp with dump_to_cpp.py
  encoder_drift.save("cpp_models/encoder_drift_weights.h5")
  encoder_diffusion.save("cpp_models/encoder_diffusion_weights.h5")

if args.debug:
  print("Checking drift_nn weights", encoder_drift.weights)
  print()
  print()
  print()
  print("Checking diffusion_nn weights", encoder_diffusion.weights)

if args.test_pairs:
  use_dns_deltat = True
else:
  use_dns_deltat = False

(times_arr_base, step_sizes, x_data, y_data) = read_filtered_data_csv(
  f"{data_dir_base}/{test_dict['data_file'].format(filter_width=delta_filt)}",
  max_num_p=args.num_p,
  use_dns_deltat=use_dns_deltat,
)
# format data so columns match variables, and data is normalised
(
  time_interval,
  x_data,
  x_interval,
  y_data,
  y_interval,
) = format_data(
  times_arr_base,
  step_sizes,
  x_data,
  y_data,
  viscosity=test_dict["viscosity"],
  mesh_size=mesh_size_test,
  train=False,
)

T_steps = time_interval.shape[1]

num_test_particles = len(x_interval)
initial_vel_inds = us_inds

initial_vel = copy(x_interval[:, 0][:, initial_vel_inds])
initial_vel = np.hstack((initial_vel, initial_vel))
if not args.quiet:
  print("initial_vel shape", initial_vel.shape)
t0 = time()
print("sampling paths for NN")

# get additional features at times_network from ground truth.
timescale = x_interval[:, :, timescale_inds]
uc = x_interval[:, :, uc_inds]
mesh_ratio_test = x_interval[:, :, mesh_size_ind]
test_time_steps_nn = time_interval

if args.load_data_only:
  exit()

sde_i = SDEIntegrator(
  model=model,
  us_n_inds=us_inds,
  up_n_inds=up_inds,
  mesh_size_ind=mesh_size_ind,
  timescale_inp_inds=timescale_inds,
  uc_inds=uc_inds,
  DRIFT_MULTIPLIER_CONST=DRIFT_MULTIPLIER_CONST,
  DIFFUSION_MULTIPLIER_CONST=DIFFUSION_MULTIPLIER_CONST,
  DIFFUSION_NORM_TIMESCALE_INDEX=config["normalisation"][
    "DIFFUSION_NORM_TIMESCALE_INDEX"
  ],
  DRIFT_NORM_TIMESCALE_INDEX=config["normalisation"][
    "DRIFT_NORM_TIMESCALE_INDEX"
  ],
  viscosity=test_dict["viscosity"],
)

# make sure ordering is consistent with indexes given!!
us_time_start = x_interval[:, :, us_inds]
up_time_start = x_interval[:, :, up_inds]
test_func = sde_i.sample_path
tau_p = 1.0e-6

us_network, drift_time_list, diffusion_list = test_func(
  initial_velocity=initial_vel,
  step_size=test_time_steps_nn,
  number_of_timesteps=T_steps,
  number_of_particles=num_test_particles,
  mesh_size_all=mesh_ratio_test,
  timescale_all=timescale,
  fluid_vel_all=uc,
  us_time_start=us_time_start,
  up_time_start=up_time_start,
  tau_p_all=tau_p,
  pairs=args.test_pairs,
  diffusion_off=args.diffusion_off,
  debug=True,
  return_nn_output=True,
)
# Get u_p when u_s = \bar{u}_c
us_up_no_model = test_func(
  initial_velocity=initial_vel,
  step_size=test_time_steps_nn,
  number_of_timesteps=T_steps,
  number_of_particles=num_test_particles,
  mesh_size_all=mesh_ratio_test,
  timescale_all=timescale,
  fluid_vel_all=uc,
  us_time_start=us_time_start,
  up_time_start=up_time_start,
  tau_p_all=tau_p,
  pairs=args.test_pairs,
  diffusion_off=args.diffusion_off,
  debug=False,
  return_nn_output=False,
  no_model=True,
)
up_no_us_model = np.array(us_up_no_model)[:, 1:, up_inds]
print(f"sampling took {round(time()-t0, 4)} seconds.")

print("getting TKE and displacement")
t0 = time()
drift_time_list = np.swapaxes(np.array(drift_time_list), 0, 1)
diffusion_list = np.swapaxes(np.array(diffusion_list), 0, 1)

us_test = y_data[:, :, us_inds]
up_test = y_data[:, :, up_inds]
initial_tke = np.sum(us_test[:, 0] ** 2, axis=-1)
tke_test = np.sum(us_test**2.0, axis=2)
normaliser = 1.0  # initial_tke.mean()

up_network = np.array(us_network)[:, 1:, up_inds]
us_network = np.array(us_network)[:, 1:, us_inds]

STOKES_TEST = "tracer"
test_vel_list = [us_test, up_test]
network_vel_list = [us_network, up_network]
plot_string_list = ["useen", "uparticle"]
plot_time_dict = dict.fromkeys(plot_string_list)
tke_dict = dict.fromkeys(plot_string_list)
tke_dict_gt = dict.fromkeys(plot_string_list)
tke_uc = np.sum(x_data[:, :, uc_inds] ** 2.0, axis=-1)
tke_up_no_us_model = np.sum(up_no_us_model**2, axis=-1)
# dns_data = np.loadtxt("meanFluidVelocityEnergyDNS.txt")
normaliser = 1.0  # dns_data[0, 1]
for vel_test, vel_network, plot_str in zip(
  test_vel_list, network_vel_list, plot_string_list
):
  initial_tke = np.sum(vel_test[:, 0] ** 2, axis=-1)
  tke_test = np.sum(vel_test**2.0, axis=2)
  # normaliser = initial_tke.mean()
  normaliser = initial_tke.mean()
  tke_test /= normaliser
  if plot_str == "useen":
    tke_uc /= normaliser
  elif plot_str == "uparticle":
    tke_up_no_us_model /= normaliser

  tke_network = np.sum(vel_network**2.0, axis=2)
  tke_network /= normaliser
  plot_times_nn = times_arr_base[:, 0]
  # save tke and times to dict for comparing two
  tke_dict[plot_str] = copy(tke_network)
  plot_time_dict[plot_str] = copy(plot_times_nn)
  tke_dict_gt[plot_str] = copy(tke_test)

  if plot_str == "uparticle":
    print_tke = tke_up_no_us_model
  else:
    print_tke = tke_uc
  error_in_tke = (
    1.0  # 100.0
    * abs(tke_dict_gt[plot_str].mean(axis=0) - tke_dict[plot_str].mean(axis=0))
    / tke_dict_gt[plot_str].mean(axis=0)
  )
  error_in_tke_nomodel = (
    1.0  # 100.0
    * abs(tke_dict_gt[plot_str].mean(axis=0) - print_tke.mean(axis=0))
    / tke_dict_gt[plot_str].mean(axis=0)
  )
  improvement_in_tke = tke_dict[plot_str].mean(axis=0) / print_tke.mean(axis=0)

  tke_results = np.c_[
    tke_dict_gt[plot_str].mean(axis=0),
    tke_dict[plot_str].mean(axis=0),
    print_tke.mean(axis=0),
    improvement_in_tke,
    # error_in_tke,
  ]
  if plot_str == "useen":
    print(f"case {args.test_dict_key} - {args.test_filt}")
    print(f"tke results - {plot_str}")
    print("\t GT \t NN \t Fluid \t NN/Fluid")
    print(tke_results)
    print("error:")
    print(error_in_tke[1:])
    print("error with no model:")
    print(error_in_tke_nomodel[1:])

  t0 = time()

if args.plots:
  plot_filename = (
    f"{FIG_DIR}/driftTime{args.test_dict_key}_test{args.test_filt}.png"
  )
  utils.plot_inputs(
    plot_time_dict["useen"],
    drift_time_list.mean(axis=0)[:, 0],
    "output drift time",
    kPaths=1,
    filename=plot_filename,
    logy=False,
  )
  plot_filename = (
    f"{FIG_DIR}/diffusion{args.test_dict_key}_test{args.test_filt}.png"
  )
  utils.plot_inputs(
    plot_time_dict["useen"],
    diffusion_list.mean(axis=0)[:, 0],
    "output diffusion",
    kPaths=1,
    filename=plot_filename,
    logy=False,
  )
  # set up plotting
  if args.test_pairs:
    pairs_string = "pairs"
  else:
    pairs_string = "nonPairs"
  plot_filename = (
    f"{FIG_DIR}/{pairs_string}TkeUsDecayWithUc{args.test_dict_key}_"
    f"{STOKES_TEST}_{args.test_filt}_{DIFFUSION_TERM_STR}.{args.extension}"
  )

  utils.plot_tke_mean_decay_with_uc(
    plot_time_dict["useen"] / PLOT_TIME_NORMALISATION,
    plot_time_dict["useen"] / PLOT_TIME_NORMALISATION,
    tke_dict["useen"].mean(axis=0),
    tke_dict_gt["uparticle"].mean(axis=0),
    tke_uc.mean(axis=0),
    ylim=[0, 1.1],
    xlim=3, # 3 integral times
    ylabel=r"$k$ / $k^{0}$ [-]",
    extra_label=r"Filtered",
    network_label=r"Filtered + NN",
  )

  plot_filename = f"{FIG_DIR}/weights_diffusion_nn.{args.extension}"
  utils.plot_weights_pdf(
    model.encoder_diffusion.weights,
    r"$B^{NN}$",
    col="blue",
    filename=plot_filename,
  )
  plot_filename = f"{FIG_DIR}/weights_drift_nn.{args.extension}"
  utils.plot_weights_pdf(
    model.encoder_drift.weights,
    r"$G^{NN}$",
    col="green",
    filename=plot_filename,
  )
