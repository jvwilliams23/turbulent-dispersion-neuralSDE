import numpy as np
import lagrangian_parser as lp
from glob import glob


def read_foam_data(
  times, x_data_fields, base_dir, quiet=False, max_num_p=1e100, suppress=False, **kwargs
):
  """

  Suppress (bool): when true, suppresses errors by fields that cannot be read,
  and creates an empty field
  """
  def read_foam_data_t_x_y(
    times,
    x_data_fields,
    y_data_fields=["UTurb"],
    base_dir="data/lagrangianDirs/",
    **kwargs,
  ):
    nonlocal quiet, max_num_p
    x_data = []
    y_data = []
    step_sizes = []
    times_arr = []
    for t, time in enumerate(times):
      # if t == 0:
      #   continue
      x_data.append([])
      y_data.append([])
      # if timestep is integer, convert so we do not try to read
      # e.g. "0.0/" instead of "0/"
      if time.is_integer():
        timetmp = int(time)
        cloudDir = f"{base_dir}/{timetmp}/lagrangian/kinematic*Cloud/"
        cloudDir = glob(cloudDir)[0]
      else:
        # solve some issues with time-precision causing long file names
        if len(str(time)) > 8:
          timetmp = float(str(time)[:-2])
          cloudDir = glob(f"{base_dir}/{timetmp}*/lagrangian/kinematic*Cloud/")[
            0
          ]
        else:
          timetmp = time
          cloudDir = f"{base_dir}/{timetmp}/lagrangian/kinematic*Cloud/"
          if not quiet:
            print("DEBUG", cloudDir)
          cloudDir = glob(cloudDir)[0]

      if not quiet:
        print(f"reading data at t={time} from {cloudDir}")
      # read lagrangian properties
      # for now, truncate final axis to make problem 2D (x, y only)
      x_data_tmp = []
      for i, field in enumerate(x_data_fields):
        field_i = lp.parse_lagrangian_field(
          cloudDir + field, returnAll=True, max_num_p=max_num_p, suppress=suppress
        )
        if type(field_i) == type(None) and suppress:
          if "x_train_indexes" in kwargs.keys():
            # set array of zeros, only works if a field has already been read
            num_columns = len(kwargs["x_train_indexes"][i])
            num_rows = x_data_tmp[-1].shape[0]
            field_i = np.zeros((num_rows, num_columns))
          else:
            # otherwise, assume only 1 column
            field_i = np.zeros(num_rows)
        if field_i.ndim == 1:
          field_i = field_i.reshape(-1, 1)
        x_data_tmp.append(field_i)
      # print(np.hstack(x_data_tmp))
      # for i in x_data_tmp:
      #   print(i.shape)
      x_data[t].extend(np.hstack(x_data_tmp))
      y_data_tmp = []
      for field in y_data_fields:
        data = lp.parse_lagrangian_field(
          cloudDir + field, returnAll=True, max_num_p=max_num_p
        )
        num_particles = data.shape[0]
        y_data_tmp.append(data)
        # y_data[t].extend(data)
      y_data[t].extend(np.hstack(y_data_tmp))
      # get step sizes. 1st time step will be -ve so set to t[0] instead of t[0] - t[-1]
      dt = time - times[t - 1]
      if dt < 0:
        dt = time
      step_sizes.append(np.ones(num_particles) * dt)
      times_arr.append(np.ones(num_particles) * time)

    return (
      np.array(times_arr),
      np.array(step_sizes),
      np.array(x_data),
      np.array(y_data),
    )

  def read_foam_data_x(times, x_data_fields, base_dir="data/lagrangianDirs/"):
    nonlocal quiet, max_num_p
    x_data = []
    y_data = []
    step_sizes = []
    times_arr = []
    for t, time in enumerate(times):
      x_data.append([])
      # if timestep is integer, convert so we do not try to read
      # e.g. "0.0/" instead of "0/"
      if time.is_integer():
        timetmp = int(time)
        cloudDir = f"{base_dir}/{timetmp}/lagrangian/kinematic*Cloud/"
        cloudDir = glob(cloudDir)[0]
      else:
        # solve some issues with time-precision causing long file names
        if len(str(time)) > 8:
          timetmp = float(str(time)[:-2])
          cloudDir = glob(f"{base_dir}/{timetmp}*/lagrangian/kinematic*Cloud/")[
            0
          ]
        else:
          timetmp = time
          cloudDir = f"{base_dir}/{timetmp}/lagrangian/kinematic*Cloud/"
          cloudDir = glob(cloudDir)[0]

      if not quiet:
        print(f"reading data at t={time} from {cloudDir}")
      # read lagrangian properties
      # for now, truncate final axis to make problem 2D (x, y only)
      x_data_tmp = []
      for field in x_data_fields:
        field_i = lp.parse_lagrangian_field(
          cloudDir + field, returnAll=True, max_num_p=max_num_p
        )
        x_data_tmp.append(field_i)
      x_data[t].extend(np.hstack(x_data_tmp))

    return np.array(x_data)

  # helper to choose which way to format reading and writing of data
  if "y_data_fields" in kwargs.keys():
    return read_foam_data_t_x_y(
      times, x_data_fields, base_dir=base_dir, **kwargs
    )
  else:
    return read_foam_data_x(times, x_data_fields, base_dir)


if __name__ == "__main__":
  """
  Some tests to ensure correct shape of data etc is read
  """
  data_dir = "data/lagrangianDirsHIT/"
  times = [0.2, 0.4, 0.6]

  p_data = read_foam_data(
    times,
    x_data_fields=["U", "d"],
    base_dir=data_dir,
  )
  assert p_data.shape[2] == 4, "should have 3 cols for up_i and 1 for dp"

  times_arr, steps, x_data, y_data = read_foam_data(
    times,
    x_data_fields=["U", "d"],
    y_data_fields=["UTurb"],
    base_dir=data_dir,
  )
  assert (
    times_arr.shape[0]
    == len(times) & steps.shape[0]
    == len(times) & x_data.shape[0]
    == len(times) & y_data.shape[0]
    == len(times)
  ), "not correct number time steps read"
