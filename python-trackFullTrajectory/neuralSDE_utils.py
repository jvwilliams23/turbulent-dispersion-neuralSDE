import matplotlib.pyplot as plt
import numpy as np

x_size = 8 / 2.54
y_size = 8 / 2.54

def plot_loss(train_loss, val_loss, filename="figures/loss.pdf"):
  plt.close()
  if (
    max(train_loss) / min(train_loss) > 100
    or max(val_loss) / min(val_loss) > 100
  ):
    if min(train_loss) > 0 and min(val_loss) > 0:
      plt.yscale("log")
  else:
    if (max(train_loss) > 50 or max(val_loss) > 50) and (
      min(train_loss) < 0 or min(val_loss) < 0
    ):
      train_loss = train_loss[10:]
      val_loss = val_loss[10:]
  plt.plot(train_loss, c="blue", lw=2, label="train")
  plt.plot(val_loss, c="orange", lw=2, label="validation")
  plt.ylabel("loss")
  plt.xlabel("epoch")
  plt.legend(frameon=False)
  # plt.show()
  for pos in ["right", "top"]:
    plt.gca().spines[pos].set_visible(False)
  plt.savefig(filename)
  return None

def plot_inputs(
  times, x_data, feature_name, kPaths=100, filename="", logy=False
):
  plt.close()
  fig, ax = plt.subplots(1, 1, figsize=(x_size, y_size))
  if kPaths == 1:
    ax.plot(
      times,
      x_data,
      "b-",
      linewidth=1.5,
      label="Network",
      alpha=1,
    )
  else:
    for k in range(kPaths):
      ax.plot(
        times,
        x_data[k],
        "b-",
        linewidth=1.5,
        label="Network" if k == 0 else None,
        alpha=0.1,
      )
  if logy:
    ax.set_yscale("log")
  ax.set_xlabel("time [s]")
  ax.set_ylabel(f"{feature_name}")
  # ax.legend(frameon=False)
  fig.tight_layout()
  for pos in ["right", "top"]:
    plt.gca().spines[pos].set_visible(False)
  if filename == "":
    plt.show()
  else:
    plt.savefig(filename, dpi=300)
  return None


def plot_tke_mean_decay_with_uc(
  time_network,
  times_test,
  tke_network,
  tke_test,
  tke_uc,
  ylim=None,
  xlim=None,
  ylabel=r"$k/k^{0}$ [-]",
  filename="",
  gt_label="DNS",
  network_label="Filtered DNS + NN",
  extra_label=r"Fluid ($\widetilde{\mathbf{u}}_f$)",
):
  plt.close()
  fig, ax = plt.subplots(1, 1, figsize=(x_size, y_size))
  size = 20
  ax.scatter(time_network, tke_network, label=network_label, marker="x", s=size, c="b")
  ax.scatter(time_network, tke_uc, label=extra_label, s=size*0.5, c="g")
  ax.plot(
    times_test,
    tke_test,
    label=gt_label,  # r"Ground truth ($u_f$)",
    lw=1.5,
    c="r",
    ls="--",
  )
  ax.set_ylim(ylim)
  ax.set_ylabel(ylabel, fontsize=11)
  ax.set_xlim([0, xlim])
  ax.set_xlabel(r"t / $\tau_L^{0}$ [-]", fontsize=11)

  ax.legend(frameon=False, fontsize=10)
  fig.tight_layout(rect=(-0.03, -0.03, 1.02, 1.02))
  for pos in ["right", "top"]:
    plt.gca().spines[pos].set_visible(False)
  if filename == "":
    plt.show()
  else:
    plt.savefig(filename, dpi=300)

def plot_weights_pdf(
  weights_list, network_name, col="black", filename="weights.png"
):
  x_size = 16 / 2.54
  y_size = x_size * 0.5
  if len(weights_list) <= 4:
    x_panels = len(weights_list)
    y_panels = 1
  else:
    y_panels = 2
    x_panels = int(np.ceil(len(weights_list) / y_panels))
  fig, axes = plt.subplots(y_panels, x_panels, figsize=(x_size, y_size))#, sharey=True)
  axes = axes.ravel()
  for i, (ax, layer) in enumerate(zip(axes, weights_list)):
    logits = layer.numpy().ravel()
    # print(logits)
    ax.hist(
      logits,
      bins="auto",
      histtype="step",
      density=True,
      ec=col,
      lw=2,
    )
    # make x limits symmetric to highlight any kurtosis
    xlim = np.array(list(ax.get_xlim()))
    xlim = np.max(abs(xlim))
    ax.set_xlim([-xlim, xlim])
    if i == 0:
      ax.set_ylabel("PDF")
    ax.set_xlabel("value")
    for pos in ["right", "top"]:
      ax.spines[pos].set_visible(False)

  fig.suptitle(network_name, fontsize=11)
  fig.tight_layout()
  plt.savefig(filename, dpi=300)
  # plt.show()
