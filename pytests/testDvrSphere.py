import numpy as np
import matplotlib.pyplot as plt
import os
import imageio

# load pyrenderer
import sys
sys.path.append('../bin')
import pyrenderer

import utils

def runTimings(mode : str):
  # CONFIGURATION
  SETTINGS_FILE = "../scenes/dvrSphere-scene2.json"
  ROOT_PATH = ".."
  RESOLUTION = (512, 512)
  NUM_FRAMES = 1
  VOLUME_RESOLUTION = 128
  KERNEL_NAMES_NORMAL, KERNEL_NAMES_TIMING = utils.getKernelNames(VOLUME_RESOLUTION)

  DENSITY_STEPS = 10
  WIDTH_START = 0.05
  WIDTH_END = 0.002
  SCALING_START = 15.0
  EXPONENT = 0.1
  TIMING_STEPS = utils.TIMING_STEPS

  OUTPUT_STATS_ALL = "../results/statistics/dvr-sphere/timings-all-%s.tsv"
  OUTPUT_STATS_AVG = "../results/statistics/dvr-sphere/timings-avg-%s.tsv"
  OUTPUT_HISTO_ALL = "../results/statistics/dvr-sphere/histograms-%s.tsv"
  OUTPUT_HISTO_CFG = "../results/statistics/dvr-sphere/histogram-cfg-%s.tsv"
  OUTPUT_STATS_USE_DOUBLE = False
  OUTPUT_IMAGE_PATH = "../results/statistics/dvr-sphere/images/"
  OUTPUT_INSTRUMENTATION = "../results/statistics/dvr-sphere/instrumentation.tsv"

  HISTO_NUM_BINS = 100
  HISTO_BIN_MIN = np.log10(1e-6)
  HISTO_BIN_MAX = np.log10(1)
  HISTO_BIN_EDGES = [0.0] + list(10 ** np.linspace(HISTO_BIN_MIN, HISTO_BIN_MAX, HISTO_NUM_BINS))
  print("histogram bins:", HISTO_BIN_EDGES)

  pyrenderer.oit.set_fragment_buffer_size(2 ** 26)
  pyrenderer.oit.set_marching_cubes_mode(pyrenderer.oit.MarchingCubesComputationMode.OnTheFly)
  pyrenderer.oit.set_max_fragments_per_pixel(512)
  pyrenderer.oit.set_tile_size(128)

  os.makedirs(OUTPUT_IMAGE_PATH, exist_ok=True)

  # load settings file
  rendererArgs, camera, volumePath = pyrenderer.load_from_json(
    SETTINGS_FILE, ROOT_PATH)
  print("settings loaded")
  rendererArgs.width = RESOLUTION[0]
  rendererArgs.height = RESOLUTION[1]
  camera.update_render_args(rendererArgs)
  multiiso_tf = rendererArgs.get_multiiso_tf()

  # create volume
  print("Create Sphere")
  volume = pyrenderer.Volume.create_implicit(pyrenderer.ImplicitEquation.Sphere, VOLUME_RESOLUTION)
  print("Loaded volumed of resolution", volume.resolution,
        "and world size", volume.world_size)
  volume.copy_to_gpu()

  def computeSettings(time, rendererArgs):
    settings = rendererArgs.clone()
    width = pow(utils.lerp(time, pow(WIDTH_START, EXPONENT),
                           pow(WIDTH_END, EXPONENT)),
                1 / EXPONENT)
    scaling = SCALING_START * WIDTH_START / width
    settings.set_linear_tf(utils.multiIso2Linear(multiiso_tf, width))
    settings.opacity_scaling = scaling
    return settings, width

  # allocate timing
  timer = pyrenderer.GpuTimer()
  times = [[None] * DENSITY_STEPS for i in range(len(KERNEL_NAMES_NORMAL))]

  if mode == "visualize":
    # allocate output
    pyrenderer.reload_kernels(enableDebugging=False, enableInstrumentation=False)
    output = pyrenderer.allocate_output(
      rendererArgs.width, rendererArgs.height,
      rendererArgs.render_mode)
    outputs = [[None] * DENSITY_STEPS for i in range(len(KERNEL_NAMES_NORMAL))]

    # render
    camera.update_render_args(rendererArgs)
    for j, (kernel_name, _, stepsize) in enumerate(KERNEL_NAMES_NORMAL):
      print("Render", kernel_name, stepsize)
      rendererArgs.stepsize = stepsize
      for i in range(DENSITY_STEPS):
        settings, width = computeSettings(i / (DENSITY_STEPS - 1), rendererArgs)
        timer.start()
        pyrenderer.render(kernel_name, volume, settings, output)
        timer.stop()
        outputs[j][i] = np.array(output.copy_to_cpu())
        times[j][i] = timer.elapsed_ms()

    def slugify(value):
      """
      Normalizes string, converts to lowercase, removes non-alpha characters,
      and converts spaces to hyphens.
      """
      import unicodedata
      import re
      value = str(unicodedata.normalize('NFKD', value))  # .encode('ascii', 'ignore'))
      value = re.sub('[^\w\s-]', '', value).strip().lower()
      value = re.sub('[-\s]+', '-', value)
      return value

    # visualize
    print("Visualize")
    # fig, axes = plt.subplots(nrows=len(KERNEL_NAMES_NORMAL), ncols=DENSITY_STEPS)
    for j, (kernel_name, human_kernel_name, _) in enumerate(KERNEL_NAMES_NORMAL):
      for i in range(DENSITY_STEPS):
        filename = os.path.join(OUTPUT_IMAGE_PATH,
                                slugify("%s__%d" % (human_kernel_name, i)) + ".png")
        imageio.imwrite(
          filename,
          outputs[j][i][:, :, 0:4])

        # axes[j][i].imshow(outputs[j][i][:,:,0:4])
        # axes[j][i].set_title("time=%.2fms"%times[j][i])
        # if j==len(KERNEL_NAMES_NORMAL)-1:
        #    axes[j][i].set_xlabel("range=%.3f"%(
        #        max_densities[i]-base_min_density))
        # if i==0:
        #    axes[j][i].set_ylabel(human_kernel_name)

    # save to numpy
    npz_output = {}
    npz_output['kernels'] = KERNEL_NAMES_NORMAL
    npz_output['widths'] = [
      computeSettings(i / (DENSITY_STEPS - 1), rendererArgs)[1] for i in range(DENSITY_STEPS)]
    for j in range(len(KERNEL_NAMES_NORMAL)):
      for i in range(DENSITY_STEPS):
        npz_output['img_%d_%d' % (j, i)] = outputs[j][i]
    np.savez(os.path.join(OUTPUT_IMAGE_PATH, "raw.npz"), **npz_output)

    # plt.subplots_adjust(left=0.03, bottom=0.05, right=0.99, top=0.97, wspace=0.20, hspace=0.23)
    # plt.show()

  elif mode == "measure":
    summed_times = [[0] * DENSITY_STEPS for i in range(len(KERNEL_NAMES_NORMAL))]

    pyrenderer.reload_kernels(
      enableDebugging=False,
      enableInstrumentation=False,
      otherPreprocessorArguments=["-DKERNEL_USE_DOUBLE=%s" % ("1" if OUTPUT_STATS_USE_DOUBLE else "0")])
    # allocate output for baseline
    output = pyrenderer.allocate_output(
      rendererArgs.width, rendererArgs.height,
      rendererArgs.render_mode);
    outputs = [[None] * TIMING_STEPS for i in range(DENSITY_STEPS)]

    histograms = [[
      np.zeros(HISTO_NUM_BINS, dtype=np.int64) for i in range(DENSITY_STEPS)]
      for j in range(len(KERNEL_NAMES_NORMAL))]

    # render and write output
    with open(OUTPUT_STATS_ALL % ("double" if OUTPUT_STATS_USE_DOUBLE else "float"), "w") as f:
      f.write("Kernel Name\tWidth\tFrame\tTime (ms)\tPSNR (dB)\n")
      # for j, (kernel_name, human_kernel_name, stepsize) in enumerate(KERNEL_NAMES):
      for j in range(len(KERNEL_NAMES_NORMAL)):
        kernel_name_normal, human_kernel_name, stepsize = KERNEL_NAMES_NORMAL[j]
        kernel_name_timings, _, _ = KERNEL_NAMES_TIMING[j]
        print("Render", kernel_name_normal, stepsize)
        rendererArgs.stepsize = stepsize
        for i in range(DENSITY_STEPS):
          settings, width = computeSettings(i / (DENSITY_STEPS - 1), rendererArgs)
          print("  width=%.4f, scaling=%.4f" % (width, settings.opacity_scaling))
          histogram = histograms[j][i]
          histogram_edges = None
          for k in range(TIMING_STEPS):
            camera.yaw = k * 360 / TIMING_STEPS
            camera.update_render_args(settings)
            timer.start()
            pyrenderer.render(kernel_name_timings, volume, settings, output)
            timer.stop()
            if kernel_name_timings != kernel_name_normal:
              pyrenderer.render(kernel_name_normal, volume, settings, output)
            out_img = np.array(output.copy_to_cpu())
            if j == 0:  # baseline
              outputs[i][k] = out_img
              psnr = 0
            else:
              # compute psnr
              maxValue = np.max(outputs[i][k][:, :, 0:4])
              mse = ((outputs[i][k][:, :, 0:4] - out_img[:, :, 0:4]) ** 2).mean(axis=None)
              psnr = 20 * np.log10(maxValue) - 10 * np.log10(mse)
              # compute histogram
              diff = outputs[i][k][:, :, 0:4] - out_img[:, :, 0:4]
              new_histogram, histogram_edges = np.histogram(
                diff, bins=HISTO_BIN_EDGES)
              histogram += new_histogram
            t = timer.elapsed_ms()
            summed_times[j][i] += t
            f.write("%s\t%.4f\t%d\t%.4f\t%.4f\n" % (
              human_kernel_name.replace("\n", " "), width, k, t, psnr))

    # write average stats
    with open(OUTPUT_STATS_AVG % ("double" if OUTPUT_STATS_USE_DOUBLE else "float"), "w") as f:
      f.write("Kernel Name\tWidth\tAvg-Time (ms)\n")
      for j, (_, human_kernel_name, _) in enumerate(KERNEL_NAMES_NORMAL):
        for i in range(DENSITY_STEPS):
          _, width = computeSettings(i / (DENSITY_STEPS - 1), rendererArgs)
          f.write("%s\t%.4f\t%.4f\n" % (
            human_kernel_name.replace("\n", " "), width,
            summed_times[j][i] / TIMING_STEPS))

    # write histograms
    with open(OUTPUT_HISTO_ALL % ("double" if OUTPUT_STATS_USE_DOUBLE else "float"), "w") as f:
      f.write("BinStart\tBinEnd")
      for j in range(len(KERNEL_NAMES_NORMAL)):
        for i in range(DENSITY_STEPS):
          f.write("\t%d-%d" % (j, i))
      f.write("\n")
      for b in range(HISTO_NUM_BINS):
        f.write("%.10f\t%.10f" % (HISTO_BIN_EDGES[b], HISTO_BIN_EDGES[b + 1]))
        for j in range(len(KERNEL_NAMES_NORMAL)):
          for i in range(DENSITY_STEPS):
            f.write("\t%d" % histograms[j][i][b])
        f.write("\n");
    with open(OUTPUT_HISTO_CFG % ("double" if OUTPUT_STATS_USE_DOUBLE else "float"), "w") as f:
      f.write("Kernel Name\tWidth\tConfig-ID\n")
      for j, (_, human_kernel_name, _) in enumerate(KERNEL_NAMES_NORMAL):
        for i in range(DENSITY_STEPS):
          f.write("%s\t%.4f\t%s\n" % (
            human_kernel_name.replace("\n", " "),
            computeSettings(i / (DENSITY_STEPS - 1), rendererArgs)[1],
            "%d-%d" % (j, i)))

  elif mode == "instrumentation":
    # recompile with instrumentation
    pyrenderer.reload_kernels(enableInstrumentation=True)

    # allocate output
    output = pyrenderer.allocate_output(
      rendererArgs.width, rendererArgs.height,
      rendererArgs.render_mode);
    outputs = [[None] * DENSITY_STEPS for i in range(len(KERNEL_NAMES_NORMAL))]

    fields = ["densityFetches", "tfFetches", "ddaSteps", "isoIntersections",
              "intervalEval", "intervalStep", "intervalMaxStep"]

    # render
    with open(OUTPUT_INSTRUMENTATION, "w") as f:
      f.write("Kernel Name\tWidth\t%s\tavgIntervalSteps\tnumTriangles\tnumFragments\n" % "\t".join(fields))
      camera.update_render_args(rendererArgs);
      for j, (kernel_name, human_kernel_name, stepsize) in enumerate(KERNEL_NAMES_NORMAL):
        print("Render", kernel_name, stepsize)
        rendererArgs.stepsize = stepsize
        for i in range(DENSITY_STEPS):
          settings, width = computeSettings(i / (DENSITY_STEPS - 1), rendererArgs)
          instrumentations, globalInstrumentations = \
            pyrenderer.render_with_instrumentation(
              kernel_name, volume, settings, output)
          f.write("%s\t%s" % (human_kernel_name.replace("\n", " "),
                              width))
          for field in fields:
            f.write("\t%.2f" % np.mean(instrumentations[field], dtype=np.float32))
          avgIntervalSteps = np.mean(
            instrumentations["intervalStep"].astype(np.float32) /
            instrumentations["intervalEval"].astype(np.float32))
          f.write("\t%.2f\t%d\t%d\n" % (
            avgIntervalSteps, globalInstrumentations.numTriangles, globalInstrumentations.numFragments))
          f.flush()

if __name__=='__main__':
  pyrenderer.oit.setup_offscreen_context()
  pyrenderer.init()
  runTimings("measure")
  runTimings("visualize")
  #runTimings("instrumentation")
  pyrenderer.oit.delete_offscreen_context()