import numpy as np
import matplotlib.pyplot as plt
import os
import imageio

# load pyrenderer
import sys
sys.path.append('../bin')
import pyrenderer

def runTimings(mode : str):
    # CONFIGURATION
    SETTINGS_FILE = "../scenes/bigBug1.json"
    ROOT_PATH = ".."
    RESOLUTION = (512, 512)
    NUM_FRAMES = 1
    VOLUME_RESOLUTION = 128
    KERNEL_NAMES = [
        ("DVR: DDA - fixed step (control points)", "dvr stepping 0.0001\n(Baseline)", 0.0001)] + \
        [("DVR: Fixed step size - trilinear", "stepping 2^-%d"%s, pow(2, -s)) for s in range(2,15)]
    DENSITY_STEPS = 7
    MIN_DENSITY_DIFFERENCE = 0.01 # minimal difference between min and max density
    TIMING_STEPS = 1
    OUTPUT_STATS_ALL = "../results/statistics/dvr-beetle/timingsConvergence-all-%s.tsv"
    OUTPUT_STATS_AVG = "../results/statistics/dvr-beetle/timingsConvergence-avg-%s.tsv"
    OUTPUT_HISTO_ALL = "../results/statistics/dvr-beetle/histogramsConvergence-%s.tsv"
    OUTPUT_HISTO_CFG = "../results/statistics/dvr-beetle/histogramConvergence-cfg-%s.tsv"
    OUTPUT_STATS_USE_DOUBLE = True
    os.makedirs("../results/statistics/dvr-beetle/", exist_ok=True)

    HISTO_NUM_BINS = 100
    HISTO_BIN_MIN = np.log10(1e-6)
    HISTO_BIN_MAX = np.log10(1)
    HISTO_BIN_EDGES = [0.0] + list(10 ** np.linspace(HISTO_BIN_MIN, HISTO_BIN_MAX, HISTO_NUM_BINS))
    print("histogram bins:", HISTO_BIN_EDGES)
    
    pyrenderer.oit.set_fragment_buffer_size(2**26)
    pyrenderer.oit.set_marching_cubes_mode(pyrenderer.oit.MarchingCubesComputationMode.OnTheFly)
    pyrenderer.oit.set_max_fragments_per_pixel(512)
    pyrenderer.oit.set_tile_size(128)

    # load settings file
    rendererArgs, camera, volumePath = pyrenderer.load_from_json(
        SETTINGS_FILE, ROOT_PATH)
    print("settings loaded")
    rendererArgs.width = RESOLUTION[0]
    rendererArgs.height = RESOLUTION[1]
    base_min_density = rendererArgs.min_density
    base_max_density = rendererArgs.max_density
    base_opacity = rendererArgs.opacity_scaling

    # create density+opacity test cases
    end_max_density = base_min_density + MIN_DENSITY_DIFFERENCE
    max_densities = np.exp(
        np.linspace(np.log(base_max_density), np.log(end_max_density), DENSITY_STEPS))
    scaling = (base_max_density-base_min_density) / (max_densities-base_min_density)
    opacities = base_opacity * scaling

    # create volume
    print("Create Marschner Lobb")
    volume = pyrenderer.Volume(volumePath)
    print("Loaded volumed of resolution", volume.resolution, 
          "and world size", volume.world_size)
    volume.copy_to_gpu();

    # allocate timing
    timer = pyrenderer.GpuTimer()
    times = [[None] * DENSITY_STEPS for i in range(len(KERNEL_NAMES))]

    if mode == "measure":
        summed_times = [[0] * DENSITY_STEPS for i in range(len(KERNEL_NAMES))]

        pyrenderer.reload_kernels(
            enableDebugging=False, 
            enableInstrumentation=False, 
            otherPreprocessorArguments=["-DKERNEL_USE_DOUBLE=%s"%("1" if OUTPUT_STATS_USE_DOUBLE else "0")])
        # allocate output for baseline
        output = pyrenderer.allocate_output(
            rendererArgs.width, rendererArgs.height,
            rendererArgs.render_mode);
        outputs = [[None] * TIMING_STEPS for i in range(DENSITY_STEPS)]

        histograms = [[
            np.zeros(HISTO_NUM_BINS, dtype=np.int64) for i in range(DENSITY_STEPS)]
                      for j in range(len(KERNEL_NAMES))]

        # render and write output
        with open(OUTPUT_STATS_ALL%("double" if OUTPUT_STATS_USE_DOUBLE else "float"), "w") as f:
            f.write("Kernel Name\tTF-Range\tFrame\tTime (ms)\tPSNR (dB)\n")  
            for j, (kernel_name, human_kernel_name, stepsize) in enumerate(KERNEL_NAMES):
                print("Render", kernel_name, stepsize)
                rendererArgs.stepsize = stepsize
                for i in range(DENSITY_STEPS):
                    rendererArgs.max_density = max_densities[i]
                    rendererArgs.opacity_scaling = opacities[i]
                    histogram = histograms[j][i]
                    histogram_edges = None
                    for k in range(TIMING_STEPS):
                        camera.yaw = k * 360 / TIMING_STEPS
                        camera.update_render_args(rendererArgs)
                        timer.start()
                        pyrenderer.render(kernel_name, volume, rendererArgs, output)
                        timer.stop()
                        out_img = np.array(output.copy_to_cpu());
                        if j == 0: # baseline
                            outputs[i][k] = out_img
                            psnr = 0
                        else:
                            # compute psnr
                            maxValue = np.max(outputs[i][k][:,:,0:4])
                            mse = ((outputs[i][k][:,:,0:4] - out_img[:,:,0:4])**2).mean(axis=None)
                            psnr = 20 * np.log10(maxValue) - 10 * np.log10(mse)
                            # compute histogram
                            diff = outputs[i][k][:,:,0:4] - out_img[:,:,0:4]
                            new_histogram, histogram_edges = np.histogram(
                                diff, bins=HISTO_BIN_EDGES)
                            histogram += new_histogram
                        t = timer.elapsed_ms()
                        summed_times[j][i] += t
                        f.write("%s\t%.4f\t%d\t%.4f\t%.4f\n"%(
                            human_kernel_name.replace("\n", " "), max_densities[i]-base_min_density,
                            k, t, psnr))

        # write average stats
        with open(OUTPUT_STATS_AVG%("double" if OUTPUT_STATS_USE_DOUBLE else "float"), "w") as f:
            f.write("Kernel Name\tTF-Range\tAvg-Time (ms)\n")
            for j, (_, human_kernel_name, _) in enumerate(KERNEL_NAMES):
                for i in range(DENSITY_STEPS):
                    f.write("%s\t%.4f\t%.4f\n"%(
                        human_kernel_name.replace("\n", " "), max_densities[i]-base_min_density,
                        summed_times[j][i] / TIMING_STEPS))

        # write histograms
        with open(OUTPUT_HISTO_ALL%("double" if OUTPUT_STATS_USE_DOUBLE else "float"), "w") as f:
            f.write("BinStart\tBinEnd");
            for j in range(len(KERNEL_NAMES)):
                for i in range(DENSITY_STEPS):
                    f.write("\t%d-%d"%(j, i))
            f.write("\n")
            for b in range(HISTO_NUM_BINS):
                f.write("%.10f\t%.10f"%(HISTO_BIN_EDGES[b], HISTO_BIN_EDGES[b+1]))
                for j in range(len(KERNEL_NAMES)):
                    for i in range(DENSITY_STEPS):
                        f.write("\t%d"%histograms[j][i][b])
                f.write("\n");
        with open(OUTPUT_HISTO_CFG%("double" if OUTPUT_STATS_USE_DOUBLE else "float"), "w") as f:
            f.write("Kernel Name\tTF-Range\tConfig-ID\n")
            for j, (_, human_kernel_name, _) in enumerate(KERNEL_NAMES):
                for i in range(DENSITY_STEPS):
                    f.write("%s\t%.4f\t%s\n"%(
                        human_kernel_name.replace("\n", " "), 
                        max_densities[i]-base_min_density,
                        "%d-%d"%(j,i)))

if __name__=='__main__':
    pyrenderer.oit.setup_offscreen_context()
    pyrenderer.init()
    runTimings("measure")
    pyrenderer.oit.delete_offscreen_context()