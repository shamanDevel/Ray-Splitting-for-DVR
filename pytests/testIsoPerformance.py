import numpy as np
import matplotlib.pyplot as plt
import os
import imageio

# load pyrenderer
import sys
sys.path.append('../bin')
import pyrenderer

def screenShading(input : np.ndarray):
    mask = input[:,:,0:1]
    normals = np.linalg.norm(input[:,:,1:4], axis=2)
    # assume light from front dir=(0,0,1)
    diffuse = input[:,:,3:4] / (normals[:,:,np.newaxis] + 1e-7)
    ambient = 0.2
    color = mask * (ambient + (1-ambient)*diffuse)
    color = (np.clip(color, 0.0, 1.0) * 255).astype(np.uint8)
    return np.concatenate([color]*3, axis=2)

def avg_and_std(values):
    mean = np.mean(values)
    variance = np.mean((values-mean)**2)
    return (mean, np.sqrt(variance))
def weighted_avg_and_std(values, count):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.

    Source and modified: https://stackoverflow.com/a/2415343/1786598
    """
    if np.sum(count)==0:
        return 0.0, 0.0 # everything is zero

    weights = count
    values = values / (count + 1e-7)

    average = np.average(values, weights=weights)
    # Fast and numerically precise:
    variance = np.average((values-average)**2, weights=weights)
    return (average, np.sqrt(variance))

def runTimings(mode : str):
    # CONFIGURATION
    SETTINGS_FILE = "../scenes/isoLobb-scene1.json"
    ROOT_PATH = ".."
    RESOLUTION = (512, 512)
    NUM_FRAMES = 1
    VOLUME_RESOLUTION = 128
    KERNEL_NAMES = [
        ("Iso: Fixed step size - nearest", "stepping nearest 0.1", 0.1),
        ("Iso: Fixed step size - trilinear", "stepping linear 0.1", 0.1),
        ("Iso: Fixed step size - tricubic", "stepping cubic 0.1", 0.1),
        ("Iso: Fixed step size - tricubic", "stepping cubic 0.01", 0.01),
        ("Iso: Fixed step size - tricubic", "stepping cubic 0.001", 0.001),

        ("Iso: DDA - [num] Marmitt (float, stable)", "DDA linear Marmitt", 1),

        ("Iso: Cubic DDA - fixed step (no poly)", "DDA cubic fixed (no poly) 0.1", 0.1),
        ("Iso: Cubic DDA - fixed step (no poly)", "DDA cubic fixed (no poly) 0.01", 0.01),
        ("Iso: Cubic DDA - fixed step (no poly)", "DDA cubic fixed (no poly) 0.001", 0.001),
        ("Iso: Cubic DDA - fixed step (loop)", "DDA cubic fixed (loop) 0.1", 0.1),
        ("Iso: Cubic DDA - fixed step (loop)", "DDA cubic fixed (loop) 0.01", 0.01),
        ("Iso: Cubic DDA - fixed step (loop)", "DDA cubic fixed (loop) 0.001", 0.001),
        ("Iso: Cubic DDA - fixed step (explicit)", "DDA cubic fixed (explicit) 0.1", 0.1),
        ("Iso: Cubic DDA - fixed step (explicit)", "DDA cubic fixed (explicit) 0.01", 0.01),
        ("Iso: Cubic DDA - fixed step (explicit)", "DDA cubic fixed (explicit) 0.001", 0.001),

        ("Iso: Cubic DDA - Sphere Simple (loop)", "DDA cubic Sphere Simple (loop)", 1),
        ("Iso: Cubic DDA - Sphere Bernstein (loop)", "DDA cubic Sphere Bernstein (loop)", 1),
        ("Iso: Cubic DDA - Sphere Simple (explicit)", "DDA cubic Sphere Simple (explicit)", 1),
        ("Iso: Cubic DDA - Sphere Bernstein (explicit)", "DDA cubic Sphere Bernstein (explicit)", 1),
        ]
    KERNEL_NAMES_MEASURE = [
        ("Iso: Cubic DDA - fixed step (no poly)", "Baseline 0.001", 0.001),

        ("Iso: Cubic DDA - fixed step (no poly)", "DDA cubic fixed (no poly) 0.1", 0.1),
        ("Iso: Cubic DDA - fixed step (no poly)", "DDA cubic fixed (no poly) 0.01", 0.01),
        ("Iso: Cubic DDA - fixed step (loop)", "DDA cubic fixed (loop) 0.1", 0.1),
        ("Iso: Cubic DDA - fixed step (loop)", "DDA cubic fixed (loop) 0.01", 0.01),
        ("Iso: Cubic DDA - fixed step (explicit)", "DDA cubic fixed (explicit) 0.1", 0.1),
        ("Iso: Cubic DDA - fixed step (explicit)", "DDA cubic fixed (explicit) 0.01", 0.01),

        ("Iso: Cubic DDA - Sphere Simple (loop)", "DDA cubic Sphere Simple (loop)", 1),
        ("Iso: Cubic DDA - Sphere Bernstein (loop)", "DDA cubic Sphere Bernstein (loop)", 1),
        ("Iso: Cubic DDA - Sphere Simple (explicit)", "DDA cubic Sphere Simple (explicit)", 1),
        ("Iso: Cubic DDA - Sphere Bernstein (explicit)", "DDA cubic Sphere Bernstein (explicit)", 1),
        ]
    TIMING_STEPS = 50
    OUTPUT_STATS_ALL = "../results/statistics/iso-marschner-lobb/timings-all-%s.tsv"
    OUTPUT_STATS_AVG = "../results/statistics/iso-marschner-lobb/timings-avg-%s.tsv"
    OUTPUT_HISTO_ALL = "../results/statistics/iso-marschner-lobb/histograms-%s.tsv"
    OUTPUT_HISTO_CFG = "../results/statistics/iso-marschner-lobb/histogram-cfg-%s.tsv"
    OUTPUT_STATS_USE_DOUBLE = False
    OUTPUT_IMAGE_PATH = "../results/statistics/iso-marschner-lobb/images/"
    OUTPUT_INSTRUMENTATION = "../results/statistics/iso-marschner-lobb/instrumentation.tsv"

    HISTO_NUM_BINS = 100
    HISTO_BIN_MIN = np.log10(1e-6)
    HISTO_BIN_MAX = np.log10(1)
    HISTO_BIN_EDGES = [0.0] + list(10 ** np.linspace(HISTO_BIN_MIN, HISTO_BIN_MAX, HISTO_NUM_BINS))
    print("histogram bins:", HISTO_BIN_EDGES)

    os.makedirs(OUTPUT_IMAGE_PATH, exist_ok=True)

    # load settings file
    rendererArgs, camera, volumePath = pyrenderer.load_from_json(
        SETTINGS_FILE, ROOT_PATH)
    print("settings loaded")
    rendererArgs.width = RESOLUTION[0]
    rendererArgs.height = RESOLUTION[1]

    # create volume
    print("Create Marschner Lobb")
    volume = pyrenderer.Volume.create_marschner_lobb(VOLUME_RESOLUTION)
    print("Loaded volumed of resolution", volume.resolution, 
          "and world size", volume.world_size)
    volume.copy_to_gpu();

    # allocate timing
    timer = pyrenderer.GpuTimer()
    times = [None]*len(KERNEL_NAMES)

    if mode == "visualize":
        # allocate output
        pyrenderer.reload_kernels(enableDebugging=False, enableInstrumentation=False)
        output = pyrenderer.allocate_output(
            rendererArgs.width, rendererArgs.height,
            rendererArgs.render_mode);
        outputs = [None]*len(KERNEL_NAMES)

        # render    
        camera.update_render_args(rendererArgs);
        for j, (kernel_name, _, stepsize) in enumerate(KERNEL_NAMES):
            print("Render", kernel_name, stepsize)
            rendererArgs.stepsize = stepsize
            timer.start()
            pyrenderer.render(kernel_name, volume, rendererArgs, output)
            timer.stop()
            outputs[j] = np.array(output.copy_to_cpu());
            times[j] = timer.elapsed_ms()


        def slugify(value):
            """
            Normalizes string, converts to lowercase, removes non-alpha characters,
            and converts spaces to hyphens.
            """
            import unicodedata
            import re
            value = str(unicodedata.normalize('NFKD', value))#.encode('ascii', 'ignore'))
            value = re.sub('[^\w\s-]', '', value).strip().lower()
            value = re.sub('[-\s]+', '-', value)
            return value

        # visualize
        print("Visualize")
        fig, axes = plt.subplots(ncols=len(KERNEL_NAMES), nrows=1)
        for j, (kernel_name, human_kernel_name, _) in enumerate(KERNEL_NAMES):
            img = screenShading(outputs[j])
            filename = os.path.join(OUTPUT_IMAGE_PATH, 
                                slugify(human_kernel_name)+".png")
            imageio.imwrite(
                filename,
                img)

            axes[j].imshow(img)
            axes[j].set_xlabel(human_kernel_name)

        # save to numpy
        npz_output = {}
        npz_output['kernels'] = KERNEL_NAMES
        for j in range(len(KERNEL_NAMES)):
            npz_output['img_%d'%j] = outputs[j]
        np.savez(os.path.join(OUTPUT_IMAGE_PATH, "raw.npz"), **npz_output)

        plt.subplots_adjust(left=0.03, bottom=0.05, right=0.99, top=0.97, wspace=0.20, hspace=0.23)
        plt.show()

    elif mode == "measure":
        summed_times = [0]*len(KERNEL_NAMES_MEASURE)

        pyrenderer.reload_kernels(
            enableDebugging=False, 
            enableInstrumentation=False, 
            otherPreprocessorArguments=["-DKERNEL_USE_DOUBLE=%s"%("1" if OUTPUT_STATS_USE_DOUBLE else "0")])
        # allocate output for baseline
        output = pyrenderer.allocate_output(
            rendererArgs.width, rendererArgs.height,
            rendererArgs.render_mode);
        gt_outputs = [None] * TIMING_STEPS

        # render and write output
        with open(OUTPUT_STATS_ALL%("double" if OUTPUT_STATS_USE_DOUBLE else "float"), "w") as f:
            f.write("Kernel Name\tFrame\tTime (ms)\tNum False Positives\tNum False Negatives\tMean Abs Depth Error\tVar Abs Depth Error\n")  
            for j, (kernel_name, human_kernel_name, stepsize) in enumerate(KERNEL_NAMES_MEASURE):
                print("Render", kernel_name, stepsize)
                rendererArgs.stepsize = stepsize
                for k in range(TIMING_STEPS):
                    camera.yaw = k * 360 / TIMING_STEPS
                    camera.update_render_args(rendererArgs)
                    timer.start()
                    pyrenderer.render(kernel_name, volume, rendererArgs, output)
                    timer.stop()
                    out_img = np.array(output.copy_to_cpu());
                    if j == 0: # baseline
                        gt_outputs[k] = out_img
                        falsePositives = 0
                        falseNegatives = 0
                        meanDepthError = 0
                        varDepthError = 0
                    else:
                        # compute false positives and negatives
                        falsePositives = np.sum((gt_outputs[k][:,:,0]<out_img[:,:,0])*1.0)
                        falseNegatives = np.sum((gt_outputs[k][:,:,0]>out_img[:,:,0])*1.0)
                        # compute mean depth error
                        mask = (out_img[:,:,0] > 0) & (gt_outputs[k][:,:,0] > 0)
                        depthDiff = np.ma.masked_array(
                            np.abs(gt_outputs[k][:,:,4] - out_img[:,:,4]), mask=mask)
                        meanDepthError = depthDiff.mean()
                        varDepthError = depthDiff.var()
                    t = timer.elapsed_ms()
                    summed_times[j] += t
                    f.write("%s\t%d\t%.4f\t%d\t%d\t%.4f\t%.4f\n"%(
                        human_kernel_name.replace("\n", " "), k, t,
                        falsePositives, falseNegatives, meanDepthError, varDepthError))

        # write average stats
        with open(OUTPUT_STATS_AVG%("double" if OUTPUT_STATS_USE_DOUBLE else "float"), "w") as f:
            f.write("Kernel Name\tAvg-Time (ms)\n")
            for j, (_, human_kernel_name, _) in enumerate(KERNEL_NAMES_MEASURE):
                f.write("%s\t%.4f\n"%(
                    human_kernel_name.replace("\n", " "),
                    summed_times[j] / TIMING_STEPS))

    elif mode == "instrumentation":
        # recompile with instrumentation
        pyrenderer.reload_kernels(enableInstrumentation=True)

        # allocate output
        output = pyrenderer.allocate_output(
            rendererArgs.width, rendererArgs.height,
            rendererArgs.render_mode);
        outputs = [None] * len(KERNEL_NAMES)

        fields = ["densityFetches-avg", "densityFetches-std",
                  "ddaSteps-avg", "ddaSteps-std",
                  "intervalEval-avg", "intervalEval-std", 
                  "intervalStep-avg", "intervalStep-std", 
                  "intervalMaxStep",
                  "timeDensityFetch-avg", "timeDensityFetch-std",
                  "timePolynomialCreation-avg", "timePolynomialCreation-std",
                  "timePolynomialSolution-avg", "timePolynomialSolution-std",
                  "timeTotal-avg", "timeTotal-std"]
        # render    
        with open(OUTPUT_INSTRUMENTATION, "w") as f:
            f.write("Kernel Name\t%s\n"%"\t".join(fields))
            camera.update_render_args(rendererArgs);
            for j, (kernel_name, human_kernel_name, stepsize) in enumerate(KERNEL_NAMES):
                print("Render", kernel_name, stepsize)
                rendererArgs.stepsize = stepsize
                instrumentations = \
                    pyrenderer.render_with_instrumentation(
                        kernel_name, volume, rendererArgs, output)
                densityFetches = avg_and_std(instrumentations["densityFetches"])
                ddaSteps = avg_and_std(instrumentations["ddaSteps"])
                intervalEval = avg_and_std(instrumentations["intervalEval"])
                intervalStep = weighted_avg_and_std(
                    instrumentations["intervalStep"], instrumentations["intervalEval"])
                intervalMaxStep = np.max(instrumentations["intervalMaxStep"])
                timeDensityFetch = weighted_avg_and_std(
                    instrumentations["timeDensityFetch"], instrumentations["timeDensityFetch_NumSamples"])
                timePolynomialCreation = weighted_avg_and_std(
                    instrumentations["timePolynomialCreation"], instrumentations["timePolynomialCreation_NumSamples"])
                timePolynomialSolution = weighted_avg_and_std(
                    instrumentations["timePolynomialSolution"], instrumentations["timePolynomialSolution_NumSamples"])
                timeTotal = avg_and_std(instrumentations["timeTotal"])
                f.write("%s\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%d\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n" % (
                    human_kernel_name.replace("\n", " "),
                    densityFetches[0],densityFetches[1], 
                    ddaSteps[0], ddaSteps[1],
                    intervalEval[0], intervalEval[1], 
                    intervalStep[0], intervalStep[1],
                    intervalMaxStep,
                    timeDensityFetch[0], timeDensityFetch[1], 
                    timePolynomialCreation[0], timePolynomialCreation[1],
                    timePolynomialSolution[0], timePolynomialSolution[1],
                    timeTotal[0], timeTotal[1]))
                f.flush()

if __name__=='__main__':
    pyrenderer.oit.setup_offscreen_context()
    pyrenderer.init()
    #runTimings("measure")
    #runTimings("visualize")
    runTimings("instrumentation")
    pyrenderer.oit.delete_offscreen_context()