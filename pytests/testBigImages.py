import numpy as np
import matplotlib.pyplot as plt
import os
import imageio
from io import StringIO
from collections import namedtuple

# load pyrenderer
import sys
sys.path.append('../bin')
import pyrenderer

OUTPUT_IMAGE_PATH = "../results/statistics/big-images/"

def blendToWhite(img):
    white = np.ones((img.shape[0], img.shape[1], 3), dtype=img.dtype)
    alpha = img[:,:,3:4]
    return alpha*img[:,:,0:3] + (1-alpha)*white
    
StepErrorTime = namedtuple("StepErrorTime", ["step", "error", "time"])
SceneResult = namedtuple("SceneResult", ["name", "simpson", "coarse", "converged"])
Results = []

def formatNumber(num):
    if num<0:
        return "--"
    exponent = np.floor(np.log10(num))
    base = num / pow(10, exponent)
    #if exponent>3 or exponent<2:
        # write as exponential form
    #return "$%.2f \\times 10^{%d}$"%(base, exponent)
    return "%.2fe%d"%(base, exponent)
    #else:
    #    return "$%.3f$"%num # regular number

def writeResultTable():
    output = StringIO()
    numScenes = len(Results)
    output.write('\\begin{tabular}{c%s} \\toprule\n'%("|ccc"*numScenes))
    output.write('\\multirow{2}{*}{Method} & %s \\\\ \n'%(' & '.join(["\\multicolumn{3}{c|}{%s}"%r.name for r in Results])))
    output.write(' & %s \\\\ \\midrule\n'%(' & '.join(["$s$ & $e$ & $t$ (ms)" for r in Results])))
    ATTRS = [("Simpson-10", "simpson"), ("Stepping", "coarse"), ("Stepping", "converged")]
    for attrName, attrField in ATTRS:
        output.write(attrName)
        for r in Results:
            set = getattr(r, attrField)
            output.write(" & %s & %s & %s"%(formatNumber(set.step), formatNumber(set.error), formatNumber(set.time)))
        output.write("\\\\\n")
    output.write('\\bottomrule\n\\end{tabular}\n')
    output_str = output.getvalue()
    print(output_str)
    with open(OUTPUT_IMAGE_PATH+"results.tex", "w") as f:
        f.write(output_str)

def renderImage(sceneFile : str, name : str, scaleIndependent : bool = False):
    # load scene file
    RESOLUTION = (1920, 1080)
    ROOT_PATH = ".."
    rendererArgs, camera, volumePath = pyrenderer.load_from_json(
        sceneFile, ROOT_PATH)
    print("\n\n============================\n")
    print("settings loaded:", sceneFile)
    rendererArgs.width = RESOLUTION[0]
    rendererArgs.height = RESOLUTION[1]
    volume = pyrenderer.Volume(volumePath)
    print("Loaded volumed of resolution", volume.resolution, 
          "and world size", volume.world_size)
    volume.copy_to_gpu();

    # configuration
    ERROR = 1/256.0
    BASELINE_STEPSIZE = 0.0001
    COARSE_STEPSIZE = 0.2
    if scaleIndependent:
        STEPPING_KERNEL = "DVR: Scale invariant - trilinear"
        ANALYTIC_KERNEL = "DVR: Scale invariant - Simpson"
    elif rendererArgs.dvr_use_shading:
        STEPPING_KERNEL = "DVR: Fixed step size - trilinear"
        ANALYTIC_KERNEL = "DVR: DDA - interval Simpson shaded (10)"
        BASELINE_KERNEL = "DVR: Fixed step size - trilinear (control points) - doubles"
    else:
        STEPPING_KERNEL = "DVR: Fixed step size - trilinear"
        ANALYTIC_KERNEL = "DVR: DDA - interval Simpson (10)"
        BASELINE_KERNEL = "DVR: Fixed step size - trilinear (control points) - doubles"
    print("stepping kernel:", STEPPING_KERNEL)
    print("analytic kernel:", ANALYTIC_KERNEL)
    INITIAL_STEPSIZE = 0.25
    MIN_STEPSIZE = 1e-5
    BINARY_SEARCH_STEPS = 0 # 10
    os.makedirs(OUTPUT_IMAGE_PATH, exist_ok=True)
    outputImagePathBaseline = OUTPUT_IMAGE_PATH + os.path.splitext(os.path.basename(sceneFile))[0] + "-baseline.png"
    outputImagePathCoarse = OUTPUT_IMAGE_PATH + os.path.splitext(os.path.basename(sceneFile))[0] + "-stepping02.png"
    outputImagePathAnalytic = OUTPUT_IMAGE_PATH + os.path.splitext(os.path.basename(sceneFile))[0] + "-simpson.png"
    outputTexPath = OUTPUT_IMAGE_PATH + os.path.splitext(os.path.basename(sceneFile))[0] + ".tex"
        
    # allocate outputs
    timer = pyrenderer.GpuTimer()
    output = pyrenderer.allocate_output(
        rendererArgs.width, rendererArgs.height,
        rendererArgs.render_mode)
    camera.update_render_args(rendererArgs)

    # render baseline
    print("Render baseline")
    rendererArgs.stepsize = BASELINE_STEPSIZE
    pyrenderer.render(BASELINE_KERNEL, volume, rendererArgs, output)
    outputBaseline = np.array(output.copy_to_cpu())
    outputBaselineBlended = blendToWhite(outputBaseline)
    imageio.imwrite(outputImagePathBaseline, outputBaselineBlended)

    # render coarse stepping
    print("Render coarse stepping")
    rendererArgs.stepsize = COARSE_STEPSIZE
    pyrenderer.render(STEPPING_KERNEL, volume, rendererArgs, output)
    outputCoarse = np.array(output.copy_to_cpu())
    outputCoarseBlended = blendToWhite(outputCoarse)
    imageio.imwrite(outputImagePathCoarse, outputCoarseBlended)

    # render analytic
    print("Render analytic")
    rendererArgs.stepsize = 1.0
    timer.start()
    pyrenderer.render(ANALYTIC_KERNEL, volume, rendererArgs, output)
    timer.stop()
    outputAnalytic = np.array(output.copy_to_cpu())
    outputAnalyticBlended = blendToWhite(outputAnalytic)
    errorAnalytic = np.mean(np.abs(outputAnalyticBlended - outputBaselineBlended))
    timeAnalytic = timer.elapsed_ms()
    # and save
    imageio.imwrite(outputImagePathAnalytic, outputAnalyticBlended)
    analyticResult = StepErrorTime(-1, errorAnalytic, timeAnalytic)
    
    def renderWithError(stepsize, prevImage):
        rendererArgs.stepsize = stepsize
        timer.start()
        pyrenderer.render(STEPPING_KERNEL, volume, rendererArgs, output)
        timer.stop()
        outputStepping = np.array(output.copy_to_cpu())
        timeStepping = timer.elapsed_ms()
        outputSteppingBlended = blendToWhite(outputStepping)
        error = np.max(np.abs(prevImage - outputSteppingBlended))
        print("step size", stepsize, "-> error", error)
        return error, timeStepping, outputSteppingBlended
    
    # render stepping (coarse)
    errorSteppingCoarse, timeSteppingCoarse, outputSteppingCoarse = renderWithError(INITIAL_STEPSIZE, outputAnalyticBlended)
    previousError = errorSteppingCoarse
    errorSteppingCoarse = np.mean(np.abs(outputSteppingCoarse - outputBaselineBlended))
    coarseResult = StepErrorTime(INITIAL_STEPSIZE, errorSteppingCoarse, timeSteppingCoarse)
    
    # search until maximal error of ERROR is reached
    stepsizeUpper = INITIAL_STEPSIZE
    stepsizeLower = INITIAL_STEPSIZE/2
    previousImage = outputSteppingCoarse
    currentTime = None
    while True:
        error, currentTime, currentImage = renderWithError(stepsizeLower, previousImage)
        if error < ERROR:
            break
        if error > previousError * 1.2:
            print("ERROR: error increased, cancel binary search")
            stepsizeMid = stepsizeUpper
            error = previousError
            currentImage = previousImage
            BINARY_SEARCH_STEPS = 0
            break
        stepsizeUpper = stepsizeLower
        stepsizeLower /= 2
        previousImage = currentImage
        previousError = error
        stepsizeMid = stepsizeUpper
    # binary search
    for i in range(BINARY_SEARCH_STEPS):
        stepsizeMid = pow(2, 0.5*(np.log2(stepsizeUpper)+np.log2(stepsizeLower)))
        #print(stepsizeUpper, stepsizeLower, "->", stepsizeMid)
        error, currentTime, currentImage = renderWithError(stepsizeMid, previousImage)
        if error < ERROR: # increase stepsize (coarsen)
            stepsizeLower = stepsizeMid
        else: # decrease stepsize (finer)
            stepsizeUpper = stepsizeMid
            previousImage = currentImage
    finalError = np.mean(np.abs(currentImage - outputBaselineBlended))
    convergedResult = StepErrorTime(stepsizeMid, finalError, currentTime)
    print("Final stepsize:", stepsizeMid, "with an error of", error, "and a time of", currentTime, "ms")
    
    Results.append(SceneResult(name, analyticResult, coarseResult, convergedResult))


if __name__=='__main__':
    pyrenderer.oit.setup_offscreen_context()
    pyrenderer.init()

    #pyrenderer.reload_kernels(enableDebugging=False, enableInstrumentation=False)
    pyrenderer.reload_kernels(
        enableDebugging=False,
        enableInstrumentation=False,
        otherPreprocessorArguments=["-DKERNEL_USE_DOUBLE=0"])

    renderImage("../scenes/bigKarpfen.json", "Carp $512^3$")
    renderImage("../scenes/bigThorax2Regular.json", "Thorax $512 \\times 512 \\times 286$")
    renderImage("../scenes/bigBug1.json", "Beetle $832 \\times 832 \\times 494$")
    renderImage("../scenes/bigHuman1.json", "Human $512 \\times 512 \\times 1884$")
    renderImage("../scenes/bigEjecta2FilledUnshaded.json", "Ejecta $512^3$")
    
    #renderImage("../bigRM1Filled.json", "Richtmyer-Meshkov")
    #renderImage("../bigThorax1ScaleInvariant.json", "Thorax", True)
    
    writeResultTable()
    
    pyrenderer.oit.delete_offscreen_context()