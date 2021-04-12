import numpy as np
import pyrenderer

TIMING_STEPS = 10

def getKernelNames(volume_resolution):
  KERNEL_NAMES_NORMAL = [
    #("DVR: DDA - fixed step (control points)", "baseline", 0.001),
    ("DVR: Fixed step size - trilinear (control points) - doubles", "baseline", 0.0001),

    ("DVR: Fixed step size - trilinear (control points)", "stepping 0.1", 0.1),
    ("DVR: Fixed step size - trilinear (control points)", "stepping 0.01", 0.01),
    ("DVR: Fixed step size - trilinear (control points)", "stepping 0.001", 0.001),
    ("DVR: Fixed step size - preintegrate 1D", "preintegrate 1D 0.1", 0.1),
    ("DVR: Fixed step size - preintegrate 1D", "preintegrate 1D 0.01", 0.01),
    ("DVR: Fixed step size - preintegrate 1D", "preintegrate 1D 0.001", 0.001),
    ("DVR: Fixed step size - preintegrate 2D", "preintegrate 2D 0.1", 0.1),
    ("DVR: Fixed step size - preintegrate 2D", "preintegrate 2D 0.01", 0.01),
    ("DVR: Fixed step size - preintegrate 2D", "preintegrate 2D 0.001", 0.001),

    ("DVR: DDA - interval simple", "interval - simple", 1),
    ("DVR: DDA - interval stepping (3)", "interval - stepping-3", 1),
    ("DVR: DDA - interval trapezoid (2)", "interval - trapezoid-2", 1),
    ("DVR: DDA - interval trapezoid (4)", "interval - trapezoid-4", 1),
    ("DVR: DDA - interval trapezoid (10)", "interval - trapezoid-10", 1),
    ("DVR: DDA - interval Simpson (2)", "interval - Simpson-2", 1),
    ("DVR: DDA - interval Simpson (4)", "interval - Simpson-4", 1),
    ("DVR: DDA - interval Simpson (10)", "interval - Simpson-10", 1),
    ("DVR: DDA - interval Simpson adapt", "interval - Simpson-adaptive e-3", 1e-3 * volume_resolution),
    ("DVR: DDA - interval Simpson adapt", "interval - Simpson-adaptive e-5", 1e-5 * volume_resolution),
    ("DVR: Marching Cubes", "marching cubes 1", 1/1-0.001), # number of subdivisions
    ("DVR: Marching Cubes", "marching cubes 2", 1/2-0.001),
    ("DVR: Marching Cubes", "marching cubes 4", 1/4-0.001),
    #("DVR: Marching Cubes", "marching cubes 8", 1/8-0.001),
    #("DVR: Marching Cubes", "marching cubes 16", 1/16-0.001),
  ]
  KERNEL_NAMES_TIMING = [
    #("DVR: DDA - fixed step (control points)", "baseline", 0.001),
    ("DVR: Fixed step size - trilinear (control points) - doubles", "baseline", 0.0001),

    ("DVR: Fixed step size - trilinear", "stepping 0.1", 0.1),
    ("DVR: Fixed step size - trilinear", "stepping 0.01", 0.01),
    ("DVR: Fixed step size - trilinear", "stepping 0.001", 0.001),
    ("DVR: Fixed step size - preintegrate 1D", "preintegrate 1D 0.1", 0.1),
    ("DVR: Fixed step size - preintegrate 1D", "preintegrate 1D 0.01", 0.01),
    ("DVR: Fixed step size - preintegrate 1D", "preintegrate 1D 0.001", 0.001),
    ("DVR: Fixed step size - preintegrate 2D", "preintegrate 2D 0.1", 0.1),
    ("DVR: Fixed step size - preintegrate 2D", "preintegrate 2D 0.01", 0.01),
    ("DVR: Fixed step size - preintegrate 2D", "preintegrate 2D 0.001", 0.001),

    ("DVR: DDA - interval simple", "interval - simple", 1),
    ("DVR: DDA - interval stepping (3)", "interval - stepping-3", 1),
    ("DVR: DDA - interval trapezoid (2)", "interval - trapezoid-2", 1),
    ("DVR: DDA - interval trapezoid (4)", "interval - trapezoid-4", 1),
    ("DVR: DDA - interval trapezoid (10)", "interval - trapezoid-10", 1),
    ("DVR: DDA - interval Simpson (2)", "interval - Simpson-2", 1),
    ("DVR: DDA - interval Simpson (4)", "interval - Simpson-4", 1),
    ("DVR: DDA - interval Simpson (10)", "interval - Simpson-10", 1),
    ("DVR: DDA - interval Simpson adapt", "interval - Simpson-adaptive e-3", 1e-3 * volume_resolution),
    ("DVR: DDA - interval Simpson adapt", "interval - Simpson-adaptive e-5", 1e-5 * volume_resolution),
    # ("DVR: DDA - interval trapezoid var", "interval - trapezoid-var 0.1", 0.1),
    # ("DVR: DDA - interval trapezoid var", "interval - trapezoid-var 0.01", 0.01),
    # ("DVR: DDA - interval trapezoid var", "interval - trapezoid-var 0.001", 0.001),
    ("DVR: Marching Cubes", "marching cubes 1", 1 / 1 - 0.001),  # number of subdivisions
    ("DVR: Marching Cubes", "marching cubes 2", 1 / 2 - 0.001),
    ("DVR: Marching Cubes", "marching cubes 4", 1 / 4 - 0.001),
    #("DVR: Marching Cubes", "marching cubes 8", 1 / 8 - 0.001),
    #("DVR: Marching Cubes", "marching cubes 16", 1 / 16 - 0.001),
  ]
  return KERNEL_NAMES_NORMAL, KERNEL_NAMES_TIMING

def smoothstep(t:float, a:float=0, b:float=1):
  if b == a: return a
  x = np.clip((t-a)/(b-a), 0.0, 1.0)
  return x * x * (3 - 2*x)
def smootherstep(t:float, a:float=0, b:float=1):
  if b==a: return a
  x = np.clip((t-a)/(b-a), 0.0, 1.0)
  return x * x * x * (x * (x * 6 - 15) + 10)
def lerp(x, a, b):
  return a + x * (b-a)


def multiIso2Linear(multiiso_tf, width : float, filling:float=0):
  """
  Converts the multi-iso tf (renderer.TfMultiIso) to renderer.TfLinear
  with the given peak width
  """
  linear_tf = pyrenderer.TfLinear()

  # convert colors
  colors_lab = []
  colors_opacities = []
  for c in multiiso_tf.colors:
    xyz = pyrenderer.float3()
    xyz.x = c.x; xyz.y = c.y; xyz.z = c.z
    lab = pyrenderer.xyzToLab(xyz)
    colors_lab.append(lab)
    colors_opacities.append(c.w)

  # set color control points
  colorDensities = []
  colorValues = []
  colorDensities.append(0)
  colorValues.append(colors_lab[0])
  for i in range(len(multiiso_tf.colors)):
    colorDensities.append(multiiso_tf.densities[i])
    colorValues.append(colors_lab[i])
  colorDensities.append(1)
  colorValues.append(colors_lab[-1])
  linear_tf.density_axis_color = colorDensities
  linear_tf.color_axis = colorValues

  # set density control points
  opacityDensities = []
  opacityValues = []
  for i in range(len(multiiso_tf.colors)):
    pos = multiiso_tf.densities[i]
    if i==0:
      minPos = 0
    else:
      minPos = 0.5 * (pos + multiiso_tf.densities[i-1])
    if i == (len(multiiso_tf.colors)-1):
      maxPos = 1
    else:
      maxPos = 0.5 * (pos + multiiso_tf.densities[i+1])
    minPos = max(minPos, pos - width)
    maxPos = min(maxPos, pos + width)
    opacityDensities.append(minPos)
    opacityValues.append(0)
    opacityDensities.append(pos)
    opacityValues.append(multiiso_tf.colors[i].w)
    opacityDensities.append(maxPos)
    opacityValues.append(0)
  opacityValues[-1] = filling
  linear_tf.density_axis_opacity = opacityDensities
  linear_tf.opacity_axis = opacityValues

  return linear_tf
