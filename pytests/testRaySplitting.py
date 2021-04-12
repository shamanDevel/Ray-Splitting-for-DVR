import numpy as np
from collections import deque
import matplotlib.pyplot as plt

def _lerp(x, a, b):
  if isinstance(a, (list, tuple)):
    return [_lerp(x, ia,ib) for (ia,ib) in zip(a,b)]
  return a + x * (b-a)

density_min_x = 0
density_max_x = 4
def f_density(x):
  #0.08 + 0.7 x - 0.52 x^2 + 0.1 x^3
  return 0.1*(x-0.2)*(x-2)*(x-3)+0.2
density_coeff = [0.08, 0.7, -0.52, 0.1]

tf_min_x = 0
tf_max_x = 1
tf_points = [(0,0), (0.25,0.01), (0.35,0.7), (0.45,0.2), (0.675,0.3), (0.725,0),(1,0)]
def f_tf(x):
  for i in range(len(tf_points)-1):
    p0,x0 = tf_points[i]
    p1,x1 = tf_points[i+1]
    if p0 <= x < p1:
      return _lerp((x-p0)/(p1-p0), x0, x1)
  return 0

intersectionsT = [0.31, 0.70425, 1.1473, 1.7422, 3.1478, 3.3520, 3.5034, 3.7588, 3.8061]
intersectionsI = [1,2,2,1,1,2,3,4,5]

def emitInterval(case, i0, i1, t0, t1, axis):
  d0, d1 = f_density(t0), f_density(t1)
  x0, x1 = tf_points[i0][1], tf_points[i1][1]
  print("EmitInterval %s: i0=%d, i1=%d, t0=%.2f, t1=%.2f, d0=%.2f, d1=%.2f, x0=%.2f, x1=%.2f"%(
    case, i0, i1, t0, t1, d0, d1, x0, x1))
  x0, x1 = _lerp((d0 - tf_points[i0][0]) / (tf_points[i1][0] - tf_points[i0][0]), x0, x1), \
           _lerp((d1 - tf_points[i0][0]) / (tf_points[i1][0] - tf_points[i0][0]), x0, x1)
  print("  after lerp: x0=%.2f, x1=%.2f"%(x0, x1))
  # compute new coefficients
  alpha = (x0-x1)/(d0-d1)
  ax = density_coeff
  bx = [x0+alpha*(ax[0]-d0), alpha*ax[1], alpha*ax[2], alpha*ax[3]]
  print("  start-mid-end:", np.polyval(bx[::-1], np.linspace(t0, t1,3)))
  # plot
  X = np.linspace(t0, t1, 50)
  Y = np.polyval(bx[::-1], X)
  axis.plot(X, Y)

def emitPoint(i, t):
  d = f_density(t)
  x = tf_points[i][1]
  print("EmitPoint: i=%d, t=%.2f, d=%.2f, x=%.2f"%(
    i, t, d, x))

def raySplitting(ax):
  t0 = density_min_x
  t1 = density_max_x
  N = len(tf_points)
  # compute isosurface intersections
  roots = [deque() for i in range(N)]
  for t,i in zip(intersectionsT, intersectionsI):
    roots[i].append(t)
  # find initial control point interval
  i = 0
  for j in range(N-1):
    if tf_points[i][0] <= f_density(t0) < tf_points[i+1][0]:
      i = j
      break
  print("Start in interval %d - %d"%(i, i+1))
  i_last = -1
  # main loop
  while True:
    t_lower = roots[i][0] if len(roots[i])>0 else t1+1 # np.infty
    t_upper = roots[i+1][0] if len(roots[i+1]) > 0 else t1 + 1  # np.infty
    if t1<t_lower and t1<t_upper: #exit
      emitInterval("exit", i, i+1, t0, t1, ax)
      return
    if t_lower < t_upper:
      roots[i].popleft()
      if i_last == i:
        emitInterval("IIIa", i, i+1, t0, t_lower, ax) # case IIIa
      else:
        emitInterval("II  ", i+1, i, t0, t_lower, ax) # case II
      emitPoint(i, t_lower)
      i_last = i; t0 = t_lower; i -= 1
    else:
      roots[i+1].popleft()
      if i_last == i+1:
        emitInterval("IIIb", i+1, i, t0, t_upper, ax) # case IIIb
      else:
        emitInterval("I   ", i, i+1, t0, t_upper, ax) # case I
      emitPoint(i+1, t_upper)
      i_last = i+1; t0 = t_upper; i += 1

if __name__ == '__main__':
  fig, axes = plt.subplots(1, 3)

  # TF
  X = np.linspace(tf_min_x, tf_max_x)
  Y = [f_tf(x) for x in X]
  axes[0].plot(X, Y)

  # density
  X = np.linspace(density_min_x, density_max_x)
  Y = [f_density(x) for x in X]
  axes[1].plot(X, Y)

  # result
  raySplitting(axes[2])

  plt.show()