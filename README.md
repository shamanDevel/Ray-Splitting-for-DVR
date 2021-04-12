# Ray Splitting for DVR

Code and datasets for "Analytic Ray Splitting for Controlled Precision DVR", Sebastian Weiss and Rüdiger Westermann, EuroVis 2021.

Currently supported:
 - Iso-surface rendering 
   - Trilinear, analytic + numerical
   - Tricubic, numerical
   - Multiple semi-transparent isosurfaces
 - Direct Volume Rendering
   - trilinear, piecewise linear transfer function
   - rectangle rule, trapezoid rule, Simpson rule. With fixed number of control points, variable number of control points (analytic bounds), iterative refinement (Simpson only)
   - scale-invariant DVR
- Hybrid of DVR+Iso



## Project structure:
 - renderer: the library that contains the rendering core (C++, CUDA)
 - gui: interactive gui, allows to test all available options (C++, OpenGL)
 - bindings: python bindings to the renderer core (C++, Python)
 - tests: test cases and evaluation written in C++python. The code that generated the data for the plots in the paper

## Data

- volumes: all test volumes used in the plots and images. 
  Note that the synthetic datasets (Marschner Lobb, Sphere, Tube) are are generated in the program, not loaded from external files
- scenes: scene settings to replicate the images in the paper

See the releases for compiled binaries.


## Requirements

 - CUDA
 - OpenGL with GLFW and GLM

Tested with CUDA 11.0 (device architecture 7.5), Windows 10, Visual Studio 2019