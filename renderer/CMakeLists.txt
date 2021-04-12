cmake_minimum_required(VERSION 3.18)

# OUTPUT CACHE VARIABLES:
# BINDING_NAME
# LIBRARY_INCLUDE_DIR
# LIBRARY_LINK_LIBRARIES

option(RENDERER_SHARED_LIB "Build renderer as a shared library, needed only when used as Python extension" OFF)
option(RENDERER_RUNTIME_COMPILATION "Build renderer with runtime compilation of the kernels" OFF)

# your configuraton
set(LIBRARY_HEADER_FILES
	commons.h
	lib.h
	errors.h
	volume.h
	renderer.h
	settings.h
	camera.h
	halton_sampler.h
	tf_texture_1d.h
	inpainting.h
	warping.h
	kernel_launcher.h
	utils.h
	renderer_math_cpu.h
	opengl_framebuffer.h
	opengl_mesh.h
	opengl_oit.h
	opengl_renderer.h
	opengl_shader.h
	opengl_utils.h
	sha1.h
	tf_preintegration.h
	)
set(LIBRARY_CUDA_HEADER_FILES
	helper_math.cuh
	renderer_commons.cuh
	renderer_settings.cuh
	renderer_utils.cuh
	renderer_impl_iso.cuh
	renderer_impl_dvr.cuh
	renderer_kernels_iso.cuh
	renderer_kernels_dvr.cuh
	renderer_color.cuh
	renderer_math.cuh
	renderer_math_iso.cuh
	)
set(LIBRARY_CUDA_FILES
	tf_texture_1d.cu
	volume.cu
	inpainting.cu
	warping.cu
	renderer_static.cu
	opengl_framebuffer.cu
	opengl_renderer.cu
	tf_preintegration.cu
	)
set(LIBRARY_SOURCE_FILES
	volume.cpp
	volume_implicit.cpp
	settings.cpp
	camera.cpp
	tf_texture_1d.cpp
	python_bindings.cpp
	kernel_launcher.cpp
	renderer.cpp
	renderer_rtc.cpp
	renderer_math_cpu.cpp
	opengl_mesh.cpp
	opengl_oit.cpp
	opengl_renderer.cpp
	opengl_shader.cpp
	sha1.cpp
	)
set(LIBRARY_RESOURCES
	shaders/PassThrough.vs
	shaders/SingleIso.vs
	shaders/SingleIso.fs
	shaders/ShowNormals.fs
	shaders/OITIsoRendering.fs
	shaders/OITIsoBlending.fs
	shaders/OITDvrBlending.fs
	shaders/ScreenQuad.vs
	)
# the target name of the library
set(LIBRARY_NAME Renderer CACHE INTERNAL "the target name of the library, also used for the binding")

#add_library(${LIBRARY_NAME}_cuda OBJECT ${LIBRARY_CUDA_FILES})

# the library, compiled as a shared library
set(LIBRARY_ALL_FILES
	${LIBRARY_HEADER_FILES}
	${LIBRARY_CUDA_HEADER_FILES}
	${LIBRARY_SOURCE_FILES}
	${LIBRARY_CUDA_FILES}
	${LIBRARY_RESOURCES}
)
if(RENDERER_SHARED_LIB)
	add_library(${LIBRARY_NAME} SHARED ${LIBRARY_ALL_FILES})
else(RENDERER_SHARED_LIB)
	add_library(${LIBRARY_NAME} STATIC ${LIBRARY_ALL_FILES})
endif(RENDERER_SHARED_LIB)
source_group("CUDA Headers" REGULAR_EXPRESSION ".*\.cuh")
source_group("Shaders" REGULAR_EXPRESSION "(.*\.vs)|(.*\.fs)")

set_target_properties(${LIBRARY_NAME} PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
)
target_compile_options(${LIBRARY_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${MY_CUDA_NVCC_FLAGS}>)
set_property(TARGET ${LIBRARY_NAME} PROPERTY CUDA_STANDARD 17)
# get include path to the first library header, to be used in the test application
list(GET ${LIBRARY_HEADER_FILES} 0 LIBRARY_INCLUDE_DIRb)
get_filename_component(LIBRARY_INCLUDE_DIRa ${LIBRARY_INCLUDE_DIRb} ABSOLUTE)
get_filename_component(LIBRARY_INCLUDE_DIRc ${LIBRARY_INCLUDE_DIRa} DIRECTORY)
set(LIBRARY_INCLUDE_DIR ${LIBRARY_INCLUDE_DIRc} CACHE FILEPATH "include directory of the custom library") 
target_include_directories(${LIBRARY_NAME}
	PRIVATE 
		${PYTHON_INCLUDE_DIR} 
		${CMAKE_SOURCE_DIR}/third-party/cuMat 
		${CMAKE_SOURCE_DIR}/third-party/cuMat/third-party
		${GLM_INCLUDE_DIRS}
		${GLEW_INCLUDE_DIR} 
		${GLFW_INCLUDE_DIRS}
		${CMAKE_SOURCE_DIR}/third-party/nlohmann
		${CMAKE_SOURCE_DIR}/third-party/tinyformat
	INTERFACE ${LIBRARY_INCLUDE_DIR}
	)
# libraries, also used in the test application
set(LIBRARY_LINK_LIBRARIES
	${PYTHON_LIBRARY}
	${CUDA_LIBRARIES}
	CUDA::cuda_driver CUDA::nvrtc
	${OPENGL_LIBRARY} 
	${GLEW_SHARED_LIBRARY_RELEASE} 
	${GLFW_LIBRARIES}
	CACHE STRING "python, cuda and OpenGL libraries")
target_link_libraries(${LIBRARY_NAME}
	${LIBRARY_LINK_LIBRARIES})
set_property(TARGET ${LIBRARY_NAME} PROPERTY CXX_STANDARD 17)
#add_dependencies(${LIBRARY_NAME} CUDA::cuda_driver CUDA::nvrtc)

if(RENDERER_SHARED_LIB)
	target_compile_definitions(${LIBRARY_NAME}
		PUBLIC RENDERER_BUILD_SHARED 
		PRIVATE BUILD_MAIN_LIB)
else(RENDERER_SHARED_LIB)
	# no extra definitions
endif(RENDERER_SHARED_LIB)

# runtime compilation details
if (RENDERER_RUNTIME_COMPILATION)
	target_compile_definitions(${LIBRARY_NAME} PUBLIC RENDERER_RUNTIME_COMPILATION=1)
	message(STATUS "Enabled runtime compilation. Use this only during development, as the kernel code is referenced using absolute paths")
	list(GET ARCH_FLAGS -1 RENDERER_NVCC_FLAGS)
	target_compile_definitions(${LIBRARY_NAME} PRIVATE RENDERER_SHADER_DIR=${CMAKE_CURRENT_SOURCE_DIR} NVCC_ARGS="${RENDERER_NVCC_FLAGS}")
else(RENDERER_RUNTIME_COMPILATION)
	target_compile_definitions(${LIBRARY_NAME} PUBLIC RENDERER_RUNTIME_COMPILATION=0)
endif(RENDERER_RUNTIME_COMPILATION)

# additional cuda stuff
set_property(TARGET ${LIBRARY_NAME} PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)

if(RENDERER_SHARED_LIB)
# post-build commands, copies the dll to the bin/ folder
get_filename_component(LIBRARY_INCLUDE_DIR ${LIBRARY_INCLUDE_DIRa} DIRECTORY)
add_custom_command(TARGET ${LIBRARY_NAME}
	POST_BUILD
	COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_SOURCE_DIR}/bin
	COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:${LIBRARY_NAME}> ${CMAKE_SOURCE_DIR}/bin/${LIBRARY_NAME}${CMAKE_SHARED_LIBRARY_SUFFIX}

	COMMENT "Copies the libarary .dll to bin/"
	WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/..
	
	VERBATIM
	)
endif(RENDERER_SHARED_LIB)

set_property(TARGET ${LIBRARY_NAME} PROPERTY
  MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

# debugging: keep .ptx files of the kernels
#set_property(TARGET ${LIBRARY_NAME}_cuda PROPERTY CUDA_PTX_COMPILATION ON)
#target_compile_options(${LIBRARY_NAME} PRIVATE "--keep")


# Include resources in executable
file(DOWNLOAD "https://raw.githubusercontent.com/vector-of-bool/cmrc/master/CMakeRC.cmake"
					"${CMAKE_BINARY_DIR}/CMakeRC.cmake")
include("${CMAKE_BINARY_DIR}/CMakeRC.cmake")
if (RENDERER_RUNTIME_COMPILATION)
	# empty resources
	cmrc_add_resource_library(
		${LIBRARY_NAME}-resources

		ALIAS ${LIBRARY_NAME}::rc
		NAMESPACE renderer
		)
	target_link_libraries(${LIBRARY_NAME} ${LIBRARY_NAME}::rc)
	set_property(TARGET ${LIBRARY_NAME}-resources PROPERTY
	  MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
else(RENDERER_RUNTIME_COMPILATION)
	cmrc_add_resource_library(
		${LIBRARY_NAME}-resources

		ALIAS ${LIBRARY_NAME}::rc
		NAMESPACE renderer

		${LIBRARY_RESOURCES}
		)
	target_link_libraries(${LIBRARY_NAME} ${LIBRARY_NAME}::rc)
	set_property(TARGET ${LIBRARY_NAME}-resources PROPERTY
	  MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif(RENDERER_RUNTIME_COMPILATION)
