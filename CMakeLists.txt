cmake_minimum_required(VERSION 3.18)
cmake_policy(SET CMP0074 NEW) # use _ROOT environment variables
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)
cmake_policy(SET CMP0091 NEW) # set runtime library

project(AnalyticVolumeRenderer)


####################################
# C++ standard
####################################
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_definitions(-DNOMINMAX)
#add_definitions(-D_SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING) # since CUDA does not work with C++17 yet

####################################
# GENERAL THIRD-PARTY DEPENDENCIES
####################################

# CUDA is always required
find_package(CUDA REQUIRED)
if(COMMAND CUDA_SELECT_NVCC_ARCH_FLAGS)
	if (WIN32) # inference-gui
		CUDA_SELECT_NVCC_ARCH_FLAGS(ARCH_FLAGS Auto)
	else() # server
		CUDA_SELECT_NVCC_ARCH_FLAGS(ARCH_FLAGS 6.1)
	endif()
	STRING(REPLACE ";" "=" ARCH_FLAGS2 "${ARCH_FLAGS}")
	LIST(APPEND CUDA_NVCC_FLAGS "${ARCH_FLAGS2}")
	message(STATUS "cuda flags: ${ARCH_FLAGS2}")
endif()
list(APPEND CUDA_NVCC_FLAGS "-std=c++17" "--use_fast_math" "--generate-line-info" "–Xptxas" "-v")
set(MY_CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS})
set(CUDA_SEPARABLE_COMPILATION ON)
#set(CMAKE_CUDA_FLAGS ${CUDA_NVCC_FLAGS})
message(STATUS "final cuda flags: ${CMAKE_CUDA_FLAGS}")
#LIST(APPEND CUDA_NVCC_FLAGS "--keep") #for debugging the .ptx files
enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)

# cuMat-Settings
add_definitions(-DCUMAT_SINGLE_THREAD_CONTEXT=1 -DTHRUST_IGNORE_CUB_VERSION_CHECK=1)

####################################
# OpenGL
####################################
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(GLFW)
find_package(GLM REQUIRED)
# if glfw failed, check using PkgConfig
if(NOT GLFW_FOUND)
	message(STATUS "GLFW could not be found with normal lookup, use PkgConfig instead")
	find_package(PkgConfig REQUIRED)
	pkg_search_module(GLFW REQUIRED glfw3)
else()
	set(GLFW_LIBRARIES ${GLFW_LIBRARY})
	set(GLFW_INCLUDE_DIRS ${GLFW_INCLUDE_DIR})
endif()
# copy shared libraries
if (WIN32)
	# glew dll if running on windows
	string(REPLACE "/lib/" "/bin/" GLEW_BINARY_RELEASEa ${GLEW_SHARED_LIBRARY_RELEASE})
	string(REPLACE ${CMAKE_STATIC_LIBRARY_SUFFIX} ${CMAKE_SHARED_LIBRARY_SUFFIX} GLEW_BINARY_RELEASE ${GLEW_BINARY_RELEASEa})
	file(COPY ${GLEW_BINARY_RELEASE} DESTINATION ${CMAKE_SOURCE_DIR}/bin/)
	# glfw dll
	get_filename_component(GLFW_BINARY_DIRECTORY ${GLFW_LIBRARY} DIRECTORY)
	file(COPY ${GLFW_BINARY_DIRECTORY}/glfw3.dll DESTINATION ${CMAKE_SOURCE_DIR}/bin/)
else()
	# copy glew, glfw, glm
	file(COPY ${GLEW_SHARED_LIBRARY_RELEASE} DESTINATION ${CMAKE_SOURCE_DIR}/bin/)
endif()

####################################
# THE LIBRARY
####################################
add_subdirectory(renderer)

####################################
# TEST APPLICATION
# depend on the library
####################################
add_subdirectory(gui)

####################################
# UNIT TESTS and BENCHMARKS
# depend on the library
####################################
add_subdirectory(tests)

####################################
# PYTHON BINDINGS
# depend on the library
####################################
add_subdirectory(third-party/pybind11)
add_subdirectory(bindings)
