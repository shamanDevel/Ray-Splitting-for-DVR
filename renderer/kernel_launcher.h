#pragma once

#include <vector>
#include <unordered_map>
#include <iostream>
#include <functional>
#include <cassert>
#include <filesystem>

#include <device_launch_parameters.h>
#include <cuda.h>
#if RENDERER_RUNTIME_COMPILATION==0
#include <cuda_runtime_api.h>
#include <cuMat/src/Context.h>
#endif

#include "commons.h"
#include "renderer_settings.cuh"
#include "volume.h"

BEGIN_RENDERER_NAMESPACE
	struct GlobalInstrumentation;
	struct RendererArgs;

#if RENDERER_RUNTIME_COMPILATION==0
//Define the kernel registration class for the runtime api

/**
 * Prototype of the function that launches the kernel. Arguments:
 * - int screenWidth
 * - int screenHeight
 * - cudaStream_t stream
 * - const kernel::RendererDeviceSettings& settings,
 * - cudaTextureObject_t volume_nearest,
 * - cudaTextureObject_t volume_linear,
 * - kernel::OutputTensor& output
 */
typedef std::function<void(
		int, int, cudaStream_t,
		kernel::OutputTensor&,
		kernel::PerPixelInstrumentation* instrumentation)>
	RuntimeApiKernelLauncher;

class KernelLauncher;
class RuntimeApiKernelRegistry
{
	friend class KernelLauncher;

	static RuntimeApiKernelRegistry INSTANCE;
	RuntimeApiKernelRegistry() = default;

	std::unordered_map<std::string, RuntimeApiKernelLauncher> launcher_;

public:
	template<typename T>
	static T reg(const std::string& name, const T& launcher)
	{
		std::cout << "Kernel " << name << " registered" << std::endl;
		INSTANCE.launcher_[name] = launcher;
		return launcher;
	}
};
#endif


#ifdef DEFINE_KERNEL_REGISTRATION

#define CONCATENATE_IMPL(s1, s2) s1##s2
#define CONCATENATE(s1, s2) CONCATENATE_IMPL(s1, s2)
#ifdef __COUNTER__
#define ANONYMOUS_VARIABLE(str) CONCATENATE(str, __COUNTER__)
#else
#define ANONYMOUS_VARIABLE(str) CONCATENATE(str, __LINE__)
#endif

//define the register kernel macro.
//Only defined in a .cpp file, creates global variables
//Because the function can contain commas, it is realized as variadic macro
#define REGISTER_KERNEL(NAME, ...)																			\
	::renderer::RuntimeApiKernelLauncher ANONYMOUS_VARIABLE(RegisteredKernel) =								\
		::renderer::RuntimeApiKernelRegistry::reg(															\
		NAME,																								\
		[](int screenWidth, int screenHeight, cudaStream_t stream,											\
			kernel::OutputTensor& output, kernel::PerPixelInstrumentation* instrumentation)					\
	{																										\
																											\
		cuMat::KernelLaunchConfig cfg = cuMat::Context::current().createLaunchConfig2D(						\
			screenWidth, screenHeight, __VA_ARGS__);														\
		__VA_ARGS__																							\
			<< < cfg.block_count, cfg.thread_per_block, 0, stream >> >										\
			(cfg.virtual_size, output, instrumentation);			\
	}																										\
	);																										\

#endif

/**
 * \brief Custom rendering kernels that are not contained in the main CUDA
 * files and are registered via.
 *
 * Note that the custom kernel launchers are expected to be Singletones.
 */
MY_API class ICustomKernelLauncher
{
public:
	virtual ~ICustomKernelLauncher() {}

	virtual bool render(
		const std::string& kernelName,
		int screenWidth, int screenHeight,
		const kernel::RendererDeviceSettings& deviceSettings,
		const RendererArgs* hostSettings,
		const Volume::MipmapLevel* data,
		kernel::OutputTensor& output,
		cudaStream_t stream,
		kernel::PerPixelInstrumentation* instrumentation,
		GlobalInstrumentation* globalInstrumentation) = 0;

	virtual void reload() {}
	virtual void cleanup() {}
};

MY_API class KernelLauncher
{
public:
	typedef std::function< ICustomKernelLauncher* (const std::string&) > Factory_t;
	
private:
	static KernelLauncher INSTANCE;
	static std::unordered_map<std::string, Factory_t> CUSTOM_KERNELS;

	static std::filesystem::path CACHE_DIR;
	
	KernelLauncher();

public:
	static KernelLauncher& Instance() { return INSTANCE; }

	static void SetCacheDir(const std::filesystem::path& path);
	static bool RegisterCustomKernelLauncher(const std::string& name, const Factory_t& factory);

	enum class KernelTypes
	{
		Iso, //prefix: Iso
		Dvr, //prefix: DVR
		MultiIso, //prefix: MultiIso
		Hybrid, //prefix: Hybrid
		__COUNT__
	};

	/**
	 * \brief Initializes the kernel launcher.
	 * This queries the device and checks for registered kernels
	 */
	bool init();

	/**
	 * \brief Runs clean-up code of any registered (custom) kernel launchers
	 */
	void cleanup();

	/**
	 * \brief Returns the current CUstream.
	 * This is based on cuMat::Context::current().stream .
	 */
	CUstream getCurrentStream();

	/**
	 * \brief Reloads the kernels (in dynamic compilation mode).
	 * \param msg compilation logs
	 * \param enableDebugging should debugging be enabled?
	 * If false, defines KERNEL_NO_DEBUG=1
	 * \return true on success
	 */
	bool reload(std::ostream& msg, bool enableDebugging=true, bool enableInstrumentation=false,
		const std::vector<std::string>& otherPreprocessorArguments = {});

	const std::vector<std::string>& getKernelNames(KernelTypes type) const
	{
		return kernelNames_[int(type)];
	}

#if RENDERER_RUNTIME_COMPILATION==1
	const std::vector<CUdeviceptr>& getConstantAddresses(const std::string& constantName) { return constants_[constantName]; }
#endif

	/**
	 * \brief Launches the kernel with the given name.
	 * The first parameter of the kernel must be the virtual size, followed by
	 * the parameters specified in 'parameters'.
	 * \tparam KernelParamaters 
	 * \param kernelName the name of the kernel
	 * \return true on success
	 */
	bool launchKernel(const std::string& kernelName,
	                  int screenWidth, int screenHeight,
	                  const kernel::RendererDeviceSettings& deviceSettings,
	                  const RendererArgs* hostSettings,
	                  const Volume::MipmapLevel* volume,
	                  kernel::OutputTensor& output,
	                  cudaStream_t stream, 
	                  kernel::PerPixelInstrumentation* instrumentation,
	                  GlobalInstrumentation* globalInstrumentation);

private:
	void sortAndPrintNames();

	CUcontext ctx_;

#if RENDERER_RUNTIME_COMPILATION==1
	const std::vector<std::string> constantNames_;
	std::unordered_map<std::string, std::vector<CUdeviceptr>> constants_;
	std::vector<CUmodule> modules_;
	std::unordered_map<std::string, CUfunction> kernels_;
#endif
	
	std::vector<std::string> kernelNames_[int(KernelTypes::__COUNT__)];
	static const std::string KERNEL_PREFIXES[int(KernelTypes::__COUNT__)];
};



END_RENDERER_NAMESPACE
