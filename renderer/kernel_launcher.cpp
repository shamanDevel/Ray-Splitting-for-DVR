#include "kernel_launcher.h"
#include <cuMat/src/Context.h>
#include <sstream>
#include <fstream>
#include <algorithm>
#include <cctype>
#include <locale>
#include <filesystem>
#include <assert.h>
#include <nvrtc.h>

#include "renderer.h"
#include "sha1.h"

namespace fs = std::filesystem;

#if RENDERER_RUNTIME_COMPILATION==0
renderer::RuntimeApiKernelRegistry renderer::RuntimeApiKernelRegistry::INSTANCE;
#endif

renderer::KernelLauncher renderer::KernelLauncher::INSTANCE;
std::unordered_map<std::string, renderer::KernelLauncher::Factory_t> renderer::KernelLauncher::CUSTOM_KERNELS;
fs::path renderer::KernelLauncher::CACHE_DIR = "kernel_cache";

const std::string renderer::KernelLauncher::KERNEL_PREFIXES[] = {
	std::string("Iso"),
	std::string("DVR"),
	std::string("MultiIso"),
	std::string("Hybrid")
};

static void throwOnError(CUresult err, const char* file, const int line)
{
	if (err != CUDA_SUCCESS)
	{
		const char* pStr;
		cuGetErrorString(err, &pStr);
		const char* pName;
		cuGetErrorName(err, &pName);
		std::stringstream ss;
		ss << "Cuda error " << pName << " at " << file << ":" << line << " : " << pStr;
		throw std::exception(ss.str().c_str());
	}
}
#define CU_SAFE_CALL( err ) throwOnError( err, __FILE__, __LINE__ )

static void throwOnNvrtcError(nvrtcResult result, const char* file, const int line)
{
	if (result != NVRTC_SUCCESS) {
		std::stringstream ss;
		ss << "NVRTC error at " << file << ":" << line << " : " << nvrtcGetErrorString(result);
		throw std::exception(ss.str().c_str());
	}
}
#define NVRTC_SAFE_CALL( err ) throwOnNvrtcError( err, __FILE__, __LINE__ )


renderer::KernelLauncher::KernelLauncher()
	: ctx_(nullptr)
#if RENDERER_RUNTIME_COMPILATION==1
	, constantNames_({ "cAOHemisphere", "cAORandomRotations", "cDeviceSettings" })
#endif
{
}

void renderer::KernelLauncher::SetCacheDir(const std::filesystem::path& path)
{
	CACHE_DIR = path;
}

bool renderer::KernelLauncher::RegisterCustomKernelLauncher(const std::string& name, const Factory_t& factory)
{
	if (auto it = CUSTOM_KERNELS.find(name); it == CUSTOM_KERNELS.end())
	{ // C++17 init-if ^^
		CUSTOM_KERNELS[name] = factory;
		std::cout << "Registered custom kernel launcher \"" << name << "\"" << std::endl;
		return true;
	}
	return false;
}

bool renderer::KernelLauncher::init()
{
	//query context
	CUMAT_SAFE_CALL(cudaSetDevice(0)); //initializes runtime api
#if RENDERER_RUNTIME_COMPILATION==0
	//load kernels from registry
	for (const auto& e : RuntimeApiKernelRegistry::INSTANCE.launcher_)
	{
		int typeIndex = -1;
		for (int i = 0; i<int(KernelTypes::__COUNT__); ++i)
		{
			if (e.first.substr(0, KERNEL_PREFIXES[i].size()) == KERNEL_PREFIXES[i])
			{
				typeIndex = i;
				kernelNames_[typeIndex].push_back(e.first);
				break;
			}
		}
		if (typeIndex < 0)
		{
			std::cerr << "Unknown kernel type: " << e.first << std::endl;
		}
	}

	//append kernels from custom registration
	for (const auto& e : CUSTOM_KERNELS)
	{
		const auto& name = e.first;
		int typeIndex = -1;
		for (int i = 0; i<int(KernelTypes::__COUNT__); ++i)
		{
			if (name.substr(0, KERNEL_PREFIXES[i].size()) == KERNEL_PREFIXES[i])
			{
				typeIndex = i;
				break;
			}
		}
		if (typeIndex >= 0)
		{
			kernelNames_[typeIndex].push_back(name);
		}
		else
		{
			std::cerr << "Unknown kernel type: " << name << " (custom registration)" << std::endl;
		}
	}

	
	sortAndPrintNames();
	return true;
#else
	
	CU_SAFE_CALL(cuCtxGetCurrent(&ctx_));
	if (reload(std::cout))
	{
		std::cout << "Kernels loaded successfully" << std::endl;
		return true;
	}
	else
	{
		std::cerr << "Failed to load kernels" << std::endl;
		return false;
	}
#endif
}

void renderer::KernelLauncher::cleanup()
{
	for (auto& l : CUSTOM_KERNELS)
		l.second(l.first)->cleanup();
}

CUstream renderer::KernelLauncher::getCurrentStream()
{
	return static_cast<CUstream>(cuMat::Context::current().stream());
}

// trim from start (in place)
static inline void ltrim(std::string &s) {
	s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](int ch) {
		return !std::isspace(ch);
	}));
}

// trim from end (in place)
static inline void rtrim(std::string &s) {
	s.erase(std::find_if(s.rbegin(), s.rend(), [](int ch) {
		return !std::isspace(ch);
	}).base(), s.end());
}

// trim from both ends (in place)
static inline std::string trim(std::string s) {
	ltrim(s);
	rtrim(s);
	return s;
}

bool renderer::KernelLauncher::reload(std::ostream& msg, bool enableDebugging, bool enableInstrumentation,
	const std::vector<std::string>& otherPreprocessorArguments)
{
#if RENDERER_RUNTIME_COMPILATION==0
	msg << "Runtime compilation disabled via CMake settings, unable to reload kernels.";
	return false;
#else
	//reload custom kernel launchers
	for (auto& e : CUSTOM_KERNELS)
		e.second(e.first)->reload();
	
	//Free previous kernels and modules
	kernels_.clear();
	for (const auto& m : modules_)
		CU_SAFE_CALL(cuModuleUnload(m));
	modules_.clear();
	for (int i = 0; i<int(KernelTypes::__COUNT__); ++i)
		kernelNames_[i].clear();
	constants_.clear();

	fs::path root(CUMAT_STR(RENDERER_SHADER_DIR));

	//enumerate all *.cuh files
	//files with __global__ are deemed source files and are compiled,
	//all other .cuh files are include files
	typedef std::pair<std::string, std::string> NameAndContent;
	std::vector<NameAndContent> includeFiles;
	std::vector<NameAndContent> sourceFiles;
	for (const auto& p : fs::directory_iterator(root))
	{
		if (p.path().extension() == ".cuh")
		{
			try
			{
				std::ifstream t(p.path());
				std::ostringstream ss;
				ss << t.rdbuf();
				std::string buffer = ss.str();
				//t.seekg(0, std::ios::end);
				//size_t size = t.tellg();
				//std::string buffer(size, ' ');
				//t.seekg(0);
				//t.read(&buffer[0], size);
				if (buffer.find("__global__") != std::string::npos)
				{
					sourceFiles.emplace_back(p.path().filename().string(), buffer);
				}
				else
				{
					includeFiles.emplace_back(p.path().filename().string(), buffer);
				}
				msg << "Loaded file " << p.path() << std::endl;
			} catch (const std::exception& ex)
			{
				msg << "Unable to read file " << p.path() << ": " << ex.what() << std::endl;
			}
		}
		else if (p.path().extension() == ".inl")
		{
			try
			{
				std::ifstream t(p.path());
				std::ostringstream ss;
				ss << t.rdbuf();
				std::string buffer = ss.str();
				includeFiles.emplace_back(p.path().filename().string(), buffer);
				msg << "Loaded file " << p.path() << std::endl;
			}
			catch (const std::exception& ex)
			{
				msg << "Unable to read file " << p.path() << ": " << ex.what() << std::endl;
			}
		}
	}

	std::vector<const char*> headerContents(includeFiles.size());
	std::vector<const char*> headerNames(includeFiles.size());
	for (size_t i=0; i<includeFiles.size(); ++i)
	{
		headerContents[i] = includeFiles[i].second.c_str();
		headerNames[i] = includeFiles[i].first.c_str();
	}
	const char * const *headers = includeFiles.empty() ? nullptr : headerContents.data();
	const char * const *includeNames = includeFiles.empty() ? nullptr : headerNames.data();

	//get compute version for compiler args
	//From CMake, we get from NVCC_ARGS something like "arch=compute_75,code=sm_75"
	//let's just take the last two character, hope that they are the compute version
	//and convert them to "--gpu-architecture=compute_75"
	std::string nvccArgsFromCMake(NVCC_ARGS);
	std::string nvccComputeVersion = nvccArgsFromCMake.substr(nvccArgsFromCMake.size() - 2, 2);
	std::string newNvccArgs = "--gpu-architecture=compute_" + nvccComputeVersion;
	msg << "NVCC args: " << newNvccArgs << "\n";
	std::vector<const char*> opts { "--std=c++17", "--use_fast_math", newNvccArgs.c_str(), "-D__NVCC__=1" };
	if (!enableDebugging)
		opts.push_back("-DKERNEL_NO_DEBUG=1");
	opts.push_back(enableInstrumentation ? "-DKERNEL_INSTRUMENTATION=1" : "-DKERNEL_INSTRUMENTATION=0");
	for (const auto& s : otherPreprocessorArguments) opts.push_back(s.c_str());
	const int numOpts = opts.size();

	//compile every source file
	for (const auto& source : sourceFiles)
	{
		msg << "Compile " << source.first << std::endl;
		try {
			//parse for kernel definitions
			msg << "Parse kernel definitions" << std::endl;
			std::istringstream iss(source.second);
			typedef std::pair<std::string, std::string> NameAndType;
			std::vector<NameAndType> kernels;
			for (std::string line; std::getline(iss, line);)
			{
				auto pos = line.find("REGISTER_KERNEL");
				if (pos == std::string::npos) continue; //no REGISTER_KERNEL
				if (line.find("#") != std::string::npos) continue; //the macro definition
				//split into name and type
				//For now, assume that there is no escaped \"
				if (line.find("\\\"") != std::string::npos)
				{
					msg << "Names with an escaped \" are not supported yet:" << line << "\n";
					continue;
				}
				auto nameStart = line.find('\"', pos)+1;
				auto nameEnd = line.find('\"', nameStart);
				std::string name = line.substr(nameStart, nameEnd - nameStart);
				auto commentPos = line.find("//");
				if (commentPos != std::string::npos && commentPos < pos)
				{
					msg << "Kernel " << name << " commented out" << "\n";
					continue;
				}
				auto typeStart = line.find(',', nameEnd) + 1;
				auto typeEnd = line.find_last_of(')');
				std::string type = trim(line.substr(typeStart, typeEnd - typeStart));
				msg << "Kernel with name \"" << name << "\" and type \"" << type << "\" found\n";

				int typeIndex = -1;
				for (int i=0; i<int(KernelTypes::__COUNT__); ++i)
				{
					if (name.substr(0, KERNEL_PREFIXES[i].size()) == KERNEL_PREFIXES[i])
					{
						typeIndex = i;
						break;
					}
				}
				if (typeIndex >= 0)
				{
					kernelNames_[typeIndex].push_back(name);
				} else
				{
					std::cerr << "Unknown kernel type: " << name << std::endl;
					continue;
				}
				kernels.emplace_back(name, type);
			}

			msg << "Create program" << std::endl;
			nvrtcProgram prog;
			NVRTC_SAFE_CALL(
				nvrtcCreateProgram(&prog,
					source.second.c_str(),
					source.first.c_str(),
					includeFiles.size(),
					headers,
					includeNames));
			for (auto& kernel : kernels)
			{
				NVRTC_SAFE_CALL(nvrtcAddNameExpression(prog, kernel.second.c_str()));
			}
			for (const auto& var : constantNames_)
			{
				NVRTC_SAFE_CALL(nvrtcAddNameExpression(prog, ("&"+var).c_str()));
			}

			//compute hash for caching the compilation result
			SHA1 sha1;
			for (const auto* const s : opts) sha1.update(s);
			for (const auto& s : includeFiles) sha1.update(s.second);
			sha1.update(source.second);
			const std::string checksum = sha1.final();
			fs::path cacheFile = CACHE_DIR / (checksum + ".kernel");
			if (!exists(CACHE_DIR)) {
				if (!create_directory(CACHE_DIR))
					msg << "Unable to create cache directory " << absolute(CACHE_DIR) << std::endl;
				else
					msg << "Cache directory created at " << absolute(CACHE_DIR) << std::endl;
			}

			std::vector<char> ptx;
			std::unordered_map<std::string, std::string> human2machineNames;
			if (exists(cacheFile))
			{
				//reuse from cache
				msg << "Read from cache: " << cacheFile << std::endl;
				std::ifstream i(cacheFile, std::ifstream::binary);
				if (i.is_open())
				{
					size_t ptxSize;
					i.read(reinterpret_cast<char*>(&ptxSize), sizeof(size_t));
					ptx.resize(ptxSize);
					i.read(ptx.data(), ptxSize);
					size_t mapSize;
					i.read(reinterpret_cast<char*>(&mapSize), sizeof(size_t));
					for (size_t j=0; j<mapSize; ++j)
					{
						size_t keySize, valueSize;
						std::string key, value;
						i.read(reinterpret_cast<char*>(&keySize), sizeof(size_t));
						key.resize(keySize);
						i.read(key.data(), keySize);
						i.read(reinterpret_cast<char*>(&valueSize), sizeof(size_t));
						value.resize(valueSize);
						i.read(value.data(), valueSize);
						human2machineNames.emplace(key, value);
					}
					if (i.bad())
					{
						msg << "Failed to read kernel cache, recompile" << std::endl;
						ptx.clear();
						human2machineNames.clear();
					}
				}
			}
			if (ptx.empty()) {
				// compile program
				msg << "Compile program" << std::endl;
				nvrtcResult compileResult = nvrtcCompileProgram(prog, numOpts, opts.data());
				// obtain log
				size_t logSize;
				NVRTC_SAFE_CALL(nvrtcGetProgramLogSize(prog, &logSize));
				std::vector<char> log(logSize);
				NVRTC_SAFE_CALL(nvrtcGetProgramLog(prog, &log[0]));
				msg << log.data();
				if (compileResult != NVRTC_SUCCESS)
				{
					msg << "Failed to compile this file, skip it" << std::endl;
					continue;
				}
				// obtain PTX
				size_t ptxSize;
				NVRTC_SAFE_CALL(nvrtcGetPTXSize(prog, &ptxSize));
				ptx.resize(ptxSize);
				NVRTC_SAFE_CALL(nvrtcGetPTX(prog, &ptx[0]));
				// create kernel mapping
				for (auto& kernel : kernels)
				{
					const auto& humanName = kernel.first;
					const char* machineName;
					NVRTC_SAFE_CALL(nvrtcGetLoweredName(prog, kernel.second.c_str(), &machineName));
					human2machineNames.emplace(humanName, std::string(machineName));
				}
				// create variable mapping
				for (const auto& var : constantNames_)
				{
					std::string humanName = "&" + var;
					const char* machineName;
					NVRTC_SAFE_CALL(nvrtcGetLoweredName(prog, humanName.c_str(), &machineName));
					human2machineNames.emplace(humanName, std::string(machineName));
				}
				// save to cache
				std::ofstream o(cacheFile, std::ofstream::binary);
				if (o.is_open())
				{
					o.write(reinterpret_cast<const char*>(&ptxSize), sizeof(size_t));
					o.write(ptx.data(), ptxSize);
					size_t mapSize = human2machineNames.size();
					o.write(reinterpret_cast<char*>(&mapSize), sizeof(size_t));
					for (const auto& h2m : human2machineNames)
					{
						size_t keySize = h2m.first.size(), valueSize = h2m.second.size();
						std::string key = h2m.first, value = h2m.second;
						o.write(reinterpret_cast<char*>(&keySize), sizeof(size_t));
						o.write(key.data(), keySize);
						o.write(reinterpret_cast<char*>(&valueSize), sizeof(size_t));
						o.write(value.data(), valueSize);
					}
				}
				msg << "Saved kernel to cache: " << cacheFile << std::endl;
			}
			// Load the generated PTX
			CUmodule module;
			CU_SAFE_CALL(cuModuleLoadDataEx(&module, ptx.data(), 0, 0, 0));
			modules_.push_back(module);
			msg << "Module loaded successfully" << std::endl;
			// load kernels
			for (auto& kernel : kernels)
			{
				const auto& humanName = kernel.first;
				const auto machineNameIt = human2machineNames.find(humanName);
				if (machineNameIt != human2machineNames.end())
				{
					const auto& machineName = machineNameIt->second;
					CUfunction fun;
					CU_SAFE_CALL(cuModuleGetFunction(&fun, module, machineName.data()));
					kernels_.emplace(humanName, fun);
				} else
				{
					msg << "Unable to find machine name for kernel " << humanName << std::endl;
				}
				
			}
			// load variables
			for (const auto& var : constantNames_)
			{
				std::string humanName = "&" + var;
				const auto machineNameIt = human2machineNames.find(humanName);
				if (machineNameIt != human2machineNames.end())
				{
					const auto& machineName = machineNameIt->second;
					CUdeviceptr addr;
					CU_SAFE_CALL(cuModuleGetGlobal(&addr, nullptr, module, machineName.data()));
					constants_[var].push_back(addr);
				}
				else
				{
					msg << "Unable to find machine name for variable " << humanName << std::endl;
				}
			}
			// Destroy the program.
			NVRTC_SAFE_CALL(nvrtcDestroyProgram(&prog));
		} catch (const std::exception& ex)
		{
			std::cerr << "Unknown error during compilation: " << ex.what() << std::endl;
			return false;
		}
	}

	//append kernels from custom registration
	for (const auto& e : CUSTOM_KERNELS)
	{
		const auto& name = e.first;
		int typeIndex = -1;
		for (int i = 0; i<int(KernelTypes::__COUNT__); ++i)
		{
			if (name.substr(0, KERNEL_PREFIXES[i].size()) == KERNEL_PREFIXES[i])
			{
				typeIndex = i;
				break;
			}
		}
		if (typeIndex >= 0)
		{
			kernelNames_[typeIndex].push_back(name);
		}
		else
		{
			std::cerr << "Unknown kernel type: " << name << " (custom registration)" << std::endl;
		}
	}
	
	sortAndPrintNames();
	initializeRenderer(); //sets AO

	return true;
#endif
}

bool renderer::KernelLauncher::launchKernel(const std::string& kernelName, int screenWidth, int screenHeight,
                                            const kernel::RendererDeviceSettings& deviceSettings,
	                                        const RendererArgs* hostSettings,
	                                        const Volume::MipmapLevel* volume,
                                            kernel::OutputTensor& output,
	                                        cudaStream_t stream,
	                                        kernel::PerPixelInstrumentation* instrumentation,
	                                        GlobalInstrumentation* globalInstrumentation)
{
	
	//check for custom kernel
	if (auto it = CUSTOM_KERNELS.find(kernelName); it != CUSTOM_KERNELS.end())
	{
		ICustomKernelLauncher* launcher = it->second(kernelName);
		return launcher->render(kernelName, screenWidth, screenHeight, 
			deviceSettings, hostSettings, volume, output, 
			stream, instrumentation, globalInstrumentation);
	}
	
#if RENDERER_RUNTIME_COMPILATION==0
	auto i = RuntimeApiKernelRegistry::INSTANCE.launcher_.find(kernelName);
	if (i == RuntimeApiKernelRegistry::INSTANCE.launcher_.end())
	{
		std::cerr << "No kernel with name " << kernelName << " found" << std::endl;
		return false;
	}
	i->second(screenWidth, screenHeight, stream, output, instrumentation);
	CUMAT_CHECK_ERROR();
	return true;
#else
	auto i = kernels_.find(kernelName);
	if (i == kernels_.end())
	{
		std::cerr << "No kernel with name " << kernelName << " found" << std::endl;
		return false;
	}
	CUfunction kernel = i->second;
	CUresult result = CUDA_SUCCESS;

	int minGridSize = 0, bestBlockSize = 0;
	result = cuOccupancyMaxPotentialBlockSize(
		&minGridSize, &bestBlockSize, kernel, NULL, 0, 0);
	if (result != CUDA_SUCCESS)
	{
		const char* pStr;
		cuGetErrorString(result, &pStr);
		std::cerr << "Failed to compute the best block size: " << pStr << std::endl;
		return false;
	}
	minGridSize = std::min(int(CUMAT_DIV_UP(screenWidth*screenHeight, bestBlockSize)), minGridSize);

	dim3 virtual_size{
		cuMat::internal::narrow_cast<unsigned>(screenWidth), 
		cuMat::internal::narrow_cast<unsigned>(screenHeight), 
		1u};
	const void* args[] = {&virtual_size, &output, &instrumentation};
	//CUstream stream = getCurrentStream();
	result = cuLaunchKernel(
		kernel, minGridSize, 1, 1, bestBlockSize, 1, 1,
		0, stream, const_cast<void**>(args), NULL);
	if (result != CUDA_SUCCESS)
	{
		const char* pStr;
		cuGetErrorString(result, &pStr);
		std::cerr << "Unable to launch kernel: " << pStr << std::endl;
		return false;
	}
	return true;
#endif
}

void renderer::KernelLauncher::sortAndPrintNames()
{
	for (int i = 0; i<int(KernelTypes::__COUNT__); ++i)
		std::sort(kernelNames_[i].begin(), kernelNames_[i].end());
	std::cout << "Isosurface Kernels:\n";
	for (const auto& n : kernelNames_[int(KernelTypes::Iso)])
		std::cout << "  " << n << "\n";
	std::cout << "DVR Kernels:\n";
	for (const auto& n : kernelNames_[int(KernelTypes::Dvr)])
		std::cout << "  " << n << "\n";
	std::cout << "Multi-Iso Kernels:\n";
	for (const auto& n : kernelNames_[int(KernelTypes::MultiIso)])
		std::cout << "  " << n << "\n";
	std::cout << std::flush;
}
