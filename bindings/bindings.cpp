#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <json.hpp>
#include <fstream>
#include <json.hpp>
#include <tinyformat.h>

#include <lib.h>
#include <renderer_settings.cuh>

#ifdef WIN32
#ifndef NOMINMAX
#define NOMINMAX 1
#endif
#include <Windows.h>
#endif

namespace py = pybind11;
using namespace renderer;

//helpers
OutputTensor allocateOutput(int width, int height, 
	RendererArgs::RenderMode type)
{
	int channels;
	if (type == RendererArgs::RenderMode::ISO)
		channels = IsoRendererOutputChannels;
	else if (type == RendererArgs::RenderMode::DVR)
		channels = DvrRendererOutputChannels;
	else
		throw std::invalid_argument("'type' must be either 'iso' or 'dvr'");
	if (width <= 0) throw std::invalid_argument("'width' must be positive");
	if (height <= 0) throw std::invalid_argument("'height' must be positive");
	return OutputTensor(height, width, channels);
}

std::filesystem::path getCacheDir()
{
	//suffix and default (if default, it is a relative path)
	static const std::filesystem::path SUFFIX{ "kernel_cache" };
#ifdef WIN32
	//get the path to this dll as base path
	char path[MAX_PATH];
	HMODULE hm = NULL;

	if (GetModuleHandleEx(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
		GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
		(LPCSTR)&getCacheDir, &hm) == 0)
	{
		int ret = GetLastError();
		fprintf(stderr, "GetModuleHandle failed, error = %d\n", ret);
		return SUFFIX;
	}
	if (GetModuleFileName(hm, path, sizeof(path)) == 0)
	{
		int ret = GetLastError();
		fprintf(stderr, "GetModuleFileName failed, error = %d\n", ret);
		return SUFFIX;
	}

	std::filesystem::path out = path;
	out = out.parent_path();
	const auto out_str = out.string();
	fprintf(stdout, "This DLL is located at %s, use that as cache dir\n", out_str.c_str());
	out /= SUFFIX;
	return out;
	
#else
	return SUFFIX //default
#endif
}

class OutputTensorCPU
{
public:
	size_t height, width, channels;
	std::vector<float> data;
	OutputTensorCPU(const OutputTensor& gpu)
		: height(gpu.rows()), width(gpu.cols()), channels(gpu.batches())
		, data(gpu.rows()*gpu.cols()*gpu.batches())
	{
		gpu.copyToHost(&data[0]);
	}
};

//class InstrumentationCPU
//{
//public:
//	size_t height, width;
//	std::vector<kernel::Instrumentation> data;
//	InstrumentationCPU(size_t height, size_t width)
//		: height(height), width(width), data(width* height)
//	{}
//};

class GPUTimer
{
	cudaEvent_t start_, stop_;
public:
	GPUTimer()
		: start_(0), stop_(0)
	{
		CUMAT_SAFE_CALL(cudaEventCreate(&start_));
		CUMAT_SAFE_CALL(cudaEventCreate(&stop_));
	}
	~GPUTimer()
	{
		CUMAT_SAFE_CALL(cudaEventDestroy(start_));
		CUMAT_SAFE_CALL(cudaEventDestroy(stop_));
	}
	void start()
	{
		CUMAT_SAFE_CALL(cudaEventRecord(start_));
	}
	void stop()
	{
		CUMAT_SAFE_CALL(cudaEventRecord(stop_));
	}
	float getElapsedMilliseconds()
	{
		CUMAT_SAFE_CALL(cudaEventSynchronize(stop_));
		float ms;
		CUMAT_SAFE_CALL(cudaEventElapsedTime(&ms, start_, stop_));
		return ms;
	}
};

auto loadFromJson(const std::string& jsonFileName, const std::string& basePath)
{
	std::ifstream i(jsonFileName);
	if (!i.is_open()) throw std::exception("Unable to load settings file");
	nlohmann::json settings;
	i >> settings;
	i.close();

	RendererArgs rendererArgs;
	Camera camera;
	std::string volumeFileName;
	RendererArgs::load(settings, basePath, rendererArgs, camera, volumeFileName);

	return std::make_tuple(rendererArgs, camera, volumeFileName);
}

// initialize context
cuMat::Context& ctx = cuMat::Context::current();
cuMat::DevicePointer<float> dummyMemory;
bool contextInitialized = false;

//BINDINGS
PYBIND11_MODULE(pyrenderer, m)
{
	m.doc() = "python bindings for the analytic volume renderer";

	// initialize context
	dummyMemory = cuMat::DevicePointer<float>(1, ctx);
	contextInitialized = true;
	auto cleanup_callback = []() {
		if (contextInitialized) {
			contextInitialized = false;
			dummyMemory = {};
			ctx.destroy();
			std::cout << "cuMat context destroyed" << std::endl;
		}
	};
	m.def("cleanup", cleanup_callback,
		py::doc("Explicit cleanup of all CUDA references"));
	m.add_object("_cleanup", py::capsule(cleanup_callback));
	
	py::enum_<RendererArgs::RenderMode>(m, "RenderMode")
		.value("Iso", RendererArgs::RenderMode::ISO)
		.value("Dvr", RendererArgs::RenderMode::DVR);
	py::enum_<RendererArgs::VolumeFilterMode>(m, "VolumeFilterMode")
		.value("Trilinear", RendererArgs::VolumeFilterMode::TRILINEAR)
		.value("Tricubic", RendererArgs::VolumeFilterMode::TRICUBIC);
	py::enum_<RendererArgs::DvrTfMode>(m, "DvrTfMode")
		.value("Linear", RendererArgs::DvrTfMode::PiecewiseLinear)
		.value("MultiIso", RendererArgs::DvrTfMode::MultipleIsosurfaces)
		.value("Hybrid", RendererArgs::DvrTfMode::Hybrid);
	//py::enum_<RendererArgs::TfPreintegration>(m, "TfPreintegration")
	//	.value("Off", RendererArgs::TfPreintegration::OFF)
	//	.value("OneD", RendererArgs::TfPreintegration::ONE_D)
	//	.value("TwoD", RendererArgs::TfPreintegration::TWO_D);
	py::enum_<Camera::Orientation>(m, "Orientation")
		.value("Xp", Camera::Orientation::Xp)
		.value("Xm", Camera::Orientation::Xm)
		.value("Yp", Camera::Orientation::Yp)
		.value("Ym", Camera::Orientation::Ym)
		.value("Zp", Camera::Orientation::Zp)
		.value("Zm", Camera::Orientation::Zm);

	py::class_<float3>(m, "float3")
		.def(py::init<>())
		.def(py::init([](float x, float y, float z) {return make_float3(x, y, z); }))
		.def_readwrite("x", &float3::x)
		.def_readwrite("y", &float3::y)
		.def_readwrite("z", &float3::z)
		.def("__str__", [](const float3& v)
		{
		return tinyformat::format("(%f, %f, %f)", v.x, v.y, v.z);
		});
	py::class_<float4>(m, "float4")
		.def(py::init<>())
		.def(py::init([](float x, float y, float z, float w)
		{
			return make_float4(x, y, z, w);
		}))
		.def_readwrite("x", &float4::x)
		.def_readwrite("y", &float4::y)
		.def_readwrite("z", &float4::z)
		.def_readwrite("w", &float4::w)
		.def("__str__", [](const float4& v)
	{
		return tinyformat::format("(%f, %f, %f, %f)", v.x, v.y, v.z, v.w);
	});

	py::class_<int3>(m, "int3")
		.def(py::init<>())
		.def_readwrite("x", &int3::x)
		.def_readwrite("y", &int3::y)
		.def_readwrite("z", &int3::z)
		.def("__str__", [](const int3& v)
	{
		return tinyformat::format("(%d, %d, %d)", v.x, v.y, v.z);
	});

	m.def("labToXyz", &kernel::labToXyz);
	m.def("xyzToLab", &kernel::xyzToLab);

	py::class_<Camera>(m, "Camera")
		.def(py::init<>())
		.def("clone", [](const Camera& args)->Camera {return args; })
		.def("update_render_args", &Camera::updateRenderArgs)
		.def("screen_to_world", &Camera::screenToWorld)
		.def_property("orientation", &Camera::orientation, &Camera::setOrientation)
		.def_property("look_at", &Camera::lookAt, &Camera::setLookAt)
		.def_property("fov", &Camera::fov, &Camera::setFov)
		.def_property_readonly("base_distance", &Camera::baseDistance)
		.def_property("pitch", &Camera::currentPitch, &Camera::setCurrentPitch)
		.def_property("yaw", &Camera::currentYaw, &Camera::setCurrentYaw)
		.def_property("zoom", &Camera::zoomvalue, &Camera::setZoomvalue);

	py::class_<ShadingSettings>(m, "ShadingSettings")
		.def(py::init<>())
		.def_readwrite("ambient_light_color", &ShadingSettings::ambientLightColor)
		.def_readwrite("diffuse_light_color", &ShadingSettings::diffuseLightColor)
		.def_readwrite("specular_light_color", &ShadingSettings::specularLightColor)
		.def_readwrite("specular_exponent", &ShadingSettings::specularExponent)
		.def_readwrite("material_color", &ShadingSettings::materialColor)
		.def_readwrite("ao_strength", &ShadingSettings::aoStrength)
		.def_readwrite("light_direction", &ShadingSettings::lightDirection);

	//colors in Lab
	py::class_<RendererArgs::TfLinear>(m, "TfLinear")
		.def(py::init<>())
		.def_readwrite("density_axis_opacity", &RendererArgs::TfLinear::densityAxisOpacity)
		.def_readwrite("opacity_axis", &RendererArgs::TfLinear::opacityAxis)
		.def_readwrite("opacity_extra_color_axis", &RendererArgs::TfLinear::opacityExtraColorAxis)
		.def_readwrite("density_axis_color", &RendererArgs::TfLinear::densityAxisColor)
		.def_readwrite("color_axis", &RendererArgs::TfLinear::colorAxis);

	//colors in XYZ
	py::class_<RendererArgs::TfMultiIso>(m, "TfMultiIso")
		.def(py::init<>())
		.def_readwrite("densities", &RendererArgs::TfMultiIso::densities)
		.def_readwrite("colors", &RendererArgs::TfMultiIso::colors);
	
	py::class_<RendererArgs>(m, "RendererArgs")
		.def(py::init<>())
		.def("clone", [](const RendererArgs& args)->RendererArgs {return args; })
		.def_readwrite("mipmap_level", &RendererArgs::mipmapLevel)
		.def_readwrite("render_mode", &RendererArgs::renderMode)
		.def_readwrite("width", &RendererArgs::cameraResolutionX)
		.def_readwrite("height", &RendererArgs::cameraResolutionY)
		.def_readwrite("isovalue", &RendererArgs::isovalue)
		.def_readwrite("stepsize", &RendererArgs::stepsize)
		.def_readwrite("volume_filter_mode", &RendererArgs::volumeFilterMode)
		.def_readwrite("shading", &RendererArgs::shading)
		.def_readwrite("dvr_use_shading", &RendererArgs::dvrUseShading)
		.def_readwrite("opacity_scaling", &RendererArgs::opacityScaling)
		.def_readwrite("min_density", &RendererArgs::minDensity)
		.def_readwrite("max_density", &RendererArgs::maxDensity)
		.def_readwrite("enable_clip_plane", &RendererArgs::enableClipPlane)
		.def_readwrite("clip_plane", &RendererArgs::clipPlane)
		.def_readonly("dvr_tf_mode", &RendererArgs::dvrTfMode)
		.def("get_linear_tf", [](const RendererArgs& args)
		{
			if (args.dvrTfMode != kernel::DvrTfMode::PiecewiseLinear)
				throw std::invalid_argument("TF mode is not 'Linear'");
			return std::get<RendererArgs::TfLinear>(args.tf);
		})
		.def("get_multiiso_tf", [](const RendererArgs& args)
		{
			if (args.dvrTfMode != kernel::DvrTfMode::MultipleIsosurfaces)
				throw std::invalid_argument("TF mode is not 'MultiIso'");
			return std::get<RendererArgs::TfMultiIso>(args.tf);
		})
		.def("get_hybrid_tf", [](const RendererArgs& args)
		{
			if (args.dvrTfMode != kernel::DvrTfMode::Hybrid)
				throw std::invalid_argument("TF mode is not 'Hybrid'");
			return std::get<RendererArgs::TfLinear>(args.tf);
		})
		.def("set_linear_tf", [](RendererArgs& args, const RendererArgs::TfLinear& tf)
		{
			args.dvrTfMode = kernel::DvrTfMode::PiecewiseLinear;
			args.tf = tf;
		})
		.def("set_multiiso_tf", [](RendererArgs& args, const RendererArgs::TfMultiIso& tf)
		{
			args.dvrTfMode = kernel::DvrTfMode::MultipleIsosurfaces;
			args.tf = tf;
		})
		.def("set_hybrid_tf", [](RendererArgs& args, const RendererArgs::TfLinear& tf)
		{
			args.dvrTfMode = kernel::DvrTfMode::Hybrid;
			args.tf = tf;
		})
		//.def_readwrite("tf_preintegration", &RendererArgs::tfPreintegration)
		.def("dumps", [](const RendererArgs& args)
		{
			const auto json = args.toJson();
			return json.dump(-1);
		}, "dumps the settings to a json string");
	

	m.def("load_from_json", &loadFromJson,
		"Loads a json file with the settings.\n"
		"Parameters: file name of the json, base path for volume path resolution.\n"
		"Returns:\n"
		"    RendererArgs instance\n"
		"    Camera instance\n"
		"    file name of the volume as string");

	py::enum_<Volume::ImplicitEquation>(m, "ImplicitEquation")
		.value("MarschnerLobb", Volume::ImplicitEquation::MARSCHNER_LOBB)
		.value("Cube", Volume::ImplicitEquation::CUBE)
		.value("Sphere", Volume::ImplicitEquation::SPHERE)
		.value("InverseSphere", Volume::ImplicitEquation::INVERSE_SPHERE)
		.value("DingDong", Volume::ImplicitEquation::DING_DONG)
		.value("Endrass", Volume::ImplicitEquation::ENDRASS)
		.value("Barth", Volume::ImplicitEquation::BARTH)
		.value("Heart", Volume::ImplicitEquation::HEART)
		.value("Kleine", Volume::ImplicitEquation::KLEINE)
		.value("Cassini", Volume::ImplicitEquation::CASSINI)
		.value("Steiner", Volume::ImplicitEquation::STEINER)
		.value("CrossCap", Volume::ImplicitEquation::CROSS_CAP)
		.value("Kummer", Volume::ImplicitEquation::KUMMER)
		.value("Blobbly", Volume::ImplicitEquation::BLOBBY)
		.value("Tube", Volume::ImplicitEquation::TUBE);
	py::class_<Volume, std::shared_ptr<Volume>>(m, "Volume")
		.def(py::init<std::string>())
		.def("create_mipmap_level", &Volume::createMipmapLevel)
		.def_property_readonly("world_size", &Volume::worldSize)
		.def_property_readonly("resolution", &Volume::baseResolution)
		.def_static("create_implicit", [](Volume::ImplicitEquation name, int resolution, py::kwargs kwargs) {
			std::unordered_map<std::string, float> args;
			for (const auto& e : kwargs)
				args.insert({ e.first.cast<std::string>(), e.second.cast<float>() });
			return std::shared_ptr<Volume>(Volume::createImplicitDataset(resolution, name, args));
		})
		.def("copy_to_gpu", [](std::shared_ptr<Volume> v)
		{
			for (int i = 0; v->getLevel(i); ++i)
				v->getLevel(i)->copyCpuToGpu();
		})
		.def("save", [](std::shared_ptr<Volume> v, const std::string& filename)
		{
			v->save(filename);
		})
		.def("resample", [](std::shared_ptr<Volume> v, int x, int y, int z, const std::string& mode)
		{
			Volume::VolumeFilterMode m;
			if (mode == "nearest")
				m = Volume::NEAREST;
			else if (mode == "linear")
				m = Volume::TRILINEAR;
			else if (mode == "cubic")
				m = Volume::TRICUBIC;
			else
				throw std::runtime_error("Unknown resampling mode, must be 'nearest', 'linear' or 'cubic'");
			return std::shared_ptr<Volume>(v->resample(make_int3(x, y, z), m));
		});

	m.def("reload_kernels", [](
		bool enableDebugging, bool enableInstrumentation,
		const std::vector<std::string>& otherPreprocessorArguments)
	{
		return KernelLauncher::Instance().reload(
			std::cout, enableDebugging, enableInstrumentation, otherPreprocessorArguments);
	}, 
		py::arg("enableDebugging")=true, 
		py::arg("enableInstrumentation")=false,
		py::arg("otherPreprocessorArguments")=std::vector<std::string>(),
		"(re)loads the kernels");
	m.def("iso_kernel_names", []() {
		return KernelLauncher::Instance().getKernelNames(KernelLauncher::KernelTypes::Iso); },
		"returns a list of the names of the isosurface kernels");
	m.def("dvr_kernel_names", []() {
		return KernelLauncher::Instance().getKernelNames(KernelLauncher::KernelTypes::Dvr); },
		"returns a list of the names of the DVR kernels");
	m.def("multiiso_kernel_names", []() {
		return KernelLauncher::Instance().getKernelNames(KernelLauncher::KernelTypes::MultiIso); },
		"returns a list of the names of the DVR kernels");

	m.def("sync", []() {CUMAT_SAFE_CALL(cudaDeviceSynchronize()); });
	
	py::class_<OutputTensor>(m, "OutputTensor")
		.def("copy_to_cpu", [](const OutputTensor& gpu)
	{
		return std::make_shared<OutputTensorCPU>(gpu);
	}, "copies the output tensor to the CPU. Use it in combination with numpy: 'np.array(output.copy_to_cpu())'");
	py::class_<OutputTensorCPU, std::shared_ptr<OutputTensorCPU>>(m, "OutputTensorCPU", py::buffer_protocol())
		.def_buffer([](OutputTensorCPU& m) -> py::buffer_info
	{
		return py::buffer_info(
			m.data.data(),
			sizeof(float),
			py::format_descriptor<float>::format(),
			3,
			{ m.height, m.width, m.channels },
			{
				sizeof(float),
				sizeof(float) * m.height,
				sizeof(float)* m.width* m.height
			}
			);
	});
	
	m.def("allocate_output", &allocateOutput, "Allocates the output tensor");
	
	m.def("render", [](
		const std::string& kernelName, std::shared_ptr<Volume> volume,
		const RendererArgs& args, OutputTensor& output)
	{
		renderer::render_gpu(kernelName, volume.get(), &args, output, 0);
	});

#if KERNEL_INSTRUMENTATION==1
	PYBIND11_NUMPY_DTYPE(kernel::PerPixelInstrumentation, 
		densityFetches, tfFetches, ddaSteps, 
		isoIntersections, intervalEval, intervalStep, intervalMaxStep,
		timeDensityFetch, timeDensityFetch_NumSamples,
		timePolynomialCreation, timePolynomialCreation_NumSamples,
		timePolynomialSolution, timePolynomialSolution_NumSamples,
		timeTotal);
	py::class_<GlobalInstrumentation>(m, "GlobalInstrumentation")
		.def(py::init<>())
		.def_readwrite("numControlPoints", &GlobalInstrumentation::numControlPoints)
		.def_readwrite("numTriangles", &GlobalInstrumentation::numTriangles)
		.def_readwrite("numFragments", &GlobalInstrumentation::numFragments);
	m.def("render_with_instrumentation", [](
		const std::string& kernelName, std::shared_ptr<Volume> volume,
		const RendererArgs& args, OutputTensor& output)
	{
		size_t size = args.cameraResolutionX * args.cameraResolutionY;
		kernel::PerPixelInstrumentation* instrumentationHost = new kernel::PerPixelInstrumentation[size];
		kernel::PerPixelInstrumentation* instrumentationDevice;
		CUMAT_SAFE_CALL(cudaMalloc(&instrumentationDevice, size * sizeof(kernel::PerPixelInstrumentation)));

		GlobalInstrumentation globalInstrumentation;
		renderer::render_gpu(kernelName, volume.get(), &args, output, 0, 
			instrumentationDevice, &globalInstrumentation);
		
		CUMAT_SAFE_CALL(cudaMemcpy(instrumentationHost, instrumentationDevice, size * sizeof(kernel::PerPixelInstrumentation), cudaMemcpyDeviceToHost));
		CUMAT_SAFE_CALL(cudaFree(instrumentationDevice));

		py::capsule free_when_done(instrumentationHost, [](void* f) {
			kernel::PerPixelInstrumentation* foo = static_cast<kernel::PerPixelInstrumentation*>(f);
			delete[] foo;
		});
		return py::make_tuple(py::array_t<kernel::PerPixelInstrumentation>(
			{ args.cameraResolutionX, args.cameraResolutionY },
			{ args.cameraResolutionY * sizeof(kernel::PerPixelInstrumentation), sizeof(kernel::PerPixelInstrumentation) },
			instrumentationHost,
			free_when_done),
			globalInstrumentation);
	});
#endif
	
	py::class_<GPUTimer>(m, "GpuTimer")
		.def(py::init<>())
		.def("start", &GPUTimer::start)
		.def("stop", &GPUTimer::stop)
		.def("elapsed_ms", &GPUTimer::getElapsedMilliseconds);

	//OpenGL / OIT
	auto mOit = m.def_submodule("oit");
	mOit.def("setup_offscreen_context", []()
	{
		OpenGLRasterization::Instance().setupOffscreenContext();
	});
	mOit.def("delete_offscreen_context", []()
	{
		OpenGLRasterization::Instance().cleanup();
		OpenGLRasterization::Instance().deleteOffscreenContext();
	});
	mOit.def("set_fragment_buffer_size", [](int size)
	{
		OpenGLRasterization::Instance().setFragmentBufferSize(size);
	});
	py::enum_<OpenGLRasterization::MarchingCubesComputationMode>(mOit, "MarchingCubesComputationMode")
		.value("PreDevice", OpenGLRasterization::MarchingCubesComputationMode::PRE_DEVICE)
		.value("PreHost", OpenGLRasterization::MarchingCubesComputationMode::PRE_HOST)
		.value("OnTheFly", OpenGLRasterization::MarchingCubesComputationMode::ON_THE_FLY);
	mOit.def("set_marching_cubes_mode", [](OpenGLRasterization::MarchingCubesComputationMode mode)
	{
		OpenGLRasterization::Instance().setMarchingCubesComputationMode(mode);
	});
	mOit.def("set_max_fragments_per_pixel", [](int size)
	{
		OpenGLRasterization::Instance().setMaxFragmentsPerPixel(size);
	});
	mOit.def("set_tile_size", [](int size)
	{
		OpenGLRasterization::Instance().setTileSize(size);
	});
	
	//Initialization
	KernelLauncher::SetCacheDir(getCacheDir());
	m.def("init", []() {
		OpenGLRasterization::Register();
		KernelLauncher::Instance().init();
	});
}