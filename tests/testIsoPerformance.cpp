#include <catch.hpp>
#include <json.hpp>
#include <fstream>
#include <memory>
#include <random>

#include <lib.h>
#include <tinyformat.h>
#include "test_utils.h"


TEST_CASE("Iso-Performance", "[!hide][Iso]")
{
	static const std::vector<std::string> SETTINGS_FILES = {
		"../tests/ejecta-far.json" };
	//(kernel name, stepsize)
	static const std::vector<std::tuple<std::string, float>> TEST_CASES = {
		{"Iso: Fixed step size - trilinear", 0.01f},
		{"Iso: Fixed step size - trilinear", 0.05f},
		{"Iso: Fixed step size - trilinear", 0.10f},
		{"Iso: Fixed step size - trilinear", 0.20f},
		{"Iso: Fixed step size - trilinear", 0.50f},

		{"Iso: DDA - fixed step", 0.05f},
		{"Iso: DDA - [ana] hyperbolic (float)", 1.0f},
		{"Iso: DDA - [ana] Schwarze (float)", 1.0f},
		{"Iso: DDA - [num] midpoint", 1.0f},
		{"Iso: DDA - [num] linear", 1.0f},
		{"Iso: DDA - [num] Neubauer", 1.0f},
		{"Iso: DDA - [num] Marmitt (float, unstable)", 1.0f},
		{"Iso: DDA - [num] Marmitt (float, stable)", 1.0f},
	};
	static const std::string FORMAT_HEADER1 = tinyformat::format(
		" %40s %5s | %11s %11s",
		"Kernel", "step", "avg time", "std time"
	);
	static const std::string FORMAT_HEADER2(FORMAT_HEADER1.size(), '-');
	static const char* FORMAT = " %40s %5.3f | %11.7f %11.7f";
	static int RESOLUTION_X = 512;
	static int RESOLUTION_Y = 512;

	static const std::vector<std::string> STATISTICS_FILES = {
		"../results/statistics/iso-ejecta/IsoPerformanceFar.txt"
	};
	static const bool WRITE_STATISTICS = true;

	static int NUM_FRAMES = 100;
	//compute camera motions
	std::default_random_engine rnd(42);
	std::uniform_real_distribution<float> pitch_distr(-80, +80);
	std::uniform_real_distribution<float> yaw_distr(0, 360);
	std::vector<float> pitches(NUM_FRAMES);
	std::vector<float> yaws(NUM_FRAMES);
	std::vector<cudaEvent_t> startEvents(NUM_FRAMES);
	std::vector<cudaEvent_t> stopEvents(NUM_FRAMES);
	std::vector<float> milliseconds(NUM_FRAMES);
	for (int i=0; i<NUM_FRAMES; ++i)
	{
		pitches[i] = pitch_distr(rnd);
		yaws[i] = yaw_distr(rnd);
		cudaEventCreate(&startEvents[i]);
		cudaEventCreate(&stopEvents[i]);
	}
	
	//allocate output
	renderer::OutputTensor output(
		RESOLUTION_Y, RESOLUTION_X, renderer::IsoRendererOutputChannels);

	//load kernels
	renderer::KernelLauncher::Instance().init();
	bool kernelsCompiledSuccessfully = renderer::KernelLauncher::Instance().reload(std::cout, false);
	REQUIRE(kernelsCompiledSuccessfully);
	auto isoKernels = renderer::KernelLauncher::Instance().getKernelNames(
		renderer::KernelLauncher::KernelTypes::Iso);

	for (size_t idx = 0; idx < SETTINGS_FILES.size(); ++idx)
	{
		const auto& settingsFile = SETTINGS_FILES[idx];

		//prepare statistics output
		std::ofstream statisticsFile;
		if (WRITE_STATISTICS) {
			statisticsFile.open(STATISTICS_FILES[idx]);
			statisticsFile << "# Kernel,\tstepsize,\ttimes. Tab-separated\n";
		}

		//load settings
		INFO("open " << std::filesystem::absolute(std::filesystem::path(settingsFile)));
		std::ifstream i(settingsFile);
		REQUIRE(i.is_open());
		nlohmann::json settings;
		i >> settings;
		i.close();

		renderer::RendererArgs rendererArgs;
		renderer::Camera camera;
		std::string volumeFileName;
		renderer::RendererArgs::load(settings, { "./" }, rendererArgs, camera, volumeFileName);
		rendererArgs.cameraResolutionX = RESOLUTION_X;
		rendererArgs.cameraResolutionY = RESOLUTION_Y;

		//load volume
		auto volume = std::make_shared<renderer::Volume>(volumeFileName);
		if (rendererArgs.mipmapLevel != 0)
			volume->createMipmapLevel(rendererArgs.mipmapLevel, renderer::Volume::MipmapFilterMode::AVERAGE);
		volume->getLevel(rendererArgs.mipmapLevel)->copyCpuToGpu();

		//update render arguments
		camera.updateRenderArgs(rendererArgs);
		rendererArgs.shading.lightDirection = camera.screenToWorld(rendererArgs.shading.lightDirection);

		//run test cases
		cudaStream_t stream = 0;
		std::cout << "\nRun " << settingsFile << std::endl;
		std::cout << FORMAT_HEADER1 << "\n" << FORMAT_HEADER2 << std::endl;
		for (const auto& testcase : TEST_CASES)
		{
			const std::string& kernelName = std::get<0>(testcase);
			const float stepsize = std::get<1>(testcase);

			rendererArgs.stepsize = stepsize;
			CUMAT_SAFE_CALL(cudaDeviceSynchronize());

			for (int j=0; j<NUM_FRAMES; ++j)
			{
				camera.setCurrentPitch(pitches[j]);
				camera.setCurrentYaw(yaws[j]);
				camera.updateRenderArgs(rendererArgs);
				cudaEventRecord(startEvents[j]);
				renderer::render_gpu(kernelName, volume.get(), &rendererArgs, output, stream);
				cudaEventRecord(stopEvents[j]);
			}
			CUMAT_SAFE_CALL(cudaDeviceSynchronize());

			double mean = 0;
			double lastmean = 0;
			double sn = 0;
			int n = 0;
			for (int j = 0; j < NUM_FRAMES; ++j)
			{
				float ms;
				CUMAT_SAFE_CALL(cudaEventElapsedTime(&ms, startEvents[j], stopEvents[j]));
				milliseconds[j] = ms;

				n += 1;
				lastmean = mean;
				mean += (ms - lastmean) / n;
				if (n == 1)
					sn = 0;
				else
					sn += (ms - lastmean) * (ms - mean);
			}
			
			std::cout << tinyformat::format(FORMAT,
				kernelName, stepsize, mean, std::sqrtf(sn/n))
				<< std::endl;

			if (WRITE_STATISTICS)
			{
				statisticsFile << "\"" << kernelName << "\"\t" << stepsize;
				for (int j = 0; j < NUM_FRAMES; ++j)
				{
					statisticsFile << "\t" << milliseconds[j];
				}
				statisticsFile << std::endl;
			}
		}
	}
}