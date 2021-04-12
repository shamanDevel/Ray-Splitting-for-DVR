#pragma once

#include <cuda_runtime.h>
#include <lib.h>
#include <GL/glew.h>
#include <sstream>

#include "quad_drawer.h"
#include "tf_editor.h"
#include "camera_gui.h"
#include "visualizer_kernels.h"
#include "background_worker.h"
#include "visualizer_commons.h"
#include "mean_variance.h"
#include "opengl_renderer.h"

struct GLFWwindow;

class Visualizer
{
public:
	Visualizer(GLFWwindow* window);
	~Visualizer();

	void specifyUI();

	void render(int display_w, int display_h);

	ImVec4 clearColor_ = ImVec4(0, 0, 0, 0);
	
private:

	enum RedrawMode
	{
		RedrawNone,
		RedrawPost,
		RedrawRenderer,

		_RedrawModeCount_
	};
	static const char* RedrawModeNames[_RedrawModeCount_];
	RedrawMode redrawMode_ = RedrawNone;

	GLFWwindow* window_;

	//volume
	int syntheticDatasetIndex_ = 0;
	int syntheticDatasetResolutionPower_ = 6;
	std::string volumeDirectory_;
	std::string volumeFilename_;
	std::string volumeFullFilename_;
	std::unique_ptr<renderer::Volume> volume_;
	static constexpr int MipmapLevels[] = { 0, 1, 2, 3, 7 };
	int volumeMipmapLevel_ = 0;
	renderer::Volume::MipmapFilterMode volumeMipmapFilterMode_
		= renderer::Volume::MipmapFilterMode::AVERAGE;
	renderer::RendererArgs rendererArgs_;

	CameraGui cameraGui_;

	//background computation
	BackgroundWorker worker_;
	std::function<void()> backgroundGui_;

	//information string that is displayed together with the FPS
	//It is cleared every frame. Use it to append per-frame information
	std::stringstream extraFrameInformation_;

	RenderMode renderMode_ = RenderMode::ISO;
	DvrTfMode renderModeDvr_ = DvrTfMode::MultipleIsosurfaces;

	//display
	int displayWidth_ = 0;
	int displayHeight_ = 0;
	unsigned int screenTextureGL_ = 0;
	cudaGraphicsResource_t screenTextureCuda_ = nullptr;
	GLubyte* screenTextureCudaBuffer_ = nullptr;
	QuadDrawer drawer_;
	float currentFPS_ = 0;

	//dvr
	TfEditorLinear editorLinear_;
	TfEditorMultiIso editorMultiIso_;
	TfEditorLinear editorHybrid_;
	std::string tfDirectory_;
	float minDensity_{ 0.0f };
	float maxDensity_{ 1.0f };
	float isoToDvrCopyWidth_{ 0.01f };
	float opacityScaling_{ 50.0f };
	bool showColorControlPoints_{ true };
	bool dvrUseShading_ = false;
	RENDERER_NAMESPACE::Volume::Histogram volumeHistogram_;
	int oitFragmentStoragePower_ = 22; //number of fragments as 2^oitPower
	int oitFragmentsPerPixel_ = 64;
	renderer::OpenGLRasterization::MarchingCubesComputationMode mcComputationMode_
		= renderer::OpenGLRasterization::MarchingCubesComputationMode::PRE_HOST;
	bool oitEnableTiling_ = false;
	int oitTilingPower_ = 8;
	int clipPlaneIndex_ = 0;

	//available kernels
	int selectedKernel_[int(renderer::KernelLauncher::KernelTypes::__COUNT__)] = { 0 };
	bool kernelDebugging_ = true;
	bool kernelInstrumentation_ = false;
	struct AccumulatedInstrumentation
	{ //mean and std of kernel::Instrumentation
		MeanVariance densityFetches;
		MeanVariance tfFetches;
		MeanVariance ddaSteps;
		MeanVariance isoIntersections;
		MeanVariance intervalEval;
		MeanVariance intervalStep;
		int intervalMaxStep;
	} accumulatedInstrumentation_;
	cudaEvent_t startEvent_ = nullptr;
	cudaEvent_t stopEvent_ = nullptr;
	float elapsedMilliseconds_;
	RENDERER_NAMESPACE::GlobalInstrumentation globalInstrumentation_;
	
	//intermediate computation results
	RENDERER_NAMESPACE::OutputTensor rendererOutput_;
	FlowTensor interpolatedFlow_;
	bool interpolatedFlowAvailable_ = false;
	RENDERER_NAMESPACE::OutputTensor previousBlendingOutput_;
	GLubyte* postOutput_ = nullptr;

	//shading
	float3 ambientLightColor{ 0.1, 0.1, 0.1 };
	float3 diffuseLightColor{ 0.8, 0.8, 0.8 };
	float3 specularLightColor{ 0.1, 0.1, 0.1 };
	float specularExponent = 16;
	float3 materialColor{ 1.0, 1.0, 1.0 };
	float aoStrength = 0.5;
	float3 lightDirectionScreen{ 0,0,+1 };
	enum ChannelMode
	{
		ChannelMask,
		ChannelNormal,
		ChannelDepth,
		ChannelAO,
		ChannelFlow,
		ChannelColor,

		_ChannelCount_
	};
	static const char* ChannelModeNames[_ChannelCount_];
	ChannelMode channelMode_ = ChannelColor;
	int temporalPostSmoothingPercentage_ = 0;
	bool flowWithInpainting_ = true;

	//screenshot
	std::string screenshotString_;
	float screenshotTimer_ = 0;

	//debug
	int2 selectedPixel_;
	bool pixelSelected_ = false;
	int3 selectedVoxel_{0,0,0};
	bool voxelFiltered_ = false;

	//settings
	std::string settingsDirectory_;
	enum SettingsToLoad
	{
		CAMERA = 1<<0,
		COMPUTATION_MODE = 1<<1,
		TF_EDITOR = 1<<2,
		RENDERER = 1<<3,
		SHADING = 1<<4,
		DATASET = 1<<5,
		_ALL_SETTINGS_ = 0x0fffffff
	};
	int settingsToLoad_ = _ALL_SETTINGS_;

private:
	void releaseResources();
	
	void settingsSave();
	void settingsLoad();
	
	void loadVolumeDialogue();
	void loadVolume(const std::string& filename, float* progress = nullptr);
	void selectMipmapLevel(int level, renderer::Volume::MipmapFilterMode filter);
	
	void uiMenuBar();
	void uiVolume();
	void uiCamera();
	void uiPixelSelection();
	void uiRenderer();
	void uiTfEditor();
	void uiComputationMode();
	void uiKernels();
	void uiShading();
	void uiRunTimings();
	void uiScreenshotOverlay();
	void uiFPSOverlay();

	void reloadKernels();

	renderer::RendererArgs setupRendererArgs(
		RenderMode renderMode, int upscaleFactor=1);

	void renderImpl(RenderMode renderMode);
	void copyBufferToOpenGL();
	void resize(int display_w, int display_h);
	void triggerRedraw(RedrawMode mode);

	//Selects the channel to write to the cuda buffer.
	//The network output has shape (1 x Channels=8 x displayHeight_ x displayWidth_)
	//See renderer::IsoRendererOutputChannels
	void selectChannelIso(ChannelMode mode,
		const RENDERER_NAMESPACE::OutputTensor& networkOutput,
		GLubyte* cudaBuffer) const;
	//Selects the channel to write to the cuda buffer.
	//The network output has shape (1 x Channels=10 x displayHeight_ x displayWidth_)
	//See renderer::DvrRendererOutputChannels
	void selectChannelDvr(ChannelMode mode,
		const RENDERER_NAMESPACE::OutputTensor& networkOutput,
		GLubyte* cudaBuffer) const;

	void screenshot();
};

