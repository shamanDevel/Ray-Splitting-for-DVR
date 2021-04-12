#include "visualizer.h"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <locale>

#include <lib.h>
#include <cuMat/src/Errors.h>
#include <GL/glew.h>
#include <cuda_gl_interop.h>
#include <cuMat/src/Context.h>

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>

#include "imgui/imgui.h"
#include "imgui/IconsFontAwesome5.h"
#include "imgui/imgui_extension.h"
#include "imgui/imgui_internal.h"

#include <json.hpp>
#include <lodepng.h>
#include <portable-file-dialogs.h>
#include <magic_enum.hpp>

#include "utils.h"
#include "kernel_launcher.h"
#include "opengl_renderer.h"

namespace nlohmann {
	template <>
	struct adl_serializer<ImVec4> {
		static void to_json(json& j, const ImVec4& v) {
			j = json::array({ v.x, v.y, v.z, v.w });
		}

		static void from_json(const json& j, ImVec4& v) {
			if (j.is_array() && j.size() == 4)
			{
				v.x = j.at(0).get<float>();
				v.y = j.at(1).get<float>();
				v.z = j.at(2).get<float>();
				v.w = j.at(3).get<float>();
			}
			else
				std::cerr << "Unable to deserialize " << j << " into a ImVec4" << std::endl;
		}
	};
}

const char* Visualizer::RedrawModeNames[] = {
	"None", "Post", "Renderer"
};
const char* Visualizer::ChannelModeNames[] = {
	"Mask", "Normal", "Depth", "AO", "Flow", "Color"
};

Visualizer::Visualizer(GLFWwindow* window)
	: window_(window)
	, editorLinear_(false), editorHybrid_(true)
{
	// Add .ini handle for ImGuiWindow type
	ImGuiSettingsHandler ini_handler;
	ini_handler.TypeName = "Visualizer";
	ini_handler.TypeHash = ImHashStr("Visualizer");
	static const auto replaceWhitespace = [](const std::string& s) -> std::string
	{
		std::string cpy = s;
		for (int i = 0; i < cpy.size(); ++i)
			if (cpy[i] == ' ') cpy[i] = '%'; //'%' is not allowed in path names
		return cpy;
	};
	static const auto insertWhitespace = [](const std::string& s) -> std::string
	{
		std::string cpy = s;
		for (int i = 0; i < cpy.size(); ++i)
			if (cpy[i] == '%') cpy[i] = ' '; //'%' is not allowed in path names
		return cpy;
	};
	auto settingsReadOpen = [](ImGuiContext*, ImGuiSettingsHandler* handler, const char* name) -> void*
	{
		return handler->UserData;
	};
	auto settingsReadLine = [](ImGuiContext*, ImGuiSettingsHandler* handler, void* entry, const char* line)
	{
		Visualizer* vis = reinterpret_cast<Visualizer*>(handler->UserData);
		char path[MAX_PATH];
		int intValue = 0;
		memset(path, 0, sizeof(char)*MAX_PATH);
		std::cout << "reading \"" << line << "\"" << std::endl;
		if (sscanf(line, "VolumeDir=%s", path) == 1)
			vis->volumeDirectory_ = insertWhitespace(std::string(path));
		if (sscanf(line, "TfDir=%s", path) == 1)
			vis->tfDirectory_ = insertWhitespace(std::string(path));
		if (sscanf(line, "SettingsDir=%s", path) == 1)
			vis->settingsDirectory_ = insertWhitespace(std::string(path));
		if (sscanf(line, "SettingsToLoad=%d", &intValue) == 1)
			vis->settingsToLoad_ = intValue;
	};
	auto settingsWriteAll = [](ImGuiContext* ctx, ImGuiSettingsHandler* handler, ImGuiTextBuffer* buf)
	{
		Visualizer* vis = reinterpret_cast<Visualizer*>(handler->UserData);
		buf->reserve(200);
		buf->appendf("[%s][Settings]\n", handler->TypeName);
		std::string volumeDir = replaceWhitespace(vis->volumeDirectory_);
		std::string tfDir = replaceWhitespace(vis->tfDirectory_);
		std::string settingsDirectory = replaceWhitespace(vis->settingsDirectory_);
		std::cout << "Write settings:" << std::endl;
		buf->appendf("VolumeDir=%s\n", volumeDir.c_str());
		buf->appendf("TfDir=%s\n", tfDir.c_str());
		buf->appendf("SettingsDir=%s\n", settingsDirectory.c_str());
		buf->appendf("SettingsToLoad=%d\n", vis->settingsToLoad_);
		buf->appendf("\n");
	};
	ini_handler.UserData = this;
	ini_handler.ReadOpenFn = settingsReadOpen;
	ini_handler.ReadLineFn = settingsReadLine;
	ini_handler.WriteAllFn = settingsWriteAll;
	GImGui->SettingsHandlers.push_back(ini_handler);

	//initialize renderer
	renderer::initializeRenderer();

	//initialize list of kernels
	renderer::KernelLauncher::Instance().init();
	CUMAT_SAFE_CALL(cudaEventCreate(&startEvent_));
	CUMAT_SAFE_CALL(cudaEventCreate(&stopEvent_));

	//initialize test volume
	//volume_ = renderer::Volume::createSphere(256);
	volume_ = renderer::Volume::createImplicitDataset(
		1 << syntheticDatasetResolutionPower_,
		renderer::Volume::ImplicitEquation(syntheticDatasetIndex_));
	volume_->getLevel(0)->copyCpuToGpu();
	volumeMipmapLevel_ = 0;
	volumeFilename_ = "Initial";
	volumeHistogram_ = volume_->extractHistogram();
	minDensity_ = (minDensity_ < volumeHistogram_.maxDensity && minDensity_ > volumeHistogram_.minDensity) ? minDensity_ : volumeHistogram_.minDensity;
	maxDensity_ = (maxDensity_ < volumeHistogram_.maxDensity && maxDensity_ > volumeHistogram_.minDensity) ? maxDensity_ : volumeHistogram_.maxDensity;
	rendererArgs_.stepsize = 0.1;
	float3 n = renderer::Camera::OrientationUp[clipPlaneIndex_];
	rendererArgs_.clipPlane = make_float4(n, 0);
	
	renderer::OpenGLRasterization::Instance().setMaxFragmentsPerPixel(oitFragmentsPerPixel_);
	renderer::OpenGLRasterization::Instance().setFragmentBufferSize(1 << oitFragmentStoragePower_);
	renderer::OpenGLRasterization::Instance().setMarchingCubesComputationMode(mcComputationMode_);
	if (oitEnableTiling_)
		renderer::OpenGLRasterization::Instance().setTileSize(1 << oitTilingPower_);
	else
		renderer::OpenGLRasterization::Instance().setTileSize(-1);
}

Visualizer::~Visualizer()
{
	releaseResources();
	if (startEvent_)
	{
		cudaEventDestroy(startEvent_);
		startEvent_ = nullptr;
	}
	if (stopEvent_)
	{
		cudaEventDestroy(stopEvent_);
		stopEvent_ = nullptr;
	}
	renderer::KernelLauncher::Instance().cleanup();
}

void Visualizer::releaseResources()
{
	if (screenTextureCuda_)
	{
		CUMAT_SAFE_CALL(cudaGraphicsUnregisterResource(screenTextureCuda_));
		screenTextureCuda_ = nullptr;
	}
	if (screenTextureGL_)
	{
		glDeleteTextures(1, &screenTextureGL_);
		screenTextureGL_ = 0;
	}
	if (screenTextureCudaBuffer_)
	{
		CUMAT_SAFE_CALL(cudaFree(screenTextureCudaBuffer_));
		screenTextureCudaBuffer_ = nullptr;
	}
	if (postOutput_)
	{
		CUMAT_SAFE_CALL(cudaFree(postOutput_));
		postOutput_ = nullptr;
	}
}

void Visualizer::settingsSave()
{
	// save file dialog
	auto fileNameStr = pfd::save_file(
		"Save settings",
		settingsDirectory_,
		{ "Json file", "*.json" },
		true
	).result();
	if (fileNameStr.empty())
		return;

	auto fileNamePath = std::filesystem::path(fileNameStr);
	fileNamePath = fileNamePath.replace_extension(".json");
	std::cout << "Save settings to " << fileNamePath << std::endl;
	settingsDirectory_ = fileNamePath.string();

	// Build json
	nlohmann::json settings;
	settings["version"] = 1;
	//camera
	settings["camera"] = cameraGui_.toJson();
	//computation mode
	settings["renderMode"] = renderMode_;
	settings["renderModeDvr"] = static_cast<int>(renderModeDvr_);
	//TF editor
	settings["tfEditor"] = {
		{"editorLinear", editorLinear_.toJson()},
		{"editorMultiIso", editorMultiIso_.toJson()},
		{"editorHybrid", editorHybrid_.toJson()},
		{"minDensity", minDensity_},
		{"maxDensity", maxDensity_},
		{"opacityScaling", opacityScaling_},
		{"showColorControlPoints", showColorControlPoints_},
		{"dvrUseShading", dvrUseShading_}
	};
	//render parameters
	settings["renderer"] = {
		{"isovalue", rendererArgs_.isovalue},
		{"stepsize", rendererArgs_.stepsize},
		{"filterMode", rendererArgs_.volumeFilterMode},
		{"binarySearchSteps", rendererArgs_.binarySearchSteps},
		{"aoSamples", rendererArgs_.aoSamples},
		{"aoRadius", rendererArgs_.aoRadius},
		{"enableClipPlane", rendererArgs_.enableClipPlane},
		{"clipPlane", rendererArgs_.clipPlane},
		{"clipPlaneIndex", clipPlaneIndex_}
	};
	//shading
	settings["shading"] = {
		{"materialColor", materialColor},
		{"ambientLight", ambientLightColor},
		{"diffuseLight", diffuseLightColor},
		{"specularLight", specularLightColor},
		{"specularExponent", specularExponent},
		{"aoStrength", aoStrength},
		{"lightDirection", lightDirectionScreen},
		{"channel", channelMode_},
		{"flowWithInpainting", flowWithInpainting_},
		{"temporalSmoothing", temporalPostSmoothingPercentage_},
		{"clearColor", clearColor_}
	};
	//dataset
	settings["dataset"] = {
		{"file", volumeFullFilename_},
		{"mipmap", volumeMipmapLevel_},
		{"filterMode", volumeMipmapFilterMode_}
	};

	//save json to file
	std::ofstream o(fileNamePath);
	o << std::setw(4) << settings << std::endl;
	screenshotString_ = std::string("Settings saved to ") + fileNamePath.string();
	screenshotTimer_ = 2.0f;
}

namespace
{
	std::string getDir(const std::string& path)
	{
		if (path.empty())
			return path;
		std::filesystem::path p(path);
		if (std::filesystem::is_directory(p))
			return path;
		return p.parent_path().string();
	}
}

void Visualizer::settingsLoad()
{
	// load file dialog
	auto results = pfd::open_file(
        "Load settings",
        getDir(settingsDirectory_),
        { "Json file", "*.json" },
        false
    ).result();
	if (results.empty())
		return;

	auto fileNameStr = results[0];
	auto fileNamePath = std::filesystem::path(fileNameStr);
	std::cout << "Load settings from " << fileNamePath << std::endl;
	settingsDirectory_ = fileNamePath.string();
	const auto basePath = fileNamePath.parent_path();

	//load json
	std::ifstream i(fileNamePath);
	nlohmann::json settings;
	try
	{
		i >> settings;
	} catch (const nlohmann::json::exception& ex)
	{
		pfd::message("Unable to parse Json", std::string(ex.what()),
			pfd::choice::ok, pfd::icon::error).result();
		return;
	}
	i.close();
	int version = settings.contains("version")
		? settings.at("version").get<int>()
		: 0;
	if (version != 1)
	{
		pfd::message("Illegal Json", "The loaded json does not contain settings in the correct format",
			pfd::choice::ok, pfd::icon::error).result();
		return;
	}

	//Ask which part should be loaded
	static bool loadCamera, loadComputationMode, loadTFeditor, loadRenderer, loadShading, loadDataset;
	static bool popupOpened;
	loadCamera = settingsToLoad_ & CAMERA;
	loadComputationMode = settingsToLoad_ & COMPUTATION_MODE;
	loadTFeditor = settingsToLoad_ & TF_EDITOR;
	loadRenderer = settingsToLoad_ & RENDERER;
	loadShading = settingsToLoad_ & SHADING;
	loadDataset = settingsToLoad_ & DATASET;
	popupOpened = false;
	auto guiTask = [this, basePath, settings]()
	{
		if (!popupOpened)
		{
			ImGui::OpenPopup("What to load");
			popupOpened = true;
			std::cout << "Open popup" << std::endl;
		}
		if (ImGui::BeginPopupModal("What to load", nullptr, ImGuiWindowFlags_AlwaysAutoResize))
		{
			ImGui::Checkbox("Camera##LoadSettings", &loadCamera);
			ImGui::Checkbox("Computation Mode##LoadSettings", &loadComputationMode);
			ImGui::Checkbox("TF Editor##LoadSettings", &loadTFeditor);
			ImGui::Checkbox("Renderer##LoadSettings", &loadRenderer);
			ImGui::Checkbox("Shading##LoadSettings", &loadShading);
			ImGui::Checkbox("Dataset##LoadSettings", &loadDataset);
			if (ImGui::Button("Load##LoadSettings") || ImGui::IsKeyPressedMap(ImGuiKey_Enter))
			{
				try
				{
					//apply new settings
					if (loadCamera)
					{
						cameraGui_.fromJson(settings.at("camera"));
					}
					if (loadComputationMode)
					{
						renderMode_ = settings.at("renderMode").get<RenderMode>();
						renderModeDvr_ = static_cast<DvrTfMode>(settings.value<int>(
							"renderModeDvr", static_cast<int>(DvrTfMode::PiecewiseLinear)));
					}
					if (loadTFeditor)
					{
						const auto& s = settings.at("tfEditor");
						if (s.contains("editor")) {
							//editor is the old name
							editorLinear_.fromJson(s.at("editor"));
						}
						else if (s.contains("editorLinear"))
						{
							editorLinear_.fromJson(s.at("editorLinear"));
						}
						if (s.contains("editorMultiIso"))
						{
							editorMultiIso_.fromJson(s.at("editorMultiIso"));
						}
						if (s.contains("editorHybrid"))
						{
							editorHybrid_.fromJson(s.at("editorHybrid"));
						}
						minDensity_ = s.at("minDensity").get<float>();
						maxDensity_ = s.at("maxDensity").get<float>();
						opacityScaling_ = s.at("opacityScaling").get<float>();
						showColorControlPoints_ = s.at("showColorControlPoints").get<bool>();
						dvrUseShading_ = s.at("dvrUseShading").get<bool>();
					}
					if (loadRenderer)
					{
						const auto& s = settings.at("renderer");
						rendererArgs_.isovalue = s.at("isovalue").get<double>();
						rendererArgs_.stepsize = s.at("stepsize").get<double>();
						rendererArgs_.volumeFilterMode = s.at("filterMode").get<renderer::RendererArgs::VolumeFilterMode>();
						rendererArgs_.binarySearchSteps = s.at("binarySearchSteps").get<int>();
						rendererArgs_.aoSamples = s.at("aoSamples").get<int>();
						rendererArgs_.aoRadius = s.at("aoRadius").get<double>();
						rendererArgs_.enableClipPlane = s.value("enableClipPlane", false);
						rendererArgs_.clipPlane = s.value("clipPlane", make_float4(0,0,0,0));
						clipPlaneIndex_ = s.value("clipPlaneIndex", 1);
					}
					if (loadShading)
					{
						const auto& s = settings.at("shading");
						materialColor = s.at("materialColor").get<float3>();
						ambientLightColor = s.at("ambientLight").get<float3>();
						diffuseLightColor = s.at("diffuseLight").get<float3>();
						specularLightColor = s.at("specularLight").get<float3>();
						specularExponent = s.at("specularExponent").get<float>();
						aoStrength = s.at("aoStrength").get<float>();
						lightDirectionScreen = s.at("lightDirection").get<float3>();
						channelMode_ = s.at("channel").get<ChannelMode>();
						flowWithInpainting_ = s.at("flowWithInpainting").get<bool>();
						temporalPostSmoothingPercentage_ = s.at("temporalSmoothing").get<int>();
						clearColor_ = s.value("clearColor", clearColor_);
					}
					if (loadDataset)
					{
						const auto& s = settings.at("dataset");
						if (!s.at("file").get<std::string>().empty())
						{
							auto targetPath = std::filesystem::path(s.at("file").get<std::string>());
							auto absPath = targetPath.is_absolute()
								? targetPath
								: std::filesystem::absolute(basePath / targetPath);
							try {
								loadVolume(absPath.string(), nullptr);
								volumeDirectory_ = absPath.string();
								selectMipmapLevel(
									s.at("mipmap").get<int>(),
									s.at("filterMode").get<renderer::Volume::MipmapFilterMode>());
							}
							catch (const std::exception& ex)
							{
								std::cerr << "Unable to load dataset with path " << absPath << ": " << ex.what() << std::endl;
							}
						}
					}
					//save last selection
					settingsToLoad_ =
						(loadCamera ? CAMERA : 0) |
						(loadComputationMode ? COMPUTATION_MODE : 0) |
						(loadTFeditor ? TF_EDITOR : 0) |
						(loadRenderer ? RENDERER : 0) |
						(loadShading ? SHADING : 0) |
						(loadDataset ? DATASET : 0);
					ImGui::MarkIniSettingsDirty();
					ImGui::SaveIniSettingsToDisk(GImGui->IO.IniFilename);
					std::cout << "Settings applied" << std::endl;
				} catch (const nlohmann::json::exception& ex)
				{
					std::cerr << "Error: id=" << ex.id << ", message: " << ex.what() << std::endl;
					pfd::message("Unable to apply settings",
						std::string(ex.what()),
						pfd::choice::ok, pfd::icon::error).result();
				}
				//close popup
				triggerRedraw(RedrawRenderer);
				ImGui::CloseCurrentPopup();
				this->backgroundGui_ = {};
			}
			ImGui::SameLine();
			if (ImGui::Button("Cancel##LoadSettings") || ImGui::IsKeyPressedMap(ImGuiKey_Escape))
			{
				//close popup
				triggerRedraw(RedrawRenderer);
				ImGui::CloseCurrentPopup();
				this->backgroundGui_ = {};
			}
			ImGui::EndPopup();
		}
	};
	worker_.wait(); //wait for current task
	this->backgroundGui_ = guiTask;
}

void Visualizer::loadVolumeDialogue()
{
	std::cout << "Open file dialog" << std::endl;

	// open file dialog
	auto results = pfd::open_file(
        "Load volume",
        getDir(volumeDirectory_),
        { "Volumes", "*.dat *.xyz *.cvol" },
        false
    ).result();
	if (results.empty())
		return;
	std::string fileNameStr = results[0];

	std::cout << "Load " << fileNameStr << std::endl;
	auto fileNamePath = std::filesystem::path(fileNameStr);
	volumeDirectory_ = fileNamePath.string();
	ImGui::MarkIniSettingsDirty();
	ImGui::SaveIniSettingsToDisk(GImGui->IO.IniFilename);

	//load the file
	worker_.wait(); //wait for current task
	std::shared_ptr<float> progress = std::make_shared<float>(0);
	auto guiTask = [progress]()
	{
		std::cout << "Progress " << *progress.get() << std::endl;
		if (ImGui::BeginPopupModal("Load Volume", nullptr, ImGuiWindowFlags_AlwaysAutoResize))
		{
			ImGui::ProgressBar(*progress.get(), ImVec2(200, 0));
			ImGui::EndPopup();
		}
	};
	this->backgroundGui_ = guiTask;
	ImGui::OpenPopup("Load Volume");
	auto loaderTask = [fileNameStr, progress, this](BackgroundWorker* worker)
	{
		loadVolume(fileNameStr, progress.get());

		//set it in the GUI and close popup
		this->backgroundGui_ = {};
		ImGui::CloseCurrentPopup();
		triggerRedraw(RedrawRenderer);
	};
	//start background task
	worker_.launch(loaderTask);
}

void Visualizer::loadVolume(const std::string& filename, float* progress)
{
	auto fileNamePath = std::filesystem::path(filename);
	//callbacks
	renderer::VolumeProgressCallback_t progressCallback = [progress](float v)
	{
		if (progress) *progress = v * 0.99f;
	};
	renderer::VolumeLoggingCallback_t logging = [](const std::string& msg)
	{
		std::cout << msg << std::endl;
	};
	int errorCode = 1;
	renderer::VolumeErrorCallback_t error = [&errorCode](const std::string& msg, int code)
	{
		errorCode = code;
		std::cerr << msg << std::endl;
		throw std::exception(msg.c_str());
	};
	//load it locally
	try {
		std::unique_ptr<renderer::Volume> volume;
		if (fileNamePath.extension() == ".dat")
			volume.reset(renderer::loadVolumeFromRaw(filename, progressCallback, logging, error));
		else if (fileNamePath.extension() == ".xyz")
			volume.reset(renderer::loadVolumeFromXYZ(filename, progressCallback, logging, error));
		else if (fileNamePath.extension() == ".cvol")
			volume = std::make_unique<renderer::Volume>(filename, progressCallback, logging, error);
		else {
			std::cerr << "Unrecognized extension: " << fileNamePath.extension() << std::endl;
		}
		if (volume != nullptr) {
			volume->getLevel(0)->copyCpuToGpu();
			std::swap(volume_, volume);
			volumeMipmapLevel_ = 0;
			volumeFilename_ = fileNamePath.filename().string();
			volumeFullFilename_ = fileNamePath.string();
			std::cout << "Loaded" << std::endl;

			volumeHistogram_ = volume_->extractHistogram();

			minDensity_ = (minDensity_ < volumeHistogram_.maxDensity&& minDensity_ > volumeHistogram_.minDensity) ? minDensity_ : volumeHistogram_.minDensity;
			maxDensity_ = (maxDensity_ < volumeHistogram_.maxDensity&& maxDensity_ > volumeHistogram_.minDensity) ? maxDensity_ : volumeHistogram_.maxDensity;
		}
	} catch (std::exception ex)
	{
		std::cerr << "Unable to load volume: " << ex.what() << std::endl;
	}
}

void Visualizer::selectMipmapLevel(int level, renderer::Volume::MipmapFilterMode filter)
{
	if (filter != volumeMipmapFilterMode_)
		volume_->deleteAllMipmapLevels();
	volume_->createMipmapLevel(level, filter);
	volume_->getLevel(level)->copyCpuToGpu();
	volumeMipmapLevel_ = level;
	volumeMipmapFilterMode_ = filter;
}

static void HelpMarker(const char* desc)
{
	//ImGui::TextDisabled(ICON_FA_QUESTION);
	ImGui::TextDisabled("(?)");
	if (ImGui::IsItemHovered())
	{
		ImGui::BeginTooltip();
		ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
		ImGui::TextUnformatted(desc);
		ImGui::PopTextWrapPos();
		ImGui::EndTooltip();
	}
}
void Visualizer::specifyUI()
{
	uiMenuBar();

	ImGui::PushItemWidth(ImGui::GetFontSize() * -8);

	uiVolume();
	uiComputationMode();
	uiCamera();
	uiPixelSelection();
	if (renderMode_ == RenderMode::DVR) {
		uiTfEditor();
	}
	uiRenderer();
	uiKernels();
	uiShading();
	uiRunTimings();

	ImGui::PopItemWidth();

	if (backgroundGui_)
		backgroundGui_();

	uiScreenshotOverlay();
	uiFPSOverlay();
}

void Visualizer::uiMenuBar()
{
	ImGui::BeginMenuBar();
	ImGui::Text("Hotkeys");
	if (ImGui::IsItemHovered())
	{
		ImGui::BeginTooltip();
		ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
		ImGui::TextUnformatted("'P': Screenshot");
		ImGui::TextUnformatted("'S' + left mouse to select a pixel\n'S' + right mouse to clear selection");
		ImGui::PopTextWrapPos();
		ImGui::EndTooltip();
	}
	if (ImGui::SmallButton("Save##Settings"))
		settingsSave();
	if (ImGui::SmallButton("Load##Settings"))
		settingsLoad();
	ImGui::EndMenuBar();
	//hotkeys
	if (ImGui::IsKeyPressed(GLFW_KEY_P, false))
	{
		screenshot();
	}
}

void Visualizer::uiVolume()
{
	if (ImGui::CollapsingHeader("Volume", ImGuiTreeNodeFlags_DefaultOpen)) {
		std::vector<std::string> datasetIndexNames;
		datasetIndexNames.push_back("External file");
		for (const auto& n : magic_enum::enum_names<renderer::Volume::ImplicitEquation>())
		{
			datasetIndexNames.push_back(static_cast<std::string>(n));
		}
		datasetIndexNames.pop_back(); //counter
		if (ImGui::BeginCombo("Source", datasetIndexNames[syntheticDatasetIndex_+1].c_str()))
		{
			for (int i=0; i<datasetIndexNames.size(); ++i)
			{
				bool isSelected = i == (syntheticDatasetIndex_ + 1);
				if (ImGui::Selectable(datasetIndexNames[i].c_str(), isSelected))
				{
					syntheticDatasetIndex_ = i - 1;
					if (syntheticDatasetIndex_ >= 0)
					{
						volume_ = renderer::Volume::createImplicitDataset(
							1 << syntheticDatasetResolutionPower_, 
							renderer::Volume::ImplicitEquation(syntheticDatasetIndex_));
						volume_->getLevel(0)->copyCpuToGpu();
						volumeMipmapLevel_ = 0;
						triggerRedraw(RedrawRenderer);
					}
					else
					{
						//external file selected -> reset volume
						volumeFilename_ = "";
						volumeFullFilename_ = "";
					}
				}
				if (isSelected)
					ImGui::SetItemDefaultFocus();
			}
			ImGui::EndCombo();
		}

		if (syntheticDatasetIndex_ == -1) {
			ImGui::InputText("", &volumeFilename_[0], volumeFilename_.size() + 1, ImGuiInputTextFlags_ReadOnly);
			ImGui::SameLine();
			if (ImGui::Button(ICON_FA_FOLDER_OPEN "##Volume"))
			{
				loadVolumeDialogue();
			}
		}
		else
		{
			std::string resolutionStr = std::to_string(1 << syntheticDatasetResolutionPower_);
			if (ImGui::SliderInt("Resolution", &syntheticDatasetResolutionPower_, 4, 10, resolutionStr.c_str()))
			{
				volume_ = renderer::Volume::createImplicitDataset(
					1 << syntheticDatasetResolutionPower_,
					renderer::Volume::ImplicitEquation(syntheticDatasetIndex_));
				volume_->getLevel(0)->copyCpuToGpu();
				volumeMipmapLevel_ = 0;
				triggerRedraw(RedrawRenderer);
			}
		}

		//functor for selecting the mipmap level, possible in a separate thread
		auto selectMipmapLevelTask = [this](int level, renderer::Volume::MipmapFilterMode filter)
		{
			if (volume_ == nullptr) return;
			if (level == volumeMipmapLevel_ &&
				filter == volumeMipmapFilterMode_) return;
			if (volume_->getLevel(level) == nullptr || filter != volumeMipmapFilterMode_)
			{
				//resample in background thread
				worker_.wait(); //wait for current task
				auto guiTask = []()
				{
					if (ImGui::BeginPopupModal("Resample", nullptr, ImGuiWindowFlags_AlwaysAutoResize))
					{
						const ImU32 col = ImGui::GetColorU32(ImGuiCol_ButtonHovered);
						ImGuiExt::Spinner("ResampleVolume", 50, 10, col);
						ImGui::EndPopup();
					}
				};
				this->backgroundGui_ = guiTask;
				ImGui::OpenPopup("Resample");
				auto resampleTask = [level, filter, this](BackgroundWorker* worker)
				{
					selectMipmapLevel(level, filter);
					//close popup
					this->backgroundGui_ = {};
					ImGui::CloseCurrentPopup();
					triggerRedraw(RedrawRenderer);
				};
				//start background task
				worker_.launch(resampleTask);
			}
			else
			{
				//just ensure, it is on the GPU
				volume_->getLevel(level)->copyCpuToGpu();
				volumeMipmapLevel_ = level;
				volumeMipmapFilterMode_ = filter; //not necessarily needed
				triggerRedraw(RedrawRenderer);
			}
		};
		//Level buttons
		for (int i = 0; i < sizeof(MipmapLevels) / sizeof(int); ++i)
		{
			int l = MipmapLevels[i];
			if (i > 0) ImGui::SameLine();
			std::string label = std::to_string(l + 1) + "x";
			if (ImGui::RadioButton(label.c_str(), volumeMipmapLevel_ == l))
				selectMipmapLevelTask(l, volumeMipmapFilterMode_);
		}
		//Filter buttons
		ImGui::TextUnformatted("Filtering:");
		ImGui::SameLine();
		if (ImGui::RadioButton("Average",
			volumeMipmapFilterMode_ == renderer::Volume::MipmapFilterMode::AVERAGE))
			selectMipmapLevelTask(volumeMipmapLevel_, renderer::Volume::MipmapFilterMode::AVERAGE);
		ImGui::SameLine();
		if (ImGui::RadioButton("Halton",
			volumeMipmapFilterMode_ == renderer::Volume::MipmapFilterMode::HALTON))
			selectMipmapLevelTask(volumeMipmapLevel_, renderer::Volume::MipmapFilterMode::HALTON);

		//print statistics
		ImGui::Text("Resolution: %d, %d, %d\nSize: %5.3f, %5.3f, %5.3f",
			volume_ && volume_->getLevel(volumeMipmapLevel_) ? static_cast<int>(volume_->getLevel(volumeMipmapLevel_)->sizeX()) : 0,
			volume_ && volume_->getLevel(volumeMipmapLevel_) ? static_cast<int>(volume_->getLevel(volumeMipmapLevel_)->sizeY()) : 0,
			volume_ && volume_->getLevel(volumeMipmapLevel_) ? static_cast<int>(volume_->getLevel(volumeMipmapLevel_)->sizeZ()) : 0,
			volume_ ? volume_->worldSizeX() : 0,
			volume_ ? volume_->worldSizeY() : 0,
			volume_ ? volume_->worldSizeZ() : 0);

		if (volume_)
		{
			ImGui::Text("Min Density: %f\nMax Density: %f", volumeHistogram_.minDensity, volumeHistogram_.maxDensity);
		}
	}
}

void Visualizer::uiCamera()
{
	if (ImGui::CollapsingHeader("Camera")) {
		if (cameraGui_.specifyUI()) triggerRedraw(RedrawRenderer);
	}
	if (cameraGui_.updateMouse())
		triggerRedraw(RedrawRenderer);
}

void Visualizer::uiPixelSelection()
{
	if (ImGui::IsKeyDown(GLFW_KEY_S))
	{
		if (ImGui::IsMouseClicked(1))
		{
			pixelSelected_ = false;
		}
		else if (ImGui::IsMouseClicked(0))
		{
			auto pos = ImGui::GetMousePos();
			selectedPixel_ = make_int2(static_cast<int>(pos.x), static_cast<int>(pos.y));
			pixelSelected_ = true;
			triggerRedraw(RedrawRenderer);
		}
	}

	if (pixelSelected_)
	{
		extraFrameInformation_ << "\nPixel selected: " << selectedPixel_.x << ", " << selectedPixel_.y;
	}
}

void Visualizer::uiRenderer()
{
	if (ImGui::CollapsingHeader("Render Parameters")) {
		if (renderMode_ == RenderMode::ISO)
		{
			double isoMin = 0.01, isoMax = 2.0;
			if (ImGui::SliderScalar("Isovalue", ImGuiDataType_Double, &rendererArgs_.isovalue, &isoMin, &isoMax, "%.5f", 2)) triggerRedraw(RedrawRenderer);
		}
		double stepMin = 0.01, stepMax = 1.0;
		if (ImGui::SliderScalar("Stepsize", ImGuiDataType_Double, &rendererArgs_.stepsize, &stepMin, &stepMax, "%.5f", 2)) triggerRedraw(RedrawRenderer);
		static const char* VolumeFilterModeNames[] = { "Trilinear", "Tricubic" };
		const char* currentFilterModeName = (int(rendererArgs_.volumeFilterMode) >= 0 && int(rendererArgs_.volumeFilterMode) < int(renderer::RendererArgs::VolumeFilterMode::_COUNT_))
			? VolumeFilterModeNames[int(rendererArgs_.volumeFilterMode)] : "Unknown";
		if (ImGui::SliderInt("Filter Mode", reinterpret_cast<int*>(&rendererArgs_.volumeFilterMode),
			0, int(renderer::RendererArgs::VolumeFilterMode::_COUNT_) - 1, currentFilterModeName))
			triggerRedraw(RedrawRenderer);
		if (renderMode_ == RenderMode::ISO)
		{
			int binaryMin = 0, binaryMax = 10;
			if (ImGui::SliderScalar("Binary Search", ImGuiDataType_S32, &rendererArgs_.binarySearchSteps, &binaryMin, &binaryMax, "%d")) triggerRedraw(RedrawRenderer);
			int aoSamplesMin = 0, aoSamplesMax = 512;
			if (ImGui::SliderScalar("AO Samples", ImGuiDataType_S32, &rendererArgs_.aoSamples, &aoSamplesMin, &aoSamplesMax, "%d", 2)) triggerRedraw(RedrawRenderer);
			double aoRadiusMin = 0.01, aoRadiusMax = 0.5;
			if (ImGui::SliderScalar("AO Radius", ImGuiDataType_Double, &rendererArgs_.aoRadius, &aoRadiusMin, &aoRadiusMax, "%.5f", 2)) triggerRedraw(RedrawRenderer);
		}
		else if (renderMode_ == RenderMode::DVR)
		{
			ImGui::TextUnformatted("2^"); ImGui::SameLine();
			if (ImGui::SliderInt("Fragment Storage", &oitFragmentStoragePower_, 20, 26))
			{
				renderer::OpenGLRasterization::Instance().setFragmentBufferSize(1 << oitFragmentStoragePower_);
				triggerRedraw(RedrawRenderer);
			}
			{
				int logNum = int(round(log2(oitFragmentsPerPixel_)));
				int minLogNum = 4, maxLogNum = 9;
				std::string logNumStr = std::to_string(oitFragmentsPerPixel_);
				if (ImGui::SliderInt("Fragments p.P.", &logNum, minLogNum, maxLogNum, logNumStr.c_str()))
				{
					oitFragmentsPerPixel_ = 1 << logNum;
					renderer::OpenGLRasterization::Instance().setMaxFragmentsPerPixel(oitFragmentsPerPixel_);
					triggerRedraw(RedrawRenderer);
				}
			}
			static const char* MCModeNames[] = { "Pre-Device", "Pre-Host", "On-the-fly" };
			if (ImGui::SliderInt("MC Mode", reinterpret_cast<int*>(&mcComputationMode_),
				0, 2, MCModeNames[int(mcComputationMode_)]))
			{
				renderer::OpenGLRasterization::Instance().setMarchingCubesComputationMode(mcComputationMode_);
				triggerRedraw(RedrawRenderer);
			}
			if (ImGui::Checkbox("##oitEnableTiling", &oitEnableTiling_))
			{
				if (oitEnableTiling_)
					renderer::OpenGLRasterization::Instance().setTileSize(1 << oitTilingPower_);
				else
					renderer::OpenGLRasterization::Instance().setTileSize(-1);
				triggerRedraw(RedrawRenderer);
			}
			ImGui::SameLine();
			std::string tilingStr = std::to_string(1 << oitTilingPower_);
			if (ImGui::SliderInt("OIT Tiling", &oitTilingPower_, 4, 10, tilingStr.c_str()))
			{
				if (oitEnableTiling_) {
					renderer::OpenGLRasterization::Instance().setTileSize(1 << oitTilingPower_);
					triggerRedraw(RedrawRenderer);
				}
			}
		}
		//clip plane
		if (ImGui::Checkbox("##ClipPlaneEnabled", &rendererArgs_.enableClipPlane))
			triggerRedraw(RedrawRenderer);
		ImGui::SameLine();
		ImGui::PushItemWidth(ImGui::GetFontSize() * 4);
		if (ImGui::Combo("##ClipPlaneNormal", &clipPlaneIndex_,
		                 renderer::Camera::OrientationNames, 6))
		{
			float3 n = renderer::Camera::OrientationUp[clipPlaneIndex_];
			rendererArgs_.clipPlane = make_float4(n, rendererArgs_.clipPlane.w);
			triggerRedraw(RedrawRenderer);
		}
		ImGui::PopItemWidth();
		ImGui::SameLine();
		if (ImGui::SliderFloat("ClipPlane##ClipPlaneOffset", &rendererArgs_.clipPlane.w, -1, 1))
			triggerRedraw(RedrawRenderer);
	}
}

void Visualizer::uiComputationMode()
{
	if (ImGui::RadioButton("Iso-surface",
		reinterpret_cast<int*>(&renderMode_), static_cast<int>(RenderMode::ISO)))
		triggerRedraw(RedrawRenderer);
	ImGui::SameLine();
	if (ImGui::RadioButton("Dvr",
		reinterpret_cast<int*>(&renderMode_), static_cast<int>(RenderMode::DVR)))
		triggerRedraw(RedrawRenderer);
	if (renderMode_ == RenderMode::DVR)
	{
		if (ImGui::RadioButton("Piecewise",
			reinterpret_cast<int*>(&renderModeDvr_), static_cast<int>(DvrTfMode::PiecewiseLinear)))
			triggerRedraw(RedrawRenderer);
		ImGui::SameLine();
		if (ImGui::RadioButton("Multi-Iso",
			reinterpret_cast<int*>(&renderModeDvr_), static_cast<int>(DvrTfMode::MultipleIsosurfaces)))
			triggerRedraw(RedrawRenderer);
		ImGui::SameLine();
		if (ImGui::RadioButton("Hybrid",
			reinterpret_cast<int*>(&renderModeDvr_), static_cast<int>(DvrTfMode::Hybrid)))
			triggerRedraw(RedrawRenderer);
	}
}

void Visualizer::uiKernels()
{
	renderer::KernelLauncher::KernelTypes kernelType;
	if (renderMode_ == RenderMode::ISO)
		kernelType = renderer::KernelLauncher::KernelTypes::Iso;
	else if (renderMode_ == RenderMode::DVR)
	{
		if (renderModeDvr_ == DvrTfMode::PiecewiseLinear)
			kernelType = renderer::KernelLauncher::KernelTypes::Dvr;
		else if (renderModeDvr_ == DvrTfMode::MultipleIsosurfaces)
			kernelType = renderer::KernelLauncher::KernelTypes::MultiIso;
		else if (renderModeDvr_ == DvrTfMode::Hybrid)
			kernelType = renderer::KernelLauncher::KernelTypes::Hybrid;
	}
	const std::vector<std::string>& kernels =
		renderer::KernelLauncher::Instance().getKernelNames(kernelType);
	int& selection = selectedKernel_[int(kernelType)];
	if (ImGui::CollapsingHeader("Kernels", ImGuiTreeNodeFlags_DefaultOpen))
	{
		std::vector<const char*> kernelNames(kernels.size());
		for (size_t i = 0; i < kernels.size(); ++i) kernelNames[i] = kernels[i].c_str();
		ImGui::PushItemWidth(-1);
		if (ImGui::ListBox("##Kernels", &selection, kernelNames.data(), kernels.size()))
		{
			triggerRedraw(RedrawRenderer);
		}
		ImGui::PopItemWidth();
#if RENDERER_RUNTIME_COMPILATION==1
		if (ImGui::Button("Reload Kernels"))
		{
			reloadKernels();
			triggerRedraw(RedrawRenderer);
		}
		if (ImGui::Checkbox("Debugging", &kernelDebugging_))
		{
			reloadKernels();
			triggerRedraw(RedrawRenderer);
		}
		ImGui::SameLine();
		if (ImGui::Checkbox("Instrumentation", &kernelInstrumentation_))
		{
			reloadKernels();
			triggerRedraw(RedrawRenderer);
		}
		if (kernelDebugging_) {
			if (ImGui::Checkbox("Filter Voxel", &voxelFiltered_))
				triggerRedraw(RedrawRenderer);
			int minv[3] = { 0,0,0 };
			int maxv[3] = {
				static_cast<int>(volume_->getLevel(volumeMipmapLevel_)->sizeX()),
				static_cast<int>(volume_->getLevel(volumeMipmapLevel_)->sizeY()) ,
				static_cast<int>(volume_->getLevel(volumeMipmapLevel_)->sizeZ()) };
			if (ImGui::SliderInt3("Voxel", reinterpret_cast<int*>(&selectedVoxel_), minv, maxv))
				triggerRedraw(RedrawRenderer);
		}
#endif
	}
}

void Visualizer::uiTfEditor()
{
	if (ImGui::CollapsingHeader("TF Editor"))
	{
		if (ImGui::Button(ICON_FA_FOLDER_OPEN " Load TF"))
		{
			// open file dialog
			std::vector<std::string> filters = { "Transfer Function (*.tf)", "*.tf" };
			if (renderModeDvr_ == DvrTfMode::PiecewiseLinear)
			{
				filters.push_back("Colormaps (*.xml");
				filters.push_back("*.xml");
			}
			auto results = pfd::open_file(
				"Load transfer function",
				tfDirectory_,
				filters,
				false
			).result();
			if (results.empty())
				return;
			std::string fileNameStr = results[0];

			auto fileNamePath = std::filesystem::path(fileNameStr);
			std::cout << "TF is loaded from " << fileNamePath << std::endl;
			tfDirectory_ = fileNamePath.string();

			if (renderModeDvr_ == DvrTfMode::PiecewiseLinear)
				editorLinear_.loadFromFile(fileNamePath.string(), minDensity_, maxDensity_);
			else if (renderModeDvr_ == DvrTfMode::MultipleIsosurfaces)
				editorMultiIso_.loadFromFile(fileNamePath.string(), minDensity_, maxDensity_);
			if (renderModeDvr_ == DvrTfMode::Hybrid)
				editorHybrid_.loadFromFile(fileNamePath.string(), minDensity_, maxDensity_);
			triggerRedraw(RedrawRenderer);
		}
		ImGui::SameLine();
		if (ImGui::Button(ICON_FA_SAVE " Save TF"))
		{
			// save file dialog
			auto fileNameStr = pfd::save_file(
				"Save transfer function",
				tfDirectory_,
				{ "Transfer Function", "*.tf" },
				true
			).result();
			if (fileNameStr.empty())
				return;

			auto fileNamePath = std::filesystem::path(fileNameStr);
			fileNamePath = fileNamePath.replace_extension(".tf");
			std::cout << "TF is saved under " << fileNamePath << std::endl;
			tfDirectory_ = fileNamePath.string();

			if (renderModeDvr_ == DvrTfMode::PiecewiseLinear)
				editorLinear_.saveToFile(fileNamePath.string(), minDensity_, maxDensity_);
			else if (renderModeDvr_ == DvrTfMode::MultipleIsosurfaces)
				editorMultiIso_.saveToFile(fileNamePath.string(), minDensity_, maxDensity_);
			if (renderModeDvr_ == DvrTfMode::Hybrid)
				editorHybrid_.saveToFile(fileNamePath.string(), minDensity_, maxDensity_);
		}
		if (renderModeDvr_ == DvrTfMode::PiecewiseLinear || renderModeDvr_ == DvrTfMode::Hybrid) {
			ImGui::SameLine();
			ImGui::Checkbox("Show CPs", &showColorControlPoints_);
		}

		ImGuiWindow* window = ImGui::GetCurrentWindow();
		ImGuiContext& g = *GImGui;
		const ImGuiStyle& style = g.Style;

		ImRect tfEditorColorRect;
		if (renderModeDvr_ == DvrTfMode::PiecewiseLinear || renderModeDvr_ == DvrTfMode::Hybrid) {
			//Color
			const ImGuiID tfEditorColorId = window->GetID("TF Editor Color");
			auto pos = window->DC.CursorPos;
			auto tfEditorColorWidth = window->WorkRect.Max.x - window->WorkRect.Min.x;
			auto tfEditorColorHeight = 50.0f;
			tfEditorColorRect = ImRect(pos, ImVec2(pos.x + tfEditorColorWidth, pos.y + tfEditorColorHeight));
			ImGui::ItemSize(tfEditorColorRect, style.FramePadding.y);
			ImGui::ItemAdd(tfEditorColorRect, tfEditorColorId);
		}

		//Opacity
		const ImGuiID tfEditorOpacityId = window->GetID("TF Editor Opacity");
		auto pos = window->DC.CursorPos;
		auto tfEditorOpacityWidth = window->WorkRect.Max.x - window->WorkRect.Min.x;
		auto tfEditorOpacityHeight = 100.0f;
		const ImRect tfEditorOpacityRect(pos, ImVec2(pos.x + tfEditorOpacityWidth, pos.y + tfEditorOpacityHeight));

		auto histogramRes = (volumeHistogram_.maxDensity - volumeHistogram_.minDensity) / volumeHistogram_.getNumOfBins();
		int histogramBeginOffset = (minDensity_ - volumeHistogram_.minDensity) / histogramRes;
		int histogramEndOffset = (volumeHistogram_.maxDensity - maxDensity_) / histogramRes;
		auto maxFractionVal = *std::max_element(std::begin(volumeHistogram_.bins) + histogramBeginOffset, std::end(volumeHistogram_.bins) - histogramEndOffset);
		ImGui::PlotHistogram("", volumeHistogram_.bins + histogramBeginOffset, volumeHistogram_.getNumOfBins() - histogramEndOffset - histogramBeginOffset,
			0, NULL, 0.0f, maxFractionVal, ImVec2(tfEditorOpacityWidth, tfEditorOpacityHeight));

		if (renderModeDvr_ == DvrTfMode::PiecewiseLinear) 
		{
			editorLinear_.init(tfEditorOpacityRect, tfEditorColorRect, showColorControlPoints_);
			editorLinear_.handleIO();
			editorLinear_.render();
		}
		else if (renderModeDvr_ == DvrTfMode::MultipleIsosurfaces)
		{
			editorMultiIso_.init(tfEditorOpacityRect);
			editorMultiIso_.handleIO();
			editorMultiIso_.render();
		}
		else if (renderModeDvr_ == DvrTfMode::Hybrid) 
		{
			editorHybrid_.init(tfEditorOpacityRect, tfEditorColorRect, showColorControlPoints_);
			editorHybrid_.handleIO();
			editorHybrid_.render();
		}

		if (renderModeDvr_ != DvrTfMode::MultipleIsosurfaces) {
			if (ImGui::SliderFloat("Opacity Scaling", &opacityScaling_, 1.0f, 500.0f))
			{
				triggerRedraw(RedrawRenderer);
			}
		}
		if (ImGui::SliderFloat("Min Density", &minDensity_, volumeHistogram_.minDensity, volumeHistogram_.maxDensity))
		{
			triggerRedraw(RedrawRenderer);
		}
		if (ImGui::SliderFloat("Max Density", &maxDensity_, volumeHistogram_.minDensity, volumeHistogram_.maxDensity))
		{
			triggerRedraw(RedrawRenderer);
		}
		//static const char* TfPreintegrationNames[] = { "Off", "1D", "2D" };
		//if (ImGui::SliderInt("Preintegration", 
		//	reinterpret_cast<int*>(&tfPreintegration_), 0, 2,
		//	TfPreintegrationNames[static_cast<int>(tfPreintegration_)]))
		//{
		//	triggerRedraw(RedrawRenderer);
		//}
		if (ImGui::Checkbox("Use Shading", &dvrUseShading_))
		{
			triggerRedraw(RedrawRenderer);
		}
		if (renderModeDvr_ == DvrTfMode::PiecewiseLinear) {
			if (ImGui::Button("from MultiIso"))
			{
				editorLinear_.fromMultiIso(&editorMultiIso_, isoToDvrCopyWidth_);
				triggerRedraw(RedrawRenderer);
			}
			ImGui::SameLine();
			ImGui::SliderFloat("Width", &isoToDvrCopyWidth_, 0.0001f, 0.01f);
		}
		else if (renderModeDvr_ == DvrTfMode::Hybrid) {
			if (ImGui::Button("from MultiIso"))
			{
				editorHybrid_.fromMultiIso(&editorMultiIso_, isoToDvrCopyWidth_);
				triggerRedraw(RedrawRenderer);
			}
			ImGui::SameLine();
			ImGui::SliderFloat("Width", &isoToDvrCopyWidth_, 0.0001f, 0.01f);
		}
		if (editorLinear_.getIsChanged() || editorMultiIso_.getIsChanged() || editorHybrid_.getIsChanged())
		{
			triggerRedraw(RedrawRenderer);
		}
	}
}

void Visualizer::uiShading()
{
	if (ImGui::CollapsingHeader("Output - Shading", ImGuiTreeNodeFlags_DefaultOpen)) {
		auto redraw = renderMode_ == RenderMode::ISO
			? RedrawPost
			: RedrawRenderer;

		ImGuiColorEditFlags colorFlags = ImGuiColorEditFlags_Float | ImGuiColorEditFlags_PickerHueWheel;
		if (ImGui::ColorEdit3("Material Color", &materialColor.x, colorFlags)) triggerRedraw(redraw);
		if (ImGui::ColorEdit3("Ambient Light", &ambientLightColor.x, colorFlags)) triggerRedraw(redraw);
		if (ImGui::ColorEdit3("Diffuse Light", &diffuseLightColor.x, colorFlags)) triggerRedraw(redraw);
		if (ImGui::ColorEdit3("Specular Light", &specularLightColor.x, colorFlags)) triggerRedraw(redraw);
		float minSpecular = 0, maxSpecular = 64;
		if (ImGui::SliderScalar("Spec. Exp.", ImGuiDataType_Float, &specularExponent, &minSpecular, &maxSpecular, "%.3f", 2)) triggerRedraw(redraw);
		float minAO = 0, maxAO = 1;
		if (ImGui::SliderScalar("AO Strength", ImGuiDataType_Float, &aoStrength, &minAO, &maxAO)) triggerRedraw(RedrawPost);
		if (ImGuiExt::DirectionPicker2D("Light direction", &lightDirectionScreen.x, ImGuiExtDirectionPickerFlags_InvertXY))
			triggerRedraw(redraw);
		const char* currentChannelName = (channelMode_ >= 0 && channelMode_ < _ChannelCount_)
			? ChannelModeNames[channelMode_] : "Unknown";
		if (ImGui::SliderInt("Channel", reinterpret_cast<int*>(&channelMode_), 0, _ChannelCount_ - 1, currentChannelName))
			triggerRedraw(RedrawPost);
		if (channelMode_ == ChannelFlow)
			if (ImGui::Checkbox("Flow with Inpainting", &flowWithInpainting_))
				triggerRedraw(RedrawPost);
		if (ImGui::SliderInt("Temporal Smoothing", &temporalPostSmoothingPercentage_, 0, 100, "%d%%"))
			triggerRedraw(RedrawRenderer);

		static const char* ClearColorNames[] = { "Black", "White" };
		int clearColorIndex = clearColor_.x > 0.5;
		if (ImGui::SliderInt("Background", &clearColorIndex, 0, 1, ClearColorNames[clearColorIndex]))
		{
			if (clearColorIndex)
				clearColor_ = ImVec4(1, 1, 1, 1);
			else
				clearColor_ = ImVec4(0, 0, 0, 1);
			triggerRedraw(RedrawPost);
		}
	}
}

void Visualizer::uiRunTimings()
{
	ImGui::PushItemWidth(ImGui::GetWindowWidth()/3);
	static int width = 1920;
	static int height = 1080;
	ImGui::InputInt("##width", &width);
	ImGui::SameLine();
	ImGui::InputInt("##height", &height);
	ImGui::SameLine();
	if (ImGui::Button("Timings"))
	{
		std::cout << "Run Timings" << std::endl;
		renderer::RendererArgs args = setupRendererArgs(renderMode_, 1);
		args.cameraResolutionX = width;
		args.cameraResolutionY = height;
		int outputChannels = renderMode_ == RenderMode::ISO
			? RENDERER_NAMESPACE::IsoRendererOutputChannels
			: RENDERER_NAMESPACE::DvrRendererOutputChannels;
		rendererOutput_ = RENDERER_NAMESPACE::OutputTensor(
			args.cameraResolutionY, args.cameraResolutionX, outputChannels);

		std::string kernelName;
		renderer::KernelLauncher::KernelTypes kernelType;
		if (renderMode_ == RenderMode::ISO)
			kernelType = renderer::KernelLauncher::KernelTypes::Iso;
		else if (renderMode_ == RenderMode::DVR)
		{
			if (renderModeDvr_ == DvrTfMode::PiecewiseLinear)
				kernelType = renderer::KernelLauncher::KernelTypes::Dvr;
			else if (renderModeDvr_ == DvrTfMode::MultipleIsosurfaces)
				kernelType = renderer::KernelLauncher::KernelTypes::MultiIso;
			else if (renderModeDvr_ == DvrTfMode::Hybrid)
				kernelType = renderer::KernelLauncher::KernelTypes::Hybrid;
		}
		const std::vector<std::string>& kernels =
			renderer::KernelLauncher::Instance().getKernelNames(kernelType);
		if (kernels.empty()) return;
		const int selection = selectedKernel_[int(kernelType)];
		if (selection < 0 || selection >= kernels.size())
		{
			std::cerr << "Invalid selection (out of bounds)" << std::endl;
			return;
		}
		kernelName = kernels[selection];

		cudaStream_t stream = 0; // cuMat::Context::current().stream();
		cudaEventRecord(startEvent_, stream);
		static const int RUNS = 10;
		for (int i=0; i<RUNS; ++i)
			render_gpu(kernelName, volume_.get(), &args, rendererOutput_, stream, nullptr, &globalInstrumentation_);
		cudaEventRecord(stopEvent_, stream);

		CUMAT_SAFE_CALL(cudaEventSynchronize(stopEvent_));
		elapsedMilliseconds_ = 0;
		CUMAT_SAFE_CALL(cudaEventElapsedTime(&elapsedMilliseconds_, startEvent_, stopEvent_));
		elapsedMilliseconds_ /= RUNS;
		std::cout << "Time: " << elapsedMilliseconds_ << " ms" << std::endl;
	}
	ImGui::PopItemWidth();
}

void Visualizer::uiScreenshotOverlay()
{
	if (screenshotTimer_ <= 0) return;

	ImGuiIO& io = ImGui::GetIO();
	ImVec2 window_pos = ImVec2(io.DisplaySize.x / 2, io.DisplaySize.y - 10);
	ImVec2 window_pos_pivot = ImVec2(0.5f, 1.0f);
	ImGui::SetNextWindowPos(window_pos, ImGuiCond_Always, window_pos_pivot);
	//ImGui::PushStyleVar(ImGuiStyleVar_Alpha, alpha);
	ImGui::Begin("Example: Simple overlay", nullptr, ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoNav);
	ImGui::TextUnformatted(screenshotString_.c_str());
	ImGui::End();
	//ImGui::PopStyleVar(ImGuiStyleVar_Alpha);

	screenshotTimer_ -= io.DeltaTime;
}

void Visualizer::uiFPSOverlay()
{
	ImGuiIO& io = ImGui::GetIO();
	ImVec2 window_pos = ImVec2(io.DisplaySize.x - 5, 5);
	ImVec2 window_pos_pivot = ImVec2(1.0f, 0.0f);
	ImGui::SetNextWindowPos(window_pos, ImGuiCond_Always, window_pos_pivot);
	ImGui::SetNextWindowBgAlpha(0.5f);
	ImGui::Begin("FPSDisplay", nullptr, ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoNav);

	//ImGui::Text("FPS: %.1f", io.Framerate);
	ImGui::Text("FPS: %.1f", currentFPS_);
	ImGui::Text("Kernel time: %.2f ms", elapsedMilliseconds_);
	
	std::string extraText = extraFrameInformation_.str();
	if (!extraText.empty())
	{
		extraText = extraText.substr(1); //strip initial '\n'
		ImGui::TextUnformatted(extraText.c_str());
	}
	extraFrameInformation_ = std::stringstream();

#if RENDERER_RUNTIME_COMPILATION==1
	if (kernelInstrumentation_)
	{
		std::stringstream str;
		str << std::setprecision(2) << std::fixed;
#define ADD_STATS(name, field) \
	       name << ": " << accumulatedInstrumentation_.field.mean() << " (" \
	    << sqrtf(accumulatedInstrumentation_.field.var()) << ")"
		str << ADD_STATS("Density fetches", densityFetches) << "\n";
		str << ADD_STATS("TF fetches", tfFetches) << "\n";
		str << ADD_STATS("DDA steps", ddaSteps) << "\n";
		str << ADD_STATS("Iso intersections", isoIntersections) << "\n";
		str << ADD_STATS("Quadratures", intervalEval) << "\n";
		str << ADD_STATS("Quadrature samples", intervalStep) << "\n";
		str << "Average samples: " << (accumulatedInstrumentation_.intervalStep.mean() /
			accumulatedInstrumentation_.intervalEval.mean()) << "\n";
		str << "Max quadrature samples:" << accumulatedInstrumentation_.intervalMaxStep;
		ImGui::TextUnformatted(str.str().c_str());
	}
#endif

	{
		std::stringstream str;
		struct thousandsdot : std::numpunct<char>
		{
			char_type do_thousands_sep() const { return ','; }
			std::string do_grouping() const { return "\3"; }
		};
		str.imbue(std::locale(std::cout.getloc(), new thousandsdot()));
		if (globalInstrumentation_.numControlPoints > 0)
			str << "Num control points: " << globalInstrumentation_.numControlPoints << "\n";
		if (globalInstrumentation_.numTriangles > 0)
			str << "Num triangles: " << globalInstrumentation_.numTriangles << "\n";
		if (globalInstrumentation_.numFragments > 0)
		{
			str << "Num fragments: " << globalInstrumentation_.numFragments << "\n";
			if (renderer::OpenGLRasterization::Instance().hadOverflow())
				str << "  Overflow!!\n";
		}
		if (!str.str().empty())
		{
			auto s = str.str();
			s[s.size()] = '\0';
			ImGui::TextUnformatted(s.c_str());
		}
	}
	
	ImGui::End();
}

void Visualizer::reloadKernels()
{
	if (renderer::KernelLauncher::Instance().reload(std::cout, 
		kernelDebugging_, kernelInstrumentation_))
	{
		screenshotString_ = "Kernels reloaded";
		screenshotTimer_ = 2.0f;
	}
	else
	{
		screenshotString_ = "Failed to reloaded kernels, see console log";
		screenshotTimer_ = 2.0f;
	}
	for (int i=0; i<int(renderer::KernelLauncher::KernelTypes::__COUNT__); ++i)
	{
		selectedKernel_[i] = std::min(selectedKernel_[i],
			static_cast<int>(renderer::KernelLauncher::Instance().getKernelNames(
				static_cast<renderer::KernelLauncher::KernelTypes>(i)).size())-1);
	}
}

renderer::RendererArgs Visualizer::setupRendererArgs(
	RenderMode renderMode, int upscaleFactor)
{
	cameraGui_.updateRenderArgs(rendererArgs_);
	renderer::RendererArgs args = rendererArgs_;
	args.cameraResolutionX = displayWidth_ / upscaleFactor;
	args.cameraResolutionY = displayHeight_ / upscaleFactor;
	args.cameraViewport = make_int4(0, 0, -1, -1);
	args.mipmapLevel = volumeMipmapLevel_;
	args.renderMode = renderMode;
	args.dvrTfMode = renderModeDvr_;
	if (renderModeDvr_ == DvrTfMode::PiecewiseLinear) 
	{
		args.tf = renderer::RendererArgs::TfLinear{
			editorLinear_.getDensityAxisOpacity(),
			editorLinear_.getOpacityAxis(),
			{},
			editorLinear_.getDensityAxisColor(),
			editorLinear_.getColorAxis()
		};
	} else if (renderModeDvr_ == DvrTfMode::MultipleIsosurfaces)
	{
		args.tf = renderer::RendererArgs::TfMultiIso{
			editorMultiIso_.getControlPointPositions(),
			editorMultiIso_.getControlPointColors()
		};
	}
	else if (renderModeDvr_ == DvrTfMode::Hybrid)
	{
		args.tf = renderer::RendererArgs::TfLinear{
			editorHybrid_.getDensityAxisOpacity(),
			editorHybrid_.getOpacityAxis(),
			editorHybrid_.getOpacityExtraColorAxis(),
			editorHybrid_.getDensityAxisColor(),
			editorHybrid_.getColorAxis()
		};
	}
	args.opacityScaling = opacityScaling_;
	args.minDensity = minDensity_;
	args.maxDensity = maxDensity_;

	renderer::ShadingSettings shading;
	shading.ambientLightColor = ambientLightColor;
	shading.diffuseLightColor = diffuseLightColor;
	shading.specularLightColor = specularLightColor;
	shading.specularExponent = specularExponent;
	shading.materialColor = materialColor;
	shading.aoStrength = aoStrength;
	shading.lightDirection = normalize(cameraGui_.screenToWorld(lightDirectionScreen));
	args.shading = shading;
	args.dvrUseShading = dvrUseShading_;

	args.pixelSelected = pixelSelected_;
	args.selectedPixel = selectedPixel_;
	args.voxelFiltered = voxelFiltered_;
	args.selectedVoxel = selectedVoxel_;

	return args;
}

void Visualizer::render(int display_w, int display_h)
{
	resize(display_w, display_h);

	if (volume_ == nullptr) return;
	if (volume_->getLevel(volumeMipmapLevel_) == nullptr) return;

	if (redrawMode_ == RedrawNone)
	{
		//just draw the precomputed texture
		drawer_.drawQuad(screenTextureGL_);
		return;
	}

	auto startTime = std::chrono::steady_clock::now();
	renderImpl(renderMode_);
	auto endTime = std::chrono::steady_clock::now();
	currentFPS_ = 1000000.0 / std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count();
}

void Visualizer::renderImpl(RenderMode renderMode)
{
	//render iso-surface to rendererOutput_
	if (redrawMode_ == RedrawRenderer)
	{
		memset(&globalInstrumentation_, 0, sizeof(RENDERER_NAMESPACE::GlobalInstrumentation));
		int upscale_factor = 1;
		renderer::RendererArgs args = setupRendererArgs(renderMode, upscale_factor);
		if (args.stepsize <= 1e-5) return; //in the middle of manual editing the stepsize
		args.aoSamples = renderMode==RenderMode::ISO ? rendererArgs_.aoSamples : 0;

		int outputChannels = renderMode == RenderMode::ISO
			? RENDERER_NAMESPACE::IsoRendererOutputChannels
			: RENDERER_NAMESPACE::DvrRendererOutputChannels;
		rendererOutput_ = RENDERER_NAMESPACE::OutputTensor(
			args.cameraResolutionY, args.cameraResolutionX, outputChannels);
		kernel::PerPixelInstrumentation* instrumentation = nullptr;
#if RENDERER_RUNTIME_COMPILATION==1
		if (kernelInstrumentation_)
			CUMAT_SAFE_CALL(cudaMalloc(&instrumentation, 
				args.cameraResolutionX * args.cameraResolutionY * sizeof(kernel::PerPixelInstrumentation)));
#endif
		
		std::string kernelName;
		renderer::KernelLauncher::KernelTypes kernelType;
		if (renderMode_ == RenderMode::ISO)
			kernelType = renderer::KernelLauncher::KernelTypes::Iso;
		else if (renderMode_ == RenderMode::DVR)
		{
			if (renderModeDvr_ == DvrTfMode::PiecewiseLinear)
				kernelType = renderer::KernelLauncher::KernelTypes::Dvr;
			else if (renderModeDvr_ == DvrTfMode::MultipleIsosurfaces)
				kernelType = renderer::KernelLauncher::KernelTypes::MultiIso;
			else if (renderModeDvr_ == DvrTfMode::Hybrid)
				kernelType = renderer::KernelLauncher::KernelTypes::Hybrid;
		}
		const std::vector<std::string>& kernels =
			renderer::KernelLauncher::Instance().getKernelNames(kernelType);
		if (kernels.empty()) return;
		const int selection = selectedKernel_[int(kernelType)];
		if (selection<0 || selection>=kernels.size())
		{
			std::cerr << "Invalid selection (out of bounds)" << std::endl;
			return;
		}
		kernelName = kernels[selection];

		cudaStream_t stream = 0; // cuMat::Context::current().stream();
		cudaEventRecord(startEvent_, stream);
		render_gpu(kernelName, volume_.get(), &args, rendererOutput_, stream, instrumentation, &globalInstrumentation_);
		cudaEventRecord(stopEvent_, stream);
		interpolatedFlowAvailable_ = false;

		CUMAT_SAFE_CALL(cudaEventSynchronize(stopEvent_));
		CUMAT_SAFE_CALL(cudaEventElapsedTime(&elapsedMilliseconds_, startEvent_, stopEvent_));
		
		if (instrumentation)
		{
			std::vector<kernel::PerPixelInstrumentation> instrumentationHost(
				args.cameraResolutionX * args.cameraResolutionY);
			cudaMemcpy(instrumentationHost.data(), instrumentation,
				args.cameraResolutionX * args.cameraResolutionY * sizeof(kernel::PerPixelInstrumentation),
				cudaMemcpyDeviceToHost);
			CUMAT_SAFE_CALL(cudaFree(instrumentation));
			//accumulate stats
			accumulatedInstrumentation_.densityFetches.reset();
			accumulatedInstrumentation_.tfFetches.reset();
			accumulatedInstrumentation_.ddaSteps.reset();
			accumulatedInstrumentation_.isoIntersections.reset();
			accumulatedInstrumentation_.intervalEval.reset();
			accumulatedInstrumentation_.intervalStep.reset();
			accumulatedInstrumentation_.intervalMaxStep = 0;
			for (int i=0; i< args.cameraResolutionX * args.cameraResolutionY; ++i)
			{
				accumulatedInstrumentation_.densityFetches.append(instrumentationHost[i].densityFetches);
				accumulatedInstrumentation_.tfFetches.append(instrumentationHost[i].tfFetches);
				accumulatedInstrumentation_.ddaSteps.append(instrumentationHost[i].ddaSteps);
				accumulatedInstrumentation_.isoIntersections.append(instrumentationHost[i].isoIntersections);
				accumulatedInstrumentation_.intervalEval.append(instrumentationHost[i].intervalEval);
				accumulatedInstrumentation_.intervalStep.append(instrumentationHost[i].intervalStep);
				accumulatedInstrumentation_.intervalMaxStep = std::max(
					accumulatedInstrumentation_.intervalMaxStep, instrumentationHost[i].intervalMaxStep);
			}
		}
		
		redrawMode_ = RedrawPost;
	}

	//flow inpainting
	bool needFlow =
		temporalPostSmoothingPercentage_ > 0 ||
		(channelMode_ == ChannelFlow && flowWithInpainting_);
	if (needFlow && !interpolatedFlowAvailable_)
	{
		std::cerr << "Inpainting disabled, there is an out-of-bounds error somewhere" << std::endl;
		//if (renderMode == RenderMode::ISO)
		//{
		//	interpolatedFlow_ = kernel::inpaintFlow(rendererOutput_, 0, 6, 7);
		//}
		//else //DVR
		//{
		//	interpolatedFlow_ = kernel::inpaintFlow(rendererOutput_, 3, 8, 9);
		//}
	}

	//select channel and write to screen texture
	//this also includes the temporal reprojection
	if (redrawMode_ == RedrawPost)
	{
		if (channelMode_ == ChannelFlow)
		{
			if (flowWithInpainting_)
			{
				kernel::selectOutputChannel(interpolatedFlow_, postOutput_,
					0, 1, -1, -2,
					10, 0.5, 1, 0);
			}
			else
			{
				kernel::selectOutputChannel(rendererOutput_, postOutput_,
					6, 7, -1, -2,
					10, 0.5, 1, 0);
			}
		}
		else {
			//temporal reprojection
			RENDERER_NAMESPACE::OutputTensor blendingOutput;
			if (previousBlendingOutput_.rows() != rendererOutput_.rows() ||
				previousBlendingOutput_.cols() != rendererOutput_.cols() ||
				previousBlendingOutput_.batches() != rendererOutput_.batches() ||
				temporalPostSmoothingPercentage_ == 0)
			{
				blendingOutput = rendererOutput_;
			}
			else
			{
				auto previousOutput = kernel::warp(
					previousBlendingOutput_,
					interpolatedFlow_);
				float blendingFactor = temporalPostSmoothingPercentage_ / 100.0f;
				blendingOutput = kernel::lerp(rendererOutput_, previousOutput, blendingFactor);
			}
			//channel selection
			if (renderMode == RenderMode::ISO)
				selectChannelIso(channelMode_, blendingOutput, postOutput_);
			else
				selectChannelDvr(channelMode_, blendingOutput, postOutput_);
			previousBlendingOutput_ = blendingOutput;
		}

		redrawMode_ = RedrawNone;
	}

	cudaMemcpy(screenTextureCudaBuffer_, postOutput_, 4 * displayWidth_*displayHeight_,
		cudaMemcpyDeviceToDevice);
	copyBufferToOpenGL();

	//draw texture
	drawer_.drawQuad(screenTextureGL_);
}

void Visualizer::copyBufferToOpenGL()
{
	CUMAT_SAFE_CALL(cudaGraphicsMapResources(1, &screenTextureCuda_, 0));
	cudaArray* texture_ptr;
	CUMAT_SAFE_CALL(cudaGraphicsSubResourceGetMappedArray(&texture_ptr, screenTextureCuda_, 0, 0));
	size_t size_tex_data = sizeof(GLubyte) * displayWidth_ * displayHeight_ * 4;
	CUMAT_SAFE_CALL(cudaMemcpyToArray(texture_ptr, 0, 0, screenTextureCudaBuffer_, size_tex_data, cudaMemcpyDeviceToDevice));
	CUMAT_SAFE_CALL(cudaGraphicsUnmapResources(1, &screenTextureCuda_, 0));
}

void Visualizer::resize(int display_w, int display_h)
{
	//NOT NEEDED as we don't apply neural networks
	////make it a nice multiplication of everything
	//const int multiply = 4 * 3;
	//display_w = display_w / multiply * multiply;
	//display_h = display_h / multiply * multiply;

	if (display_w == displayWidth_ && display_h == displayHeight_)
		return;
	if (display_w == 0 || display_h == 0)
		return;
	releaseResources();
	displayWidth_ = display_w;
	displayHeight_ = display_h;

	//create texture
	glGenTextures(1, &screenTextureGL_);
	glBindTexture(GL_TEXTURE_2D, screenTextureGL_);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE); // clamp s coordinate
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE); // clamp t coordinate
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	glTexImage2D(
		GL_TEXTURE_2D, 0, GL_RGBA8,
		displayWidth_, displayHeight_, 0
		, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);

	//register with cuda
	CUMAT_SAFE_CALL(cudaGraphicsGLRegisterImage(
		&screenTextureCuda_, screenTextureGL_,
		GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard));

	//create channel output buffer
	CUMAT_SAFE_CALL(cudaMalloc(&screenTextureCudaBuffer_, displayWidth_ * displayHeight_ * 4 * sizeof(GLubyte)));
	CUMAT_SAFE_CALL(cudaMalloc(&postOutput_, displayWidth_ * displayHeight_ * 4 * sizeof(GLubyte)));

	glBindTexture(GL_TEXTURE_2D, 0);

	triggerRedraw(RedrawRenderer);
	std::cout << "Visualizer::resize(): " << displayWidth_ << ", " << displayHeight_ << std::endl;
}

void Visualizer::triggerRedraw(RedrawMode mode)
{
	redrawMode_ = std::max(redrawMode_, mode);
}

void Visualizer::selectChannelIso(ChannelMode mode, const RENDERER_NAMESPACE::OutputTensor& output, GLubyte* cudaBuffer) const
{
	if (output.rows() != displayHeight_ || output.cols() != displayWidth_)
	{
		std::cout << "Tensor shape does not match: expected=(1 * Channels * "
			<< displayHeight_ << " * " << displayWidth_ << "), but got: "
			<< output.rows() << " * " << output.cols() << std::endl;
		throw std::exception("Tensor shape does not match screen size");
	}
	switch (mode)
	{
	case ChannelMask:
		kernel::selectOutputChannel(output, cudaBuffer,
			0, 0, 0, 0,
			1, 0, 0, 1);
		break;
	case ChannelNormal:
		kernel::selectOutputChannel(output, cudaBuffer,
			1, 2, 3, 0,
			0.5, 0.5, 1, 0);
		break;
	case ChannelDepth: {
		auto minMaxDepth = kernel::extractMinMaxDepth(output, 4);
		float minDepth = 0.0f;//minMaxDepth.first;
		float maxDepth = minMaxDepth.second;
		//std::cout << "depth: min=" << minDepth << ", max=" << maxDepth << std::endl;
		kernel::selectOutputChannel(output, cudaBuffer,
			4, 4, 4, 0,
			1 / (maxDepth - minDepth), -minDepth / (maxDepth - minDepth), 1, 0);
		break;
	}
	case ChannelAO:
		kernel::selectOutputChannel(output, cudaBuffer,
			5, 5, 5, 0,
			1, 0, 1, 0);
		break;
	case ChannelFlow:
		kernel::selectOutputChannel(output, cudaBuffer,
			6, 7, -1, -2,
			10, 0.5, 1, 0);
		break;
	case ChannelColor: {
		renderer::ShadingSettings settings;
		settings.ambientLightColor = ambientLightColor;
		settings.diffuseLightColor = diffuseLightColor;
		settings.specularLightColor = specularLightColor;
		settings.specularExponent = specularExponent;
		settings.materialColor = materialColor;
		settings.aoStrength = aoStrength;
		settings.lightDirection = lightDirectionScreen;
		kernel::screenShading(output, cudaBuffer, settings);
		break;
	}
	default:
		throw std::exception("unknown enum");
	}
}

void Visualizer::selectChannelDvr(ChannelMode mode, const RENDERER_NAMESPACE::OutputTensor& output, GLubyte* cudaBuffer) const
{
	if (output.rows() != displayHeight_ || output.cols() != displayWidth_)
	{
		std::cout << "Tensor shape does not match: expected=(1 * Channels * "
			<< displayHeight_ << " * " << displayWidth_ << "), but got: "
			<< output.rows() << " * " << output.cols() << std::endl;
		throw std::exception("Tensor shape does not match screen size");
	}
	switch (mode)
	{
	case ChannelColor:
		kernel::selectOutputChannel(output, cudaBuffer,
			0, 1, 2, 3,
			1, 0, 1, 0); //color is pre-multiplied with alpha
		break;
	case ChannelMask:
		kernel::selectOutputChannel(output, cudaBuffer,
			3, 3, 3, 3,
			1, 0, 0, 1);
		break;
	case ChannelNormal:
		kernel::selectOutputChannel(output, cudaBuffer,
			4, 5, 6, 3,
			0.5, 0.5, 1, 0);
		break;
	case ChannelDepth: {
		auto minMaxDepth = kernel::extractMinMaxDepth(output, 7);
		float minDepth = minMaxDepth.first;
		float maxDepth = minMaxDepth.second;
		//std::cout << "depth: min=" << minDepth << ", max=" << maxDepth << std::endl;
		kernel::selectOutputChannel(output, cudaBuffer,
			7, 7, 7, 3,
			1 / (maxDepth - minDepth), -minDepth / (maxDepth - minDepth), 1, 0);
		break;
	}
	case ChannelAO:
		kernel::selectOutputChannel(output, cudaBuffer,
			-1, -1, -1, -1, //disabled
			1, 0, 1, 0);
		break;
	case ChannelFlow:
		kernel::selectOutputChannel(output, cudaBuffer,
			8, 9, -1, -2,
			10, 0.5, 1, 0);
		break;
	default:
		throw std::exception("unknown enum");
	}
}

void Visualizer::screenshot()
{
	std::string folder = "screenshots";

	char time_str[128];
	time_t now = time(0);
	struct tm tstruct;
	localtime_s(&tstruct, &now);
	strftime(time_str, sizeof(time_str), "%Y%m%d-%H%M%S", &tstruct);

	char output_name[512];
	sprintf(output_name, "%s/screenshot_%s_%s.png", folder.c_str(), time_str, ChannelModeNames[channelMode_]);

	std::cout << "Take screenshot: " << output_name << std::endl;
	std::filesystem::create_directory(folder);

	std::vector<GLubyte> textureCpu(4 * displayWidth_ * displayHeight_);
	CUMAT_SAFE_CALL(cudaMemcpy(&textureCpu[0], screenTextureCudaBuffer_, 4 * displayWidth_*displayHeight_, cudaMemcpyDeviceToHost));

	if (lodepng_encode32_file(output_name, textureCpu.data(), displayWidth_, displayHeight_) != 0)
	{
		std::cerr << "Unable to save image" << std::endl;
		screenshotString_ = std::string("Unable to save screenshot to ") + output_name;
	}
	else
	{
		screenshotString_ = std::string("Screenshot saved to ") + output_name;
	}
	screenshotTimer_ = 2.0f;
}
