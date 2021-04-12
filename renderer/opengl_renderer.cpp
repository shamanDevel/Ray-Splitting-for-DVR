#include "opengl_renderer.h"
#include "opengl_utils.h"

#include <iostream>
#include <fstream>
#include <cuMat/src/Errors.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <tinyformat.h>

#include "camera.h"
#include "renderer_color.cuh"

#if RENDERER_RUNTIME_COMPILATION==0
#include <cmrc/cmrc.hpp>
CMRC_DECLARE(renderer);
#endif

BEGIN_RENDERER_NAMESPACE

OpenGLRasterization OpenGLRasterization::INSTANCE;
bool OpenGLRasterization::registered_;

const std::string OpenGLRasterization::SINGLE_ISO_NAME = "Iso: Marching Cubes";
const std::string OpenGLRasterization::MULTIPLE_ISO_NAME = "MultiIso: Marching Cubes";
const std::string OpenGLRasterization::DVR_NAME = "DVR: Marching Cubes";

namespace
{
	GLFWwindow* offscreenWindow = nullptr;
}
static void glfw_error_callback(int error, const char* description)
{
	fprintf(stderr, "Glfw Error %d: %s\n", error, description);
}

OpenGLRasterization::OpenGLRasterization()
	: mcMode_(MarchingCubesComputationMode::PRE_HOST)
	, maxFragmentsPerPixel_(64)
	, tileSize_(-1)
{
}

void OpenGLRasterization::initialize()
{
	if (!initialized_)
	{
		initMarchingCubes();
		oit_ = std::make_unique<OIT>();
		dynamicMesh_ = std::make_unique<Mesh>();
		
		//load shaders
		loadShaders();
		
		initialized_ = true;
	}
}

void OpenGLRasterization::loadShaders()
{
	std::stringstream settings;
	settings << "#define MAX_FRAGMENTS " << maxFragmentsPerPixel_ << "\n";
	const auto settingsStr = settings.str();
	isosurfaceShader_ = std::make_unique<renderer::Shader>(
		"SingleIso.vs", "ShowNormals.fs", settingsStr);
	multiIsoRenderShader_ = std::make_unique<renderer::Shader>(
		"PassThrough.vs", "OITIsoRendering.fs", settingsStr);
	multiIsoBlendingShader_ = std::make_unique<renderer::Shader>(
		"ScreenQuad.vs", "OITIsoBlending.fs", settingsStr);
	dvrBlendingShader_ = std::make_unique<renderer::Shader>(
		"ScreenQuad.vs", "OITDvrBlending.fs", settingsStr);
}

void OpenGLRasterization::Register()
{
	const std::vector<std::string> names = {
		SINGLE_ISO_NAME,
		MULTIPLE_ISO_NAME,
		DVR_NAME
	};

	registered_ = true;
	for (const std::string& name : names)
	{
		registered_ = registered_ && KernelLauncher::RegisterCustomKernelLauncher(
			name,
			[name](const std::string& kernelName)
		{
			assert(name == name);
			return &OpenGLRasterization::Instance();
		});
	}
}

void OpenGLRasterization::setupOffscreenContext()
{
	std::cout << "Setup offscreen context" << std::endl;
	if (offscreenWindow != nullptr)
		throw std::runtime_error("offscreen context already created");

	// Setup window
	glfwSetErrorCallback(glfw_error_callback);
	if (!glfwInit())
		throw std::runtime_error("Unable to initialize GLFW");

	// GL 3.0 + GLSL 130
	const char* glsl_version = "#version 130";
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);  // 3.2+ only
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);            // 3.0+ only
	glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
#if !defined(NDEBUG)
	glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GLFW_TRUE);
#endif

	offscreenWindow = glfwCreateWindow(640, 480, "", NULL, NULL);
	if (offscreenWindow == nullptr)
		throw std::runtime_error("Unable to create offscreen window");
	glfwMakeContextCurrent(offscreenWindow);

	if (glewInit() != GLEW_OK)
	{
		std::cerr << "Unable to initialize GLEW" << std::endl;
		deleteOffscreenContext();
		throw std::runtime_error("Unable to initialize GLEW");
	}
	std::cout << "Offscreen context created" << std::endl;
}

void OpenGLRasterization::deleteOffscreenContext()
{
	if (offscreenWindow == nullptr)
		throw std::runtime_error("offscreen context already destroyed or none created");

	glfwDestroyWindow(offscreenWindow);
	glfwTerminate();
	
	offscreenWindow = nullptr;
}

void OpenGLRasterization::reload()
{
	loadShaders();
}

void OpenGLRasterization::cleanup()
{
	framebuffer_.reset();
	isosurfaceShader_.reset();
	isosurfaceMesh_.reset();
	multiIsoBlendingShader_.reset();
	multiIsoRenderShader_.reset();
	dvrBlendingShader_.reset();
	multiIsoMeshes_.clear();
	oit_.reset();
	dynamicMesh_.reset();
}

void OpenGLRasterization::setFragmentBufferSize(int size)
{
	initialize();
	oit_->resizeBuffer(size);
}

int OpenGLRasterization::getFragmentBufferSize() const
{
	return oit_->getNumFragments();
}

void OpenGLRasterization::setMarchingCubesComputationMode(MarchingCubesComputationMode mode)
{
	mcMode_ = mode;
	multiIsoMeshes_.clear();
}

OpenGLRasterization::MarchingCubesComputationMode OpenGLRasterization::getMarchingCubesComputationMode() const
{
	return mcMode_;
}

void OpenGLRasterization::setMaxFragmentsPerPixel(int count)
{
	maxFragmentsPerPixel_ = count;
	reload();
}

int OpenGLRasterization::getMaxFragmentsPerPixel() const
{
	return maxFragmentsPerPixel_;
}

void OpenGLRasterization::setTileSize(int tileSize)
{
	tileSize_ = tileSize;
}

int OpenGLRasterization::getTileSize() const
{
	return tileSize_;
}

static glm::vec3 toGLM3(float3 v)
{
	const auto r = glm::vec3(v.x, v.y, v.z);
	return r;
}
static glm::vec4 toGLM4(float4 v) { return glm::vec4(v.x, v.y, v.z, v.w); }

void OpenGLRasterization::fillMarchingCubesMeshes(const Volume::MipmapLevel* data, cudaStream_t stream)
{
	for (auto& m : multiIsoMeshes_)
	{
		if (m.changed)
		{
			if (mcMode_ == MarchingCubesComputationMode::PRE_DEVICE) {
				if (!m.mesh)
					m.mesh = std::make_unique<Mesh>();
				fillMarchingCubesMeshPreDevice(data, m.iso,
				                               m.mesh.get(), stream);
			}
			else if (mcMode_ == MarchingCubesComputationMode::PRE_HOST)
			{
				fillMarchingCubesMeshPreHost(data, m.iso,
				                             &m.meshCpu, dynamicMesh_.get(), stream);
			}
		}
	}
}

bool OpenGLRasterization::render(const std::string& kernelName, int screenWidth, int screenHeight,
                                 const kernel::RendererDeviceSettings& deviceSettings, const RendererArgs* hostSettings,
                                 const Volume::MipmapLevel* data, kernel::OutputTensor& output,
                                 cudaStream_t stream,
                                 kernel::PerPixelInstrumentation* perPixelInstrumentation,
                                 GlobalInstrumentation* globalInstrumentation)
{
	initialize();
	hadOverflow_ = false;

	struct Viewport
	{
		float x, y, width, height;
	} oldViewport;
	glGetFloatv(GL_VIEWPORT, &oldViewport.x);
	glViewport(0, 0, screenWidth, screenHeight);
	checkOpenGLError();

	if (!framebuffer_ || framebuffer_->width()!=screenWidth || framebuffer_->height()!=screenHeight)
	{
		framebuffer_ = std::make_unique<Framebuffer>(screenWidth, screenHeight);
		oit_->resizeScreen(screenWidth, screenHeight);
		//for now, just use 20 fragments per pixel
		if (!oit_->getNumFragments()) //if not already specified by the GUI
			oit_->resizeBuffer(20 * screenWidth * screenHeight);
	}

	//matrices
	glm::vec3 boxSize(deviceSettings.boxSize.x, deviceSettings.boxSize.y, deviceSettings.boxSize.z);
	glm::vec3 voxelSize = boxSize / (glm::vec3(
		deviceSettings.volumeResolution.x, deviceSettings.volumeResolution.y, deviceSettings.volumeResolution.z) - glm::vec3(1.0f));
	glm::vec3 boxMin = glm::vec3(-boxSize.x / 2, -boxSize.y / 2, -boxSize.z / 2) - (voxelSize * 0.5f);
	
	glm::mat4 model = glm::mat4(1.0f);
	model = glm::scale(model, voxelSize);
	model = glm::translate(glm::mat4(1.0f), boxMin) * model;
	glm::mat4 view, projection, normal;
	renderer::Camera::computeMatricesOpenGL(
		hostSettings->cameraOrigin, hostSettings->cameraLookAt, hostSettings->cameraUp,
		hostSettings->cameraFovDegrees, screenWidth, screenHeight,
		hostSettings->nearClip, hostSettings->farClip,
		view, projection, normal);
	//normal = transpose(normal);
	const glm::vec3 viewPos = toGLM3(hostSettings->cameraOrigin);
	const glm::vec3 lightDir = toGLM3(hostSettings->shading.lightDirection);
	const glm::vec3 ambientLightColor = toGLM3(hostSettings->shading.ambientLightColor);
	const glm::vec3 diffuseLightColor = toGLM3(hostSettings->shading.diffuseLightColor);
	const glm::vec3 specularLightColor = toGLM3(hostSettings->shading.specularLightColor);

#if 0
	//test transformations
	for (int i = 0; i < 2; ++i) for (int j = 0; j < 2; ++j) for (int k = 0; k < 2; ++k)
	{
		glm::vec3 objectPos(
			i * data->sizeX(),
			j * data->sizeY(),
			k * data->sizeZ());
		glm::vec4 worldPos = model * glm::vec4(objectPos, 1.0f);
		glm::vec4 viewPos = view * worldPos;
		glm::vec4 screenPos = projection * viewPos;
		screenPos /= screenPos.w;
		float4 viewport; glGetFloatv(GL_VIEWPORT, &viewport.x);
		int xScreen = static_cast<int>(round((screenPos.x + 1) * (viewport.z / 2) + viewport.x));
		int yScreen = static_cast<int>(round((screenPos.y + 1) * (viewport.w / 2) + viewport.y));
		tinyformat::format(std::cout,
			"Object (%5.2f, %5.2f, %5.2f) -> World (%5.2f, %5.2f, %5.2f) -> Screen (%5.2f, %5.2f, %5.2f) (%4d, %4d)\n",
			objectPos.x, objectPos.y, objectPos.z,
			viewPos.x, viewPos.y, viewPos.z,
			screenPos.x, screenPos.y, screenPos.z,
			xScreen, yScreen);
	}
#endif

	framebuffer_->bind();
	if (kernelName == SINGLE_ISO_NAME)
	{
		//(re)create marching cubes mesh
		if (!isosurfaceMesh_)
			isosurfaceMesh_ = std::make_unique<Mesh>();
		if (previousSettings_.isovalue != deviceSettings.isovalue 
			|| previousData_ != data
			|| previousKernelName_ != kernelName)
			fillMarchingCubesMeshPreDevice(data, 
				deviceSettings.isovalue, 
				isosurfaceMesh_.get(),
				stream);

		//render
		isosurfaceShader_->use();

		glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glEnable(GL_DEPTH_TEST);
		glDisable(GL_CULL_FACE);

		isosurfaceShader_->setMat4("model", model);
		isosurfaceShader_->setMat4("transpInvModel", normal);
		isosurfaceShader_->setMat4("view", view);
		isosurfaceShader_->setMat4("projection", projection);
		
		isosurfaceMesh_->draw();
		if (globalInstrumentation)
			globalInstrumentation->numTriangles = isosurfaceMesh_->getNumIndices() / 3;
	}
	else if (kernelName == MULTIPLE_ISO_NAME || kernelName == DVR_NAME)
	{
		bool globalChanged = previousData_ != data
			|| previousKernelName_ != kernelName;

		//1. prepare meshes (choose isovalues)
		if (kernelName == MULTIPLE_ISO_NAME)
		{
			multiIsoMeshes_.resize(deviceSettings.tfPoints.numPoints - 2);
			for (auto i = 0u; i < multiIsoMeshes_.size(); ++i)
			{
				multiIsoMeshes_[i].changed = globalChanged ||
					(multiIsoMeshes_[i].iso != deviceSettings.tfPoints.positions[i + 1]);
				multiIsoMeshes_[i].iso = deviceSettings.tfPoints.positions[i + 1];
				float4 colXYZ = deviceSettings.tfPoints.valuesIso[i + 1];
				multiIsoMeshes_[i].color.x = colXYZ.x;
				multiIsoMeshes_[i].color.y = colXYZ.y;
				multiIsoMeshes_[i].color.z = colXYZ.z;
				multiIsoMeshes_[i].color.w = colXYZ.w;
			}
		}
		else if (kernelName == DVR_NAME)
		{
			//Note: the first and second isovalue is for the bounds, not needed here
			// they are needed for raytracing.
			const int numPointsWithBounds = 1 + max(1, static_cast<int>(1 / hostSettings->stepsize));
			const int minControlPoint = deviceSettings.tfPoints.valuesDvr[0].w > 0 ? 0 : 1;
			const int maxControlPoint = 
				deviceSettings.tfPoints.valuesDvr[deviceSettings.tfPoints.numPoints-1].w > 0
				? deviceSettings.tfPoints.numPoints
				: deviceSettings.tfPoints.numPoints-1;
			const int numControlPoints = maxControlPoint-minControlPoint;
			const int numTotalPoints = 1 + (numControlPoints - 1) * (numPointsWithBounds - 1);
			multiIsoMeshes_.resize(numTotalPoints);
			//1. find isovalue and color
			int meshIndex = 0;
			for (int i = minControlPoint; i < maxControlPoint-1; ++i)
			{
				float isoStart = deviceSettings.tfPoints.positions[i];
				float isoEnd = deviceSettings.tfPoints.positions[i + 1];
				float4 colorStart = deviceSettings.tfPoints.valuesDvr[i];
				float4 colorEnd = deviceSettings.tfPoints.valuesDvr[i + 1];
				for (int j = 0; j < numPointsWithBounds; ++j)
				{
					if (i > 1 && j == 0) continue; //skip duplicate boundary
					float f = j / (numPointsWithBounds - 1.0f);
					float iso = isoStart + f * (isoEnd - isoStart);
					glm::vec4 color = toGLM4(colorStart + f * (colorEnd - colorStart));
					color.w *= deviceSettings.opacityScaling;
					multiIsoMeshes_[meshIndex].changed =
						(multiIsoMeshes_[meshIndex].iso != iso) ||
						globalChanged;
					multiIsoMeshes_[meshIndex].iso = iso;
					multiIsoMeshes_[meshIndex].color = color;
					++meshIndex;
				}
			}
		}
		
		//2. create mesh
		fillMarchingCubesMeshes(data, stream);

		//3. render
		const auto renderIsosurfaces = [&](
			Shader* renderShader, Shader* blendingShader)
		{
			int tileSize = max(screenHeight, screenWidth);
			int numFragments = 0, numTriangles = 0;
			if (tileSize_ < 0)
				glDisable(GL_SCISSOR_TEST);
			else
			{
				//tiled rendering
				glEnable(GL_SCISSOR_TEST);
				tileSize = tileSize_;
			}

			for (int scissorX = 0; scissorX<screenWidth; scissorX += tileSize)
				for (int scissorY = 0; scissorY < screenHeight; scissorY += tileSize)
				{
					glScissor(scissorX, scissorY, tileSize, tileSize);
					
					renderShader->use();
					checkOpenGLError();
					renderShader->setMat4("model", model);
					//renderShader->setMat4("transpInvModel", normal); //not needed
					renderShader->setMat4("view", view);
					renderShader->setMat4("projection", projection);
					renderShader->setVec3("lightDir", lightDir);
					renderShader->setVec3("viewPos", viewPos);
					renderShader->setVec3("ambientLightColor", ambientLightColor);
					renderShader->setVec3("diffuseLightColor", diffuseLightColor);
					renderShader->setVec3("specularLightColor", specularLightColor);
					renderShader->setInt("specularExponent", deviceSettings.shading.specularExponent);
					renderShader->setBool("useShading", deviceSettings.useShading);
					checkOpenGLError();
					oit_->start();
					oit_->setShaderParams(renderShader);
					glDisable(GL_DEPTH_TEST);
					glDisable(GL_CULL_FACE);
					glDisable(GL_BLEND);
					int numIndices = 0;
					for (const auto& m : multiIsoMeshes_) {
						renderShader->setVec4("objectColor", m.color);
						checkOpenGLError();
						if (mcMode_ == MarchingCubesComputationMode::PRE_DEVICE) {
							m.mesh->draw();
							numIndices += m.mesh->getNumIndices();
						}
						else if (mcMode_ == MarchingCubesComputationMode::PRE_HOST)
						{
							dynamicMesh_->copyFromCpu(m.meshCpu);
							dynamicMesh_->draw();
							numIndices += dynamicMesh_->getNumIndices();
						}
						else if (mcMode_ == MarchingCubesComputationMode::ON_THE_FLY)
						{
							fillMarchingCubesMeshPreDevice(data, m.iso,
								dynamicMesh_.get(), stream);
							dynamicMesh_->draw();
							numIndices += dynamicMesh_->getNumIndices();
						}
					}
					numTriangles += numIndices / 3;

					//blend fragment list
					glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
					glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
					checkOpenGLError();

					blendingShader->use();
					int tmpFragments = oit_->blend(blendingShader);
					if (tmpFragments > oit_->getNumFragments())
						hadOverflow_ = true;
					numFragments += tmpFragments;
				}
					
			if (globalInstrumentation) {
				globalInstrumentation->numFragments = numFragments;
				globalInstrumentation->numTriangles = numTriangles;
			}
			oit_->finish();

			glDisable(GL_SCISSOR_TEST);
		};

		if (kernelName == MULTIPLE_ISO_NAME)
		{
			renderIsosurfaces(multiIsoRenderShader_.get(), multiIsoBlendingShader_.get());
		}
		else if (kernelName == DVR_NAME)
		{
			renderIsosurfaces(multiIsoRenderShader_.get(), dvrBlendingShader_.get());
		}
	}
	
	framebuffer_->unbind();

#if 0
	std::vector<float> rgba;
	framebuffer_->readRGBA(rgba);
	//save as image
	{
		using namespace std;
		ofstream ofs("testMarchingCubes.ppm", ios_base::out | ios_base::binary);
		ofs << "P6" << endl << screenWidth << ' ' << screenHeight << endl << "255" << endl;

		for (auto j = 0; j < screenHeight; ++j) {
			for (auto i = 0; i < screenWidth; ++i) {
				uint8_t r = static_cast<uint8_t>(rgba[0 + 4 * (i + screenWidth * j)] * 255);
				uint8_t g = static_cast<uint8_t>(rgba[1 + 4 * (i + screenWidth * j)] * 255);
				uint8_t b = static_cast<uint8_t>(rgba[2 + 4 * (i + screenWidth * j)] * 255);
				uint8_t a = static_cast<uint8_t>(rgba[3 + 4 * (i + screenWidth * j)] * 255);
				ofs << r << g << b;
			}
		}
		ofs.close();
	}
#endif

	//copy from OpenGL to CUDA for further processing
	if (kernelName == SINGLE_ISO_NAME)
	{
		framebuffer_->copyToCudaIso(output, stream);
	}
	else if (kernelName == MULTIPLE_ISO_NAME 
		|| kernelName == DVR_NAME)
	{
		framebuffer_->copyToCudaDvr(output, stream);
	}
	
	previousSettings_ = deviceSettings;
	previousData_ = data;
	previousKernelName_ = kernelName;

	glViewport(oldViewport.x, oldViewport.y, oldViewport.width, oldViewport.height);
	checkOpenGLError();
	
	return true;
}

END_RENDERER_NAMESPACE
