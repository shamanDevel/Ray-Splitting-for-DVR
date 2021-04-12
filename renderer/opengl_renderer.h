#pragma once

#include "commons.h"

#include <GL/glew.h>
#include <cuda_gl_interop.h>
#include <glm/glm.hpp>

#include "opengl_shader.h"
#include "opengl_framebuffer.h"
#include "opengl_mesh.h"
#include "opengl_oit.h"

#include "kernel_launcher.h"
#include "settings.h"
#include "volume.h"
#include "renderer_settings.cuh"

BEGIN_RENDERER_NAMESPACE

#define MARCHING_CUBES 1
#define MARCHING_TETS 2
//the algorithm to select:
#define TRIANGULATION_ALGORITHM MARCHING_TETS

/**
 * \brief Class for rendering algorithms that make use of the
 * OpenGL rasterization pipeline.
 * This is called from within \ref render_gpu() of renderer.h.
 */
MY_API class OpenGLRasterization : public ICustomKernelLauncher
{
private:
	static OpenGLRasterization INSTANCE;

	//kernel names
	static const std::string SINGLE_ISO_NAME;
	static const std::string MULTIPLE_ISO_NAME;
	static const std::string DVR_NAME;
	
	static bool registered_;
	OpenGLRasterization();
public:
	static OpenGLRasterization& Instance() { return INSTANCE; }
	/**
	 * \brief Registers the OpenGL-Rasterization kernels.
	 * This has to be done manually since all static registrations
	 * are deleted when including the static library.
	 */
	static void Register();

	//for Python interop
	void setupOffscreenContext();
	void deleteOffscreenContext();

	//main entry from renderer.h
	//Not reentrant!
	virtual bool render(
		const std::string& kernelName,
		int screenWidth, int screenHeight,
		const kernel::RendererDeviceSettings& deviceSettings,
		const RendererArgs* hostSettings,
		const Volume::MipmapLevel* data,
		kernel::OutputTensor& output,
		cudaStream_t stream,
		kernel::PerPixelInstrumentation* perPixelInstrumentation,
		GlobalInstrumentation* globalInstrumentation) override;
	bool hadOverflow() const { return hadOverflow_; }
	
	void fillMarchingCubesMeshPreDevice(
		const Volume::MipmapLevel* data,
		float isosurface,
		Mesh* output,
		cudaStream_t stream);
	void fillMarchingCubesMeshPreHost(
		const Volume::MipmapLevel* data,
		float isosurface,
		MeshCpu* output, Mesh* tmp,
		cudaStream_t stream);

	void reload() override;
	void cleanup() override;

	/**
	 * \brief Sets the number of fragments the OIT-Buffer can hold.
	 * Takes effect after the next draw call.
	 */
	void setFragmentBufferSize(int size);
	int getFragmentBufferSize() const;

	enum class MarchingCubesComputationMode
	{
		/**
		 * Precomputed, stored as OpenGL buffers.
		 * Fastest, but requires a lot of device memory.
		 */
		PRE_DEVICE,
		/**
		 * Precomputed, but stored on host memory.
		 * During runtime streamed to the device.
		 */
		PRE_HOST,
		/**
		 * MC mesh is computed on-the-fly for every isosurface.
		 * Slowest, but least memory.
		 */
		ON_THE_FLY
	};
	/**
	 * \brief specifies how the marching cubes/tets are computed.
	 * Takes effect after the next draw call.
	 */
	void setMarchingCubesComputationMode(MarchingCubesComputationMode mode);
	MarchingCubesComputationMode getMarchingCubesComputationMode() const;

	/**
	 * \brief Sets the maximal depth complexity per pixel.
	 * This is needed because the blending shader allocates space
	 * for this number of fragments for sorting and blending.
	 */
	void setMaxFragmentsPerPixel(int count);
	int getMaxFragmentsPerPixel() const;

	/**
	 * \brief Specifies the size for tiled rendering.
	 * 
	 * If the fragment storage runs out of memory,
	 * instead of increasing it, tiled rendering allows
	 * to render a smaller screen area at the same time,
	 * thus reducing the number of fragments in the buffer.
	 * But it requires multiple draw calls of the isosurface.
	 * 
	 * Pass -1 to disable and draw the whole screen at once.
	 */
	void setTileSize(int tileSize);
	int getTileSize() const;

private:
	void initialize();
	void loadShaders();
	void initMarchingCubes();

	void fillMarchingCubesMeshes(const Volume::MipmapLevel* data, cudaStream_t stream);
	
private:
	bool initialized_ = false;
#if TRIANGULATION_ALGORITHM == MARCHING_CUBES
	static const int3 offsets[8];
	static const int edgeTable[256];
	static const int triTable[256][16];
#elif TRIANGULATION_ALGORITHM == MARCHING_TETS
	static const int3 offsets[8];
	static const int tetrahedra[6][4];
	static const int edgeTable[16][10]; //16 cases, 4 vertices (index pairs)
	static const int triTable[16][7]; //16 cases, 2 tris
#endif

	//settings
	kernel::RendererDeviceSettings previousSettings_ = { 0 };
	const Volume::MipmapLevel* previousData_ = nullptr;
	std::string previousKernelName_;
	bool hadOverflow_;

	//output
	std::unique_ptr<Framebuffer> framebuffer_;

	//OIT settings
	std::unique_ptr<OIT> oit_;
	MarchingCubesComputationMode mcMode_;
	int maxFragmentsPerPixel_;
	int tileSize_;

	//single opaque isosurface
	std::unique_ptr<Shader> isosurfaceShader_;
	std::unique_ptr<Mesh> isosurfaceMesh_;

	//multiple transparent isosurfaces
	std::unique_ptr<Shader> multiIsoRenderShader_;
	std::unique_ptr<Shader> multiIsoBlendingShader_;
	struct isoMesh_t {
		float iso;
		std::shared_ptr<Mesh> mesh;
		MeshCpu meshCpu;
		glm::vec4 color;
		bool changed = false;
	};
	std::vector<isoMesh_t> multiIsoMeshes_; //reused for DVR
	std::unique_ptr<Mesh> dynamicMesh_;

	//DVR
	std::unique_ptr<Shader> dvrBlendingShader_;
};

END_RENDERER_NAMESPACE
