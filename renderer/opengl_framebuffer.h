#pragma once

#include "commons.h"

#include <GL/glew.h>
#include <cuda_gl_interop.h>
#include <glm/glm.hpp>

#include "kernel_launcher.h"
#include "settings.h"
#include "volume.h"
#include "renderer_settings.cuh"

BEGIN_RENDERER_NAMESPACE

class Framebuffer
{
	int width_, height_;
	GLuint fbo_;
	GLuint colorTexture_;
	GLuint depthRbo_;
	GLuint prevBinding_;
	cudaGraphicsResource_t colorTextureCuda_;
	
	Framebuffer(Framebuffer const&) = delete;
	Framebuffer& operator=(Framebuffer const&) = delete;

public:
	Framebuffer(int width, int height);
	~Framebuffer();

	int width() const { return width_; }
	int height() const { return height_; }

	void bind();
	void unbind();

	void copyToCudaIso(kernel::OutputTensor& output, cudaStream_t stream);
	void copyToCudaDvr(kernel::OutputTensor& output, cudaStream_t stream);
	
	void readRGBA(std::vector<float>& data);
};

END_RENDERER_NAMESPACE
