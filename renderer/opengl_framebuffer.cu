#include "opengl_framebuffer.h"
#include "opengl_utils.h"

#include <cuMat/src/Matrix.h>
#include <cuMat/src/Errors.h>

#include "opengl_mesh.h"

texture<float4, cudaTextureType2D, cudaReadModeElementType> framebufferTexRef;

BEGIN_RENDERER_NAMESPACE


Framebuffer::Framebuffer(int width, int height)
	: width_(width), height_(height), fbo_(0), colorTexture_(0), depthRbo_(0), prevBinding_(0)
{
	GLint oldRbo, oldFbo;
	glGetIntegerv(GL_RENDERBUFFER_BINDING, &oldRbo);
	glGetIntegerv(GL_FRAMEBUFFER_BINDING, &oldFbo);
	checkOpenGLError();

	glGenFramebuffers(1, &fbo_);
	glBindFramebuffer(GL_DRAW_FRAMEBUFFER, fbo_);
	checkOpenGLError();

	glGenTextures(1, &colorTexture_);
	glBindTexture(GL_TEXTURE_2D, colorTexture_);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, nullptr);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE); // clamp s coordinate
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE); // clamp t coordinate
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, colorTexture_, 0);
	checkOpenGLError();

	glGenRenderbuffers(1, &depthRbo_);
	glBindRenderbuffer(GL_RENDERBUFFER, depthRbo_);
	glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, width, height);
	glFramebufferRenderbuffer(GL_DRAW_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, depthRbo_);
	checkOpenGLError();

	CUMAT_SAFE_CALL(cudaGraphicsGLRegisterImage(&colorTextureCuda_, colorTexture_, GL_TEXTURE_2D,
		cudaGraphicsRegisterFlagsReadOnly));

	glBindRenderbuffer(GL_RENDERBUFFER, oldRbo);
	glBindFramebuffer(GL_DRAW_FRAMEBUFFER, oldFbo);
	checkOpenGLError();
}

Framebuffer::~Framebuffer()
{
	CUMAT_SAFE_CALL(cudaGraphicsUnregisterResource(colorTextureCuda_));
	glDeleteFramebuffers(1, &fbo_);
	glDeleteRenderbuffers(1, &depthRbo_);
	glDeleteRenderbuffers(1, &colorTexture_);
	checkOpenGLError();
}

void Framebuffer::bind()
{
	glGetIntegerv(GL_FRAMEBUFFER_BINDING, reinterpret_cast<int*>(&prevBinding_));
	glBindFramebuffer(GL_DRAW_FRAMEBUFFER, fbo_);
	checkOpenGLError();
}

void Framebuffer::unbind()
{
	glBindFramebuffer(GL_DRAW_FRAMEBUFFER, prevBinding_);
	checkOpenGLError();
}

void Framebuffer::readRGBA(std::vector<float>& data)
{
	glBindFramebuffer(GL_READ_FRAMEBUFFER, fbo_);
	data.resize(width_ * height_ * 4);
	glReadBuffer(GL_COLOR_ATTACHMENT0);
	glReadPixels(0, 0, width_, height_, GL_RGBA, GL_FLOAT, data.data());
	checkOpenGLError();
	glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
}

namespace
{
	__global__ void FramebufferCopyToCudaIso(dim3 virtual_size,
		kernel::OutputTensor output)
	{
		CUMAT_KERNEL_2D_LOOP(i, j, virtual_size)
		{
			float4 rgba = tex2D(framebufferTexRef, i, j);
			float3 normal = make_float3(rgba) * 2 - 1;
			float mask = rgba.w;
			output.coeff(j, i, 0) = mask;
			output.coeff(j, i, 1) = normal.x;
			output.coeff(j, i, 2) = normal.y;
			output.coeff(j, i, 3) = normal.z;
#pragma unroll
			output.coeff(j, i, 4) = 0.0f; //depth
			output.coeff(j, i, 5) = 1.0f; //ao
			output.coeff(j, i, 6) = 0.0f; //flow x
			output.coeff(j, i, 7) = 0.0f; //flow y
		}
		CUMAT_KERNEL_2D_LOOP_END
	}

	__global__ void FramebufferCopyToCudaDvr(dim3 virtual_size,
		kernel::OutputTensor output)
	{
		CUMAT_KERNEL_2D_LOOP(i, j, virtual_size)
		{
			float4 rgba = tex2D(framebufferTexRef, i, j);
			output.coeff(j, i, 0) = rgba.x;
			output.coeff(j, i, 1) = rgba.y;
			output.coeff(j, i, 2) = rgba.z;
			output.coeff(j, i, 3) = rgba.w;
#pragma unroll
			for (int b = 4; b < 10; ++b)
				output.coeff(j, i, b) = 0.0f;
		}
		CUMAT_KERNEL_2D_LOOP_END
	}
}

void Framebuffer::copyToCudaIso(kernel::OutputTensor& output, cudaStream_t stream)
{
#if 1
	CUMAT_SAFE_CALL(cudaDeviceSynchronize());
	
	assert(output.cols == width_);
	assert(output.rows == height_);
	cuMat::Context& ctx = cuMat::Context::current();

	CUMAT_SAFE_CALL(cudaGraphicsMapResources(1, &colorTextureCuda_, stream));
	cudaArray_t array;
	CUMAT_SAFE_CALL(cudaGraphicsSubResourceGetMappedArray(&array, colorTextureCuda_, 0, 0));
	CUMAT_SAFE_CALL(cudaBindTextureToArray(framebufferTexRef, array));
	
	//copy kernel
	const auto cfg = ctx.createLaunchConfig2D(width_, height_, FramebufferCopyToCudaIso);
	FramebufferCopyToCudaIso
		<<< cfg.block_count, cfg.thread_per_block, 0, stream >>>
		(cfg.virtual_size, output);
	//CUMAT_CHECK_ERROR();
	CUMAT_SAFE_CALL(cudaDeviceSynchronize());

	//CUMAT_SAFE_CALL(cudaDestroyTextureObject(tex));
	CUMAT_SAFE_CALL(cudaUnbindTexture(framebufferTexRef));
	CUMAT_SAFE_CALL(cudaGraphicsUnmapResources(1, &colorTextureCuda_, stream));
#else
	//CPU-Path for debugging
	std::vector<float> rgba;
	readRGBA(rgba);
	const float4* rgba_v = reinterpret_cast<const float4*>(rgba.data());
	std::vector<float> hostMemory_v(output.rows * output.cols * output.batches);
	float* hostMemory = hostMemory_v.data();
	for (int x=0; x<output.cols; ++x) for (int y=0; y<output.rows; ++y)
	{
		float4 in = rgba_v[x + output.cols * y];
		float3 normal = safeNormalize(make_float3(in) * 2 - 1);
		float mask = in.w;
		hostMemory[output.idx(y, x, 0)] = mask;
		hostMemory[output.idx(y, x, 1)] = normal.x;
		hostMemory[output.idx(y, x, 2)] = normal.y;
		hostMemory[output.idx(y, x, 3)] = normal.z;
	}
	CUMAT_SAFE_CALL(cudaMemcpy(output.memory, hostMemory,
		sizeof(float) * hostMemory_v.size(), cudaMemcpyHostToDevice));
#endif
}

void Framebuffer::copyToCudaDvr(kernel::OutputTensor& output, cudaStream_t stream)
{
	CUMAT_SAFE_CALL(cudaDeviceSynchronize());

	assert(output.cols == width_);
	assert(output.rows == height_);
	cuMat::Context& ctx = cuMat::Context::current();

	CUMAT_SAFE_CALL(cudaGraphicsMapResources(1, &colorTextureCuda_, stream));
	cudaArray_t array;
	CUMAT_SAFE_CALL(cudaGraphicsSubResourceGetMappedArray(&array, colorTextureCuda_, 0, 0));

	cudaChannelFormatDesc desc;
	desc.f = cudaChannelFormatKindFloat;
	desc.x = desc.y = desc.z = desc.w = 32;
	CUMAT_SAFE_CALL(cudaBindTextureToArray(framebufferTexRef, array));

	//copy kernel
	const auto cfg = ctx.createLaunchConfig2D(width_, height_, FramebufferCopyToCudaDvr);
	FramebufferCopyToCudaDvr
		<<< cfg.block_count, cfg.thread_per_block, 0, stream >>>
		(cfg.virtual_size, output);
	//CUMAT_CHECK_ERROR();
	CUMAT_SAFE_CALL(cudaDeviceSynchronize());

	//CUMAT_SAFE_CALL(cudaDestroyTextureObject(tex));
	CUMAT_SAFE_CALL(cudaUnbindTexture(framebufferTexRef));
	CUMAT_SAFE_CALL(cudaGraphicsUnmapResources(1, &colorTextureCuda_, stream));
}

END_RENDERER_NAMESPACE
