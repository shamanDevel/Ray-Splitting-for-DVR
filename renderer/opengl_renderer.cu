#include "opengl_renderer.h"
#include "opengl_utils.h"

#include <cuMat/src/Matrix.h>


#if TRIANGULATION_ALGORITHM == MARCHING_CUBES
__constant__ int3 offsetsDevice[8];
__constant__ int edgeTableDevice[256];
__constant__ int triTableDevice[256][16];
#elif TRIANGULATION_ALGORITHM == MARCHING_TETS
//Source: https://github.com/Calvin-L/MarchingTetrahedrons/blob/master/Decimate.cpp
__constant__ int3 offsetsDevice[8];
__constant__ int tetrahedraDevice[6][4];
__constant__ int edgeTableDevice[16][10]; //16 cases, 4 vertices
__constant__ int triTableDevice[16][7]; //16 cases, 2 tris
#endif


BEGIN_RENDERER_NAMESPACE

namespace
{
#if TRIANGULATION_ALGORITHM == MARCHING_CUBES
	__device__ __inline__ unsigned int getCubeIndex(float val[8], float isovalue)
	{
		unsigned int cubeindex = 0;
		if (val[0] < isovalue) cubeindex |= 1;
		if (val[1] < isovalue) cubeindex |= 2;
		if (val[2] < isovalue) cubeindex |= 4;
		if (val[3] < isovalue) cubeindex |= 8;
		if (val[4] < isovalue) cubeindex |= 16;
		if (val[5] < isovalue) cubeindex |= 32;
		if (val[6] < isovalue) cubeindex |= 64;
		if (val[7] < isovalue) cubeindex |= 128;
		return cubeindex;
	}

	__device__ __inline__ void computeCounts(
		unsigned int cubeindex, int& outNumVertices, int& outNumIndices)
	{
		//vertices
		outNumVertices = __popc(edgeTableDevice[cubeindex]); //CUDA: __popc, MSVC: __popcnt
		
		//triangles
		int ntriang = 0;
		for (int i = 0; triTableDevice[cubeindex][i] != -1; i += 3)
			ntriang++;
		outNumIndices = ntriang * 3;
	}

	struct ComputeVertexCount
	{
		static __device__ __inline__ void call(
			int x, int y, int z, float val[8], float iso,
			cudaTextureObject_t volumeTexLinear,
			unsigned int* countVertices, unsigned int* countIndices,
			Vertex* vertices, unsigned int* indices)
		{
			const unsigned int cubeindex = getCubeIndex(val, iso);
			
			int numVertices, numIndices;
			computeCounts(cubeindex, numVertices, numIndices);

			//printf("[%02d, %02d, %02d] (%.4f) cube=%02x, #vert=%2d, #ind=%2d\n",
			//	x, y, z, val[0], cubeindex, numVertices, numIndices);
			
			if (numVertices>0)
			{
				atomicAdd(countVertices, numVertices);
				atomicAdd(countIndices, numIndices);
			}
		}
	};

	struct FillBuffers
	{
		static __device__ __inline__ float vertexInterpolate(
			float isovalue, float val1, float val2)
		{
			if (fabs(isovalue - val1) < 0.00001)
				return 0;
			if (fabs(isovalue - val2) < 0.00001)
				return 1;
			if (fabs(val1 - val2) < 0.00001)
				return 0;
			float w = (isovalue - val1) / (val2 - val1);
			assert(w > 0 && w < 1);
			return w;
		}

		static __device__ __inline__ float3 normalInterp(
			const float vals[8], const float3& p)
		{
			float3 normal;
			normal.x =
				lerp(lerp(vals[1], vals[3], p.y), lerp(vals[5], vals[7], p.y), p.z) -
				lerp(lerp(vals[0], vals[2], p.y), lerp(vals[4], vals[6], p.y), p.z);
			normal.y =
				lerp(lerp(vals[2], vals[3], p.x), lerp(vals[6], vals[7], p.x), p.z) -
				lerp(lerp(vals[0], vals[1], p.x), lerp(vals[4], vals[5], p.x), p.z);
			normal.z =
				lerp(lerp(vals[4], vals[5], p.x), lerp(vals[6], vals[7], p.x), p.y) -
				lerp(lerp(vals[0], vals[1], p.x), lerp(vals[2], vals[3], p.x), p.y);
			normal = -normal;
			return normal;
		}
		
		static __device__ __inline__ unsigned int createVertex(
			int x, int y, int z, float val[8], float iso,
			cudaTextureObject_t volumeTexLinear,
			int idx1, int idx2,
			Vertex* vertices, unsigned int& vertexOffset)
		{
			float weight = vertexInterpolate(iso, val[idx1], val[idx2]);
			float3 posBase = make_float3(x, y, z);
			float3 posVertex = (1 - weight) * make_float3(offsetsDevice[idx1])
				+ weight * make_float3(offsetsDevice[idx2]);
			vertices[vertexOffset].position = make_float4(posBase + posVertex, 1.0f);

#if 0
			float3 normalVertex = safeNormalize(normalInterp(val, posVertex));
			vertices[vertexOffset].normals = make_float4(normalVertex, 0);
#else
			float3 volPos = posBase + posVertex + make_float3(0.5);
			float3 normal;
			const float normalStepSize = 0.5f;
			normal.x = 0.5 * (tex3D<float>(volumeTexLinear, volPos.x + normalStepSize, volPos.y, volPos.z)
				- tex3D<float>(volumeTexLinear, volPos.x - normalStepSize, volPos.y, volPos.z));
			normal.y = 0.5 * (tex3D<float>(volumeTexLinear, volPos.x, volPos.y + normalStepSize, volPos.z)
				- tex3D<float>(volumeTexLinear, volPos.x, volPos.y - normalStepSize, volPos.z));
			normal.z = 0.5 * (tex3D<float>(volumeTexLinear, volPos.x, volPos.y, volPos.z + normalStepSize)
				- tex3D<float>(volumeTexLinear, volPos.x, volPos.y, volPos.z - normalStepSize));
			//normal = -normal;
			vertices[vertexOffset].normals = make_float4(safeNormalize(normal), 0.0f);
#endif
			
			//printf("Vertex created at %.2f, %.2f, %.2f\n",
			//	vertices[vertexOffset].position.x, vertices[vertexOffset].position.y, vertices[vertexOffset].position.z);
			unsigned int old = vertexOffset;
			++vertexOffset;
			return old;
		}
		
		static __device__ __inline__ void call(
			int x, int y, int z, float val[8], float iso,
			cudaTextureObject_t volumeTexLinear,
			unsigned int* countVertices, unsigned int* countIndices,
			Vertex* vertices, unsigned int* indices)
		{
			//compute counts again
			const unsigned int cubeindex = getCubeIndex(val, iso);

			int numVertices, numIndices;
			computeCounts(cubeindex, numVertices, numIndices);
			if (numVertices > 0)
			{
				unsigned int vertexOffset = atomicAdd(countVertices, numVertices);
				unsigned int indexOffset = atomicAdd(countIndices, numIndices);

				//create vertices
				unsigned int vertlist[12];
				if (edgeTableDevice[cubeindex] & 1)
					vertlist[0] = createVertex(x, y, z, val, iso, volumeTexLinear, 0, 1, vertices, vertexOffset);
				if (edgeTableDevice[cubeindex] & 2)
					vertlist[1] = createVertex(x, y, z, val, iso, volumeTexLinear, 1, 2, vertices, vertexOffset);
				if (edgeTableDevice[cubeindex] & 4)
					vertlist[2] = createVertex(x, y, z, val, iso, volumeTexLinear, 2, 3, vertices, vertexOffset);
				if (edgeTableDevice[cubeindex] & 8)
					vertlist[3] = createVertex(x, y, z, val, iso, volumeTexLinear, 3, 0, vertices, vertexOffset);
				if (edgeTableDevice[cubeindex] & 16)
					vertlist[4] = createVertex(x, y, z, val, iso, volumeTexLinear, 4, 5, vertices, vertexOffset);
				if (edgeTableDevice[cubeindex] & 32)
					vertlist[5] = createVertex(x, y, z, val, iso, volumeTexLinear, 5, 6, vertices, vertexOffset);
				if (edgeTableDevice[cubeindex] & 64)
					vertlist[6] = createVertex(x, y, z, val, iso, volumeTexLinear, 6, 7, vertices, vertexOffset);
				if (edgeTableDevice[cubeindex] & 128)
					vertlist[7] = createVertex(x, y, z, val, iso, volumeTexLinear, 7, 4, vertices, vertexOffset);
				if (edgeTableDevice[cubeindex] & 256)
					vertlist[8] = createVertex(x, y, z, val, iso, volumeTexLinear, 0, 4, vertices, vertexOffset);
				if (edgeTableDevice[cubeindex] & 512)
					vertlist[9] = createVertex(x, y, z, val, iso, volumeTexLinear, 1, 5, vertices, vertexOffset);
				if (edgeTableDevice[cubeindex] & 1024)
					vertlist[10] = createVertex(x, y, z, val, iso, volumeTexLinear, 2, 6, vertices, vertexOffset);
				if (edgeTableDevice[cubeindex] & 2048)
					vertlist[11] = createVertex(x, y, z, val, iso, volumeTexLinear, 3, 7, vertices, vertexOffset);
				
				//create indices
				for (int i = 0; triTableDevice[cubeindex][i] != -1; i += 3)
				{
					indices[indexOffset++] = vertlist[triTableDevice[cubeindex][i]];
					indices[indexOffset++] = vertlist[triTableDevice[cubeindex][i + 1]];
					indices[indexOffset++] = vertlist[triTableDevice[cubeindex][i + 2]];
				}
			}
		}
	};

	template<typename Functor>
	__global__ void ProcessGrid(dim3 virtual_size, 
		cudaTextureObject_t volumeTexNearest, cudaTextureObject_t volumeTexLinear,
		float isovalue,
		unsigned int* countVertices, unsigned int* countIndices,
		Vertex* vertices, unsigned int* indices)
	{
		CUMAT_KERNEL_3D_LOOP(i, j, k, virtual_size)
		{
			//fetch voxels
			float vals[8];
#pragma unroll
			for (int a = 0; a < 8; ++a)
				vals[a] = tex3D<float>(volumeTexNearest,
					i + offsetsDevice[a].x, j + offsetsDevice[a].y, k + offsetsDevice[a].z);

			//call functor
			Functor::call(i, j, k, vals, isovalue, volumeTexLinear, countVertices, countIndices, vertices, indices);
		}
		CUMAT_KERNEL_3D_LOOP_END
	}
#elif TRIANGULATION_ALGORITHM == MARCHING_TETS

	__device__ __inline__ unsigned int getTetIndex(float val[8], int t, float isovalue)
	{
		unsigned int tetindex = 0;
		if (val[tetrahedraDevice[t][0]] < isovalue) tetindex |= 1;
		if (val[tetrahedraDevice[t][1]] < isovalue) tetindex |= 2;
		if (val[tetrahedraDevice[t][2]] < isovalue) tetindex |= 4;
		if (val[tetrahedraDevice[t][3]] < isovalue) tetindex |= 8;
		return tetindex;
	}

	struct ComputeVertexCount
	{
		static __device__ __inline__ void call(
			int x, int y, int z, float val[8], float iso,
			cudaTextureObject_t volumeTexLinear,
			unsigned int* countVertices, unsigned int* countIndices,
			Vertex* vertices, unsigned int* indices)
		{
			int numVertices = 0, numTris = 0;
			for (int t=0; t<6; ++t)
			{
				unsigned int tetindex = getTetIndex(val, t, iso);
				for (int i = 0; edgeTableDevice[tetindex][i] != -1; i += 2)
					numVertices++;
				for (int i = 0; triTableDevice[tetindex][i] != -1; i += 3)
					numTris++;
			}
			if (numVertices>0)
			{
				atomicAdd(countVertices, numVertices);
				atomicAdd(countIndices, numTris*3);
			}
		}
	};

	struct FillBuffers
	{
		static __device__ __inline__ float vertexInterpolate(
			float isovalue, float val1, float val2)
		{
			if (fabs(isovalue - val1) < 0.00001)
				return 0;
			if (fabs(isovalue - val2) < 0.00001)
				return 1;
			if (fabs(val1 - val2) < 0.00001)
				return 0.5;
			float w = (isovalue - val1) / (val2 - val1);
			assert(w > 0 && w < 1);
			return w;
		}
		
		static __device__ __inline__ unsigned int createVertex(
			int x, int y, int z, float val[8], float iso,
			cudaTextureObject_t volumeTexLinear,
			int idx1, int idx2,
			Vertex* vertices, unsigned int& vertexOffset)
		{
			float weight = vertexInterpolate(iso, val[idx1], val[idx2]);
			float3 posBase = make_float3(x, y, z);
			float3 posVertex = (1 - weight) * make_float3(offsetsDevice[idx1])
				+ weight * make_float3(offsetsDevice[idx2]);
			vertices[vertexOffset].position = make_float4(posBase + posVertex, 1.0f);

			float3 volPos = posBase + posVertex + make_float3(0.5);
			float3 normal;
			const float normalStepSize = 0.5f;
			normal.x = 0.5 * (tex3D<float>(volumeTexLinear, volPos.x + normalStepSize, volPos.y, volPos.z)
				- tex3D<float>(volumeTexLinear, volPos.x - normalStepSize, volPos.y, volPos.z));
			normal.y = 0.5 * (tex3D<float>(volumeTexLinear, volPos.x, volPos.y + normalStepSize, volPos.z)
				- tex3D<float>(volumeTexLinear, volPos.x, volPos.y - normalStepSize, volPos.z));
			normal.z = 0.5 * (tex3D<float>(volumeTexLinear, volPos.x, volPos.y, volPos.z + normalStepSize)
				- tex3D<float>(volumeTexLinear, volPos.x, volPos.y, volPos.z - normalStepSize));
			//normal = -normal;
			vertices[vertexOffset].normals = make_float4(safeNormalize(normal), 0.0f);
			
			//printf("Vertex created at %.2f, %.2f, %.2f\n",
			//	vertices[vertexOffset].position.x, vertices[vertexOffset].position.y, vertices[vertexOffset].position.z);
			unsigned int old = vertexOffset;
			++vertexOffset;
			return old;
		}
		
		static __device__ __inline__ void call(
			int x, int y, int z, float val[8], float iso,
			cudaTextureObject_t volumeTexLinear,
			unsigned int* countVertices, unsigned int* countIndices,
			Vertex* vertices, unsigned int* indices)
		{
			//compute counts again
			int numVertices = 0, numTris = 0;
			for (int t = 0; t < 6; ++t)
			{
				unsigned int tetindex = getTetIndex(val, t, iso);
				for (int i = 0; edgeTableDevice[tetindex][i] != -1; i += 2)
					numVertices++;
				for (int i = 0; triTableDevice[tetindex][i] != -1; i += 3)
					numTris++;
			}

			if (numVertices > 0)
			{
				unsigned int vertexOffset = atomicAdd(countVertices, numVertices);
				unsigned int indexOffset = atomicAdd(countIndices, numTris*3);

				//create individual tets
				for (int t=0; t<6; ++t)
				{
					unsigned int tetindex = getTetIndex(val, t, iso);
					//create vertices
					unsigned int vertlist[4];
					for (int i = 0; edgeTableDevice[tetindex][2*i] != -1; ++i)
					{
						int idx1 = tetrahedraDevice[t][edgeTableDevice[tetindex][2*i]];
						int idx2 = tetrahedraDevice[t][edgeTableDevice[tetindex][2*i+1]];
						vertlist[i] = createVertex(x, y, z, val, iso, volumeTexLinear, idx1, idx2, vertices, vertexOffset);
					}
					//create indices
					for (int i = 0; triTableDevice[tetindex][i] != -1; i += 3)
					{
						indices[indexOffset++] = vertlist[triTableDevice[tetindex][i]];
						indices[indexOffset++] = vertlist[triTableDevice[tetindex][i+1]];
						indices[indexOffset++] = vertlist[triTableDevice[tetindex][i+2]];
					}
				}
			}
		}
	};

	template<typename Functor>
	__global__ void ProcessGrid(dim3 virtual_size, 
		cudaTextureObject_t volumeTexNearest, cudaTextureObject_t volumeTexLinear,
		float isovalue,
		unsigned int* countVertices, unsigned int* countIndices,
		Vertex* vertices, unsigned int* indices)
	{
		CUMAT_KERNEL_3D_LOOP(i, j, k, virtual_size)
		{
			//fetch voxels
			float vals[8];
#pragma unroll
			for (int a = 0; a < 8; ++a)
				vals[a] = tex3D<float>(volumeTexNearest,
					i + offsetsDevice[a].x, j + offsetsDevice[a].y, k + offsetsDevice[a].z);

			//call functor
			Functor::call(i, j, k, vals, isovalue, volumeTexLinear, countVertices, countIndices, vertices, indices);
		}
		CUMAT_KERNEL_3D_LOOP_END
	}
	
#endif
}

void OpenGLRasterization::initMarchingCubes()
{
	CUMAT_SAFE_CALL(cudaDeviceSynchronize());

#if TRIANGULATION_ALGORITHM == MARCHING_CUBES
	CUMAT_SAFE_CALL(cudaMemcpyToSymbol(offsetsDevice, OpenGLRasterization::offsets,
		sizeof(int3) * 8, 0, cudaMemcpyHostToDevice));
	CUMAT_SAFE_CALL(cudaMemcpyToSymbol(edgeTableDevice, OpenGLRasterization::edgeTable,
		sizeof(int) * 256, 0, cudaMemcpyHostToDevice));
	CUMAT_SAFE_CALL(cudaMemcpyToSymbol(triTableDevice, OpenGLRasterization::triTable,
		sizeof(int) * 256 * 16, 0, cudaMemcpyHostToDevice));
#elif TRIANGULATION_ALGORITHM == MARCHING_TETS
	CUMAT_SAFE_CALL(cudaMemcpyToSymbol(offsetsDevice, OpenGLRasterization::offsets,
		sizeof(int3) * 8, 0, cudaMemcpyHostToDevice));
	CUMAT_SAFE_CALL(cudaMemcpyToSymbol(tetrahedraDevice, OpenGLRasterization::tetrahedra,
		sizeof(int) * 6 * 4, 0, cudaMemcpyHostToDevice));
	CUMAT_SAFE_CALL(cudaMemcpyToSymbol(edgeTableDevice, OpenGLRasterization::edgeTable,
		sizeof(int) * 16 * 10, 0, cudaMemcpyHostToDevice));
	CUMAT_SAFE_CALL(cudaMemcpyToSymbol(triTableDevice, OpenGLRasterization::triTable,
		sizeof(int) * 16 * 7, 0, cudaMemcpyHostToDevice));
#endif

	std::cout << "Marching Cubes initialized" << std::endl;
}

void OpenGLRasterization::fillMarchingCubesMeshPreDevice(const Volume::MipmapLevel* data,
	float isosurface, Mesh* output, cudaStream_t stream)
{
	initialize();
	
	//1. collect count
	cuMat::Context& ctx = cuMat::Context::current();
	typedef cuMat::Matrix<unsigned int, 1, 1, 1, cuMat::ColumnMajor> Scalarui;
	Scalarui countVertices, countIndices;
	countVertices.setZero();
	countIndices.setZero();

	auto cfg = ctx.createLaunchConfig3D(
		data->sizeX() - 1, data->sizeY() - 1, data->sizeZ() - 1,
		ProcessGrid<ComputeVertexCount>);
	ProcessGrid<ComputeVertexCount>
		<<< cfg.block_count, cfg.thread_per_block, 0, stream >>>
		(cfg.virtual_size, data->dataTexNearestGpu(), data->dataTexLinearGpu(), isosurface,
			countVertices.data(), countIndices.data(), 
			nullptr, nullptr);
	CUMAT_CHECK_ERROR();

	//2. resize mesh
	const int countVerticesHost = static_cast<unsigned int>(countVertices);
	const int countIndicesHost = static_cast<unsigned int>(countIndices);
	//std::cout << "MC mesh: #vertices=" << countVerticesHost << ", #indices=" << countIndicesHost << std::endl;
	output->reserve(countVerticesHost, countIndicesHost);

	//3. fill buffers
	if (countVerticesHost > 0) {
		Vertex* vertices;
		GLuint* indices;
		countVertices.setZero();
		countIndices.setZero();
		output->cudaMap(&vertices, &indices);
		cfg = ctx.createLaunchConfig3D(
			data->sizeX() - 1, data->sizeY() - 1, data->sizeZ() - 1,
			ProcessGrid<FillBuffers>);
		ProcessGrid<FillBuffers>
			<<< cfg.block_count, cfg.thread_per_block, 0, stream >>>
			(cfg.virtual_size, data->dataTexNearestGpu(), data->dataTexLinearGpu(), isosurface,
				countVertices.data(), countIndices.data(),
				vertices, indices);
		CUMAT_CHECK_ERROR();
		output->cudaUnmap();
	}
}

void OpenGLRasterization::fillMarchingCubesMeshPreHost(const Volume::MipmapLevel* data,
	float isosurface, MeshCpu* output, Mesh* tmp, cudaStream_t stream)
{
	fillMarchingCubesMeshPreDevice(data, isosurface, tmp, stream);
	tmp->copyToCpu(*output);
}

#if TRIANGULATION_ALGORITHM == MARCHING_CUBES

const int3 OpenGLRasterization::offsets[8] = {
	make_int3(0, 0, 0),
	make_int3(1, 0, 0),
	make_int3(1, 1, 0),
	make_int3(0, 1, 0),
	make_int3(0, 0, 1),
	make_int3(1, 0, 1),
	make_int3(1, 1, 1),
	make_int3(0, 1, 1)
};

const int OpenGLRasterization::edgeTable[256] = {
0x0  , 0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c,
0x80c, 0x905, 0xa0f, 0xb06, 0xc0a, 0xd03, 0xe09, 0xf00,
0x190, 0x99 , 0x393, 0x29a, 0x596, 0x49f, 0x795, 0x69c,
0x99c, 0x895, 0xb9f, 0xa96, 0xd9a, 0xc93, 0xf99, 0xe90,
0x230, 0x339, 0x33 , 0x13a, 0x636, 0x73f, 0x435, 0x53c,
0xa3c, 0xb35, 0x83f, 0x936, 0xe3a, 0xf33, 0xc39, 0xd30,
0x3a0, 0x2a9, 0x1a3, 0xaa , 0x7a6, 0x6af, 0x5a5, 0x4ac,
0xbac, 0xaa5, 0x9af, 0x8a6, 0xfaa, 0xea3, 0xda9, 0xca0,
0x460, 0x569, 0x663, 0x76a, 0x66 , 0x16f, 0x265, 0x36c,
0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a, 0x963, 0xa69, 0xb60,
0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0xff , 0x3f5, 0x2fc,
0xdfc, 0xcf5, 0xfff, 0xef6, 0x9fa, 0x8f3, 0xbf9, 0xaf0,
0x650, 0x759, 0x453, 0x55a, 0x256, 0x35f, 0x55 , 0x15c,
0xe5c, 0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53, 0x859, 0x950,
0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6, 0x2cf, 0x1c5, 0xcc ,
0xfcc, 0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0,
0x8c0, 0x9c9, 0xac3, 0xbca, 0xcc6, 0xdcf, 0xec5, 0xfcc,
0xcc , 0x1c5, 0x2cf, 0x3c6, 0x4ca, 0x5c3, 0x6c9, 0x7c0,
0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f, 0xf55, 0xe5c,
0x15c, 0x55 , 0x35f, 0x256, 0x55a, 0x453, 0x759, 0x650,
0xaf0, 0xbf9, 0x8f3, 0x9fa, 0xef6, 0xfff, 0xcf5, 0xdfc,
0x2fc, 0x3f5, 0xff , 0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0,
0xb60, 0xa69, 0x963, 0x86a, 0xf66, 0xe6f, 0xd65, 0xc6c,
0x36c, 0x265, 0x16f, 0x66 , 0x76a, 0x663, 0x569, 0x460,
0xca0, 0xda9, 0xea3, 0xfaa, 0x8a6, 0x9af, 0xaa5, 0xbac,
0x4ac, 0x5a5, 0x6af, 0x7a6, 0xaa , 0x1a3, 0x2a9, 0x3a0,
0xd30, 0xc39, 0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c,
0x53c, 0x435, 0x73f, 0x636, 0x13a, 0x33 , 0x339, 0x230,
0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f, 0x895, 0x99c,
0x69c, 0x795, 0x49f, 0x596, 0x29a, 0x393, 0x99 , 0x190,
0xf00, 0xe09, 0xd03, 0xc0a, 0xb06, 0xa0f, 0x905, 0x80c,
0x70c, 0x605, 0x50f, 0x406, 0x30a, 0x203, 0x109, 0x0 };

const int OpenGLRasterization::triTable[256][16] =
{ {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{0, 1, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{1, 8, 3, 9, 8, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{0, 8, 3, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{9, 2, 10, 0, 2, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{2, 8, 3, 2, 10, 8, 10, 9, 8, -1, -1, -1, -1, -1, -1, -1},
{3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{0, 11, 2, 8, 11, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{1, 9, 0, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{1, 11, 2, 1, 9, 11, 9, 8, 11, -1, -1, -1, -1, -1, -1, -1},
{3, 10, 1, 11, 10, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{0, 10, 1, 0, 8, 10, 8, 11, 10, -1, -1, -1, -1, -1, -1, -1},
{3, 9, 0, 3, 11, 9, 11, 10, 9, -1, -1, -1, -1, -1, -1, -1},
{9, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{4, 3, 0, 7, 3, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{0, 1, 9, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{4, 1, 9, 4, 7, 1, 7, 3, 1, -1, -1, -1, -1, -1, -1, -1},
{1, 2, 10, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{3, 4, 7, 3, 0, 4, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1},
{9, 2, 10, 9, 0, 2, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1},
{2, 10, 9, 2, 9, 7, 2, 7, 3, 7, 9, 4, -1, -1, -1, -1},
{8, 4, 7, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{11, 4, 7, 11, 2, 4, 2, 0, 4, -1, -1, -1, -1, -1, -1, -1},
{9, 0, 1, 8, 4, 7, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1},
{4, 7, 11, 9, 4, 11, 9, 11, 2, 9, 2, 1, -1, -1, -1, -1},
{3, 10, 1, 3, 11, 10, 7, 8, 4, -1, -1, -1, -1, -1, -1, -1},
{1, 11, 10, 1, 4, 11, 1, 0, 4, 7, 11, 4, -1, -1, -1, -1},
{4, 7, 8, 9, 0, 11, 9, 11, 10, 11, 0, 3, -1, -1, -1, -1},
{4, 7, 11, 4, 11, 9, 9, 11, 10, -1, -1, -1, -1, -1, -1, -1},
{9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{9, 5, 4, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{0, 5, 4, 1, 5, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{8, 5, 4, 8, 3, 5, 3, 1, 5, -1, -1, -1, -1, -1, -1, -1},
{1, 2, 10, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{3, 0, 8, 1, 2, 10, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1},
{5, 2, 10, 5, 4, 2, 4, 0, 2, -1, -1, -1, -1, -1, -1, -1},
{2, 10, 5, 3, 2, 5, 3, 5, 4, 3, 4, 8, -1, -1, -1, -1},
{9, 5, 4, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{0, 11, 2, 0, 8, 11, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1},
{0, 5, 4, 0, 1, 5, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1},
{2, 1, 5, 2, 5, 8, 2, 8, 11, 4, 8, 5, -1, -1, -1, -1},
{10, 3, 11, 10, 1, 3, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1},
{4, 9, 5, 0, 8, 1, 8, 10, 1, 8, 11, 10, -1, -1, -1, -1},
{5, 4, 0, 5, 0, 11, 5, 11, 10, 11, 0, 3, -1, -1, -1, -1},
{5, 4, 8, 5, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1},
{9, 7, 8, 5, 7, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{9, 3, 0, 9, 5, 3, 5, 7, 3, -1, -1, -1, -1, -1, -1, -1},
{0, 7, 8, 0, 1, 7, 1, 5, 7, -1, -1, -1, -1, -1, -1, -1},
{1, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{9, 7, 8, 9, 5, 7, 10, 1, 2, -1, -1, -1, -1, -1, -1, -1},
{10, 1, 2, 9, 5, 0, 5, 3, 0, 5, 7, 3, -1, -1, -1, -1},
{8, 0, 2, 8, 2, 5, 8, 5, 7, 10, 5, 2, -1, -1, -1, -1},
{2, 10, 5, 2, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1},
{7, 9, 5, 7, 8, 9, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1},
{9, 5, 7, 9, 7, 2, 9, 2, 0, 2, 7, 11, -1, -1, -1, -1},
{2, 3, 11, 0, 1, 8, 1, 7, 8, 1, 5, 7, -1, -1, -1, -1},
{11, 2, 1, 11, 1, 7, 7, 1, 5, -1, -1, -1, -1, -1, -1, -1},
{9, 5, 8, 8, 5, 7, 10, 1, 3, 10, 3, 11, -1, -1, -1, -1},
{5, 7, 0, 5, 0, 9, 7, 11, 0, 1, 0, 10, 11, 10, 0, -1},
{11, 10, 0, 11, 0, 3, 10, 5, 0, 8, 0, 7, 5, 7, 0, -1},
{11, 10, 5, 7, 11, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{0, 8, 3, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{9, 0, 1, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{1, 8, 3, 1, 9, 8, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1},
{1, 6, 5, 2, 6, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{1, 6, 5, 1, 2, 6, 3, 0, 8, -1, -1, -1, -1, -1, -1, -1},
{9, 6, 5, 9, 0, 6, 0, 2, 6, -1, -1, -1, -1, -1, -1, -1},
{5, 9, 8, 5, 8, 2, 5, 2, 6, 3, 2, 8, -1, -1, -1, -1},
{2, 3, 11, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{11, 0, 8, 11, 2, 0, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1},
{0, 1, 9, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1},
{5, 10, 6, 1, 9, 2, 9, 11, 2, 9, 8, 11, -1, -1, -1, -1},
{6, 3, 11, 6, 5, 3, 5, 1, 3, -1, -1, -1, -1, -1, -1, -1},
{0, 8, 11, 0, 11, 5, 0, 5, 1, 5, 11, 6, -1, -1, -1, -1},
{3, 11, 6, 0, 3, 6, 0, 6, 5, 0, 5, 9, -1, -1, -1, -1},
{6, 5, 9, 6, 9, 11, 11, 9, 8, -1, -1, -1, -1, -1, -1, -1},
{5, 10, 6, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{4, 3, 0, 4, 7, 3, 6, 5, 10, -1, -1, -1, -1, -1, -1, -1},
{1, 9, 0, 5, 10, 6, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1},
{10, 6, 5, 1, 9, 7, 1, 7, 3, 7, 9, 4, -1, -1, -1, -1},
{6, 1, 2, 6, 5, 1, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1},
{1, 2, 5, 5, 2, 6, 3, 0, 4, 3, 4, 7, -1, -1, -1, -1},
{8, 4, 7, 9, 0, 5, 0, 6, 5, 0, 2, 6, -1, -1, -1, -1},
{7, 3, 9, 7, 9, 4, 3, 2, 9, 5, 9, 6, 2, 6, 9, -1},
{3, 11, 2, 7, 8, 4, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1},
{5, 10, 6, 4, 7, 2, 4, 2, 0, 2, 7, 11, -1, -1, -1, -1},
{0, 1, 9, 4, 7, 8, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1},
{9, 2, 1, 9, 11, 2, 9, 4, 11, 7, 11, 4, 5, 10, 6, -1},
{8, 4, 7, 3, 11, 5, 3, 5, 1, 5, 11, 6, -1, -1, -1, -1},
{5, 1, 11, 5, 11, 6, 1, 0, 11, 7, 11, 4, 0, 4, 11, -1},
{0, 5, 9, 0, 6, 5, 0, 3, 6, 11, 6, 3, 8, 4, 7, -1},
{6, 5, 9, 6, 9, 11, 4, 7, 9, 7, 11, 9, -1, -1, -1, -1},
{10, 4, 9, 6, 4, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{4, 10, 6, 4, 9, 10, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1},
{10, 0, 1, 10, 6, 0, 6, 4, 0, -1, -1, -1, -1, -1, -1, -1},
{8, 3, 1, 8, 1, 6, 8, 6, 4, 6, 1, 10, -1, -1, -1, -1},
{1, 4, 9, 1, 2, 4, 2, 6, 4, -1, -1, -1, -1, -1, -1, -1},
{3, 0, 8, 1, 2, 9, 2, 4, 9, 2, 6, 4, -1, -1, -1, -1},
{0, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{8, 3, 2, 8, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1},
{10, 4, 9, 10, 6, 4, 11, 2, 3, -1, -1, -1, -1, -1, -1, -1},
{0, 8, 2, 2, 8, 11, 4, 9, 10, 4, 10, 6, -1, -1, -1, -1},
{3, 11, 2, 0, 1, 6, 0, 6, 4, 6, 1, 10, -1, -1, -1, -1},
{6, 4, 1, 6, 1, 10, 4, 8, 1, 2, 1, 11, 8, 11, 1, -1},
{9, 6, 4, 9, 3, 6, 9, 1, 3, 11, 6, 3, -1, -1, -1, -1},
{8, 11, 1, 8, 1, 0, 11, 6, 1, 9, 1, 4, 6, 4, 1, -1},
{3, 11, 6, 3, 6, 0, 0, 6, 4, -1, -1, -1, -1, -1, -1, -1},
{6, 4, 8, 11, 6, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{7, 10, 6, 7, 8, 10, 8, 9, 10, -1, -1, -1, -1, -1, -1, -1},
{0, 7, 3, 0, 10, 7, 0, 9, 10, 6, 7, 10, -1, -1, -1, -1},
{10, 6, 7, 1, 10, 7, 1, 7, 8, 1, 8, 0, -1, -1, -1, -1},
{10, 6, 7, 10, 7, 1, 1, 7, 3, -1, -1, -1, -1, -1, -1, -1},
{1, 2, 6, 1, 6, 8, 1, 8, 9, 8, 6, 7, -1, -1, -1, -1},
{2, 6, 9, 2, 9, 1, 6, 7, 9, 0, 9, 3, 7, 3, 9, -1},
{7, 8, 0, 7, 0, 6, 6, 0, 2, -1, -1, -1, -1, -1, -1, -1},
{7, 3, 2, 6, 7, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{2, 3, 11, 10, 6, 8, 10, 8, 9, 8, 6, 7, -1, -1, -1, -1},
{2, 0, 7, 2, 7, 11, 0, 9, 7, 6, 7, 10, 9, 10, 7, -1},
{1, 8, 0, 1, 7, 8, 1, 10, 7, 6, 7, 10, 2, 3, 11, -1},
{11, 2, 1, 11, 1, 7, 10, 6, 1, 6, 7, 1, -1, -1, -1, -1},
{8, 9, 6, 8, 6, 7, 9, 1, 6, 11, 6, 3, 1, 3, 6, -1},
{0, 9, 1, 11, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{7, 8, 0, 7, 0, 6, 3, 11, 0, 11, 6, 0, -1, -1, -1, -1},
{7, 11, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{3, 0, 8, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{0, 1, 9, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{8, 1, 9, 8, 3, 1, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1},
{10, 1, 2, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{1, 2, 10, 3, 0, 8, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1},
{2, 9, 0, 2, 10, 9, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1},
{6, 11, 7, 2, 10, 3, 10, 8, 3, 10, 9, 8, -1, -1, -1, -1},
{7, 2, 3, 6, 2, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{7, 0, 8, 7, 6, 0, 6, 2, 0, -1, -1, -1, -1, -1, -1, -1},
{2, 7, 6, 2, 3, 7, 0, 1, 9, -1, -1, -1, -1, -1, -1, -1},
{1, 6, 2, 1, 8, 6, 1, 9, 8, 8, 7, 6, -1, -1, -1, -1},
{10, 7, 6, 10, 1, 7, 1, 3, 7, -1, -1, -1, -1, -1, -1, -1},
{10, 7, 6, 1, 7, 10, 1, 8, 7, 1, 0, 8, -1, -1, -1, -1},
{0, 3, 7, 0, 7, 10, 0, 10, 9, 6, 10, 7, -1, -1, -1, -1},
{7, 6, 10, 7, 10, 8, 8, 10, 9, -1, -1, -1, -1, -1, -1, -1},
{6, 8, 4, 11, 8, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{3, 6, 11, 3, 0, 6, 0, 4, 6, -1, -1, -1, -1, -1, -1, -1},
{8, 6, 11, 8, 4, 6, 9, 0, 1, -1, -1, -1, -1, -1, -1, -1},
{9, 4, 6, 9, 6, 3, 9, 3, 1, 11, 3, 6, -1, -1, -1, -1},
{6, 8, 4, 6, 11, 8, 2, 10, 1, -1, -1, -1, -1, -1, -1, -1},
{1, 2, 10, 3, 0, 11, 0, 6, 11, 0, 4, 6, -1, -1, -1, -1},
{4, 11, 8, 4, 6, 11, 0, 2, 9, 2, 10, 9, -1, -1, -1, -1},
{10, 9, 3, 10, 3, 2, 9, 4, 3, 11, 3, 6, 4, 6, 3, -1},
{8, 2, 3, 8, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1},
{0, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{1, 9, 0, 2, 3, 4, 2, 4, 6, 4, 3, 8, -1, -1, -1, -1},
{1, 9, 4, 1, 4, 2, 2, 4, 6, -1, -1, -1, -1, -1, -1, -1},
{8, 1, 3, 8, 6, 1, 8, 4, 6, 6, 10, 1, -1, -1, -1, -1},
{10, 1, 0, 10, 0, 6, 6, 0, 4, -1, -1, -1, -1, -1, -1, -1},
{4, 6, 3, 4, 3, 8, 6, 10, 3, 0, 3, 9, 10, 9, 3, -1},
{10, 9, 4, 6, 10, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{4, 9, 5, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{0, 8, 3, 4, 9, 5, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1},
{5, 0, 1, 5, 4, 0, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1},
{11, 7, 6, 8, 3, 4, 3, 5, 4, 3, 1, 5, -1, -1, -1, -1},
{9, 5, 4, 10, 1, 2, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1},
{6, 11, 7, 1, 2, 10, 0, 8, 3, 4, 9, 5, -1, -1, -1, -1},
{7, 6, 11, 5, 4, 10, 4, 2, 10, 4, 0, 2, -1, -1, -1, -1},
{3, 4, 8, 3, 5, 4, 3, 2, 5, 10, 5, 2, 11, 7, 6, -1},
{7, 2, 3, 7, 6, 2, 5, 4, 9, -1, -1, -1, -1, -1, -1, -1},
{9, 5, 4, 0, 8, 6, 0, 6, 2, 6, 8, 7, -1, -1, -1, -1},
{3, 6, 2, 3, 7, 6, 1, 5, 0, 5, 4, 0, -1, -1, -1, -1},
{6, 2, 8, 6, 8, 7, 2, 1, 8, 4, 8, 5, 1, 5, 8, -1},
{9, 5, 4, 10, 1, 6, 1, 7, 6, 1, 3, 7, -1, -1, -1, -1},
{1, 6, 10, 1, 7, 6, 1, 0, 7, 8, 7, 0, 9, 5, 4, -1},
{4, 0, 10, 4, 10, 5, 0, 3, 10, 6, 10, 7, 3, 7, 10, -1},
{7, 6, 10, 7, 10, 8, 5, 4, 10, 4, 8, 10, -1, -1, -1, -1},
{6, 9, 5, 6, 11, 9, 11, 8, 9, -1, -1, -1, -1, -1, -1, -1},
{3, 6, 11, 0, 6, 3, 0, 5, 6, 0, 9, 5, -1, -1, -1, -1},
{0, 11, 8, 0, 5, 11, 0, 1, 5, 5, 6, 11, -1, -1, -1, -1},
{6, 11, 3, 6, 3, 5, 5, 3, 1, -1, -1, -1, -1, -1, -1, -1},
{1, 2, 10, 9, 5, 11, 9, 11, 8, 11, 5, 6, -1, -1, -1, -1},
{0, 11, 3, 0, 6, 11, 0, 9, 6, 5, 6, 9, 1, 2, 10, -1},
{11, 8, 5, 11, 5, 6, 8, 0, 5, 10, 5, 2, 0, 2, 5, -1},
{6, 11, 3, 6, 3, 5, 2, 10, 3, 10, 5, 3, -1, -1, -1, -1},
{5, 8, 9, 5, 2, 8, 5, 6, 2, 3, 8, 2, -1, -1, -1, -1},
{9, 5, 6, 9, 6, 0, 0, 6, 2, -1, -1, -1, -1, -1, -1, -1},
{1, 5, 8, 1, 8, 0, 5, 6, 8, 3, 8, 2, 6, 2, 8, -1},
{1, 5, 6, 2, 1, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{1, 3, 6, 1, 6, 10, 3, 8, 6, 5, 6, 9, 8, 9, 6, -1},
{10, 1, 0, 10, 0, 6, 9, 5, 0, 5, 6, 0, -1, -1, -1, -1},
{0, 3, 8, 5, 6, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{10, 5, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{11, 5, 10, 7, 5, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{11, 5, 10, 11, 7, 5, 8, 3, 0, -1, -1, -1, -1, -1, -1, -1},
{5, 11, 7, 5, 10, 11, 1, 9, 0, -1, -1, -1, -1, -1, -1, -1},
{10, 7, 5, 10, 11, 7, 9, 8, 1, 8, 3, 1, -1, -1, -1, -1},
{11, 1, 2, 11, 7, 1, 7, 5, 1, -1, -1, -1, -1, -1, -1, -1},
{0, 8, 3, 1, 2, 7, 1, 7, 5, 7, 2, 11, -1, -1, -1, -1},
{9, 7, 5, 9, 2, 7, 9, 0, 2, 2, 11, 7, -1, -1, -1, -1},
{7, 5, 2, 7, 2, 11, 5, 9, 2, 3, 2, 8, 9, 8, 2, -1},
{2, 5, 10, 2, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1},
{8, 2, 0, 8, 5, 2, 8, 7, 5, 10, 2, 5, -1, -1, -1, -1},
{9, 0, 1, 5, 10, 3, 5, 3, 7, 3, 10, 2, -1, -1, -1, -1},
{9, 8, 2, 9, 2, 1, 8, 7, 2, 10, 2, 5, 7, 5, 2, -1},
{1, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{0, 8, 7, 0, 7, 1, 1, 7, 5, -1, -1, -1, -1, -1, -1, -1},
{9, 0, 3, 9, 3, 5, 5, 3, 7, -1, -1, -1, -1, -1, -1, -1},
{9, 8, 7, 5, 9, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{5, 8, 4, 5, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1},
{5, 0, 4, 5, 11, 0, 5, 10, 11, 11, 3, 0, -1, -1, -1, -1},
{0, 1, 9, 8, 4, 10, 8, 10, 11, 10, 4, 5, -1, -1, -1, -1},
{10, 11, 4, 10, 4, 5, 11, 3, 4, 9, 4, 1, 3, 1, 4, -1},
{2, 5, 1, 2, 8, 5, 2, 11, 8, 4, 5, 8, -1, -1, -1, -1},
{0, 4, 11, 0, 11, 3, 4, 5, 11, 2, 11, 1, 5, 1, 11, -1},
{0, 2, 5, 0, 5, 9, 2, 11, 5, 4, 5, 8, 11, 8, 5, -1},
{9, 4, 5, 2, 11, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{2, 5, 10, 3, 5, 2, 3, 4, 5, 3, 8, 4, -1, -1, -1, -1},
{5, 10, 2, 5, 2, 4, 4, 2, 0, -1, -1, -1, -1, -1, -1, -1},
{3, 10, 2, 3, 5, 10, 3, 8, 5, 4, 5, 8, 0, 1, 9, -1},
{5, 10, 2, 5, 2, 4, 1, 9, 2, 9, 4, 2, -1, -1, -1, -1},
{8, 4, 5, 8, 5, 3, 3, 5, 1, -1, -1, -1, -1, -1, -1, -1},
{0, 4, 5, 1, 0, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{8, 4, 5, 8, 5, 3, 9, 0, 5, 0, 3, 5, -1, -1, -1, -1},
{9, 4, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{4, 11, 7, 4, 9, 11, 9, 10, 11, -1, -1, -1, -1, -1, -1, -1},
{0, 8, 3, 4, 9, 7, 9, 11, 7, 9, 10, 11, -1, -1, -1, -1},
{1, 10, 11, 1, 11, 4, 1, 4, 0, 7, 4, 11, -1, -1, -1, -1},
{3, 1, 4, 3, 4, 8, 1, 10, 4, 7, 4, 11, 10, 11, 4, -1},
{4, 11, 7, 9, 11, 4, 9, 2, 11, 9, 1, 2, -1, -1, -1, -1},
{9, 7, 4, 9, 11, 7, 9, 1, 11, 2, 11, 1, 0, 8, 3, -1},
{11, 7, 4, 11, 4, 2, 2, 4, 0, -1, -1, -1, -1, -1, -1, -1},
{11, 7, 4, 11, 4, 2, 8, 3, 4, 3, 2, 4, -1, -1, -1, -1},
{2, 9, 10, 2, 7, 9, 2, 3, 7, 7, 4, 9, -1, -1, -1, -1},
{9, 10, 7, 9, 7, 4, 10, 2, 7, 8, 7, 0, 2, 0, 7, -1},
{3, 7, 10, 3, 10, 2, 7, 4, 10, 1, 10, 0, 4, 0, 10, -1},
{1, 10, 2, 8, 7, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{4, 9, 1, 4, 1, 7, 7, 1, 3, -1, -1, -1, -1, -1, -1, -1},
{4, 9, 1, 4, 1, 7, 0, 8, 1, 8, 7, 1, -1, -1, -1, -1},
{4, 0, 3, 7, 4, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{4, 8, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{9, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{3, 0, 9, 3, 9, 11, 11, 9, 10, -1, -1, -1, -1, -1, -1, -1},
{0, 1, 10, 0, 10, 8, 8, 10, 11, -1, -1, -1, -1, -1, -1, -1},
{3, 1, 10, 11, 3, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{1, 2, 11, 1, 11, 9, 9, 11, 8, -1, -1, -1, -1, -1, -1, -1},
{3, 0, 9, 3, 9, 11, 1, 2, 9, 2, 11, 9, -1, -1, -1, -1},
{0, 2, 11, 8, 0, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{3, 2, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{2, 3, 8, 2, 8, 10, 10, 8, 9, -1, -1, -1, -1, -1, -1, -1},
{9, 10, 2, 0, 9, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{2, 3, 8, 2, 8, 10, 0, 1, 8, 1, 10, 8, -1, -1, -1, -1},
{1, 10, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{1, 3, 8, 9, 1, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{0, 9, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{0, 3, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1} };

#elif TRIANGULATION_ALGORITHM == MARCHING_TETS

const int3 OpenGLRasterization::offsets[8] = {
	make_int3(0, 0, 0),
	make_int3(1, 0, 0),
	make_int3(1, 1, 0),
	make_int3(0, 1, 0),
	make_int3(0, 0, 1),
	make_int3(1, 0, 1),
	make_int3(1, 1, 1),
	make_int3(0, 1, 1)
};

const int OpenGLRasterization::tetrahedra[6][4] = {
	{ 0, 7, 3, 2 },
	{ 0, 7, 2, 6 },
	{ 0, 4, 6, 7 },
	{ 0, 6, 1, 2 },
	{ 0, 6, 1, 4 },
	{ 5, 6, 1, 4 }
};

const int OpenGLRasterization::edgeTable[16][10] = {
	{-1,-1},
	{0,1, 0,3, 0,2, -1,-1},
	{1,0, 1,2, 1,3, -1,-1},
	{3,0, 2,0, 1,3, 2,1, -1,-1},
	{2,0, 2,3, 2,1, -1,-1},
	{3,0, 1,2, 1,0, 2,3, -1,-1},
	{0,1, 0,2, 1,3, 3,2, -1,-1},
	{3,0, 3,2, 3,1, -1,-1},
	{3,1, 3,2, 3,0, -1,-1},
	{0,1, 1,3, 0,2, 3,2, -1,-1},
	{3,0, 1,0, 1,2, 2,3, -1,-1},
	{2,1, 2,3, 2,0, -1,-1},
	{1,3, 2,0, 3,0, 2,1, -1,-1},
	{1,0, 1,3, 1,2, -1,-1},
	{0,1, 0,2, 0,3, -1,-1},
	{-1,-1}
};

const int OpenGLRasterization::triTable[16][7] = {
	{-1},
	{0,1,2,-1},
	{0,1,2,-1},
	{0,1,2,1,3,2,-1},
	{0,1,2,-1},
	{0,1,2,1,0,3,-1},
	{0,1,2,2,1,3,-1},
	{0,1,2,-1},
	{0,1,2,-1},
	{0,1,2,1,3,2,-1},
	{0,1,2,2,3,0,-1},
	{0,1,2,-1},
	{0,1,2,1,0,3,-1},
	{0,1,2,-1},
	{0,1,2,-1},
	{-1}
};

#endif



END_RENDERER_NAMESPACE
