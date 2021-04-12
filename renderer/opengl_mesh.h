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

struct Vertex
{
	float4 position;
	float4 normals;
};

struct MeshCpu
{
	std::vector<Vertex> vertices;
	std::vector<GLuint> indices;
};

/**
 * Stores a single mesh.
 * Each vertex is represented by the structure "Vertex",
 * positions are bound to attribute index 0 and normals
 * to attribute index 1.
 */
class Mesh
{
	GLuint vbo = 0, vao = 0, ibo = 0;
	int numVertices = 0, numIndices = 0;
	int availableVertices = 0, availableIndices = 0;
	cudaGraphicsResource_t vboCuda = 0, iboCuda = 0;

	Mesh(Mesh const&) = delete;
	Mesh& operator=(Mesh const&) = delete;
	
public:
	static constexpr int POSITION_INDEX = 0;
	static constexpr int NORMAL_INDEX = 1;

	Mesh() = default;
	~Mesh();
	void free();
	//resize to exactly the given size
	void resize(int vertices, int indices);
	//reserves the given size, i.e. resize only if smaller
	void reserve(int vertices, int indices);
	void draw();

	void cudaMap(Vertex** vertices, GLuint** indices);
	void cudaUnmap();

	void copyToCpu(MeshCpu& meshCpu);
	void copyFromCpu(const MeshCpu& meshCpu);

	bool isValid() const { return vbo != 0; }
	int getNumVertices() const { return numVertices; }
	int getNumIndices() const { return numIndices; }
};

END_RENDERER_NAMESPACE
