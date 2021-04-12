#include "opengl_mesh.h"
#include "opengl_utils.h"

#include <cuMat/src/Errors.h>

BEGIN_RENDERER_NAMESPACE

Mesh::~Mesh()
{
	free();
}

void Mesh::free()
{
	if (vbo == 0) return; //already freed

	CUMAT_SAFE_CALL(cudaGraphicsUnregisterResource(vboCuda));
	CUMAT_SAFE_CALL(cudaGraphicsUnregisterResource(iboCuda));

	glBindVertexArray(vao);
	glDisableVertexAttribArray(POSITION_INDEX);
	glDisableVertexAttribArray(NORMAL_INDEX);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	glDeleteBuffers(1, &ibo);
	glDeleteBuffers(1, &vbo);
	glBindVertexArray(0);
	glDeleteVertexArrays(1, &vao);
	checkOpenGLError();

	vbo = 0;
	vao = 0;
	ibo = 0;
	numVertices = 0;
	numIndices = 0;
	vboCuda = 0;
	iboCuda = 0;
}

void Mesh::resize(int vertices, int indices)
{
	free();

	glGenVertexArrays(1, &vao);
	glBindVertexArray(vao);
	checkOpenGLError();

	glGenBuffers(1, &vbo); checkOpenGLError();
	glBindBuffer(GL_ARRAY_BUFFER, vbo); checkOpenGLError();
	glBufferData(GL_ARRAY_BUFFER, sizeof(Vertex) * vertices, nullptr, GL_DYNAMIC_DRAW); checkOpenGLError();

	glGenBuffers(1, &ibo); checkOpenGLError();
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo); checkOpenGLError();
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(GLuint) * indices, nullptr, GL_DYNAMIC_DRAW); checkOpenGLError();

	glEnableVertexAttribArray(POSITION_INDEX); checkOpenGLError();
	glEnableVertexAttribArray(NORMAL_INDEX); checkOpenGLError();
	glVertexAttribPointer(POSITION_INDEX, 4, GL_FLOAT, false, 8 * sizeof(GLfloat), nullptr); checkOpenGLError();
	glVertexAttribPointer(NORMAL_INDEX, 4, GL_FLOAT, false, 8 * sizeof(GLfloat), (GLvoid*)(4 * sizeof(GLfloat))); checkOpenGLError();

	CUMAT_SAFE_CALL(cudaGraphicsGLRegisterBuffer(&vboCuda, vbo, cudaGraphicsRegisterFlagsNone));
	CUMAT_SAFE_CALL(cudaGraphicsGLRegisterBuffer(&iboCuda, ibo, cudaGraphicsRegisterFlagsNone));
	
	glBindVertexArray(0); checkOpenGLError();

	numVertices = availableVertices = vertices;
	numIndices = availableIndices = indices;
}

void Mesh::reserve(int vertices, int indices)
{
	if (availableVertices < vertices || availableIndices < indices)
	{
		resize(max(availableVertices, vertices), max(availableIndices, indices));
	}
	numVertices = vertices;
	numIndices = indices;
}

void Mesh::draw()
{
	if (numVertices == 0) return;
	glBindVertexArray(vao);
	checkOpenGLError();
	glDrawElements(GL_TRIANGLES, numIndices, GL_UNSIGNED_INT, nullptr);
	checkOpenGLError();
	glBindVertexArray(0);
}

void Mesh::cudaMap(Vertex** vertices, GLuint** indices)
{
	//glFinish();
	//checkOpenGLError();
	CUMAT_SAFE_CALL(cudaGraphicsMapResources(1, &vboCuda));
	CUMAT_SAFE_CALL(cudaGraphicsMapResources(1, &iboCuda));

	size_t s;
	CUMAT_SAFE_CALL(cudaGraphicsResourceGetMappedPointer((void**)(vertices), &s, vboCuda));
	CUMAT_SAFE_CALL(cudaGraphicsResourceGetMappedPointer((void**)(indices), &s, iboCuda));
	//CUMAT_SAFE_CALL(cudaDeviceSynchronize());
}

void Mesh::cudaUnmap()
{
	//CUMAT_SAFE_CALL(cudaDeviceSynchronize());
	CUMAT_SAFE_CALL(cudaGraphicsUnmapResources(1, &vboCuda));
	CUMAT_SAFE_CALL(cudaGraphicsUnmapResources(1, &iboCuda));
}

void Mesh::copyToCpu(MeshCpu& meshCpu)
{
	meshCpu.vertices.resize(numVertices);
	meshCpu.indices.resize(numIndices);
	Vertex* v; GLuint* i;
	cudaMap(&v, &i);
	cudaMemcpy(meshCpu.vertices.data(), v, sizeof(Vertex) * numVertices,
		cudaMemcpyDeviceToHost);
	cudaMemcpy(meshCpu.indices.data(), i, sizeof(GLuint) * numIndices,
		cudaMemcpyDeviceToHost);
	cudaUnmap();
}

void Mesh::copyFromCpu(const MeshCpu& meshCpu)
{
	reserve(meshCpu.vertices.size(), meshCpu.indices.size());
	Vertex* v; GLuint* i;
	cudaMap(&v, &i);
	cudaMemcpy(v, meshCpu.vertices.data(), sizeof(Vertex) * meshCpu.vertices.size(),
		cudaMemcpyHostToDevice);
	cudaMemcpy(i, meshCpu.indices.data(), sizeof(GLuint) * meshCpu.indices.size(),
		cudaMemcpyHostToDevice);
	cudaUnmap();
}


END_RENDERER_NAMESPACE
