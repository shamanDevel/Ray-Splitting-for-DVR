#include <catch.hpp>

#include <volume.h>
#include <renderer.h>
#include <opengl_renderer.h>
#include <cuMat/src/Errors.h>
#include <glm/gtc/matrix_transform.hpp>
#include <tinyformat.h>

#include <fstream>
#include <chrono>
#include <thread>

#include "test_utils.h"

void testMarchingCubes()
{
	CUMAT_SAFE_CALL(cudaSetDevice(0));

#if 1
	//create test volume
	const auto volume = renderer::Volume::createImplicitDataset(
		64, renderer::Volume::ImplicitEquation::MARSCHNER_LOBB);
	//const auto volume = renderer::Volume::createSphere(32);
	volume->getLevel(0)->copyCpuToGpu();
	std::cout << "Test volume created" << std::endl;
	
	//create offscreen context
	renderer::OpenGLRasterization::Instance().setupOffscreenContext();
	std::cout << "Offscreen context created" << std::endl;
	
	{
		//create iso shader
		const auto shader = std::make_unique<renderer::Shader>(
			"PassThrough.vs", "SingleIso.fs");
		shader->use();
		
		//create test mesh
		const auto mesh = std::make_unique<renderer::Mesh>();
		renderer::MeshCpu meshCpu;
		renderer::OpenGLRasterization::Instance().fillMarchingCubesMeshPreHost(
			volume->getLevel(0), 0.5f, &meshCpu, mesh.get(), 0);
		std::cout << "The Marching-Cubes mesh for isovalue 0.5 contains " <<
			meshCpu.vertices.size() << " vertices and " <<
			meshCpu.indices.size() << " indices" << std::endl;

		//for (int i=10; i>0; --i)
		//{
		//	std::cout << "Wait " << i << " sec" << std::endl;
		//	using namespace std::chrono_literals;
		//	std::this_thread::sleep_for(1s);
		//}
		
		//create framebuffer
		int width = 640;
		int height = 480;
		const auto fb = std::make_unique<renderer::Framebuffer>(width, height);
		fb->bind();

		//render mesh
		glm::vec3 cameraPosition = glm::vec3( 0, -1.5, 0.4 ) * 0.5f;
		glm::vec3 cameraTarget{ 0,0,0 };
		glm::vec3 cameraUp{ 0,0,1 };
		float fovDegree = 45.0f;
		glm::mat4 projection = glm::perspective(glm::radians(fovDegree), (float)width / (float)height, 0.1f, 100.0f);
		glm::mat4 view = glm::lookAt(cameraPosition, cameraTarget, cameraUp);
		glm::mat4 model = glm::mat4(1.0f);
		model = glm::scale(model, glm::vec3(1.0 / volume->getLevel(0)->sizeX()));
		model = glm::translate(glm::mat4(1.0f), glm::vec3(-0.5, -0.5, -0.5)) * model;
		glm::mat4 transpInvModel = glm::transpose(glm::inverse(model));

		//test transformations
		for (int i=0; i<2; ++i) for (int j=0; j<2; ++j) for (int k=0; k<2; ++k)
		{
			glm::vec3 objectPos(
				i * volume->getLevel(0)->sizeX(),
				j * volume->getLevel(0)->sizeY(),
				k * volume->getLevel(0)->sizeZ());
			glm::vec4 worldPos = model * glm::vec4(objectPos, 1.0f);
			glm::vec4 screenPos = projection * view * worldPos;
			screenPos /= screenPos.w;
			tinyformat::format(std::cout,
				"Object (%.2f, %.2f, %.2f) -> World (%.2f, %.2f, %.2f) -> Screen (%.2f, %.2f, %.2f)\n",
				objectPos.x, objectPos.y, objectPos.z,
				worldPos.x, worldPos.y, worldPos.z,
				screenPos.x, screenPos.y, screenPos.z);
		}
		
		shader->setVec3("objectColor", 1.0f, 1.0f, 1.0f);
		shader->setVec3("lightColor", 1.0f, 1.0f, 1.0f);
		shader->setVec3("lightDir", {0,1,0});
		shader->setVec3("viewPos", cameraPosition);
		shader->setMat4("model", model);
		shader->setMat4("transpInvModel", transpInvModel);
		shader->setMat4("view", view);
		shader->setMat4("projection", projection);

		glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glEnable(GL_DEPTH_TEST);
		glDisable(GL_CULL_FACE);

		mesh->copyFromCpu(meshCpu);
		mesh->draw();

		fb->unbind();

		std::vector<float> rgba;
		fb->readRGBA(rgba);
		//save as image
		{
			using namespace std;
			ofstream ofs("testMarchingCubes.ppm", ios_base::out | ios_base::binary);
			ofs << "P6" << endl << width << ' ' << height << endl << "255" << endl;

			for (auto j = 0; j < height; ++j) {
				for (auto i = 0; i < width; ++i) {
					uint8_t r = static_cast<uint8_t>(rgba[0 + 4 * (i + width * j)] * 255);
					uint8_t g = static_cast<uint8_t>(rgba[1 + 4 * (i + width * j)] * 255);
					uint8_t b = static_cast<uint8_t>(rgba[2 + 4 * (i + width * j)] * 255);
					uint8_t a = static_cast<uint8_t>(rgba[3 + 4 * (i + width * j)] * 255);
					ofs << r << g << b;
				}
			}
			ofs.close();
			std::cout << "Image written" << std::endl;
		}
	}

	renderer::OpenGLRasterization::Instance().cleanup();
	renderer::OpenGLRasterization::Instance().deleteOffscreenContext();
#else
	testConstantMemory();
#endif
}
