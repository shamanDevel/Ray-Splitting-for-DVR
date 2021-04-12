#include "camera.h"

#include <iostream>
#include <iomanip>

#include "glm/glm.hpp"
#include "glm/gtc/quaternion.hpp"
#include "glm/gtx/quaternion.hpp"
#include "glm/gtc/matrix_access.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include "helper_math.cuh"
#include "utils.h"

namespace std
{
	std::ostream& operator<<(std::ostream& o, const glm::vec3 v)
	{
		o << std::fixed << std::setw(5) << std::setprecision(3)
			<< v.x << "," << v.y << "," << v.z;
		return o;
	}
	std::ostream& operator<<(std::ostream& o, const glm::vec4 v)
	{
		o << std::fixed << std::setw(5) << std::setprecision(3)
			<< v.x << "," << v.y << "," << v.z << "," << v.w;
		return o;
	}
	std::ostream& operator<<(std::ostream& o, const glm::mat4 m)
	{
		o << m[0] << "\n" << m[1] << "\n" << m[2] << "\n" << m[3];
		return o;
	}
}

BEGIN_RENDERER_NAMESPACE

namespace{
	// copy of glm::perspectiveFovLH_ZO, seems to not be defined in unix
	glm::mat4 perspectiveFovLH_ZO(float fov, float width, float height, float zNear, float zFar)
	{
		assert(width > static_cast<float>(0));
		assert(height > static_cast<float>(0));
		assert(fov > static_cast<float>(0));

		float const rad = fov;
		float const h = glm::cos(static_cast<float>(0.5) * rad) / glm::sin(static_cast<float>(0.5) * rad);
		float const w = h * height / width; ///todo max(width , Height) / min(width , Height)?

		glm::mat4 Result(static_cast<float>(0));
		Result[0][0] = w;
		Result[1][1] = h;
		Result[2][2] = zFar / (zFar - zNear);
		Result[2][3] = static_cast<float>(1);
		Result[3][2] = -(zFar * zNear) / (zFar - zNear);
		return Result;
	}
}

const char* Camera::OrientationNames[6] = {
	"Xp", "Xm", "Yp", "Ym", "Zp", "Zm"
};
const float3 Camera::OrientationUp[6] = {
	float3{1,0,0}, float3{-1,0,0},
	float3{0,1,0}, float3{0,-1,0},
	float3{0,0,1}, float3{0,0,-1}
};
const int3 Camera::OrientationPermutation[6] = {
	int3{2,-1,-3}, int3{-2, 1, 3},
	int3{1,2,3}, int3{-1,-2,-3},
	int3{-3,-1,2}, int3{3,1,-2}
};
const bool Camera::OrientationInvertYaw[6] = {
	true, false, false, true, false, true
};
const bool Camera::OrientationInvertPitch[6] = {
	false, false, false, false, false, false
};

void Camera::updateRenderArgs(renderer::RendererArgs& args) const
{
	float3 origin, up;
	float distance;
	computeParameters(origin, up, distance);

	args.cameraOrigin = origin;
	args.cameraFovDegrees = fov_;
	args.cameraLookAt = lookAt_;
	args.cameraUp = up;
}

float3 Camera::screenToWorld(const float3& screenDirection) const
{
	float3 origin, up;
	float distance;
	computeParameters(origin, up, distance);

	float3 viewDir = normalize(lookAt_ - origin);
	up = normalize(up - dot(viewDir, up) * viewDir);
	float3 right = cross(up, viewDir);

	return right * screenDirection.x +
		up * screenDirection.y +
		viewDir * screenDirection.z;
}

nlohmann::json Camera::toJson() const
{
	return {
		{"orientation", orientation_},
		{"lookAt", lookAt_},
		{"rotateSpeed", rotateSpeed_},
		{"zoomSpeed", zoomSpeed_},
		{"fov", fov_},
		{"currentPitch", currentPitch_},
		{"currentYaw", currentYaw_},
		{"zoomValue", zoomvalue_}
	};
}

void Camera::fromJson(const nlohmann::json& s)
{
	orientation_ = s.at("orientation").get<Orientation>();
	lookAt_ = s.at("lookAt").get<float3>();
	rotateSpeed_ = s.at("rotateSpeed").get<float>();
	zoomSpeed_ = s.at("zoomSpeed").get<float>();
	fov_ = s.at("fov").get<float>();
	currentPitch_ = s.at("currentPitch").get<float>();
	currentYaw_ = s.at("currentYaw").get<float>();
	zoomvalue_ = s.at("zoomValue").get<float>();
}

void Camera::computeParameters(float3& origin, float3& up, float& distance) const
{
	distance = baseDistance_ * std::pow(zoomSpeed_, zoomvalue_);
	up = OrientationUp[orientation_];

	float yaw = glm::radians(!OrientationInvertYaw[orientation_] ? -currentYaw_ : +currentYaw_);
	float pitch = glm::radians(!OrientationInvertPitch[orientation_] ? -currentPitch_ : +currentPitch_);
	float pos[3];
	pos[1] = std::sin(pitch) * distance;
	pos[0] = std::cos(pitch) * std::cos(yaw) * distance;
	pos[2] = std::cos(pitch) * std::sin(yaw) * distance;
	float pos2[3];
	for (int i = 0; i < 3; ++i)
	{
		int p = (&OrientationPermutation[orientation_].x)[i];
		pos2[i] = pos[std::abs(p) - 1] * (p > 0 ? 1 : -1);
	}
	origin = make_float3(pos2[0], pos2[1], pos2[2]) + lookAt_;
}


void Camera::computeMatrices(float3 cameraOrigin_, float3 cameraLookAt_, float3 cameraUp_, float fovDegrees,
	int width, int height, float nearClip, float farClip, float4 viewMatrixOut[4], float4 viewMatrixInverseOut[4],
	float4 normalMatrixOut[4])
{
	const glm::vec3 cameraOrigin = *reinterpret_cast<glm::vec3*>(&cameraOrigin_.x);
	const glm::vec3 cameraLookAt = *reinterpret_cast<glm::vec3*>(&cameraLookAt_.x);
	const glm::vec3 cameraUp = *reinterpret_cast<glm::vec3*>(&cameraUp_.x);

	float fovRadians = glm::radians(fovDegrees);

	glm::mat4 viewMatrix = glm::lookAtLH(cameraOrigin, cameraLookAt, normalize(cameraUp));
	glm::mat4 projMatrix = glm::perspectiveFovLH_NO(fovRadians, float(width), float(height), nearClip, farClip);

	glm::mat4 viewProjMatrix = projMatrix * viewMatrix;
	glm::mat4 invViewProjMatrix = glm::inverse(viewProjMatrix);
	glm::mat4 normalMatrix = glm::inverse(glm::transpose(glm::mat4(glm::mat3(viewMatrix))));

	viewProjMatrix = glm::transpose(viewProjMatrix);
	invViewProjMatrix = glm::transpose(invViewProjMatrix);
	normalMatrix = glm::transpose(normalMatrix);
	//normalMatrix[0] = -normalMatrix[0]; //somehow, the networks were trained with normal-x inverted
	for (int i = 0; i < 4; ++i) viewMatrixOut[i] = *reinterpret_cast<float4*>(&viewProjMatrix[i].x);
	for (int i = 0; i < 4; ++i) viewMatrixInverseOut[i] = *reinterpret_cast<float4*>(&invViewProjMatrix[i].x);
	for (int i = 0; i < 4; ++i) normalMatrixOut[i] = *reinterpret_cast<float4*>(&normalMatrix[i].x);
}

void Camera::computeMatricesOpenGL(float3 cameraOrigin_, float3 cameraLookAt_, float3 cameraUp_, float fovDegrees,
	int width, int height, float nearClip, float farClip, 
	glm::mat4& viewMatrixOut, glm::mat4& projectionMatrixOut, glm::mat4& normalMatrixOut)
{
	const glm::vec3 cameraOrigin = *reinterpret_cast<glm::vec3*>(&cameraOrigin_.x);
	const glm::vec3 cameraLookAt = *reinterpret_cast<glm::vec3*>(&cameraLookAt_.x);
	const glm::vec3 cameraUp = *reinterpret_cast<glm::vec3*>(&cameraUp_.x);

	float fovRadians = glm::radians(fovDegrees);

	viewMatrixOut = glm::lookAtLH(cameraOrigin, cameraLookAt, normalize(cameraUp));
	projectionMatrixOut = glm::perspectiveFovLH_NO(fovRadians, float(width), float(height), nearClip, farClip);
	normalMatrixOut = glm::inverse(glm::transpose(glm::mat4(glm::mat3(viewMatrixOut))));
}

END_RENDERER_NAMESPACE
