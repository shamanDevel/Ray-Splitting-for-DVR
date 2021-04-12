#pragma once

#include <cuda_runtime.h>
#include <json.hpp>
#include <glm/fwd.hpp>


#include "commons.h"
#include "settings.h"

BEGIN_RENDERER_NAMESPACE

class MY_API Camera
{
public:
	
	/**
	 * \brief Computes the perspective camera matrices for raytracing
	 * \param cameraOrigin camera origin / eye pos
	 * \param cameraLookAt look at / target
	 * \param cameraUp up vector
	 * \param fovDegrees vertical field-of-views in degree
	 * \param width the width of the screen
	 * \param height the height of the screen
	 * \param nearClip the near clipping plane
	 * \param farClip the far clipping plane
	 * \param viewMatrixOut view-projection matrix in Row Major order (viewMatrixOut[0] is the first row), [OUT]
	 * \param viewMatrixInverseOut inverse view-projection matrix in Row Major order (viewMatrixInverseOut[0] is the first row), [OUT]
	 * \param normalMatrixOut normal matrix in Row Major order (normalMatrixOut[0] is the first row), [OUT]
	 */
	static void computeMatrices(
		float3 cameraOrigin, float3 cameraLookAt, float3 cameraUp, 
		float fovDegrees, int width, int height, float nearClip, float farClip,
		float4 viewMatrixOut[4], float4 viewMatrixInverseOut[4], float4 normalMatrixOut[4]
	);

	static void computeMatricesOpenGL(
		float3 cameraOrigin, float3 cameraLookAt, float3 cameraUp,
		float fovDegrees, int width, int height, float nearClip, float farClip,
		glm::mat4& viewMatrixOut, glm::mat4& projectionMatrixOut, glm::mat4& normalMatrixOut);
	
	/**
	 * \brief Sets cameraOrigin, cameraLookAt,
	 *  cameraUp and cameraFovDegrees.
	 */
	void updateRenderArgs(RendererArgs& args) const;

	/**
	 * \brief Converts a direction in screen space (X,Y,Z) to
	 * the direction in world space (right,up,viewDir).
	 * \param screenDirection the screen space direction
	 */
	float3 screenToWorld(const float3& screenDirection) const;

	nlohmann::json toJson() const;
	void fromJson(const nlohmann::json& s);

	enum Orientation
	{
		Xp, Xm, Yp, Ym, Zp, Zm
	};
	
	::renderer::Camera::Orientation orientation() const
	{
		return orientation_;
	}

	void setOrientation(::renderer::Camera::Orientation orientation)
	{
		orientation_ = orientation;
	}

	float3 lookAt() const
	{
		return lookAt_;
	}

	void setLookAt(const float3& lookAt)
	{
		lookAt_ = lookAt;
	}

	float fov() const
	{
		return fov_;
	}

	void setFov(float fov)
	{
		fov_ = fov;
	}

	float baseDistance() const
	{
		return baseDistance_;
	}

	float currentPitch() const
	{
		return currentPitch_;
	}

	void setCurrentPitch(float currentPitch)
	{
		currentPitch_ = currentPitch;
	}

	float currentYaw() const
	{
		return currentYaw_;
	}

	void setCurrentYaw(float currentYaw)
	{
		currentYaw_ = currentYaw;
	}

	float zoomvalue() const
	{
		return zoomvalue_;
	}

	void setZoomvalue(float zoomvalue)
	{
		zoomvalue_ = zoomvalue;
	}

	float rotateSpeed() const
	{
		return rotateSpeed_;
	}

	void setRotateSpeed(float rotateSpeed)
	{
		rotateSpeed_ = rotateSpeed;
	}

	float zoomSpeed() const
	{
		return zoomSpeed_;
	}

	void setZoomSpeed(float zoomSpeed)
	{
		zoomSpeed_ = zoomSpeed;
	}

public:
	static const char* OrientationNames[6];
	static const float3 OrientationUp[6];
	static const int3 OrientationPermutation[6];
	static const bool OrientationInvertYaw[6];
	static const bool OrientationInvertPitch[6];
protected:
	Orientation orientation_ = Zm;

	float3 lookAt_{ 0,0,0 };

	float fov_ = 45.0f;
	const float baseDistance_ = 1.0f;
	float currentPitch_ = 67.0f;
	float currentYaw_ = 96.0f;
	float zoomvalue_ = 0;

	float rotateSpeed_ = 0.5f;
	float zoomSpeed_ = 1.1;

	void computeParameters(
		float3& origin, float3& up, float& distance) const;
};

END_RENDERER_NAMESPACE

