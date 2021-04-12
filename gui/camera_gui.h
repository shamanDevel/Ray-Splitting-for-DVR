#pragma once

#include <lib.h>
#include <json.hpp>

#include "camera.h"

class CameraGui : public renderer::Camera
{
public:
	/**
	 * \brief Specifies the UI
	 * \return true if properties have changed
	 */
	bool specifyUI();

	/**
	 * \brief Updates movement with mouse dragging
	 * and scroll wheel zooming
	 * \return true if properties have changed
	 */
	bool updateMouse();
};
