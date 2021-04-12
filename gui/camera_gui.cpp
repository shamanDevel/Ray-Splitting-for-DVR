#include "camera_gui.h"

#include "imgui/imgui.h"
#include <cmath>
#include <helper_math.cuh>
#include <glm/glm.hpp>
#include "utils.h"

bool CameraGui::specifyUI()
{
	bool changed = false;

	float3 origin, up;
	float distance;
	computeParameters(origin, up, distance);

	float fovMin = 0.1, fovMax = 90;
	if (ImGui::SliderScalar("FoV", ImGuiDataType_Float, &fov_, &fovMin, &fovMax, u8"%.5f\u00b0", 2)) changed = true;
	ImGui::InputFloat3("Camera Origin", &origin.x, "%.3f", ImGuiInputTextFlags_ReadOnly);
	if (ImGui::InputFloat3("Camera Look At", &lookAt_.x)) changed = true;
	ImGui::InputFloat3("Camera Up", &up.x);

	for (int i = 0; i < 6; ++i) {
		if (ImGui::RadioButton(OrientationNames[i], orientation_ == Orientation(i))) {
			orientation_ = Orientation(i);
			changed = true;
		}
		if (i<5) ImGui::SameLine();
	}
	
	float minPitch = -80, maxPitch = +80;
	if (ImGui::SliderScalar("Pitch", ImGuiDataType_Float, &currentPitch_, &minPitch, &maxPitch, u8"%.5f\u00b0")) changed = true;
	if (ImGui::InputFloat("Yaw", &currentYaw_, 0, 0, u8"%.5f\u00b0")) changed = true;

	if (ImGui::InputFloat("Zoom", &zoomvalue_)) changed = true;
	ImGui::InputFloat("Distance", &distance, 0, 0, ".3f", ImGuiInputTextFlags_ReadOnly);

	return changed;
}

bool CameraGui::updateMouse()
{
	ImGuiIO& io = ImGui::GetIO();
	if (io.WantCaptureMouse) return false;

	if (io.MouseDown[0])
	{
		//dragging
		currentPitch_ = std::max(-80.0f, std::min(80.0f, 
			currentPitch_ + rotateSpeed_ * io.MouseDelta.y));
		currentYaw_ += rotateSpeed_ * io.MouseDelta.x;
	}
	//zoom
	auto mouseWheel = ImGui::GetIO().MouseWheel;
	zoomvalue_ += mouseWheel;

	bool changed = mouseWheel != 0 || (io.MouseDown[0] && (io.MouseDelta.x != 0 || io.MouseDelta.y != 0));
	return changed;
}

