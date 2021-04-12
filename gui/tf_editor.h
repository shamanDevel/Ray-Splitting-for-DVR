#pragma once

#include <cuda_runtime.h>
#include <GL/glew.h>
#include <deque>
#include <vector>
#include <json.hpp>

#include "imgui/imgui.h"
#include "imgui/imgui_internal.h"
#include "tf_texture_1d.h"

class TfEditorMultiIso;
class TfEditorLinear;

class TfEditorLinearOpacity
{
public:
	TfEditorLinearOpacity(bool extraColor);
	void init(const ImRect& rect);
	void updateControlPoints(
		const std::vector<float>& densityAxis, const std::vector<float>& opacityAxis,
		const std::vector<float4>& extraColorAxis = {});
	void handleIO();
	void render();

	const std::vector<float>& getDensityAxis() const { return densityAxis_; }
	const std::vector<float>& getOpacityAxis() const { return opacityAxis_; }
	bool getIsChanged() const { return isChanged_; }

	bool hasExtraColor() const { return hasExtraColor_; }
	const std::vector<float4>& getExtraColorAxis() const { assert(hasExtraColor());  return extraColorAxis_; }  //XYZ alpha color space

private:
	const bool hasExtraColor_;
	const float circleRadius_{ 4.0f };

	ImRect tfEditorRect_;
	int clickedControlPoint_{ -1 };
	int selectedControlPoint_{ -1 };
	struct ControlPoint
	{
		ImVec2 posOpacity;
		ImVec4 extraColor; //XYZ alpha color space
	};
	std::deque<ControlPoint> controlPoints_;
	std::vector<float> densityAxis_;
	std::vector<float> opacityAxis_;
	std::vector<float4> extraColorAxis_;

	bool isChanged_{ false };

private:
	ImRect createControlPointRect(const ImVec2& controlPoint);
	ImVec2 screenToEditor(const ImVec2& screenPosition);
	ImVec2 editorToScreen(const ImVec2& editorPosition);
};

class TfEditorLinearColor
{
public:
	//Non-copyable and non-movable
	TfEditorLinearColor();
	~TfEditorLinearColor();
	TfEditorLinearColor(const TfEditorLinearColor&) = delete;
	TfEditorLinearColor(TfEditorLinearColor&&) = delete;

	void init(const ImRect& rect, bool showControlPoints);
	//LAB color space
	void updateControlPoints(const std::vector<float>& densityAxis, const std::vector<float3>& colorAxis);
	void handleIO();
	void render();

	const std::vector<float>& getDensityAxis() const { return densityAxis_; }
	const std::vector<float3>& getColorAxis() const { return colorAxis_; }
	bool getIsChanged() const { return isChanged_; }

private:
	const float cpWidth_{ 8.0f };

	ImVec4 pickedColor_{ 0.0f, 0.0f, 1.0f, 1.0f };
	ImRect tfEditorRect_;
	int selectedControlPointForMove_{ -1 };
	int selectedControlPointForColor_{ -1 };
	std::deque<ImVec4> controlPoints_;
	std::vector<float> densityAxis_;
	std::vector<float3> colorAxis_;
	bool isChanged_{ false };
	bool showControlPoints_{ true };

	//Variables for color map texture.
	cudaGraphicsResource* resource_{ nullptr };
	GLuint colorMapImage_{ 0 };
	cudaSurfaceObject_t content_{ 0 };
	cudaArray_t contentArray_{ nullptr };
	RENDERER_NAMESPACE::TfTexture1D tfTexture_;

private:
	void destroy();
	ImRect createControlPointRect(float x);
	float screenToEditor(float screenPositionX);
	float editorToScreen(float editorPositionX);
};

/**
 * \brief Transfer function editor for piecewise-linear TFs.
 */
class TfEditorLinear
{
public:
	explicit TfEditorLinear(bool extraColorPerOpacityPoint);
	void init(const ImRect& tfEditorOpacityRect, const ImRect& tfEditorColorRect, bool showColorControlPoints);
	void handleIO();
	void render();
	void saveToFile(const std::string& path, float minDensity, float maxDensity) const;
	void loadFromFile(const std::string& path, float& minDensity, float& maxDensity);
	nlohmann::json toJson() const;
	void fromJson(const nlohmann::json& s);

	const std::vector<float>& getDensityAxisOpacity() const { return editorOpacity_.getDensityAxis(); }
	const std::vector<float>& getOpacityAxis() const { return editorOpacity_.getOpacityAxis(); }
	const std::vector<float4>& getOpacityExtraColorAxis() const { return editorOpacity_.getExtraColorAxis(); }
	const std::vector<float>& getDensityAxisColor() const { return editorColor_.getDensityAxis(); }
	const std::vector<float3>& getColorAxis() const { return editorColor_.getColorAxis(); }
	bool getIsChanged() const { return editorOpacity_.getIsChanged() || editorColor_.getIsChanged(); }

	static bool testIntersectionRectPoint(const ImRect& rect, const ImVec2& point);
	//Loads the colormap (position + rgb) from an xml.
	static std::pair< std::vector<float>, std::vector<float3> >
		loadColormapFromXML(const std::string& path);

	void fromMultiIso(const TfEditorMultiIso* multiIso, float peakWidth);
	
private:
	TfEditorLinearOpacity editorOpacity_;
	TfEditorLinearColor editorColor_;
};

/**
 * \brief Transfer Function editor for multi-isosurfaces
 */
class TfEditorMultiIso
{
public:
	TfEditorMultiIso();
	void init(const ImRect& tfEditorRect);
	void handleIO();
	void render();
	void saveToFile(const std::string& path, float minDensity, float maxDensity) const;
	void loadFromFile(const std::string& path, float& minDensity, float& maxDensity);
	nlohmann::json toJson() const;
	void fromJson(const nlohmann::json& s);

	const std::vector<float>& getControlPointPositions() const { return positionAxis_; }
	//XYZ alpha color space
	const std::vector<float4>& getControlPointColors() const { return colorAxis_; }
	bool getIsChanged();

private:
	const float circleRadius_{ 6.0f };

	struct PointData
	{
		ImVec2 pos;
		float3 color; //xyz
	};
	
	ImRect tfEditorRect_;
	int selectedControlPoint_{ -1 };
	int draggedControlPoint_{ -1 };
	std::deque<PointData> controlPoints_;
	std::vector<float> positionAxis_;
	std::vector<float4> colorAxis_;
	ImVec4 pickedColor_{ 0.0f, 0.0f, 1.0f, 1.0f };

	bool isChanged_{ false };

	ImRect createControlPointRect(const ImVec2& controlPoint);
	ImVec2 screenToEditor(const ImVec2& screenPosition);
	ImVec2 editorToScreen(const ImVec2& editorPosition);
};
