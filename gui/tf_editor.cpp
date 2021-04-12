#include "tf_editor.h"
#include "visualizer_kernels.h"

#include <algorithm>
#include <cuda_gl_interop.h>
#include <cuMat/src/Context.h>
#include <fstream>

#include "utils.h"
#include <tinyxml2.h>
#include <tinyformat.h>

static float4 ImVec4_to_floa4(const ImVec4& v)
{
	return make_float4(v.x, v.y, v.z, v.w);
}
static ImVec4 float4_to_ImVec4(const float4& v)
{
	return ImVec4(v.x, v.y, v.z, v.w);
}

TfEditorLinearOpacity::TfEditorLinearOpacity(bool extraColor)
	: hasExtraColor_(extraColor)
	, controlPoints_({ 
			{ImVec2(0.45f, 0.0f), ImVec4(0,0,0,0)},
			{ImVec2(0.5f, 0.8f), ImVec4(0,0,0,0)},
			{ImVec2(0.55f, 0.0f), ImVec4(0,0,0,0)} })
{
	for (const auto& e : controlPoints_)
	{
		densityAxis_.push_back(e.posOpacity.x);
		opacityAxis_.push_back(e.posOpacity.y);
		extraColorAxis_.push_back(ImVec4_to_floa4(e.extraColor));
	}
}

void TfEditorLinearOpacity::init(const ImRect& rect)
{
	isChanged_ = false;
	tfEditorRect_ = rect;

	if (hasExtraColor()) {
		bool canEdit = selectedControlPoint_ >= 0;
		ImVec4 data = canEdit ? controlPoints_[selectedControlPoint_].extraColor : ImVec4(0, 0, 0, 0);
		float3 rgb = kernel::xyzToRgb(make_float3(ImVec4_to_floa4(data)));
		
		ImGui::TextUnformatted("Iso:");
		ImGui::SameLine();
		if (ImGui::SliderFloat("##ExtraColorOpacity", &data.w, 0, 1))
		{
			if (canEdit)
			{
				controlPoints_[selectedControlPoint_].extraColor.w = data.w;
				isChanged_ = true;
			}
		}
		ImGui::SameLine();
		ImGuiColorEditFlags colorFlags = ImGuiColorEditFlags_Float | ImGuiColorEditFlags_RGB | ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_NoLabel;
		if (ImGui::ColorEdit3("##ExtraColor", &rgb.x, colorFlags))
		{
			if (canEdit)
			{
				float3 xyz = kernel::rgbToXyz(rgb);
				controlPoints_[selectedControlPoint_].extraColor.x = xyz.x;
				controlPoints_[selectedControlPoint_].extraColor.y = xyz.y;
				controlPoints_[selectedControlPoint_].extraColor.z = xyz.z;
				isChanged_ = true;
			}
		}
	}
}

void TfEditorLinearOpacity::updateControlPoints(
	const std::vector<float>& densityAxis, const std::vector<float>& opacityAxis,
	const std::vector<float4>& extraColorAxis)
{
	assert(densityAxis.size() == opacityAxis.size());
	assert(densityAxis.size() >= 1 && opacityAxis.size() >= 1);
	const int size = densityAxis.size();
	
	clickedControlPoint_ = -1;
	selectedControlPoint_ = -1;
	controlPoints_.clear();
	densityAxis_ = densityAxis;
	opacityAxis_ = opacityAxis;
	if (extraColorAxis.empty())
	{
		extraColorAxis_.resize(size);
		for (int i = 0; i < size; ++i) extraColorAxis_[i] = make_float4(0, 0, 0, 0);
	} else
	{
		assert(extraColorAxis.size() == size);
		extraColorAxis_ = extraColorAxis;
	}

	for (int i = 0; i < size; ++i)
	{
		controlPoints_.push_back({
			ImVec2(densityAxis_[i], opacityAxis_[i]),
			float4_to_ImVec4(extraColorAxis_[i]) });
	}
}

void TfEditorLinearOpacity::handleIO()
{
	auto mousePosition = ImGui::GetMousePos();

	//Early leave if mouse is not on opacity editor and no control point is selected.
	if (!TfEditorLinear::testIntersectionRectPoint(tfEditorRect_, mousePosition) && clickedControlPoint_ == -1)
	{
		return;
	}

	//0=left, 1=right, 2=middle
	bool isLeftDoubleClicked = ImGui::IsMouseDoubleClicked(0);
	bool isLeftClicked = ImGui::IsMouseDown(0);
	bool isRightClicked = ImGui::IsMouseClicked(1);
	bool isLeftReleased = ImGui::IsMouseReleased(0);

	if (isLeftDoubleClicked)
	{
		isChanged_ = true;

		controlPoints_.push_back({
			screenToEditor(mousePosition),
			ImVec4(0,0,0,0) });
		selectedControlPoint_ = -1;
		clickedControlPoint_ = -1;
	}
	else if (isLeftClicked)
	{
		//Move selected point.
		if (clickedControlPoint_ >= 0)
		{
			isChanged_ = true;

			ImVec2 center(std::min(std::max(mousePosition.x, tfEditorRect_.Min.x), tfEditorRect_.Max.x),
				std::min(std::max(mousePosition.y, tfEditorRect_.Min.y), tfEditorRect_.Max.y));

			controlPoints_[clickedControlPoint_].posOpacity = screenToEditor(center);
		}
		//Check whether new point is selected.
		else
		{
			int size = controlPoints_.size();
			for (int idx = 0; idx < size; ++idx)
			{
				auto cp = createControlPointRect(editorToScreen(controlPoints_[idx].posOpacity));
				if (TfEditorLinear::testIntersectionRectPoint(cp, mousePosition))
				{
					clickedControlPoint_ = idx;
					selectedControlPoint_ = idx;
					break;
				}
			}
		}
	}
	else if (isRightClicked)
	{
		int size = controlPoints_.size();
		for (int idx = 0; idx < size; ++idx)
		{
			auto cp = createControlPointRect(editorToScreen(controlPoints_[idx].posOpacity));
			if (TfEditorLinear::testIntersectionRectPoint(cp, mousePosition) && controlPoints_.size() > 1)
			{
				isChanged_ = true;

				controlPoints_.erase(controlPoints_.begin() + idx);
				clickedControlPoint_ = -1;
				selectedControlPoint_ = -1;
				break;
			}
		}
	}
	else if (isLeftReleased)
	{
		clickedControlPoint_ = -1;
	}
}

void TfEditorLinearOpacity::render()
{
	//Draw the bounding rectangle.
	ImGuiWindow* window = ImGui::GetCurrentWindow();
	window->DrawList->AddRect(tfEditorRect_.Min, tfEditorRect_.Max, ImColor(ImVec4(0.3f, 0.3f, 0.3f, 1.0f)), 0.0f, ImDrawCornerFlags_All, 1.0f);

	//Copy the control points and sort them. We don't sort original one in order not to mess up with control point indices.
	auto controlPointsRender = controlPoints_;
	std::sort(controlPointsRender.begin(), controlPointsRender.end(),
		[](const ControlPoint& p1, const ControlPoint& p2)
		{
			return p1.posOpacity.x < p2.posOpacity.x;
		});

	//Fill densityAxis_ and opacityAxis_ and convert coordinates from editor space to screen space.
	densityAxis_.clear();
	opacityAxis_.clear();
	extraColorAxis_.clear();
	for (auto& cp : controlPointsRender)
	{
		densityAxis_.push_back(cp.posOpacity.x);
		opacityAxis_.push_back(cp.posOpacity.y);
		extraColorAxis_.push_back(ImVec4_to_floa4(cp.extraColor));
		cp.posOpacity = editorToScreen(cp.posOpacity);
	}

	//Draw lines between the control points.
	const int size = controlPointsRender.size();
	for (int i = 0; i < size + 1; ++i)
	{
		auto left = (i == 0) ? ImVec2(tfEditorRect_.Min.x, controlPointsRender.front().posOpacity.y) : controlPointsRender[i - 1].posOpacity;
		auto right = (i == size) ? ImVec2(tfEditorRect_.Max.x, controlPointsRender.back().posOpacity.y) : controlPointsRender[i].posOpacity;

		window->DrawList->AddLine(left, right, ImColor(ImVec4(1.0f, 1.0f, 1.0f, 1.0f)), 1.0f);
	}

	
	if (hasExtraColor())
	{
		//draw extra color points lines
		for (int i = 0; i < size; ++i) {
			ImVec2 bottom = editorToScreen(ImVec2(0, 0));
			ImVec2 top = editorToScreen(ImVec2(0, controlPointsRender[i].extraColor.w));
			bottom.x = top.x = controlPointsRender[i].posOpacity.x;
			window->DrawList->AddLine(bottom, top,
				ImColor(0.0f, 0.0f, 0.2f), 3);
		}
		
		//Draw the control points with selection
		for (int i=0; i<size; ++i)
		{
			float3 rgb = kernel::xyzToRgb(make_float3(ImVec4_to_floa4(controlPointsRender[i].extraColor)));
			window->DrawList->AddCircleFilled(controlPointsRender[i].posOpacity, circleRadius_, ImColor(ImVec4(rgb.x, rgb.y, rgb.z, 1.0f)), 16);
			if (selectedControlPoint_ == i)
				window->DrawList->AddCircle(controlPointsRender[i].posOpacity, circleRadius_+1, ImColor(ImVec4(1.0f, 1.0f, 1.0f, 1.0f)), 16);
			else
				window->DrawList->AddCircle(controlPointsRender[i].posOpacity, circleRadius_ + 1, ImColor(ImVec4(0.2f, 0.2f, 0.2f, 1.0f)), 16);
		}		
	} else
	{
		//Draw the control points
		for (const auto& cp : controlPointsRender)
		{
			window->DrawList->AddCircleFilled(cp.posOpacity, circleRadius_, ImColor(ImVec4(0.0f, 1.0f, 0.0f, 1.0f)), 16);
		}
	}
}

ImRect TfEditorLinearOpacity::createControlPointRect(const ImVec2& controlPoint)
{
	return ImRect(ImVec2(controlPoint.x - circleRadius_, controlPoint.y - circleRadius_),
		ImVec2(controlPoint.x + circleRadius_, controlPoint.y + circleRadius_));
}

ImVec2 TfEditorLinearOpacity::screenToEditor(const ImVec2& screenPosition)
{
	ImVec2 editorPosition;
	editorPosition.x = (screenPosition.x - tfEditorRect_.Min.x) / (tfEditorRect_.Max.x - tfEditorRect_.Min.x);
	editorPosition.y = 1.0f - (screenPosition.y - tfEditorRect_.Min.y) / (tfEditorRect_.Max.y - tfEditorRect_.Min.y);

	return editorPosition;
}

ImVec2 TfEditorLinearOpacity::editorToScreen(const ImVec2& editorPosition)
{
	ImVec2 screenPosition;
	screenPosition.x = editorPosition.x * (tfEditorRect_.Max.x - tfEditorRect_.Min.x) + tfEditorRect_.Min.x;
	screenPosition.y = (1.0f - editorPosition.y) * (tfEditorRect_.Max.y - tfEditorRect_.Min.y) + tfEditorRect_.Min.y;

	return screenPosition;
}

TfEditorLinearColor::TfEditorLinearColor()
	: densityAxis_({ 0.0f, 1.0f })
{
	auto red = renderer::TfTexture1D::rgbToLab(make_float3(1.0f, 0.0f, 0.0f));
	auto white = renderer::TfTexture1D::rgbToLab(make_float3(1.0f, 1.0f, 1.0f));

	controlPoints_.emplace_back(0.0f, red.x, red.y, red.z);
	controlPoints_.emplace_back(1.0f, white.x, white.y, white.z);

	colorAxis_.push_back(red);
	colorAxis_.push_back(white);
}

TfEditorLinearColor::~TfEditorLinearColor()
{
	destroy();
}

void TfEditorLinearColor::init(const ImRect& rect, bool showControlPoints)
{
	ImGuiColorEditFlags colorFlags = ImGuiColorEditFlags_Float | ImGuiColorEditFlags_InputHSV | ImGuiColorEditFlags_NoLabel;
	ImGui::ColorEdit3("", &pickedColor_.x, colorFlags);

	showControlPoints_ = showControlPoints;

	//If editor is created for the first time or its size is changed, create CUDA texture.
	if (tfEditorRect_.Min.x == FLT_MAX ||
		!(rect.Min.x == tfEditorRect_.Min.x &&
			rect.Min.y == tfEditorRect_.Min.y &&
			rect.Max.x == tfEditorRect_.Max.x &&
			rect.Max.y == tfEditorRect_.Max.y))
	{
		destroy();
		tfEditorRect_ = rect;

		auto colorMapWidth = tfEditorRect_.Max.x - tfEditorRect_.Min.x;
		auto colorMapHeight = tfEditorRect_.Max.y - tfEditorRect_.Min.y;
		glGenTextures(1, &colorMapImage_);

		glBindTexture(GL_TEXTURE_2D, colorMapImage_);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, colorMapWidth, colorMapHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
		CUMAT_SAFE_CALL(cudaGraphicsGLRegisterImage(&resource_, colorMapImage_, GL_TEXTURE_2D, cudaGraphicsMapFlagsWriteDiscard));

		glBindTexture(GL_TEXTURE_2D, 0);

		cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
		CUMAT_SAFE_CALL(cudaMallocArray(&contentArray_, &channelDesc, colorMapWidth, colorMapHeight, cudaArraySurfaceLoadStore));

		cudaResourceDesc resDesc;
		memset(&resDesc, 0, sizeof(resDesc));
		resDesc.resType = cudaResourceTypeArray;

		resDesc.res.array.array = contentArray_;
		CUMAT_SAFE_CALL(cudaCreateSurfaceObject(&content_, &resDesc));
	}
}

void TfEditorLinearColor::updateControlPoints(const std::vector<float>& densityAxis, const std::vector<float3>& colorAxis)
{
	selectedControlPointForMove_ = -1;
	selectedControlPointForColor_ = -1;
	controlPoints_.clear();
	densityAxis_ = densityAxis;
	colorAxis_ = colorAxis;

	int size = densityAxis.size();
	for (int i = 0; i < size; ++i)
	{
		controlPoints_.emplace_back(densityAxis_[i], colorAxis_[i].x, colorAxis_[i].y, colorAxis_[i].z);
	}
}

void TfEditorLinearColor::handleIO()
{
	isChanged_ = false;

	auto mousePosition = ImGui::GetMousePos();

	if (selectedControlPointForColor_ >= 0)
	{
		auto& cp = controlPoints_[selectedControlPointForColor_];

		float3 pickedColorLab;
		ImGui::ColorConvertHSVtoRGB(pickedColor_.x, pickedColor_.y, pickedColor_.z, pickedColorLab.x, pickedColorLab.y, pickedColorLab.z);
		pickedColorLab = renderer::TfTexture1D::rgbToLab(pickedColorLab);

		if (cp.y != pickedColorLab.x || cp.z != pickedColorLab.y ||
			cp.w != pickedColorLab.z)
		{
			cp.y = pickedColorLab.x;
			cp.z = pickedColorLab.y;
			cp.w = pickedColorLab.z;
			isChanged_ = true;
		}
	}

	//Early leave if mouse is not on color editor.
	if (!TfEditorLinear::testIntersectionRectPoint(tfEditorRect_, mousePosition) && selectedControlPointForMove_ == -1)
	{
		return;
	}

	//0=left, 1=right, 2=middle
	bool isLeftDoubleClicked = ImGui::IsMouseDoubleClicked(0);
	bool isLeftClicked = ImGui::IsMouseDown(0);
	bool isRightClicked = ImGui::IsMouseClicked(1);
	bool isLeftReleased = ImGui::IsMouseReleased(0);

	if (isLeftDoubleClicked)
	{
		isChanged_ = true;

		float3 pickedColorLab;
		ImGui::ColorConvertHSVtoRGB(pickedColor_.x, pickedColor_.y, pickedColor_.z, pickedColorLab.x, pickedColorLab.y, pickedColorLab.z);
		pickedColorLab = renderer::TfTexture1D::rgbToLab(pickedColorLab);
		controlPoints_.emplace_back(screenToEditor(mousePosition.x), pickedColorLab.x, pickedColorLab.y, pickedColorLab.z);
	}
	else if (isLeftClicked)
	{
		//Move selected point.
		if (selectedControlPointForMove_ >= 0)
		{
			isChanged_ = true;

			float center = std::min(std::max(mousePosition.x, tfEditorRect_.Min.x), tfEditorRect_.Max.x);

			controlPoints_[selectedControlPointForMove_].x = screenToEditor(center);
		}
		//Check whether new point is selected.
		else
		{
			int size = controlPoints_.size();
			int idx;
			for (idx = 0; idx < size; ++idx)
			{
				auto cp = createControlPointRect(editorToScreen(controlPoints_[idx].x));
				if (TfEditorLinear::testIntersectionRectPoint(cp, mousePosition))
				{
					selectedControlPointForColor_ = selectedControlPointForMove_ = idx;

					auto colorRgb = renderer::TfTexture1D::labToRgb(make_float3(controlPoints_[selectedControlPointForMove_].y,
						controlPoints_[selectedControlPointForMove_].z,
						controlPoints_[selectedControlPointForMove_].w));

					ImGui::ColorConvertRGBtoHSV(colorRgb.x, colorRgb.y, colorRgb.z, pickedColor_.x, pickedColor_.y, pickedColor_.z);
					break;
				}
			}

			//In case of no hit on any control point, unselect for color pick as well.
			if (idx == size)
			{
				selectedControlPointForColor_ = -1;
			}
		}
	}
	else if (isRightClicked)
	{
		int size = controlPoints_.size();
		int idx;
		for (idx = 0; idx < size; ++idx)
		{
			auto cp = createControlPointRect(editorToScreen(controlPoints_[idx].x));
			if (TfEditorLinear::testIntersectionRectPoint(cp, mousePosition) && controlPoints_.size() > 1)
			{
				isChanged_ = true;

				controlPoints_.erase(controlPoints_.begin() + idx);
				selectedControlPointForColor_ = selectedControlPointForMove_ = -1;
				break;
			}
		}
	}
	else if (isLeftReleased)
	{
		selectedControlPointForMove_ = -1;
	}
}

void TfEditorLinearColor::render()
{
	ImGuiWindow* window = ImGui::GetCurrentWindow();

	//Copy the control points and sort them. We don't sort original one in order not to mess up with control point indices.
	auto controlPointsRender = controlPoints_;
	std::sort(controlPointsRender.begin(), controlPointsRender.end(),
		[](const ImVec4& cp1, const ImVec4& cp2)
		{
			return cp1.x < cp2.x;
		});

	//Fill densityAxis_ and colorAxis_.
	densityAxis_.clear();
	colorAxis_.clear();
	for (auto& cp : controlPointsRender)
	{
		densityAxis_.push_back(cp.x);
		colorAxis_.push_back(make_float3(cp.y, cp.z, cp.w));
	}

	auto colorMapWidth = tfEditorRect_.Max.x - tfEditorRect_.Min.x;
	auto colorMapHeight = tfEditorRect_.Max.y - tfEditorRect_.Min.y;

	//Write to color map texture.
	tfTexture_.updateIfChanged({ 0.0f, 1.0f }, { 0.0f, 1.0f }, {}, densityAxis_, colorAxis_);
	kernel::fillColorMap(content_, tfTexture_.getTextureObjectRGB(), colorMapWidth, colorMapHeight);

	//Draw color interpolation between control points.
	cudaArray_t texturePtr = nullptr;
	CUMAT_SAFE_CALL(cudaGraphicsMapResources(1, &resource_, 0));
	CUMAT_SAFE_CALL(cudaGraphicsSubResourceGetMappedArray(&texturePtr, resource_, 0, 0));
	CUMAT_SAFE_CALL(cudaMemcpyArrayToArray(texturePtr, 0, 0, contentArray_, 0, 0, colorMapWidth * colorMapHeight * 4, cudaMemcpyDeviceToDevice));
	CUMAT_SAFE_CALL(cudaGraphicsUnmapResources(1, &resource_, 0));

	window->DrawList->AddImage((void*)colorMapImage_, tfEditorRect_.Min, tfEditorRect_.Max);

	if (showControlPoints_)
	{
		//Draw the control points
		int cpIndex = 0;
		for (const auto& cp : controlPoints_)
		{
			//If this is the selected control point, use different color.
			auto rect = createControlPointRect(editorToScreen(cp.x));
			if (selectedControlPointForColor_ == cpIndex++)
			{
				window->DrawList->AddRect(rect.Min, rect.Max, ImColor(ImVec4(1.0f, 0.8f, 0.1f, 1.0f)), 16.0f, ImDrawCornerFlags_All, 3.0f);
			}
			else
			{
				window->DrawList->AddRect(rect.Min, rect.Max, ImColor(ImVec4(1.0f, 1.0f, 1.0f, 1.0f)), 16.0f, ImDrawCornerFlags_All, 2.0f);
			}
		}
	}
}

void TfEditorLinearColor::destroy()
{
	if (colorMapImage_)
	{
		glDeleteTextures(1, &colorMapImage_);
		colorMapImage_ = 0;
	}
	if (content_)
	{
		CUMAT_SAFE_CALL(cudaDestroySurfaceObject(content_));
		content_ = 0;
	}
	if (contentArray_)
	{
		CUMAT_SAFE_CALL(cudaFreeArray(contentArray_));
		contentArray_ = nullptr;
	}
}

ImRect TfEditorLinearColor::createControlPointRect(float x)
{
	return ImRect(ImVec2(x - 0.5f * cpWidth_, tfEditorRect_.Min.y),
		ImVec2(x + 0.5f * cpWidth_, tfEditorRect_.Max.y));
}

float TfEditorLinearColor::screenToEditor(float screenPositionX)
{
	float editorPositionX;
	editorPositionX = (screenPositionX - tfEditorRect_.Min.x) / (tfEditorRect_.Max.x - tfEditorRect_.Min.x);

	return editorPositionX;
}

float TfEditorLinearColor::editorToScreen(float editorPositionX)
{
	float screenPositionX;
	screenPositionX = editorPositionX * (tfEditorRect_.Max.x - tfEditorRect_.Min.x) + tfEditorRect_.Min.x;

	return screenPositionX;
}

TfEditorLinear::TfEditorLinear(bool extraColorPerOpacityPoint)
	: editorOpacity_(extraColorPerOpacityPoint)
{
}

void TfEditorLinear::init(const ImRect& tfEditorOpacityRect, const ImRect& tfEditorColorRect, bool showColorControlPoints)
{
	editorOpacity_.init(tfEditorOpacityRect);
	editorColor_.init(tfEditorColorRect, showColorControlPoints);
}

void TfEditorLinear::handleIO()
{
	editorOpacity_.handleIO();
	editorColor_.handleIO();
}

void TfEditorLinear::render()
{
	editorOpacity_.render();
	editorColor_.render();
}

void TfEditorLinear::saveToFile(const std::string& path, float minDensity, float maxDensity) const
{
	const auto& densityAxisOpacity = editorOpacity_.getDensityAxis();
	const auto& opacityAxis = editorOpacity_.getOpacityAxis();
	const auto& densityAxisColor = editorColor_.getDensityAxis();
	const auto& colorAxis = editorColor_.getColorAxis();

	assert(densityAxisOpacity.size() == opacityAxis.size());
	assert(densityAxisColor.size() == colorAxis.size());

	nlohmann::json json;
	json["densityAxisOpacity"] = editorOpacity_.getDensityAxis();
	json["opacityAxis"] = editorOpacity_.getOpacityAxis();
	json["densityAxisColor"] = editorColor_.getDensityAxis();
	json["colorAxis"] = editorColor_.getColorAxis();
	json["minDensity"] = minDensity;
	json["maxDensity"] = maxDensity;

	std::ofstream out(path);
	out << json;
	out.close();
}

void TfEditorLinear::loadFromFile(const std::string& path, float& minDensity, float& maxDensity)
{
	if (path.size() > 4 && path.substr(path.size()-4, 4) == ".xml")
	{
		//try to load colormap (no opacities)
		try
		{
			auto colormap = loadColormapFromXML(path);
			//convert rgb to lab
			for (auto& color : colormap.second)
				color = renderer::TfTexture1D::rgbToLab(color);
			editorColor_.updateControlPoints(colormap.first, colormap.second);
		} catch (const std::runtime_error& ex)
		{
			std::cerr << "Unable to load colormap from xml: " << ex.what() << std::endl;
		}
	}
	
	try {
		nlohmann::json json;
		std::ifstream file(path);
		file >> json;
		file.close();

		std::vector<float> densityAxisOpacity = json["densityAxisOpacity"];
		std::vector<float> opacityAxis = json["opacityAxis"];
		std::vector<float> densityAxisColor = json["densityAxisColor"];
		std::vector<float3> colorAxis = json["colorAxis"];
		minDensity = json["minDensity"];
		maxDensity = json["maxDensity"];

		assert(densityAxisOpacity.size() == opacityAxis.size());
		assert(densityAxisColor.size() == colorAxis.size());

		editorOpacity_.updateControlPoints(densityAxisOpacity, opacityAxis);
		editorColor_.updateControlPoints(densityAxisColor, colorAxis);
	}
	catch (const nlohmann::json::exception& ex)
	{
		std::cerr << "Unable to load dvr transfer function from json: " << ex.what() << std::endl;
	}
}

nlohmann::json TfEditorLinear::toJson() const
{
	const auto& densityAxisOpacity = editorOpacity_.getDensityAxis();
	const auto& opacityAxis = editorOpacity_.getOpacityAxis();
	std::vector<float4> opacityExtraColorAxis;
	if (editorOpacity_.hasExtraColor())
		opacityExtraColorAxis = editorOpacity_.getExtraColorAxis();
	const auto& densityAxisColor = editorColor_.getDensityAxis();
	const auto& colorAxis = editorColor_.getColorAxis();
	if (editorOpacity_.hasExtraColor())
		return {
			{"densityAxisOpacity", nlohmann::json(densityAxisOpacity)},
			{"opacityAxis", nlohmann::json(opacityAxis)},
			{"opacityExtraColorAxis", nlohmann::json(opacityExtraColorAxis)},
			{"densityAxisColor", nlohmann::json(densityAxisColor)},
			{"colorAxis", nlohmann::json(colorAxis)}
		};
	else
		return {
			{"densityAxisOpacity", nlohmann::json(densityAxisOpacity)},
			{"opacityAxis", nlohmann::json(opacityAxis)},
			{"densityAxisColor", nlohmann::json(densityAxisColor)},
			{"colorAxis", nlohmann::json(colorAxis)}
		};
}

void TfEditorLinear::fromJson(const nlohmann::json& s)
{
	const std::vector<float> densityAxisOpacity = s.at("densityAxisOpacity");
	const std::vector<float> opacityAxis = s.at("opacityAxis");
	std::vector<float4> opacityExtraColorAxis;
	if (editorOpacity_.hasExtraColor())
		opacityExtraColorAxis = s.at("opacityExtraColorAxis").get<std::vector<float4>>();
	const std::vector<float> densityAxisColor = s.at("densityAxisColor");
	const std::vector<float3> colorAxis = s.at("colorAxis");
	editorOpacity_.updateControlPoints(densityAxisOpacity, opacityAxis, opacityExtraColorAxis);
	editorColor_.updateControlPoints(densityAxisColor, colorAxis);
}

bool TfEditorLinear::testIntersectionRectPoint(const ImRect& rect, const ImVec2& point)
{
	return (rect.Min.x <= point.x &&
		rect.Max.x >= point.x &&
		rect.Min.y <= point.y &&
		rect.Max.y >= point.y);
}

std::pair<std::vector<float>, std::vector<float3>> TfEditorLinear::loadColormapFromXML(const std::string& path)
{
	using namespace tinyxml2;
	XMLDocument doc;
	doc.LoadFileThrow(path.c_str());
	const XMLElement* element = doc.FirstChildElementThrow("ColorMaps")
		->FirstChildElementThrow("ColorMap");

	std::vector<float> positions;
	std::vector<float3> rgbColors;
	const XMLElement* point = element->FirstChildElementThrow("Point");
	do
	{
		positions.push_back(point->FloatAttribute("x"));
		rgbColors.push_back(make_float3(
			point->FloatAttribute("r"),
			point->FloatAttribute("g"),
			point->FloatAttribute("b")
		));
		point = point->NextSiblingElement("Point");
	} while (point != nullptr);

	return std::make_pair(positions, rgbColors);
}

void TfEditorLinear::fromMultiIso(const TfEditorMultiIso* multiIso, float peakWidth)
{
	const auto& isoPos = multiIso->getControlPointPositions();
	auto isoVal = multiIso->getControlPointColors();
	if (isoVal.empty()) return;
	for (auto& val : isoVal)
	{
		val = make_float4(
			kernel::xyzToLab(make_float3(val)),
			val.w
		);
	}
	
	//set color control points
	std::vector<float> colorDensities;
	std::vector<float3> colorValues;
	colorDensities.push_back(0);
	colorValues.push_back(make_float3(isoVal[0]));
	for (size_t i=0; i<isoVal.size(); ++i)
	{
		colorDensities.push_back(isoPos[i]);
		colorValues.push_back(make_float3(isoVal[i]));
	}
	colorDensities.push_back(1);
	colorValues.push_back(make_float3(isoVal[isoVal.size()-1]));
	editorColor_.updateControlPoints(colorDensities, colorValues);
	
	std::vector<float> opacityDensities;
	std::vector<float> opacityValues;
	for (size_t i = 0; i < isoVal.size(); ++i)
	{
		float pos = isoPos[i];
		float minPos, maxPos;
		if (i == 0)
			minPos = 0;
		else
			minPos = 0.5 * (pos + isoPos[i - 1]);
		if (i == isoVal.size() - 1)
			maxPos = 1;
		else
			maxPos = 0.5 * (pos + isoPos[i + 1]);
		minPos = std::max(minPos, pos - peakWidth);
		maxPos = std::min(maxPos, pos + peakWidth);
		opacityDensities.push_back(minPos);
		opacityValues.push_back(0);
		opacityDensities.push_back(pos);
		opacityValues.push_back(isoVal[i].w);
		opacityDensities.push_back(maxPos);
		opacityValues.push_back(0);
	}
	editorOpacity_.updateControlPoints(opacityDensities, opacityValues);
}

TfEditorMultiIso::TfEditorMultiIso()
{
	controlPoints_.push_back({
	ImVec2(0.5, 0.7),
	kernel::rgbToXyz(make_float3(0.2f, 0.3f, 1.0f)) });

	for (auto& cp : controlPoints_)
	{
		positionAxis_.push_back(cp.pos.x);
		colorAxis_.push_back(make_float4(cp.color, cp.pos.y));
	}
}

void TfEditorMultiIso::init(const ImRect& tfEditorRect)
{
	tfEditorRect_ = tfEditorRect;

	ImGuiColorEditFlags colorFlags = ImGuiColorEditFlags_Float | ImGuiColorEditFlags_InputRGB | ImGuiColorEditFlags_NoLabel;
	if (ImGui::ColorEdit3("", &pickedColor_.x, colorFlags))
	{
		if (selectedControlPoint_ >= 0)
		{
			float3 rgb = make_float3(pickedColor_.x, pickedColor_.y, pickedColor_.z);
			float3 xyz = kernel::rgbToXyz(rgb);
			//float3 rgb2 = kernel::xyzToRgb(xyz);
			//std::cout << "rgb: " << rgb.x << "," << rgb.y << "," << rgb.z <<
			//	"; xyz: " << xyz.x << "," << xyz.y << "," << xyz.z <<
			//	"; rgb2: " << rgb2.x << "," << rgb2.y << "," << rgb2.z <<
			//	std::endl;
			controlPoints_[selectedControlPoint_].color = xyz;
			isChanged_ = true;
		}
	}
}

void TfEditorMultiIso::handleIO()
{
	auto mousePosition = ImGui::GetMousePos();

	//Early leave if mouse is not on opacity editor and no control point is selected.
	if (!TfEditorLinear::testIntersectionRectPoint(tfEditorRect_, mousePosition) && selectedControlPoint_ == -1)
	{
		return;
	}

	//0=left, 1=right, 2=middle
	bool isLeftDoubleClicked = ImGui::IsMouseDoubleClicked(0);
	bool isLeftClicked = ImGui::IsMouseDown(0);
	bool isRightClicked = ImGui::IsMouseClicked(1);
	bool isLeftReleased = ImGui::IsMouseReleased(0);

	if (isLeftDoubleClicked)
	{
		//create control point
		isChanged_ = true;
		controlPoints_.push_back({
			screenToEditor(mousePosition),
			kernel::rgbToXyz(make_float3(1,1,1)) }
		);
		pickedColor_ = ImVec4(1,1,1,1);
		selectedControlPoint_ = controlPoints_.size() - 1;
	}
	else if (isLeftClicked)
	{
		//Move selected point.
		if (draggedControlPoint_ >= 0)
		{
			isChanged_ = true;

			ImVec2 center(std::min(std::max(mousePosition.x, tfEditorRect_.Min.x), tfEditorRect_.Max.x),
				std::min(std::max(mousePosition.y, tfEditorRect_.Min.y), tfEditorRect_.Max.y));

			controlPoints_[draggedControlPoint_].pos = screenToEditor(center);
		}
		//Check whether new point is selected.
		else
		{
			int size = controlPoints_.size();
			for (int idx = 0; idx < size; ++idx)
			{
				auto cp = createControlPointRect(editorToScreen(controlPoints_[idx].pos));
				if (TfEditorLinear::testIntersectionRectPoint(cp, mousePosition))
				{
					selectedControlPoint_ = idx;
					draggedControlPoint_ = idx;
					float3 rgb = kernel::xyzToRgb(controlPoints_[selectedControlPoint_].color);
					pickedColor_ = ImVec4(rgb.x, rgb.y, rgb.z, 1.0f);
					break;
				}
			}
		}
	}
	else if (isRightClicked)
	{
		int size = controlPoints_.size();
		for (int idx = 0; idx < size; ++idx)
		{
			auto cp = createControlPointRect(editorToScreen(controlPoints_[idx].pos));
			if (TfEditorLinear::testIntersectionRectPoint(cp, mousePosition) && controlPoints_.size() > 1)
			{
				isChanged_ = true;

				controlPoints_.erase(controlPoints_.begin() + idx);
				selectedControlPoint_ = -1;
				draggedControlPoint_ = -1;
				break;
			}
		}
	}
	else if (isLeftReleased)
	{
		draggedControlPoint_ = -1;
	}
}

void TfEditorMultiIso::render()
{
	//Draw the bounding rectangle.
	ImGuiWindow* window = ImGui::GetCurrentWindow();
	window->DrawList->AddRect(tfEditorRect_.Min, tfEditorRect_.Max, ImColor(ImVec4(0.3f, 0.3f, 0.3f, 1.0f)), 0.0f, ImDrawCornerFlags_All, 1.0f);

	//Copy the control points and sort them. We don't sort original one in order not to mess up with control point indices.
	auto controlPointsRender = controlPoints_;
	std::sort(controlPointsRender.begin(), controlPointsRender.end(),
		[](const PointData& p1, const PointData& p2)
	{
		return p1.pos.x < p2.pos.x;
	});

	//Fill densityAxis_ and opacityAxis_ and convert coordinates from editor space to screen space.
	positionAxis_.clear();
	colorAxis_.clear();
	for (auto& cp : controlPointsRender)
	{
		positionAxis_.push_back(cp.pos.x);
		colorAxis_.push_back(make_float4(cp.color, cp.pos.y));
	}

	//Draw the control points
	ImVec2 zeroPos = editorToScreen(ImVec2(0, 0));
	int index = 0;
	for (const auto& cp : controlPointsRender)
	{
		ImVec2 pos = editorToScreen(cp.pos);
		ImVec2 bottom(pos.x, zeroPos.y);
		float3 rgb = kernel::xyzToRgb(cp.color);
		window->DrawList->AddLine(bottom, pos, 
			ImColor(0.0f, 0.0f, 0.0f));
		window->DrawList->AddCircleFilled(pos, circleRadius_, 
			ImColor(rgb.x, rgb.y, rgb.z));
		if (index == selectedControlPoint_)
			window->DrawList->AddCircle(pos, circleRadius_,
				ImColor(1.0f, 1.0f, 1.0f), 12, 2);
		else
			window->DrawList->AddCircle(pos, circleRadius_,
				ImColor(0, 0, 0), 12, 2);
		index++;
	}
}

void TfEditorMultiIso::saveToFile(const std::string& path, float minDensity, float maxDensity) const
{
	nlohmann::json json = toJson();
	json["minDensity"] = minDensity;
	json["maxDensity"] = maxDensity;
	std::ofstream out(path);
	out << json;
	out.close();
}

void TfEditorMultiIso::loadFromFile(const std::string& path, float& minDensity, float& maxDensity)
{
	try {
		nlohmann::json json;
		std::ifstream file(path);
		file >> json;
		file.close();

		fromJson(json);
		minDensity = json["minDensity"];
		maxDensity = json["maxDensity"];
	} catch (const nlohmann::json::exception& ex)
	{
		std::cerr << "Unable to load multi-iso transfer function from json: " << ex.what() << std::endl;
	}
}

nlohmann::json TfEditorMultiIso::toJson() const
{
	return {
		{"positions", nlohmann::json(positionAxis_)},
		{"colors", nlohmann::json(colorAxis_)}
	};
}

void TfEditorMultiIso::fromJson(const nlohmann::json& s)
{
	positionAxis_ = s.at("positions").get<std::vector<float>>();
	colorAxis_ = s.at("colors").get<std::vector<float4>>();
	assert(positionAxis_.size() == colorAxis_.size());
	
	selectedControlPoint_ = -1;
	draggedControlPoint_ = -1;
	controlPoints_.clear();
	int size = positionAxis_.size();
	for (int i = 0; i < size; ++i)
	{
		controlPoints_.push_back(PointData{
			ImVec2(positionAxis_[i], colorAxis_[i].w),
			make_float3(colorAxis_[i])
		});
	}
}

bool TfEditorMultiIso::getIsChanged()
{
	bool changed = isChanged_;
	isChanged_ = false;
	return changed;
}

ImRect TfEditorMultiIso::createControlPointRect(const ImVec2& controlPoint)
{
	return ImRect(ImVec2(controlPoint.x - circleRadius_, controlPoint.y - circleRadius_),
		ImVec2(controlPoint.x + circleRadius_, controlPoint.y + circleRadius_));
}

ImVec2 TfEditorMultiIso::screenToEditor(const ImVec2& screenPosition)
{
	ImVec2 editorPosition;
	editorPosition.x = (screenPosition.x - tfEditorRect_.Min.x) / (tfEditorRect_.Max.x - tfEditorRect_.Min.x);
	editorPosition.y = 1.0f - (screenPosition.y - tfEditorRect_.Min.y) / (tfEditorRect_.Max.y - tfEditorRect_.Min.y);

	return editorPosition;
}

ImVec2 TfEditorMultiIso::editorToScreen(const ImVec2& editorPosition)
{
	ImVec2 screenPosition;
	screenPosition.x = editorPosition.x * (tfEditorRect_.Max.x - tfEditorRect_.Min.x) + tfEditorRect_.Min.x;
	screenPosition.y = (1.0f - editorPosition.y) * (tfEditorRect_.Max.y - tfEditorRect_.Min.y) + tfEditorRect_.Min.y;

	return screenPosition;
}
