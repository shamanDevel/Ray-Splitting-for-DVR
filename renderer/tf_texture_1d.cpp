#include "tf_texture_1d.h"

#include <cuMat/src/Context.h>
#include <algorithm>

#include "renderer_color.cuh"

BEGIN_RENDERER_NAMESPACE

TfTexture1D::TfTexture1D(int size)
{
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat); //XYZA

	gpuData_.cudaArraySize_ = size;
	CUMAT_SAFE_CALL(cudaMallocArray(&gpuData_.cudaArrayRGB_, &channelDesc, size, 0, cudaArraySurfaceLoadStore));
	CUMAT_SAFE_CALL(cudaMallocArray(&gpuData_.cudaArrayXYZ_, &channelDesc, size, 0, cudaArraySurfaceLoadStore));

	cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeArray;

	cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(texDesc));
	texDesc.addressMode[0] = cudaTextureAddressMode(cudaAddressModeClamp);
	texDesc.filterMode = cudaTextureFilterMode(cudaFilterModeLinear);
	texDesc.normalizedCoords = 1;
	
	//Create the surface and texture object.
	resDesc.res.array.array = gpuData_.cudaArrayRGB_;
	CUMAT_SAFE_CALL(cudaCreateSurfaceObject(&gpuData_.surfaceObjectRGB_, &resDesc));
	CUMAT_SAFE_CALL(cudaCreateTextureObject(&gpuData_.textureObjectRGB_, &resDesc, &texDesc, nullptr));
	resDesc.res.array.array = gpuData_.cudaArrayXYZ_;
	CUMAT_SAFE_CALL(cudaCreateSurfaceObject(&gpuData_.surfaceObjectXYZ_, &resDesc));
	CUMAT_SAFE_CALL(cudaCreateTextureObject(&gpuData_.textureObjectXYZ_, &resDesc, &texDesc, nullptr));

	//create storage for TfGpuSettings
	tfGpuSettings_ = new ::kernel::TfGpuSettings();
}

TfTexture1D::~TfTexture1D()
{
	destroy();

	CUMAT_SAFE_CALL(cudaDestroyTextureObject(gpuData_.textureObjectRGB_));
	CUMAT_SAFE_CALL(cudaDestroySurfaceObject(gpuData_.surfaceObjectRGB_));
	CUMAT_SAFE_CALL(cudaFreeArray(gpuData_.cudaArrayRGB_));

	CUMAT_SAFE_CALL(cudaDestroyTextureObject(gpuData_.textureObjectXYZ_));
	CUMAT_SAFE_CALL(cudaDestroySurfaceObject(gpuData_.surfaceObjectXYZ_));
	CUMAT_SAFE_CALL(cudaFreeArray(gpuData_.cudaArrayXYZ_));

	delete tfGpuSettings_;
}

bool TfTexture1D::updateIfChanged(const std::vector<float>& densityAxisOpacity, const std::vector<float>& opacityAxis,
	const std::vector<float4>& opacityExtraColorAxis0,
	const std::vector<float>& densityAxisColor, const std::vector<float3>& colorAxis)
{
	std::vector<float4> opacityExtraColorAxis = opacityExtraColorAxis0;
	if (opacityExtraColorAxis.empty())
		opacityExtraColorAxis.resize(opacityAxis.size(), make_float4(0, 0, 0, 0));
	
	//First check if we have the same values for axes and values. If so, do not allocate new memory or create texture object again. 

	//std::vector cannot compare const vs non-const vectors. I don't want to use const_cast here.
	//Also, float3 has an equal operator which returns element-wise boolean. Since I cannot overload it, std::equal is the best option.
	bool changed = densityAxisOpacity.size() != densityAxisOpacity_.size() ||
		opacityAxis.size() != opacityAxis_.size() ||
		opacityExtraColorAxis.size() != opacityExtraColorAxis_.size() ||
		densityAxisColor.size() != densityAxisColor_.size() ||
		colorAxis.size() != colorAxis_.size();

	changed = changed ||
		!std::equal(densityAxisOpacity.cbegin(), densityAxisOpacity.cend(), densityAxisOpacity_.cbegin()) ||
		!std::equal(opacityAxis.cbegin(), opacityAxis.cend(), opacityAxis_.cbegin()) ||
		!std::equal(opacityExtraColorAxis.cbegin(), opacityExtraColorAxis.cend(), opacityExtraColorAxis_.cbegin(), [](const float4& l, const float4& r)
			{
				return l.x == r.x && l.y == r.y && l.z == r.z && l.w == r.w;
			}) ||
		!std::equal(densityAxisColor.cbegin(), densityAxisColor.cend(), densityAxisColor_.cbegin()) ||
		!std::equal(colorAxis.cbegin(), colorAxis.cend(), colorAxis_.cbegin(), [](const float3& l, const float3& r)
			{
				return l.x == r.x && l.y == r.y && l.z == r.z;
			});

	if (changed)
	{
		destroy();

		gpuData_.sizeOpacity_ = cuMat::internal::narrow_cast<int>(densityAxisOpacity.size());
		assert(gpuData_.sizeOpacity_ >= 1);

		gpuData_.sizeColor_ = cuMat::internal::narrow_cast<int>(densityAxisColor.size());
		assert(gpuData_.sizeColor_ >= 1);

		//Transfer from host to device
		CUMAT_SAFE_CALL(cudaMalloc(&gpuData_.densityAxisOpacity_, gpuData_.sizeOpacity_ * sizeof(float)));
		CUMAT_SAFE_CALL(cudaMemcpy(gpuData_.densityAxisOpacity_, densityAxisOpacity.data(), gpuData_.sizeOpacity_ * sizeof(float), cudaMemcpyHostToDevice));

		CUMAT_SAFE_CALL(cudaMalloc(&gpuData_.opacityAxis_, gpuData_.sizeOpacity_ * sizeof(float)));
		CUMAT_SAFE_CALL(cudaMemcpy(gpuData_.opacityAxis_, opacityAxis.data(), gpuData_.sizeOpacity_ * sizeof(float), cudaMemcpyHostToDevice));

		CUMAT_SAFE_CALL(cudaMalloc(&gpuData_.densityAxisColor_, gpuData_.sizeColor_ * sizeof(float)));
		CUMAT_SAFE_CALL(cudaMemcpy(gpuData_.densityAxisColor_, densityAxisColor.data(), gpuData_.sizeColor_ * sizeof(float), cudaMemcpyHostToDevice));

		CUMAT_SAFE_CALL(cudaMalloc(&gpuData_.colorAxis_, gpuData_.sizeColor_ * sizeof(float3)));
		CUMAT_SAFE_CALL(cudaMemcpy(gpuData_.colorAxis_, colorAxis.data(), gpuData_.sizeColor_ * sizeof(float3), cudaMemcpyHostToDevice));

		densityAxisOpacity_ = densityAxisOpacity;
		opacityAxis_ = opacityAxis;
		opacityExtraColorAxis_ = opacityExtraColorAxis;
		densityAxisColor_ = densityAxisColor;
		colorAxis_ = colorAxis;

		computeCudaTexture(gpuData_);
		assembleGpuControlPoints();
	}

	return changed;
}

void TfTexture1D::assembleGpuControlPoints()
{
	struct Point
	{
		float pos;
		float4 val;
		float4 extraColor;
	};
	std::vector<Point> points;

	auto colorValues = colorAxis_;
	auto colorPositions = densityAxisColor_;
	auto opacityValues = opacityAxis_;
	auto opacityExtraColors = opacityExtraColorAxis_;
	auto opacityPositions = densityAxisOpacity_;

	//add control points at t=0 if not existing
	//but not directly 0 and 1, better -1 and 2 as this is better for the iso-intersections
	if (colorPositions.front()>0)
	{
		colorPositions.insert(colorPositions.begin(), -1.0f);
		colorValues.insert(colorValues.begin(), colorValues.front());
	}
	if (opacityPositions.front() > 0)
	{
		opacityPositions.insert(opacityPositions.begin(), -1.0f);
		opacityExtraColors.insert(opacityExtraColors.begin(), make_float4(0,0,0,0));
		opacityValues.insert(opacityValues.begin(), opacityValues.front());
	}
	//same with t=1
	if (colorPositions.back()<1)
	{
		colorPositions.push_back(2);
		colorValues.push_back(colorValues.back());
	}
	if (opacityPositions.back() < 1)
	{
		opacityPositions.push_back(2);
		opacityExtraColors.push_back(make_float4(0, 0, 0, 0));
		opacityValues.push_back(opacityValues.back());
	}
	
	//first point at pos=0
	points.push_back({ 0.f, 
		make_float4(kernel::labToXyz(colorValues[0]), opacityValues[0]) });

	//assemble the points
	int64_t iOpacity = 0; //next indices
	int64_t iColor = 0;
	while (iOpacity<opacityPositions.size()-1 && iColor<colorPositions.size()-1)
	{
		if (opacityPositions[iOpacity+1] < colorPositions[iColor+1])
		{
			//next point is an opacity point
			points.push_back({
				opacityPositions[iOpacity + 1] ,
				make_float4(
					::kernel::labToXyz(lerp(
						colorValues[iColor], 
						colorValues[iColor+1], 
						(opacityPositions[iOpacity+1]-colorPositions[iColor])/(colorPositions[iColor+1]-colorPositions[iColor]))),
					opacityValues[iOpacity+1]) ,
				opacityExtraColors[iOpacity+1]});
			iOpacity++;
		}
		else
		{
			points.push_back({
				colorPositions[iColor + 1],
				make_float4(
					::kernel::labToXyz(colorValues[iColor + 1]),
					lerp(
						opacityValues[iOpacity],
						opacityValues[iOpacity + 1],
						(colorPositions[iColor + 1] - opacityPositions[iOpacity]) / (opacityPositions[iOpacity + 1] - opacityPositions[iOpacity]))),
				make_float4(0,0,0,0)
				});
			iColor++;
		}
		
	}
	//std::cout << opacityPositions.size() << " opacity and " << colorPositions.size() <<
	//	" control points combined into " << points.size() << " joint points" << std::endl;

	//filter the points
	//removes all color control points in areas of zero density
	int numPointsRemoved = 0;
	constexpr float EPS = 1e-7;
	for (int64_t i=0; i<static_cast<int64_t>(points.size())-2; )
	{
		if (points[i].val.w < EPS && points[i + 1].val.w < EPS && points[i + 2].val.w < EPS &&
			points[i].extraColor.w < EPS && points[i + 1].extraColor.w < EPS && points[i + 2].extraColor.w < EPS) {
			points.erase(points.begin() + (i + 1));
			numPointsRemoved++;
		}
		else
			i++;
	}
	//std::cout << numPointsRemoved << " points removed with zero density" << std::endl;

	if (points[1].val.w>0)
	{
		//opacity goes from p[0] where the opacity should be >0 directly to something positive.
		//Since the points are moved by min/max density, this would also lead to a shift.
		//We therefore add another zero-point at the beginning that will be expanded to a
		//plateau of zeros upon min/max transformation in renderer.cpp
		points.insert(points.begin(), points[0]);
		points[0].extraColor = make_float4(0, 0, 0, 0);
	}
	
	if (points.size() > TF_MAX_CONTROL_POINTS)
	{
		std::cerr << "Still too many control points, can't copy to GPU" << std::endl;
	}
	else
	{
		//update gpu structure
		tfGpuSettings_->numPoints = points.size();
		for (size_t i=0; i<points.size(); ++i)
		{
			tfGpuSettings_->positions[i] = points[i].pos;
			tfGpuSettings_->valuesDvr[i] = points[i].val;
			tfGpuSettings_->valuesIso[i] = points[i].extraColor;
		}
	}
}

float3 TfTexture1D::rgbToXyz(const float3& rgb)
{
	return kernel::rgbToXyz(rgb);
}

float3 TfTexture1D::rgbToLab(const float3& rgb)
{
	return kernel::rgbToLab(rgb);
}

float3 TfTexture1D::xyzToRgb(const float3& xyz)
{
	return kernel::xyzToRgb(xyz);
}

float3 TfTexture1D::labToRgb(const float3& lab)
{
	return kernel::labToRgb(lab);
}

void TfTexture1D::destroy()
{
	if (gpuData_.densityAxisOpacity_)
	{
		CUMAT_SAFE_CALL(cudaFree(gpuData_.densityAxisOpacity_));
		gpuData_.densityAxisOpacity_ = nullptr;
	}
	if (gpuData_.opacityAxis_)
	{
		CUMAT_SAFE_CALL(cudaFree(gpuData_.opacityAxis_));
		gpuData_.opacityAxis_ = nullptr;
	}
	if (gpuData_.densityAxisColor_)
	{
		CUMAT_SAFE_CALL(cudaFree(gpuData_.densityAxisColor_));
		gpuData_.densityAxisColor_ = nullptr;
	}
	if (gpuData_.colorAxis_)
	{
		CUMAT_SAFE_CALL(cudaFree(gpuData_.colorAxis_));
		gpuData_.colorAxis_ = nullptr;
	}
	gpuData_.sizeOpacity_ = 0;
	gpuData_.sizeColor_ = 0;

	densityAxisOpacity_.clear();
	opacityAxis_.clear();
	densityAxisColor_.clear();
	colorAxis_.clear();
}

END_RENDERER_NAMESPACE
