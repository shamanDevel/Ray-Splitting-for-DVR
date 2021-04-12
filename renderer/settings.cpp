#include "settings.h"

#include <json.hpp>

#include "utils.h"
#include "camera.h"

BEGIN_RENDERER_NAMESPACE

void RendererArgs::load(const nlohmann::json& settings, const std::filesystem::path& basePath, 
	RendererArgs& renderer, Camera& camera, std::string& filename)
{
	camera.fromJson(settings.at("camera"));
	
	renderer.renderMode = settings.at("renderMode").get<RenderMode>();
	renderer.dvrTfMode = static_cast<DvrTfMode>(settings.value<int>(
		"renderModeDvr", 
		static_cast<int>(DvrTfMode::PiecewiseLinear)));
	{
		const auto& s = settings.at("tfEditor");
		if (renderer.dvrTfMode == DvrTfMode::PiecewiseLinear) {
			nlohmann::basic_json<>::value_type tf;
			if (s.contains("editor"))
				tf = s.at("editor");
			if (s.contains("editorLinear"))
				tf = s.at("editorLinear");
			renderer.tf = RendererArgs::TfLinear();
			std::get<RendererArgs::TfLinear>(renderer.tf).densityAxisOpacity = tf.at("densityAxisOpacity").get<std::vector<float>>();
			std::get<RendererArgs::TfLinear>(renderer.tf).opacityAxis = tf.at("opacityAxis").get<std::vector<float>>();
			std::get<RendererArgs::TfLinear>(renderer.tf).densityAxisColor = tf.at("densityAxisColor").get<std::vector<float>>();
			std::get<RendererArgs::TfLinear>(renderer.tf).colorAxis = tf.at("colorAxis").get<std::vector<float3>>();
		} else if (renderer.dvrTfMode == DvrTfMode::MultipleIsosurfaces)
		{
			const auto& tf = s.at("editorMultiIso");
			renderer.tf = RendererArgs::TfMultiIso();
			std::get<RendererArgs::TfMultiIso>(renderer.tf).densities = tf.at("positions").get<std::vector<float>>();
			std::get<RendererArgs::TfMultiIso>(renderer.tf).colors = tf.at("colors").get<std::vector<float4>>();
		}
		else if (renderer.dvrTfMode == DvrTfMode::Hybrid) {
			nlohmann::basic_json<>::value_type tf = s.at("editorHybrid");
			renderer.tf = RendererArgs::TfLinear();
			std::get<RendererArgs::TfLinear>(renderer.tf).densityAxisOpacity = tf.at("densityAxisOpacity").get<std::vector<float>>();
			std::get<RendererArgs::TfLinear>(renderer.tf).opacityAxis = tf.at("opacityAxis").get<std::vector<float>>();
			std::get<RendererArgs::TfLinear>(renderer.tf).opacityExtraColorAxis = tf.at("opacityExtraColorAxis").get<std::vector<float4>>();
			std::get<RendererArgs::TfLinear>(renderer.tf).densityAxisColor = tf.at("densityAxisColor").get<std::vector<float>>();
			std::get<RendererArgs::TfLinear>(renderer.tf).colorAxis = tf.at("colorAxis").get<std::vector<float3>>();
		}
		renderer.minDensity = s.at("minDensity").get<float>();
		renderer.maxDensity = s.at("maxDensity").get<float>();
		renderer.opacityScaling = s.at("opacityScaling").get<float>();
		renderer.dvrUseShading = s.at("dvrUseShading").get<bool>();
	}

	{
		const auto& s = settings.at("renderer");
		renderer.isovalue = s.at("isovalue").get<double>();
		renderer.stepsize = s.at("stepsize").get<double>();
		renderer.volumeFilterMode = s.at("filterMode").get<VolumeFilterMode>();
		renderer.binarySearchSteps = s.at("binarySearchSteps").get<int>();
		renderer.aoSamples = s.at("aoSamples").get<int>();
		renderer.aoRadius = s.at("aoRadius").get<double>();
		renderer.enableClipPlane = s.value("enableClipPlane", false);
		renderer.clipPlane = s.value("clipPlane", make_float4(0, 0, 0, 0));
	}

	{
		const auto& s = settings.at("shading");
		renderer.shading.materialColor = s.at("materialColor").get<float3>();
		renderer.shading.ambientLightColor = s.at("ambientLight").get<float3>();
		renderer.shading.diffuseLightColor = s.at("diffuseLight").get<float3>();
		renderer.shading.specularLightColor = s.at("specularLight").get<float3>();
		renderer.shading.specularExponent = s.at("specularExponent").get<float>();
		renderer.shading.aoStrength = s.at("aoStrength").get<float>();
		renderer.shading.lightDirection = s.at("lightDirection").get<float3>();
	}

	{
		const auto& s = settings.at("dataset");
		auto targetPath = std::filesystem::path(s.at("file").get<std::string>());
		auto absPath = targetPath.is_absolute()
			? targetPath
			: std::filesystem::absolute(basePath / targetPath);
		filename = absPath.string();
		renderer.mipmapLevel = s.at("mipmap").get<int>();
	}
}

nlohmann::json RendererArgs::toJson() const
{
	nlohmann::json settings;
	settings["renderMode"] = renderMode;

	settings["renderModeDvr"] = static_cast<int>(dvrTfMode);
	{
		nlohmann::json& s = settings["tfEditor"];
		if (dvrTfMode == DvrTfMode::PiecewiseLinear)
		{
			nlohmann::json& tf = s["editorLinear"];
			tf["densityAxisOpacity"] = std::get<TfLinear>(this->tf).densityAxisOpacity;
			tf["opacityAxis"] = std::get<TfLinear>(this->tf).opacityAxis;
			tf["densityAxisColor"] = std::get<TfLinear>(this->tf).densityAxisColor;
			tf["colorAxis"] = std::get<TfLinear>(this->tf).colorAxis;
		}
		else if (dvrTfMode == DvrTfMode::MultipleIsosurfaces)
		{
			nlohmann::json& tf = s["editorMultiIso"];
			tf["positions"] = std::get<TfMultiIso>(this->tf).densities;
			tf["colors"] = std::get<TfMultiIso>(this->tf).colors;
		}
		else if (dvrTfMode == DvrTfMode::Hybrid)
		{
			nlohmann::json& tf = s["editorHybrid"];
			tf["densityAxisOpacity"] = std::get<TfLinear>(this->tf).densityAxisOpacity;
			tf["opacityAxis"] = std::get<TfLinear>(this->tf).opacityAxis;
			tf["opacityExtraColorAxis"] = std::get<TfLinear>(this->tf).opacityExtraColorAxis;
			tf["densityAxisColor"] = std::get<TfLinear>(this->tf).densityAxisColor;
			tf["colorAxis"] = std::get<TfLinear>(this->tf).colorAxis;
		}
		s["minDensity"] = minDensity;
		s["maxDensity"] = maxDensity;
		s["opacityScaling"] = opacityScaling;
		s["dvrUseShading"] = dvrUseShading;
	}

	{
		nlohmann::json& s = settings["renderer"];
		s["isovalue"] = isovalue;
		s["stepsize"] = stepsize;
		s["filterMode"] = volumeFilterMode;
		s["binarySearchSteps"] = binarySearchSteps;
		s["aoSamples"] = aoSamples;
		s["aoRadius"] = aoRadius;
		s["enableClipPlane"] = enableClipPlane;
		s["clipPlane"] = clipPlane;
	}

	{
		nlohmann::json& s = settings["shading"];
		s["materialColor"] = shading.materialColor;
		s["ambientLight"] = shading.ambientLightColor;
		s["diffuseLight"] = shading.diffuseLightColor;
		s["specularLight"] = shading.specularLightColor;
		s["specularExponent"] = shading.specularExponent;
		s["aoStrength"] = shading.aoStrength;
		s["lightDirection"] = shading.lightDirection;
	}
	
	return settings;
}

END_RENDERER_NAMESPACE
