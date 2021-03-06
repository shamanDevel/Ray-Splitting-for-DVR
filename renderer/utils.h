#pragma once

#include <json.hpp>
#include <cuda_runtime.h>
#include <iostream>

namespace nlohmann {
	template <>
	struct adl_serializer<float3> {
		static void to_json(json& j, const float3& v) {
			j = json::array({ v.x, v.y, v.z });
		}

		static void from_json(const json& j, float3& v) {
			if (j.is_array() && j.size()==3)
			{
				v.x = j.at(0).get<float>();
				v.y = j.at(1).get<float>();
				v.z = j.at(2).get<float>();
			}
			else
				std::cerr << "Unable to deserialize " << j << " into a float3" << std::endl;
		}
	};
	template <>
	struct adl_serializer<float4> {
		static void to_json(json& j, const float4& v) {
			j = json::array({ v.x, v.y, v.z, v.w });
		}

		static void from_json(const json& j, float4& v) {
			if (j.is_array() && j.size() == 4)
			{
				v.x = j.at(0).get<float>();
				v.y = j.at(1).get<float>();
				v.z = j.at(2).get<float>();
				v.w = j.at(3).get<float>();
			}
			else
				std::cerr << "Unable to deserialize " << j << " into a float4" << std::endl;
		}
	};
}
