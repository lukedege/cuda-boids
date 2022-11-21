#pragma once
#include <glm/glm.hpp>
#include <array>

#include "../utils/utils.h"

namespace utils::runners
{
	//all data should be in unified memory or both host and device memory
	class boid_runner 
	{
	public:
		inline static struct simulation_parameters
		{
			size_t boid_amount     { 1024 };
			float  boid_speed      { 3.0f };
			float  boid_fov        { 10.f };
			float  alignment_coeff { 1.0f };
			float  cohesion_coeff  { 0.8f };
			float  separation_coeff{ 0.8f };
			float  cube_size       { 5.0f };
		} simulation_params; 

		virtual void calculate(const float delta_time) = 0;
		virtual void draw(const glm::mat4& view_matrix, const glm::mat4& projection_matrix) = 0;
	protected:
		struct
		{
			utils::math::plane zp{ { 0,0, simulation_params.cube_size,1 }, { 0,0,-1,0 } };
			utils::math::plane zm{ { 0,0,-simulation_params.cube_size,1 }, { 0,0, 1,0 } };
			utils::math::plane xp{ {  simulation_params.cube_size,0,0,1 }, { -1,0,0,0 } };
			utils::math::plane xm{ { -simulation_params.cube_size,0,0,1 }, {  1,0,0,0 } };
			utils::math::plane yp{ { 0, simulation_params.cube_size,0,1 }, { 0,-1,0,0 } };
			utils::math::plane ym{ { 0,-simulation_params.cube_size,0,1 }, { 0, 1,0,0 } };
		} planes;

		utils::math::plane planes_array[6]{planes.zp, planes.zm, planes.xp, planes.xm, planes.yp, planes.ym};
	};
}