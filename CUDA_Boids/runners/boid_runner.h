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
		struct simulation_parameters
		{
			size_t boid_amount          { 50 };
			float  boid_speed           { 3.0f };
			float  boid_fov             { 10.f };
			float  alignment_coeff      { 1.0f };
			float  cohesion_coeff       { 0.9f };
			float  separation_coeff     { 0.5f };
			float  wall_separation_coeff{ 3.0f };
			float  cube_size            { 20.f };
		};

		virtual void calculate(const float delta_time) = 0;
		virtual void draw(const glm::mat4& view_matrix, const glm::mat4& projection_matrix) = 0;

		virtual simulation_parameters get_simulation_parameters() = 0;
		virtual void set_simulation_parameters(simulation_parameters new_params) = 0;

	protected:
		boid_runner() : 
			simulation_volume_planes
			{
				{{  sim_params.cube_size,0,0,1 }, { -1,0,0,0 }}, //xp
				{{ -sim_params.cube_size,0,0,1 }, {  1,0,0,0 }}, //xm
				{{ 0, sim_params.cube_size,0,1 }, { 0,-1,0,0 }}, //yp
				{{ 0,-sim_params.cube_size,0,1 }, { 0, 1,0,0 }}, //ym
				{{ 0,0, sim_params.cube_size,1 }, { 0,0,-1,0 }}, //zp
				{{ 0,0,-sim_params.cube_size,1 }, { 0,0, 1,0 }}, //zm
			}
		{}

		simulation_parameters sim_params;

		utils::math::plane simulation_volume_planes[6];
	};
}