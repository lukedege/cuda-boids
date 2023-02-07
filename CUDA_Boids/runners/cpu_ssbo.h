#pragma once
#include "ssbo_runner.h"

namespace utils::runners
{
	class cpu_ssbo : public ssbo_runner
	{
	public:
		cpu_ssbo(simulation_parameters params);

		void calculate(const float delta_time);

		void draw(const glm::mat4& view_matrix, const glm::mat4& projection_matrix);

		simulation_parameters get_simulation_parameters();
		void set_dynamic_simulation_parameters(simulation_parameters::dynamic_parameters new_dyn_params);

	private:
		void naive_calculation(const float delta_time);
		void uniform_grid_calculation(const float delta_time);
		void coherent_grid_calculation(const float delta_time);

		size_t amount; // possibly redundant but comfy to avoid writing params.boid_amount etc...
		std::vector<float4> positions; 
		std::vector<float4> velocities;
	};
}