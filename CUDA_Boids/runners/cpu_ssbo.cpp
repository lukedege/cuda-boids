#include "cpu_ssbo.h"

// std libraries
#include <vector>
#include <math.h>

#include <glad.h>
#include <glm/glm.hpp>

// utils libraries
#include "../utils/utils.h"
#include "boid_behaviours.h"

namespace utils::runners
{
	cpu_vel_ssbo::cpu_vel_ssbo(simulation_parameters params) :
		ssbo_runner{ params },
		shader{ "shaders/ssbo.vert", "shaders/basic.frag"},
		amount{ sim_params.boid_amount },
		triangle_mesh{ setup_mesh() },
		positions { std::vector<glm::vec4>(amount) },
		velocities{ std::vector<glm::vec4>(amount) }
	{
		utils::containers::random_vec4_fill_cpu(positions, -10, 10);
		utils::containers::random_vec4_fill_cpu(velocities, -1, 1);

		setup_buffer_object(ssbo_positions , GL_SHADER_STORAGE_BUFFER, sizeof(glm::vec4), amount, 0, positions.data());
		setup_buffer_object(ssbo_velocities, GL_SHADER_STORAGE_BUFFER, sizeof(glm::vec4), amount, 1, velocities.data());
	}

	void cpu_vel_ssbo::calculate(const float delta_time)
	{
		glm::vec4 accel_blend;
		for (size_t i = 0; i < amount; i++)
		{
			accel_blend =  sim_params.alignment_coeff       * behaviours::cpu::naive::alignment      (i, positions.data(), velocities.data(), amount, sim_params.boid_fov)
				         + sim_params.cohesion_coeff        * behaviours::cpu::naive::cohesion       (i, positions.data(), amount, sim_params.boid_fov)
				         + sim_params.separation_coeff      * behaviours::cpu::naive::separation     (i, positions.data(), amount, sim_params.boid_fov)
				         + sim_params.wall_separation_coeff * behaviours::cpu::naive::wall_separation(i, positions.data(), simulation_volume_planes.data(), amount);

			//velocities[i] = normalize(velocities[i]) + normalize(accel_blend) * delta_time; //v = u + at
			velocities[i] = normalize(velocities[i] + accel_blend * delta_time); //v = u + at
			positions [i] += velocities[i] * sim_params.boid_speed * delta_time; //s = vt
		}

		update_buffer_object(ssbo_positions , GL_SHADER_STORAGE_BUFFER, 0, sizeof(glm::vec4), amount, positions.data());
		update_buffer_object(ssbo_velocities, GL_SHADER_STORAGE_BUFFER, 0, sizeof(glm::vec4), amount, velocities.data());
	}

	void cpu_vel_ssbo::draw(const glm::mat4& view_matrix, const glm::mat4& projection_matrix)
	{
		// Update references to view and projection matrices
		update_buffer_object(ubo_matrices, GL_UNIFORM_BUFFER, 0                , sizeof(glm::mat4), 1, (void*) glm::value_ptr(view_matrix)      );
		update_buffer_object(ubo_matrices, GL_UNIFORM_BUFFER, sizeof(glm::mat4), sizeof(glm::mat4), 1, (void*) glm::value_ptr(projection_matrix));

		// Setup and draw debug info (simulation volume, ...)
		debug_shader.use();
		glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
		cube_mesh.draw(GL_LINES);
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

		// Setup and draw boids
		shader.use();
		triangle_mesh.draw_instanced(amount);
	}

	cpu_vel_ssbo::simulation_parameters cpu_vel_ssbo::get_simulation_parameters()
	{
		return sim_params;
	}

	void cpu_vel_ssbo::set_simulation_parameters(cpu_vel_ssbo::simulation_parameters new_params)
	{
		sim_params = new_params;
	}
}