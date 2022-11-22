#include "cpu_ssbo.h"

// std libraries
#include <vector>
#include <math.h>

#include <glm/glm.hpp>

// utils libraries
#include "../utils/utils.h"
#include "boid_behaviours.h"

namespace utils::runners
{
	cpu_vel_ssbo::cpu_vel_ssbo() :
		shader{ "shaders/ssbo.vert", "shaders/basic.frag"},
		amount{ simulation_params.boid_amount },
		triangle_mesh{ setup_mesh() },
		positions { std::vector<glm::vec4>(amount) },
		velocities{ std::vector<glm::vec4>(amount) }
	{
		utils::containers::random_vec4_fill_cpu(positions, -10, 10);
		utils::containers::random_vec4_fill_cpu(velocities, -1, 1);

		setup_ssbo(ssbo_positions , sizeof(glm::vec4), amount, 0, positions.data());
		setup_ssbo(ssbo_velocities, sizeof(glm::vec4), amount, 1, velocities.data());
	}

	void cpu_vel_ssbo::calculate(const float delta_time)
	{
		glm::vec4 accel_blend;
		for (size_t i = 0; i < amount; i++)
		{
			accel_blend =  simulation_params.alignment_coeff       * behaviours::cpu::naive::alignment      (i, positions.data(), velocities.data(), amount, simulation_params.boid_fov)
				         + simulation_params.cohesion_coeff        * behaviours::cpu::naive::cohesion       (i, positions.data(), amount, simulation_params.boid_fov)
				         + simulation_params.separation_coeff      * behaviours::cpu::naive::separation     (i, positions.data(), amount)
				         + simulation_params.wall_separation_coeff * behaviours::cpu::naive::wall_separation(i, positions.data(), planes_array, amount);

			//velocities[i] = normalize(velocities[i]) + normalize(accel_blend) * delta_time; //v = u + at
			velocities[i] = normalize(velocities[i] + accel_blend * delta_time); //v = u + at
			positions [i] += velocities[i] * simulation_params.boid_speed * delta_time; //s = vt
		}

		glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo_positions);
		glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, amount * sizeof(glm::vec4), positions.data());

		glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo_velocities);
		glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, amount * sizeof(glm::vec4), velocities.data());
		
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
	}

	void cpu_vel_ssbo::draw(const glm::mat4& view_matrix, const glm::mat4& projection_matrix)
	{
		shader.use();
		shader.setMat4("view_matrix", view_matrix);
		shader.setMat4("projection_matrix", projection_matrix);
		triangle_mesh.draw_instanced(amount);
	}
}