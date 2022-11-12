#include "cpu_vel_based.h"

// std libraries
#include <vector>
#include <math.h>

#include <glad.h>
#include <glm/glm.hpp>
#include <glm/gtx/vector_angle.hpp> 

// utils libraries
#include "../utils/utils.h"

namespace utils::runners
{
	cpu_vel_based::cpu_vel_based(const size_t amount) :
		shader{ "shaders/ssbo_instanced_vel.vert", "shaders/basic.frag"},
		amount{ amount },
		triangle_mesh{ setup_mesh() },
		positions { std::vector<glm::vec4>(amount) },
		velocities{ std::vector<glm::vec4>(amount) }
	{
		utils::containers::random_vec4_fill_cpu(positions, -20, 20);

		setup_ssbo(ssbo_positions , sizeof(glm::vec4), amount, 0, positions.data());
		setup_ssbo(ssbo_velocities, sizeof(glm::vec4), amount, 1, velocities.data());
	}

	void cpu_vel_based::calculate(const float delta_time)
	{
		glm::vec4 vel{ 1,1,0,0 };

		for (size_t i = 0; i < amount; i++)
		{
			velocities[i]    = vel * delta_time;
			positions [i] += velocities[i];
		}

		glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo_positions);
		glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, amount * sizeof(glm::vec4), positions.data());

		glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo_velocities);
		glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, amount * sizeof(glm::vec4), velocities.data());
		
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
	}

	void cpu_vel_based::draw(const glm::mat4& view_matrix, const glm::mat4& projection_matrix)
	{
		shader.use();
		shader.setMat4("view_matrix", view_matrix);
		shader.setMat4("projection_matrix", projection_matrix);
		triangle_mesh.draw_instanced(amount);
	}
}