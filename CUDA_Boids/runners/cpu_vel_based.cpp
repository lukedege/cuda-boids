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
		triangles { setup_mesh(), amount },
		velocities{ std::vector<glm::vec2>(amount) }
	{
		int pos_alloc_size = sizeof(glm::vec2) * amount;
		int ang_alloc_size = pos_alloc_size;

		utils::containers::random_vec2_fill_cpu(triangles.positions, -20, 20);

		setup_ssbo(ssbo_positions, pos_alloc_size, 0, triangles.positions.data());
		setup_ssbo(ssbo_velocities, ang_alloc_size, 1, velocities.data());
	}

	void cpu_vel_based::calculate(const float delta_time)
	{
		glm::vec2 vel{ 1,1 };

		for (size_t i = 0; i < amount; i++)
		{
			velocities[i]    = vel;
			triangles.positions[i] += vel * delta_time;
		}

		glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo_positions);
		glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, amount * sizeof(glm::vec2), triangles.positions.data());

		glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo_velocities);
		glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, amount * sizeof(glm::vec2), velocities.data());

		glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
	}

	void cpu_vel_based::draw(const glm::mat4& view_matrix, const glm::mat4& projection_matrix)
	{
		shader.use();
		shader.setMat4("view_matrix", view_matrix);
		shader.setMat4("projection_matrix", projection_matrix);
		triangles.draw(shader, view_matrix);
	}
}