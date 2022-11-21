#include "cpu_vao.h"

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
	cpu_vel_vao::cpu_vel_vao(const size_t amount) :
		shader{ "shaders/vao.vert", "shaders/basic.frag", "shaders/vao.geom"},
		amount{ amount },
		positions { std::vector<glm::vec3>(amount) },
		velocities{ std::vector<glm::vec3>(amount) }
	{
		utils::containers::random_vec3_fill_cpu(positions , -20, 20);
		//utils::containers::random_vec3_fill_cpu(velocities, -2, 2);

		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);

		setup_vbo(vbo_positions, sizeof(glm::vec3), positions.size(), 0, positions.data());
		setup_vbo(vbo_velocities, sizeof(glm::vec3), velocities.size(), 1, velocities.data());

		glBindVertexArray(0);
	}
	
	void cpu_vel_vao::calculate(const float delta_time)
	{
		glm::vec3 vel{1,1,0};
		for (size_t i = 0; i < amount; i++)
		{
			velocities[i] = vel;
			positions[i] += velocities[i] * delta_time;
		}

		glBindBuffer(GL_ARRAY_BUFFER, vbo_positions);
		glBufferSubData(GL_ARRAY_BUFFER, 0, amount * sizeof(glm::vec3), positions.data());

		glBindBuffer(GL_ARRAY_BUFFER, vbo_velocities);
		glBufferSubData(GL_ARRAY_BUFFER, 0, amount * sizeof(glm::vec3), velocities.data());

		glBindBuffer(GL_ARRAY_BUFFER, 0);
	}

	void cpu_vel_vao::draw(const glm::mat4& view_matrix, const glm::mat4& projection_matrix)
	{
		shader.use();
		shader.setMat4("view_matrix", view_matrix);
		shader.setMat4("projection_matrix", projection_matrix);

		glBindVertexArray(vao);
		glDrawArrays(GL_POINTS, 0, positions.size());
		glBindVertexArray(0);
	}
	
}