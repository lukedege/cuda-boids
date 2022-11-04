#include "cpu_angle_based.h"

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
	cpu_angle_based::cpu_angle_based(utils::graphics::opengl::Mesh& mesh, const size_t amount) :
		shader{ "shaders/ssbo_instanced_angle.vert", "shaders/basic.frag", {}, 4, 3 },
		amount{ amount },
		triangles{ mesh, amount },
		angles { std::vector<float>(amount) }
	{
		int pos_alloc_size = sizeof(glm::vec2) * amount;
		int ang_alloc_size = sizeof(float)     * amount;

		utils::containers::random_vec2_fill_cpu(triangles.positions, -20, 20);

		utils::gl::setup_ssbo(ssbo_positions, pos_alloc_size, 0, triangles.positions.data());
		utils::gl::setup_ssbo(ssbo_angles, ang_alloc_size, 1, angles.data());
	}

	void cpu_angle_based::calculate(const float delta_time)
	{
		glm::vec2 vel{ 1,1 };
		glm::vec2 vel_norm{ glm::normalize(vel) };
		glm::vec2 x{ 1,0 };

		for (size_t i = 0; i < amount; i++)
		{
			angles[i]     = glm::angle(vel_norm, x);
			triangles.positions[i] += vel * delta_time;
		}

		glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo_positions);
		glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, amount * sizeof(glm::vec2), triangles.positions.data());

		glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo_angles);
		glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, amount * sizeof(float), angles.data());

		glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
	}

	void cpu_angle_based::draw(const glm::mat4& view_matrix, const glm::mat4& projection_matrix)
	{
		shader.use();
		shader.setMat4("view_matrix", view_matrix);
		shader.setMat4("projection_matrix", projection_matrix);
		triangles.draw(shader, view_matrix);
	}
}