#include "cpu_vel_ssbo.h"

// std libraries
#include <vector>
#include <math.h>

#include <glad.h>
#include <glm/glm.hpp>
#include <glm/gtx/vector_angle.hpp> 
#include <glm/gtx/norm.hpp>
#include <glm/gtx/quaternion.hpp>

// utils libraries
#include "../utils/utils.h"

namespace utils::runners
{
	//TODO move behaviours into a common boid behaviour class (for both cpu/gpu)
	glm::vec4 alignment(size_t current, glm::vec4* positions, glm::vec4* velocities, size_t amount, size_t max_radius)
	{
		glm::vec4 alignment{0};
		for (size_t i = 0; i < amount; i++)
		{
			// conditions as multipliers (avoids divergence)
			float in_radius = glm::distance2(positions[current], positions[i]) < max_radius * max_radius;
			alignment += velocities[i] * in_radius;
		}

		return utils::math::normalize(alignment);
	}

	glm::vec4 cohesion(size_t current, glm::vec4* positions, glm::vec4* velocities, size_t amount, size_t max_radius)
	{
		glm::vec4 cohesion{ 0 };
		float counter{ 0 };
		for (size_t i = 0; i < amount; i++)
		{
			// conditions as multipliers (avoids divergence)
			float in_radius = glm::distance2(positions[current], positions[i]) < max_radius * max_radius;
			cohesion += positions[i] * in_radius;
			counter  += 1.f * in_radius;
		}
		cohesion /= (float)counter;
		cohesion -= positions[current];
		return utils::math::normalize(cohesion);
	}

	glm::vec4 separation(size_t current, glm::vec4* positions, glm::vec4* velocities, size_t amount)
	{
		glm::vec4 separation{ 0 };
		glm::vec4 repulsion;
		for (size_t i = 0; i < amount; i++)
		{
			repulsion = positions[current] - positions[i];
			separation += utils::math::normalize(repulsion) / (glm::length(repulsion) + 0.0001f);
		}

		return utils::math::normalize(separation);
	}

	cpu_vel_ssbo::cpu_vel_ssbo(const size_t amount) :
		shader{ "shaders/ssbo_instanced_vel.vert", "shaders/basic.frag"},
		amount{ amount },
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
			accel_blend =  0.8f * alignment (i, positions.data(), velocities.data(), amount, 10)
				         + 0.5f * cohesion  (i, positions.data(), velocities.data(), amount, 10)
				         + 0.5f * separation(i, positions.data(), velocities.data(), amount    ); 
			//TODO proper velocity steering, such as: velocities[i] = (velocities[i] * t + 0.5f * normalize(blend) * t * t) * 10.f; 
			velocities[i] = velocities[i] * delta_time + normalize(accel_blend) * delta_time;
			positions [i] += velocities[i];
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