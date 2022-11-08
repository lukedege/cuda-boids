#pragma once
#include "ssbo_runner.h"

#include "../utils/shader.h"
#include "../utils/mesh.h"
#include "../utils/flock.h"

namespace utils::runners
{
	class cpu_vel_based : public ssbo_runner
	{
	public:
		cpu_vel_based(const size_t amount);

		void calculate(const float delta_time);

		void draw(const glm::mat4& view_matrix, const glm::mat4& projection_matrix);

	private:
		utils::graphics::opengl::Shader shader;

		size_t amount;
		utils::graphics::opengl::Flock triangles;
		std::vector<glm::vec2> velocities;

		GLuint ssbo_positions; // shader_storage_buffer_object
		GLuint ssbo_velocities;    // shader_storage_buffer_object
	};
}