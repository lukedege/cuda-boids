#pragma once
#include "vao_runner.h"

#include "../utils/shader.h"
#include "../utils/mesh.h"
#include "../utils/flock.h"

namespace utils::runners
{
	class cpu_vel_vao : public vao_runner
	{
	public:
		cpu_vel_vao(const size_t amount);

		void calculate(const float delta_time);

		void draw(const glm::mat4& view_matrix, const glm::mat4& projection_matrix);

	private:
		utils::graphics::opengl::Shader shader;

		size_t amount;
		std::vector<glm::vec3> positions;
		std::vector<glm::vec3> velocities;

		GLuint vao;
		GLuint vbo_positions;
		GLuint vbo_velocities;
	};
}