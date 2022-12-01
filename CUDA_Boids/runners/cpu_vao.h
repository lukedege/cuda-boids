#pragma once
#include "vao_runner.h"

namespace utils::runners
{
	class cpu_vao : public vao_runner
	{
	public:
		cpu_vao(simulation_parameters params);

		void calculate(const float delta_time);

		void draw(const glm::mat4& view_matrix, const glm::mat4& projection_matrix);

	private:
		size_t amount;
		std::vector<glm::vec3> positions;
		std::vector<glm::vec3> velocities;

		GLuint vao;
		GLuint vbo_positions;
		GLuint vbo_velocities;
	};
}