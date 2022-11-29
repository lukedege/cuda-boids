#pragma once
#include "ssbo_runner.h"

#include "../utils/shader.h"
#include "../utils/mesh.h"
#include "../utils/flock.h"

namespace utils::runners
{
	class cpu_vel_ssbo : public ssbo_runner
	{
	public:
		cpu_vel_ssbo(simulation_parameters params);

		void calculate(const float delta_time);

		void draw(const glm::mat4& view_matrix, const glm::mat4& projection_matrix);

		simulation_parameters get_simulation_parameters();
		void set_simulation_parameters(simulation_parameters new_params);

	private:
		void naive_calculation(const float delta_time);

		utils::graphics::opengl::Shader shader; //TODO moveable to ssbo/vao parent class

		size_t amount; //TODO possibly redundant
		utils::graphics::opengl::Mesh triangle_mesh; //TODO possibly moveable to ssbo/vao parent class
		std::vector<glm::vec4> positions; //TODO possibly moveable to boid_runner grandparent class
		std::vector<glm::vec4> velocities;//TODO possibly moveable to boid_runner grandparent class

		GLuint ssbo_positions;  //TODO possibly moveable to ssbo/vao parent class | shader_storage_buffer_object
		GLuint ssbo_velocities; //TODO possibly moveable to ssbo/vao parent class | shader_storage_buffer_object
	};
}