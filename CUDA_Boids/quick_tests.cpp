#include <iostream>

#include "utils/shader.h"
#include "utils/window.h"
#include "utils/mesh.h"
#include "utils/camera.h"

#include "utils/utils.h"

namespace ugo = utils::graphics::opengl;

void opengl_main()
{
	ugo::window wdw
	{
		ugo::window::window_create_info
		{
			{ "Prova" }, //.title
			{ 4       }, //.gl_version_major
			{ 3       }, //.gl_version_minor
			{ 1200    }, //.window_width
			{ 1200     }, //.window_height
			{ 1200    }, //.viewport_width
			{ 1200     }, //.viewport_height
			{ false   }, //.resizable
			{ false   }, //.debug_gl
		}
	};

	GLFWwindow* glfw_window = wdw.get();
	auto window_size = wdw.get_size();

	std::vector<glm::vec3> positions
	{
		{  0.0f,  1.0f, 0.0f,},// top right
		{  1.0f, -1.0f, 0.0f },// bottom right
		{ -1.0f, -1.0f, 0.0f },// bottom left
	};

	std::vector<glm::vec3> velocities
	{
		{ 1.f, 0.f, 0.0f,},// top right
		{ 0.f, 1.f, 0.0f },// bottom right
		{ 1.f, 1.f, 0.0f },// bottom left
	};

	GLuint VAO, VBO_positions, VBO_velocities;

	glGenVertexArrays(1, &VAO);
	glGenBuffers(1, &VBO_positions);
	glGenBuffers(1, &VBO_velocities);

	// VAO is made "active"    
	glBindVertexArray(VAO);

	// positions
	glBindBuffer(GL_ARRAY_BUFFER, VBO_positions);
	glBufferData(GL_ARRAY_BUFFER, positions.size() * sizeof(glm::vec3), NULL, GL_DYNAMIC_DRAW);
	glBufferSubData(GL_ARRAY_BUFFER, 0, positions.size() * sizeof(glm::vec3), positions.data());
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (GLvoid*)0);

	// velocities
	glBindBuffer(GL_ARRAY_BUFFER, VBO_velocities);
	glBufferData(GL_ARRAY_BUFFER, velocities.size() * sizeof(glm::vec3), NULL, GL_DYNAMIC_DRAW);
	glBufferSubData(GL_ARRAY_BUFFER, 0, velocities.size() * sizeof(glm::vec3), velocities.data());
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (GLvoid*)0);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);


	utils::graphics::opengl::Shader shader{ "shaders/vao.vert", "shaders/basic.frag", "shaders/vao.geom" };

	shader.use();

	ugo::Camera camera{ glm::vec3(0, 0, 50), GL_TRUE };
	glm::mat4 projection_matrix = glm::perspective(45.0f, (float)window_size.first / (float)window_size.second, 0.1f, 10000.0f);
	glm::mat4 view_matrix = glm::mat4(1);

	while (wdw.is_open())
	{
		// we "clear" the frame and z buffer
		glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT);

		for (int i = 0; i < positions.size(); i++)
		{
			positions[i] += velocities[i] * 0.0001f;
		}

		glBindBuffer(GL_ARRAY_BUFFER, VBO_positions);
		glBufferSubData(GL_ARRAY_BUFFER, 0, positions.size() * sizeof(glm::vec3), positions.data());

		glBindBuffer(GL_ARRAY_BUFFER, VBO_velocities);
		glBufferSubData(GL_ARRAY_BUFFER, 0, velocities.size() * sizeof(glm::vec3), velocities.data());

		glBindBuffer(GL_ARRAY_BUFFER, 0);

		view_matrix = camera.GetViewMatrix();
		shader.setMat4("projection_matrix", projection_matrix);
		shader.setMat4("view_matrix", view_matrix);

		glBindVertexArray(VAO);
		glDrawArrays(GL_POINTS, 0, positions.size());
		glBindVertexArray(0);

		glfwSwapBuffers(glfw_window);
		glfwPollEvents();
	}
}

int mainz()
{
	float cube_size = 10;
	glm::vec4 boid = { -1,3,9,0 };
	utils::math::plane zp{ { 0,0, cube_size,0 }, { 0,0,-1,0 } };
	utils::math::plane zm{ { 0,0,-cube_size,0 }, { 0,0, 1,0 } };
	utils::math::plane xp{ {  cube_size,0,0,0 }, { -1,0,0,0 } };
	utils::math::plane xm{ { -cube_size,0,0,0 }, {  1,0,0,0 } };
	utils::math::plane yp{ { 0, cube_size,0,0 }, { 0,-1,0,0 } };
	utils::math::plane ym{ { 0,-cube_size,0,0 }, { 0, 1,0,0 } };
	std::vector<utils::math::plane> planes{zp,zm,xp,xm,yp,ym};
	for (auto& p : planes)
	{
		float dist = utils::math::distance_point_plane(boid, p);
		std::cout << dist << " - ";
		std::cout << 1 / dist;
		std::cout << std::endl;
	}
	
	return 0;
}