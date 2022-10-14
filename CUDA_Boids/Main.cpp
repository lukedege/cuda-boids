#pragma once

// std libraries
#include <iostream>
#include <vector>
#include <string>

// OpenGL libraries
#ifdef _WIN32
#define APIENTRY __stdcall
#endif

#include <glad.h>
#include <glfw/glfw3.h>
#include <glm/glm.hpp>

// CUDA libraries

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// utils libraries
#include "utils/cuda_utils.cuh"
#include "utils/window.h"
#include "utils/shader.h"
#include "utils/mesh.h"
#include "utils/entity.h"
#include "utils/camera.h"

namespace ugo = utils::graphics::opengl;

void calculate_positions(ugo::Shader shader_to_setup, std::vector<glm::vec2>& positions, float delta_time)
{
	for (size_t i = 0; i < positions.size(); i++)
	{
		positions[i][0] += ( i % 2? -1 : 1 ) * delta_time;
		shader_to_setup.setVec2("positions["+ std::to_string(i) + "]", positions[i]);
	}
}

int main()
{
	ugo::window wdw
	{
		ugo::window::window_create_info
		{
			{ "Prova" }, //.title
			{ 4       }, //.gl_version_major
			{ 3       }, //.gl_version_minor
			{ 1200    }, //.window_width
			{ 900     }, //.window_height
			{ 1200    }, //.viewport_width
			{ 900     }, //.viewport_height
			{ false   }, //.resizable
		}
	};

	GLFWwindow* glfw_window = wdw.get();
	auto window_size = wdw.get_size();

	// we put in relation the window and the callbacks
	//glfwSetKeyCallback(glfw_window, key_callback);
	//glfwSetCursorPosCallback(glfw_window, mouse_pos_callback);

	ugo::Shader basic_shader{ "shaders/mvp_instanced.vert", "shaders/basic.frag", {}, 4, 3 };

	std::vector<ugo::Vertex> vertices
	{
		{{  0.0f,  0.5f, 0.0f,}},// top right
		{{  0.5f, -0.5f, 0.0f }},// bottom right
		{{ -0.5f, -0.5f, 0.0f }},// bottom left
	};
	std::vector<GLuint> indices
	{
		0, 1, 2,   // first triangle
	};

	std::vector<glm::vec2> positions
	{
		{{0,0}, {2, 2}, {-2, -2}}
	};

	ugo::Mesh   triangle_mesh{ vertices, indices };
	// TODO split concept of entity and fleet of entities; entity = base object, fleet = vec of positions + represented entity
	ugo::Entity triangle     { triangle_mesh, positions.size() };


	// Camera setup
	ugo::Camera camera{ glm::vec3(0, 0, 50), GL_TRUE };
	glm::mat4 projection_matrix = glm::perspective(45.0f, (float)window_size.first / (float)window_size.second, 0.1f, 10000.0f);
	glm::mat4 view_matrix = glm::mat4(1);

	GLfloat delta_time = 0.0f;
	GLfloat last_frame = 0.0f;
	//triangle.rotate_deg(90.f, { 0,0,1 });
	while (wdw.is_open())
	{
		// we determine the time passed from the beginning
		// and we calculate the time difference between current frame rendering and the previous one
		GLfloat currentFrame = glfwGetTime();
		delta_time = currentFrame - last_frame;
		last_frame = currentFrame;

		view_matrix = camera.GetViewMatrix();
		basic_shader.use();
		basic_shader.setMat4("view_matrix", view_matrix);
		basic_shader.setMat4("projection_matrix", projection_matrix);

		// we "clear" the frame and z buffer
		glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT);
		
		/*
		TODO: cuda kernel calculates position for boids and returns them to opengl for drawing
		*/
		calculate_positions(basic_shader, positions, delta_time);

		//triangle.translate({ 0.f, deltaTime * 3.f, 0.f });
		//triangle.rotate_deg(deltaTime * 30.f, { 1,0,0 });
		triangle.draw(basic_shader, view_matrix);

		glfwSwapBuffers(glfw_window);
		glfwPollEvents();
	}
	return 0;
}