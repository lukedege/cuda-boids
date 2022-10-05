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

namespace ugo = utils::graphics::opengl;

// callback function for keyboard events
//void key_callback(GLFWwindow* window, int key, int scancode, int action, int mode);
//void mouse_pos_callback(GLFWwindow* window, double xPos, double yPos);

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

	// we put in relation the window and the callbacks
	//glfwSetKeyCallback(glfw_window, key_callback);
	//glfwSetCursorPosCallback(glfw_window, mouse_pos_callback);

	ugo::Shader basic_shader{ "shaders/basic.vert", "shaders/basic.frag", {}, 4, 3 };

	std::vector<ugo::Vertex> vertices
	{
		{{  0.5f,  0.5f, 0.0f,}},// top right
		{{  0.5f, -0.5f, 0.0f }},// bottom right
		{{ -0.5f, -0.5f, 0.0f }},// bottom left
		{{ -0.5f,  0.5f, 0.0f }},// top left 
	};
	std::vector<GLuint> indices
	{
		0, 1, 3,   // first triangle
		1, 2, 3    // second triangle
	};

	ugo::Mesh triangle{ vertices, indices };

	GLfloat deltaTime = 0.0f;
	GLfloat lastFrame = 0.0f;
	while (wdw.is_open())
	{
		// we determine the time passed from the beginning
		// and we calculate the time difference between current frame rendering and the previous one
		GLfloat currentFrame = glfwGetTime();
		deltaTime = currentFrame - lastFrame;
		lastFrame = currentFrame;

		// we "clear" the frame and z buffer
		glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT);

		basic_shader.use();
		triangle.draw();

		glfwSwapBuffers(glfw_window);
		glfwPollEvents();
	}
	return 0;
}