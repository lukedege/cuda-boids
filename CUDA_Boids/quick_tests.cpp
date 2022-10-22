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
#include <glm/gtx/vector_angle.hpp> 

// CUDA libraries

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_gl_interop.h>

// utils libraries
#include "utils/cuda_utils.cuh"
#include "utils/window.h"
#include "utils/shader.h"
#include "utils/mesh.h"
#include "utils/entity.h"
#include "utils/flock.h"
#include "utils/camera.h"

namespace ugo = utils::graphics::opengl;

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
			{ false   }, //.debug_gl
		}
	};

	GLFWwindow* glfw_window = wdw.get();
	auto window_size = wdw.get_size();

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

	ugo::Mesh   triangle_mesh{ vertices, indices };

	GLuint ubo; // uniform_buffer_object
	int size = 152;

	glGenBuffers(1, &ubo);
	glBindBuffer(GL_UNIFORM_BUFFER, ubo);

	// initialize buffer object
	glBufferData(GL_UNIFORM_BUFFER, size, NULL, GL_STATIC_DRAW); // allocate 152 bytes of memory
	glBindBuffer(GL_UNIFORM_BUFFER, 0);

	// register this buffer object with CUDA
	cudaGraphicsResource* cuda_ubo_res;
	cudaGraphicsGLRegisterBuffer(&cuda_ubo_res, ubo, cudaGraphicsMapFlags::cudaGraphicsMapFlagsWriteDiscard);
}