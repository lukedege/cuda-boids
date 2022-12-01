// std libraries
#include <iostream>
#include <iomanip>
#include <string>

// OpenGL libraries
#ifdef _WIN32
#define APIENTRY __stdcall
#endif

#include <glad.h>
#include <glfw/glfw3.h>
#include <glm/glm.hpp>

// utils libraries
#include "utils/window.h"
#include "utils/orbit_camera.h"

#include "runners/gpu_ssbo.h"
#include "runners/cpu_ssbo.h"
#include "runners/cpu_vao.h"
#include "runners/gpu_vao.h"

namespace ugl = utils::graphics::opengl;

// input stuff TODO: MOVE SOMEWHERE ELSE
void key_callback(GLFWwindow* window, int key, int scancode, int action, int mode);
void mouse_pos_callback(GLFWwindow* window, double x_pos, double y_pos);
void mouse_button_callback(GLFWwindow* window, int button, int action, int mode);
void mouse_scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
void process_keys(ugl::window& window, GLfloat delta_time);

GLfloat mouse_last_x, mouse_last_y, x_offset, y_offset;
bool left_mouse_pressed;

bool keys[1024];

ugl::orbit_camera camera{ glm::vec3(0, 0, 0), 50.f };

int main()
{
	ugl::window wdw
	{
		ugl::window::window_create_info
		{
			{ "Prova" }, //.title
			{ 4       }, //.gl_version_major
			{ 3       }, //.gl_version_minor
			{ 1200    }, //.window_width
			{ 900     }, //.window_height
			{ 1200    }, //.viewport_width
			{ 900     }, //.viewport_height
			{ false   }, //.resizable
			{ true   }, //.debug_gl
		}
	};

	GLFWwindow* glfw_window = wdw.get();
	ugl::window::window_size ws = wdw.get_size();
	float width = static_cast<float>(ws.width), height = static_cast<float>(ws.height);

	// setup callbacks
	glfwSetKeyCallback(glfw_window, key_callback);
	glfwSetCursorPosCallback(glfw_window, mouse_pos_callback);
	glfwSetMouseButtonCallback(glfw_window, mouse_button_callback);
	glfwSetScrollCallback(glfw_window, mouse_scroll_callback);
	
	// Runner setup
	utils::runners::boid_runner::simulation_parameters params
	{
		{ 1000 },//boid_amount
		{ 5.0f },//boid_speed
		{ 10.0f },//boid_fov
		{ 1.0f },//alignment_coeff
		{ 0.8f },//cohesion_coeff
		{ 1.0f },//separation_coeff
		{ 10.0f },//wall_separation_coeff
		{ 20.f },//cube_size
	};
	//utils::runners::boid_runner* runner;

	utils::runners::gpu_ssbo runner{params};
	//runner = &spec_runner;
	
	// Camera setup
	
	glm::mat4 projection_matrix = glm::perspective(45.0f, width / height, 0.1f, 10000.0f);
	glm::mat4 view_matrix = glm::mat4(1);

	GLfloat delta_time = 0.0f, last_frame = 0.0f, current_fps = 0.0f;
	GLfloat before_calculations = 0.0f, after_calculations = 0.0f, delta_calculations = 0.0f;
	GLfloat avg_calc = 1.f, avg_fps = 1.f;
	GLfloat alpha = 0.9;
	std::cout << std::setprecision(4) << std::fixed;
	while (wdw.is_open())
	{
		// we determine the time passed from the beginning
		// and we calculate the time difference between current frame rendering and the previous one
		GLfloat currentFrame = glfwGetTime();
		delta_time = currentFrame - last_frame;
		last_frame = currentFrame;

		glfwPollEvents();
		process_keys(wdw, delta_time);
		
		// we "clear" the frame and z buffer
		glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT);

		before_calculations = glfwGetTime();

		runner.calculate(delta_time);

		after_calculations = glfwGetTime(); // TODO measure time for kernel using CudaEvents
		delta_calculations = (after_calculations - before_calculations) * 1000; //in ms
		
		view_matrix = camera.view_matrix();
		runner.draw(view_matrix, projection_matrix); //TODO la projection matrix è fissa magari non serve aggiornarla ogni frame tbh

		current_fps = (1 / delta_time);
		//std::cout << "Calcs: " << delta_calculations << "ms | ";
		//std::cout << "FPS: " << current_fps << "\n";
		avg_fps = alpha * avg_fps + (1.0 - alpha) * (1 / delta_time);
		avg_calc = alpha * avg_calc + (1.0 - alpha) * (delta_calculations);

		glfwSwapBuffers(glfw_window);
	}
	std::cout << "-----------------------------------\n";
	std::cout << "Average FPS  : " << avg_fps << "\n";
	std::cout << "Average calcs: " << avg_calc << "ms \n";

	return 0;
}

void process_keys(ugl::window& window, GLfloat delta_time)
{
	if (keys[GLFW_KEY_ESCAPE])
		window.close();
}

void key_callback(GLFWwindow* window, int key, int scancode, int action, int mode)
{
	if (action == GLFW_PRESS)
		keys[key] = true;
	else if (action == GLFW_RELEASE)
		keys[key] = false;
}

void mouse_button_callback(GLFWwindow* window, int button, int action, int mode)
{
	left_mouse_pressed  = (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS);
}

void mouse_pos_callback(GLFWwindow* window, double xpos, double ypos) {
	int width, height;
	glfwGetWindowSize(window, &width, &height);
	float fwidth = static_cast<float>(width), fheight = static_cast<float>(height);
	float phi = camera.get_phi(), theta = camera.get_theta();

	if (left_mouse_pressed) 
	{
		// compute new camera parameters with polar (spherical) coordinates
		phi   += (xpos - mouse_last_x) / fwidth ;
		theta -= (ypos - mouse_last_y) / fheight;
		theta = std::clamp(theta, 0.01f, 3.14f);
		camera.update_angle(phi, theta);
	}

	mouse_last_x = xpos;
	mouse_last_y = ypos;
}

void mouse_scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
	float zoom = camera.get_distance();
	zoom -= yoffset;
	zoom = std::clamp(zoom, 0.1f, 100.0f);
	camera.update_distance(zoom);
}