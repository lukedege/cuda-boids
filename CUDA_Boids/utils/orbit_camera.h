#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

namespace utils::graphics::opengl
{
	// Control camera using spherical coordinates (distance, polar angle, azimuth angle)
	class orbit_camera
	{
		const float SPEED = 6.f;
		const float SENSITIVITY = 0.05f;

	public:
		orbit_camera(glm::vec3 look_at, float distance) :
			distance{ distance }, phi{ glm::radians(-40.f) }, theta{ glm::radians(70.f) },
			pos{ position(distance, phi, theta) },
			look_at{ look_at },
			mov_speed{ SPEED }, mouse_sensitivity{ SENSITIVITY }
		{}

		glm::mat4 view_matrix()
		{
			return glm::lookAt(pos, look_at, glm::vec3(0,0,1));
		}

		void update_angle(float phi, float theta)
		{
			this->phi = phi;
			this->theta = theta;
			pos = position(distance, phi, theta);
		}

		void update_distance(float distance)
		{
			this->distance = distance;
			pos = position(distance, phi, theta);
		}

		glm::vec3 position(float distance, float phi, float theta)
		{
			glm::vec3 pos;
			pos.x = distance * sin(phi) * sin(theta);
			pos.y = distance * cos(phi) * sin(theta);
			pos.z = distance * cos(theta);
			return pos;
		}

		float get_distance()
		{
			return distance;
		}

		float get_phi()
		{
			return phi;
		}

		float get_theta()
		{
			return theta;
		}

	private:
		float distance, phi, theta;
		glm::vec3 pos, look_at;

		float mov_speed, mouse_sensitivity;

	};
}