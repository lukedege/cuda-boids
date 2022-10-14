#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

namespace utils::graphics::opengl
{
	enum camdir
	{
		FORWARD, BACKWARD, LEFT, RIGHT
	};

	const float YAW = -90.f;
	const float PITCH = 0.f;
	const float ROLL = 0.f;

	const float SPEED = 3.f;
	const float SENSITIVITY = 0.25f;

	class Camera
	{
		glm::vec3 pos, front, up, right;
		glm::vec3 world_front, world_up;

		float yaw, pitch, roll;
		float mov_speed, mouse_sensitivity;

		bool on_ground;

	public:
		Camera(glm::vec3 pos, bool on_ground) : pos(pos), on_ground(on_ground),
			yaw(YAW), pitch(PITCH), roll(ROLL),
			mov_speed(SPEED), mouse_sensitivity(SENSITIVITY)
		{
			world_up = glm::vec3(0.f, 1.f, 0.f);
			updateCameraVectors();
		}
		glm::mat4 GetViewMatrix()
		{
			return glm::lookAt(pos, pos + front, up);
		}

		void ProcessKeyboard(camdir dir, float deltaTime)
		{
			float vel = mov_speed * deltaTime;

			if (dir == camdir::FORWARD)
			{
				pos += (on_ground ? world_front : front) * vel;
			}
			if (dir == camdir::BACKWARD)
			{
				pos -= (on_ground ? world_front : front) * vel;
			}
			if (dir == camdir::LEFT)
			{
				pos -= right * vel;
			}
			if (dir == camdir::RIGHT)
			{
				pos += right * vel;
			}
		}

		void ProcessMouseMovement(float x_offset, float y_offset, bool pitch_constraint = true)
		{
			x_offset *= mouse_sensitivity; y_offset *= mouse_sensitivity;

			yaw += x_offset; pitch += y_offset;

			if (pitch_constraint) // avoids gimbal lock
			{
				glm::clamp(pitch, -89.f, 89.f);
			}

			updateCameraVectors();
		}

		glm::vec3 position()
		{
			return pos;
		}

	private:
		void updateCameraVectors()
		{
			// TODO add roll
			float yaw_r = glm::radians(yaw),
				pitch_r = glm::radians(pitch),
				roll_r = glm::radians(roll);

			front.x = cos(yaw_r) * cos(pitch_r);
			front.y = sin(pitch_r);
			front.z = sin(yaw_r) * cos(pitch_r);

			world_front = front = glm::normalize(front);
			world_front.y = 0;

			right = glm::normalize(glm::cross(front, world_up));
			up = glm::normalize(glm::cross(right, front));
		}
	};
}