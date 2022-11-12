// Vertex shader
// The output of a vertex shader is the position of the vertex
// after any kind of elaboration
// #version 430 core
#define PI 3.1415926538

layout (location = 0) in vec3 position;

layout (std430, binding = 0) buffer Positions
{
	vec4 positions[];
};

layout (std430, binding = 1) buffer Velocities
{
	vec4 velocities[];
};

uniform mat4 view_matrix;
uniform mat4 projection_matrix;

out gl_PerVertex { vec4 gl_Position; };

mat4 calculate_instance_rotation_matrix()
{
	vec3 vel_norm = normalize(velocities[gl_InstanceID].xyz);
	vec3 ref      = { 1, 0, 0 };
	float angle = (PI/2) - acos(clamp(dot(vel_norm, ref), -1.f, 1.f));
	mat4 ret = {
		vec4( cos(angle), -sin(angle), 0.0,  0.0 ),
		vec4( sin(angle), cos(angle),  0.0,  0.0 ),
		vec4( 0.0,        0.0,         1.0,  0.0 ),
		vec4( 0.0,        0.0,         0.0,  1.0 ) };
	return ret;
}

void main()
{
	mat4 instance_rotation = calculate_instance_rotation_matrix();
	mat4 instance_matrix   = mat4(1);
	vec4 instance_position = ((instance_rotation * instance_matrix) * vec4(position, 1)) + vec4(positions[gl_InstanceID].xyz, 0);

	gl_Position = projection_matrix * view_matrix * instance_position;
}