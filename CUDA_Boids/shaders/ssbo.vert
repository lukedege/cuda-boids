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

layout (std140, binding = 2) uniform Matrices
{
	mat4 view_matrix;
	mat4 projection_matrix;
};

out gl_PerVertex { vec4 gl_Position; };

mat4 rotation_towards(vec3 vel)
{
	vec3 up = { 0, 1, 0 };
	vec3 vel_norm = normalize(vel);
	vec3 k = cross(vel_norm, up); //k is the rotation axis
	float c = dot(vel_norm, up);
	float s = length(k);
	
	mat4 rot = {
		vec4(k.x*k.x*(1-c) + c    , k.x*k.y*(1-c) - k.z*s, k.x*k.z*(1-c) + k.y*s, 0),
		vec4(k.y*k.x*(1-c) + k.z*s, k.y*k.y*(1-c) + c    , k.y*k.z*(1-c) - k.x*s, 0),
		vec4(k.z*k.x*(1-c) - k.y*s, k.z*k.y*(1-c) + k.x*s, k.z*k.z*(1-c) + c    , 0),
		vec4(0, 0, 0, 1) };
	return rot;
}

void main()
{
	mat4 instance_rotation = rotation_towards(velocities[gl_InstanceID].xyz);
	vec4 instance_position = instance_rotation * vec4(position, 1) + vec4(positions[gl_InstanceID].xyz, 0);
	
	gl_Position = projection_matrix * view_matrix * instance_position;
}