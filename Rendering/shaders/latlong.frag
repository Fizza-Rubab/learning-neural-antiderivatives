#version 420 core

in vec3 WorldPos;

out vec4 color;

uniform sampler2D equirectangularMap;

const float PI = 3.141592653;
const vec2 scale = vec2(1/(2*PI), 1/PI);

vec2 SampleSphericalMap(vec3 v){
    vec2 uv = vec2(atan(-v.z, v.x), asin(v.y));
    uv *= scale;
    uv += 0.5;
    return uv;
}


void main() {
    vec2 uv = SampleSphericalMap(normalize(WorldPos));
    color = texture2D(equirectangularMap, uv);
}
