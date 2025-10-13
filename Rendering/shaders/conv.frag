#version 430 core

layout(location=0) out vec4 color;
in vec3 WorldPos;
in vec3 aNormal;

uniform sampler2D equiMap;
uniform sampler2D brdfMap;
uniform float brdfRes;
uniform vec3 cameraPos;
uniform float shininess;
uniform float kd;
uniform int impSampling;
uniform int reference;
const float PI = 3.141592653;




float random2(vec2 n) { 
	return fract(sin(dot(n, vec2(12.9898, 4.1414))) * 43758.5453);
}

vec2 directionToSphericalEnvmap(vec3 dir) {
  float s = 1.0 - mod(1.0 / (2.0*PI) * atan(-dir.z, -dir.x), 1.0); //[-pi, pi]
  float t = 1.0 / (PI) * acos(-dir.y); // [0, pi]
  return vec2(s, t);
}


mat3 getNormalSpace(in vec3 normal) {
   vec3 someVec = vec3(1.0, 0.0, 0.0);
   float dd = dot(someVec, normal);
   vec3 tangent = vec3(0.0, 1.0, 0.0);
   if(abs(dd) > 1e-8) {
     tangent = normalize(cross(someVec, normal));
   }
   vec3 bitangent = cross(normal, tangent);
   return mat3(tangent, bitangent, normal);
}

mat3 getTangentBasis(vec3 N) {
    vec3 up = abs(N.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(0.0, 1.0, 0.0);
    vec3 tangent = normalize(cross(up, N));
    vec3 bitangent = cross(N, tangent);
    return mat3(tangent, bitangent, N); // Columns = local-to-world
}


float radicalInverse(uint bits) {
  bits = (bits << 16u) | (bits >> 16u);
  bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
  bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
  bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
  bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
  return float(bits) * 2.3283064365386963e-10; // / 0x100000000
}

vec2 hammersley(uint n, uint N) {
  return vec2(float(n) / float(N), radicalInverse(n));
}

mat3 rotation_matrix(vec3 axis, float theta) {
    axis = normalize(axis);
    float a = cos(theta / 2.0);
    float b = -axis.x * sin(theta / 2.0);
    float c = -axis.y * sin(theta / 2.0);
    float d = -axis.z * sin(theta / 2.0);
    return mat3(
        a*a + b*b - c*c - d*d, 2.0 * (b*c + a*d), 2.0 * (b*d - a*c),
        2.0 * (b*c - a*d), a*a + c*c - b*b - d*d, 2.0 * (c*d + a*b),
        2.0 * (b*d + a*c), 2.0 * (c*d - a*b), a*a + d*d - b*b - c*c
    );
}


void main() {
    float ks = 1.0 - kd;
    vec4 ambient = vec4(1., 1. ,1., 1.);
    vec3 diffuse = vec3(0.0);  
    float exponent = shininess + 4.0;
    vec3 specular = vec3(0.0);  
    float sampleDelta = 1.0/(120);
    float nrSamples = 0.0; 
    vec3 norm = normalize(aNormal);
    vec3 N = normalize(aNormal);
    vec3 V = normalize(cameraPos - WorldPos);
    vec3 RV = normalize(reflect(-V, N));
    mat3 reflectedSpace = getNormalSpace(RV);
    mat3 normalSpace = getNormalSpace(N);
    
    if (reference == 0){
      float phiRV = PI - atan(-RV.z, -RV.x);
      float thetaRV = acos(-RV.y)  -  PI / 2.0; 
      for (float dTheta = -PI/2.0; dTheta < PI/2.0; dTheta += sampleDelta) {
          float theta = thetaRV + dTheta;
          float factor = 1.0;
          for (float dPhi = -PI; dPhi < PI; dPhi += sampleDelta) {
              float phi = phiRV + dPhi;
              float uEnv = mod((phi / (2.0 * PI) + 0.5), 1.0); 
              float vEnv = mod((theta / PI + 0.5), 1.0);
              float uK = mod(((dPhi*factor) / (2.0 * PI))  + 0.5, 1.0);
              float vK = mod((dTheta / (2.0 * PI)) + 0.5, 1.0);
              float brdf = max(texture(brdfMap, vec2(uK, vK)).r, 0.0);
              vec3 li = texture(equiMap, vec2(uEnv, vEnv)).rgb;
              specular += li * brdf * sampleDelta * sampleDelta * factor;
              nrSamples++;
          }
      }
      color.rgb =  (shininess) * vec3(specular);
      color.a = 1.0;
    }
    else if (reference == 1) {
      int NUM_SAMPLES = 100;
      float r = random2(RV.xy);
      float phiRV = PI - atan(-RV.z, -RV.x);
      float thetaRV = acos(-RV.y) - PI / 2.0;
      for (uint n = 1u; n <= NUM_SAMPLES; n++) {
          vec2 p = hammersley(n, NUM_SAMPLES);
          float dPhi = (p.x - 0.5) * 2.0 * PI;
          float dTheta = (p.y - 0.5) * PI;  
          float phi = phiRV + dPhi;
          float theta = thetaRV + dTheta;
          float uEnv = mod(phi / (2.0 * PI) + 0.5, 1.0);
          float vEnv = mod(theta / PI + 0.5, 1.0);
          float uK = mod((dPhi / (2.0 * PI)) + 0.5, 1.0);
          float vK = mod((dTheta / PI) + 0.5, 1.0);
          vec3 li = texture(equiMap, vec2(uEnv, vEnv)).rgb;
          float brdf = texture(brdfMap, vec2(uK, vK)).r;
          specular += li * brdf;
      }
      specular *= (2.0 * PI * PI) / float(NUM_SAMPLES);
      color.rgb = shininess * specular;
      color.a = 1.0;
  }
    else if (reference == 2){
      for (float phi = 0.0; phi < 2.0 * PI; phi += sampleDelta)
      {
          for (float theta = 0.0; theta < 0.5 * PI; theta += sampleDelta / 2.0)
          {   
              vec3 tangentSample = vec3(sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta));
              vec3 L = normalize(reflectedSpace * tangentSample);
              // if (dot(L, N) >= 0){
              vec2 uv = directionToSphericalEnvmap(L);
              vec3 envcolor = texture2D(equiMap, uv).rgb;
              float brdf = pow(max(dot(L, RV), 0.0), exponent);
              specular += envcolor * brdf * sin(theta)  * sampleDelta * sampleDelta / 2.0 ;
              nrSamples++;
              // }
          }
      }
      color.rgb = (exponent) * vec3(specular);
      color.a = 1.0;
      // vec2 uv = directionToSphericalEnvmap(normalize(RV));
      // color = vec4(texture(equiMap, uv).rgb, 1.0);
      }
    else if (reference == 3){
      int NUM_SAMPLES = int(2.*PI/(sampleDelta) * PI/(sampleDelta/2.));
      float r = random2(RV.xy);
      for(uint n = 1u; n <= NUM_SAMPLES; n++) {
          vec2 p = hammersley(n, NUM_SAMPLES);
          float theta = acos(pow(1.0 - p.y, 1.0/(exponent + 1.0)));
          float phi = 2.0 * PI * p.x;
          vec3 pos = vec3(sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta));
          vec3 posGlob = reflectedSpace * pos;
          vec2 uv = directionToSphericalEnvmap(posGlob);
          vec3 radiance = texture2D(equiMap, uv).rgb;
          float brdf = pow(cos(theta), exponent);
          float pdf = (exponent + 1.0) * pow(cos(theta), exponent) / (2.0 * PI);
          specular += radiance * cos(theta) * (2.0 * PI) / (exponent + 1.0);
      }

      specular = specular / float(NUM_SAMPLES);
      color.rgb = (exponent + 1.0) * specular;
      color.a = 1.0;
    }
    

}
