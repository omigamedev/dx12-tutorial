struct vertex_in
{
	float3 pos : POS;
	float2 tex : TEX;
	float3 nor : NOR;
	float3 tan : TAN;
	float3 bin : BIN;
};

struct vertex_out
{
	float4 pos : SV_POSITION;
	float2 tex : TEX;
	float3 nor : NOR;
	float3 tan : TAN;
	float3 bin : BIN;
};

cbuffer uniform_t : register(b1)
{
	matrix model;
	matrix view;
	matrix proj;
};

vertex_out main(vertex_in input)
{
	vertex_out output;
	output.pos = mul(model, float4(input.pos, 1.0));
	//output.pos = mul(view, output.pos);
	//output.pos = mul(proj, output.pos);
	output.nor = mul(model, float4(input.nor, 0.0)).xyz;
	output.tan = mul(model, float4(input.tan, 0.0)).xyz;
	output.bin = mul(model, float4(input.bin, 0.0)).xyz;
	output.tex = input.tex;
	return output;
}