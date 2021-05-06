struct pixel_in
{
	float4 pos : SV_POSITION;
	float2 tex : TEX;
};

struct pixel_out
{
	float4 col : SV_TARGET;
};

cbuffer uniform_t : register(b0)
{
	float4 color;
};

Texture2D Texture : register(t0);
sampler   Sampler : register(s0);

pixel_out main(pixel_in input)
{
	pixel_out output;
	output.col = Texture.Sample(Sampler, input.tex) * color;
	return output;
}
