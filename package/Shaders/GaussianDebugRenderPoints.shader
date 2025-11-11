// SPDX-License-Identifier: MIT
Shader "Gaussian Splatting/Debug/Render Points"
{
    SubShader
    {
        Tags { "RenderType"="Transparent" "Queue"="Transparent" }

        Pass
        {
            ZWrite On
            Cull Off

CGPROGRAM
#pragma vertex vert
#pragma fragment frag
#pragma require compute
#pragma use_dxc

#include "GaussianSplatting.hlsl"

struct v2f
{
    half3 color : TEXCOORD0;
    float4 vertex : SV_POSITION;
};

float _SplatSize;
bool _DisplayIndex;
bool _DisplayCLIPDotProducts;
StructuredBuffer<float> _SplatCLIPDotProducts;
StructuredBuffer<float> _SplatRelevancyScores;
int _SplatMaxRelevancyScore;
int _SplatCount;

v2f vert (uint vtxID : SV_VertexID, uint instID : SV_InstanceID)
{
    v2f o;
    uint splatIndex = instID;

    SplatData splat = LoadSplatData(splatIndex);

    float3 centerWorldPos = splat.pos;
    centerWorldPos = mul(unity_ObjectToWorld, float4(centerWorldPos,1)).xyz;

    float4 centerClipPos = mul(UNITY_MATRIX_VP, float4(centerWorldPos, 1));

    o.vertex = centerClipPos;
	uint idx = vtxID;
    float2 quadPos = float2(idx&1, (idx>>1)&1) * 2.0 - 1.0;
    o.vertex.xy += (quadPos * _SplatSize / _ScreenParams.xy) * o.vertex.w;

    o.color.rgb = saturate(splat.sh.col);
    if (_DisplayIndex)
    {
        o.color.r = frac((float)splatIndex / (float)_SplatCount * 100);
        o.color.g = frac((float)splatIndex / (float)_SplatCount * 10);
        o.color.b = (float)splatIndex / (float)_SplatCount;
    }
    if (_DisplayCLIPDotProducts)
    {
        float relevancyScore = _SplatRelevancyScores[splatIndex];
        if (relevancyScore < 0.5)
        {
            // Do not show low relevancy score splats.
            o.color = half3(0.1, 0.1, 0.1);
        }
        else
        {
            // Normalize between 0.5 and the max relevancy score.
            float normalizedScore = saturate((relevancyScore - 0.5) / (_SplatMaxRelevancyScore - 0.5));

            // Linearly interpolate between red (low) and green (high).
            float3 lowColor = float3(1, 0, 0);   // red
            float3 highColor = float3(0, 1, 0);  // green
            o.color.rgb = lerp(lowColor, highColor, normalizedScore);
        }
    }

    FlipProjectionIfBackbuffer(o.vertex);
    return o;
}

half4 frag (v2f i) : SV_Target
{
    return half4(i.color.rgb, 1);
}
ENDCG
        }
    }
}
