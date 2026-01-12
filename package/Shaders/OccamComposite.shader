// SPDX-License-Identifier: MIT
// Composites colored gaussian splats with occam relevancy heatmap overlay
Shader "Hidden/Gaussian Splatting/OccamComposite"
{
    SubShader
    {
        Pass
        {
            ZWrite Off
            ZTest Always
            Cull Off
            Blend SrcAlpha OneMinusSrcAlpha

CGPROGRAM
#pragma vertex vert
#pragma fragment frag
#pragma require compute
#pragma use_dxc
#include "UnityCG.cginc"

struct v2f
{
    float4 vertex : SV_POSITION;
    float2 uv : TEXCOORD0;
};

v2f vert (float4 vertex : POSITION, float2 uv : TEXCOORD0)
{
    v2f o;
    o.vertex = UnityObjectToClipPos(vertex);
    o.uv = uv;
    return o;
}

sampler2D _BaseTex;     // Colored gaussian splats
sampler2D _MainTex;     // Processed occam heatmap overlay (from Blit source)

half4 frag (v2f i) : SV_Target
{
    half4 baseCol = tex2D(_BaseTex, i.uv);
    half4 occamCol = tex2D(_MainTex, i.uv);

    // Convert base splats from premultiplied alpha
    float3 baseRGB = baseCol.a > 0.001 ? baseCol.rgb / baseCol.a : float3(0,0,0);
    baseRGB = GammaToLinearSpace(baseRGB);

    // Blend: where occam has alpha, overlay the heatmap on the base color
    float3 finalColor = lerp(baseRGB, occamCol.rgb, occamCol.a);
    float finalAlpha = max(baseCol.a, occamCol.a);

    return half4(finalColor, finalAlpha);
}
ENDCG
        }
    }
}
