// SPDX-License-Identifier: MIT
Shader "Hidden/Gaussian Splatting/Composite"
{
    SubShader
    {
        Pass
        {
            ZWrite Off
            ZTest Always
            Cull Off
            Blend One Zero

CGPROGRAM
#pragma vertex vert
#pragma fragment frag
#pragma require compute
#pragma use_dxc
#include "UnityCG.cginc"


struct v2f
{
    float4 vertex : SV_POSITION;
};

v2f vert (uint vtxID : SV_VertexID)
{
    v2f o;
    float2 quadPos = float2(vtxID&1, (vtxID>>1)&1) * 4.0 - 1.0;
	o.vertex = float4(quadPos, 1, 1);
    return o;
}

Texture2D _GaussianSplatRT;
Texture2D _MainTex;
Texture2D _BaseTex;


half4 frag (v2f i) : SV_Target
{
    int3 pixelCoord = int3(i.vertex.xy, 0);

    half4 csOutput = _MainTex.Load(pixelCoord);  // Mask texture (csOutputRT)
    half4 baseCol = _BaseTex.Load(pixelCoord);   // Base Splat texture (GaussianSplatRT)

    // 1. Stabilize the Base Splat Color
    half3 base_unpre = float4((baseCol.rgb/baseCol.a),baseCol.a);
    base_unpre = saturate(base_unpre); // DEBUG CLAMP

    // 2. Stabilize the Mask Color
    half3 occam_color = saturate(csOutput.rgb); 
    
    // --- WORKAROUND START ---
    
    // Simple black/white=0, colored=1 classifier
    half luminance = dot(occam_color, half3(0.299, 0.587, 0.114));

    // Saturation = max channel - min channel (0 for gray/black/white, 1 for pure color)
    half max_channel = max(max(occam_color.r, occam_color.g), occam_color.b);
    half min_channel = min(min(occam_color.r, occam_color.g), occam_color.b);
    half saturation = saturate((max_channel - min_channel) / max(luminance, 0.001));  // Avoid div0

    half mask_blending_factor = step(0.3, saturation);  // Colored = 1, gray/black/white = 0

    // --- WORKAROUND END ---

    // 3. Blend: lerp(Background, Foreground, Factor)
    // The foreground is the mask color (occam_color), the factor is the calculated brightness.
    half3 final_unpre = lerp(base_unpre, occam_color, mask_blending_factor);

    // 4. Final Output (Opaque)
    return float4(GammaToLinearSpace(final_unpre), 1.0); 

    // return float4(1.0, 0.0, 1.0, 1.0); // Output Magenta only if ALL channels are zero
}
ENDCG
        }
    }
}


// Shader "Hidden/Gaussian Splatting/Occam Composite"
// {
//     Properties
//     {
//         _BaseTex ("Base Splats", 2D) = "white" {}
//         _Threshold ("Occam Alpha Threshold", Range(0,1)) = 0.5
//     }
//     SubShader
//     {
//         Tags { "RenderType"="Opaque" }
//         Pass
//         {
//             ZTest Always
//             ZWrite Off
//             Cull Off
//             Blend One Zero

// CGPROGRAM
// #pragma vertex vert
// #pragma fragment frag
// #pragma target 3.0

// sampler2D _MainTex; // occam texture
// sampler2D _BaseTex;
// float _Threshold;

// struct v2f {
//     float4 pos : SV_POSITION;
//     float2 uv : TEXCOORD0;
// };

// v2f vert (uint vtxID : SV_VertexID)
// {
//     v2f o;
//     float2 quadPos = float2((vtxID<<1) & 2, vtxID & 2);
//     quadPos = quadPos * float2(1.0, 1.0);
//     o.uv = float2((vtxID == 2) ? 1.0 : 0.0, (vtxID == 1) ? 1.0 : 0.0);
//     o.pos = float4(quadPos * 2.0 - 1.0, 0.0, 1.0);
//     return o;
// }

// half4 frag (v2f i) : SV_Target
// {
//     // _MainTex is the CSOutputRT (Colorized Mask/Overlay)
//     half4 csOutput = tex2D(_MainTex, i.uv); 
//     // _BaseTex is the GaussianSplatRT (Base RGB Splats)
//     half4 baseCol = tex2D(_BaseTex, i.uv); 

//     // Extract colors and blending factor
//     half3 occam_color = csOutput.rgb; 
//     half mask_alpha = saturate(csOutput.a); // The opacity/blend factor from the mask pass

//     half3 base_unpre = baseCol.a > 0.0001 ? baseCol.rgb / baseCol.a : baseCol.rgb;

//     // --- DEBUG CLAMP ---
//     // If the issue is instability, clamping here should make the background visible again.
//     base_unpre = saturate(base_unpre); 
//     // -------------------

//     // Blend: final_unpre = lerp(Background, Foreground, BlendFactor)
//     half3 final_unpre = lerp(base_unpre, occam_color, mask_alpha);

//     return float4(final_unpre, 1.0);
// }
// ENDCG
//         }
//     }
// }
