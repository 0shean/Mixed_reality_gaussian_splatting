using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using GaussianSplatting.Runtime;
using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Assertions;

namespace GaussianSplatting.Editor.Utils
{
    // input file splat data is read into this format
    public struct InputSplatData
    {
        public Vector3 pos;
        public Vector3 nor;
        public Vector3 dc0;
        public Vector3 sh1, sh2, sh3, sh4, sh5, sh6, sh7, sh8, sh9, shA, shB, shC, shD, shE, shF;
        public float lang_logit0, lang_logit1, lang_logit2, lang_logit3, lang_logit4, lang_logit5, lang_logit6, lang_logit7, lang_logit8, lang_logit9, lang_logit10, lang_logit11, lang_logit12, lang_logit13, lang_logit14, lang_logit15;
        public float lang_logit16, lang_logit17, lang_logit18, lang_logit19, lang_logit20, lang_logit21, lang_logit22, lang_logit23, lang_logit24, lang_logit25, lang_logit26, lang_logit27, lang_logit28, lang_logit29, lang_logit30, lang_logit31;
        public float lang_logit32, lang_logit33, lang_logit34, lang_logit35, lang_logit36, lang_logit37, lang_logit38, lang_logit39, lang_logit40, lang_logit41, lang_logit42, lang_logit43, lang_logit44, lang_logit45, lang_logit46, lang_logit47;
        public float lang_logit48, lang_logit49, lang_logit50, lang_logit51, lang_logit52, lang_logit53, lang_logit54, lang_logit55, lang_logit56, lang_logit57, lang_logit58, lang_logit59, lang_logit60, lang_logit61, lang_logit62, lang_logit63;
        public float opacity;
        public Vector3 scale;
        public Quaternion rot;
    }

    [BurstCompile]
    public class GaussianFileReader
    {
        // Returns splat count
        public static int ReadFileHeader(string filePath)
        {
            int vertexCount = 0;
            if (File.Exists(filePath))
            {
                if (isPLY(filePath))
                    PLYFileReader.ReadFileHeader(filePath, out vertexCount, out _, out _);
                else if (isSPZ(filePath))
                    SPZFileReader.ReadFileHeader(filePath, out vertexCount);
            }
            return vertexCount;
        }

        public static unsafe void ReadFile(string filePath, out NativeArray<InputSplatData> splats)
        {
            if (isPLY(filePath))
            {
                NativeArray<byte> plyRawData;
                List<(string, PLYFileReader.ElementType)> attributes;
                PLYFileReader.ReadFile(filePath, out var splatCount, out var vertexStride, out attributes, out plyRawData);
                string attrError = CheckPLYAttributes(attributes);
                if (!string.IsNullOrEmpty(attrError))
                    throw new IOException($"PLY file is probably not a Gaussian Splat file? Missing properties: {attrError}");
                splats = PLYDataToSplats(plyRawData, splatCount, vertexStride, attributes);
                ReorderSHs(splatCount, (float*)splats.GetUnsafePtr());
                LinearizeData(splats);
                return;
            }
            if (isSPZ(filePath))
            {
                SPZFileReader.ReadFile(filePath, out splats);
                return;
            }
            throw new IOException($"File {filePath} is not a supported format");
        }

        static bool isPLY(string filePath) => filePath.EndsWith(".ply", true, CultureInfo.InvariantCulture);
        static bool isSPZ(string filePath) => filePath.EndsWith(".spz", true, CultureInfo.InvariantCulture);

        static string CheckPLYAttributes(List<(string, PLYFileReader.ElementType)> attributes)
        {
            string[] required = { "x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3" };
            List<string> missing = required.Where(req => !attributes.Contains((req, PLYFileReader.ElementType.Float))).ToList();
            if (missing.Count == 0)
                return null;
            return string.Join(",", missing);
        }

        static unsafe NativeArray<InputSplatData> PLYDataToSplats(NativeArray<byte> input, int count, int stride, List<(string, PLYFileReader.ElementType)> attributes)
        {
            NativeArray<int> fileAttrOffsets = new NativeArray<int>(attributes.Count, Allocator.Temp);
            int offset = 0;
            for (var ai = 0; ai < attributes.Count; ai++)
            {
                var attr = attributes[ai];
                fileAttrOffsets[ai] = offset;
                offset += PLYFileReader.TypeToSize(attr.Item2);
            }

            string[] splatAttributes =
            {
                "x",
                "y",
                "z",
                "nx",
                "ny",
                "nz",
                "f_dc_0",
                "f_dc_1",
                "f_dc_2",
                "f_rest_0",
                "f_rest_1",
                "f_rest_2",
                "f_rest_3",
                "f_rest_4",
                "f_rest_5",
                "f_rest_6",
                "f_rest_7",
                "f_rest_8",
                "f_rest_9",
                "f_rest_10",
                "f_rest_11",
                "f_rest_12",
                "f_rest_13",
                "f_rest_14",
                "f_rest_15",
                "f_rest_16",
                "f_rest_17",
                "f_rest_18",
                "f_rest_19",
                "f_rest_20",
                "f_rest_21",
                "f_rest_22",
                "f_rest_23",
                "f_rest_24",
                "f_rest_25",
                "f_rest_26",
                "f_rest_27",
                "f_rest_28",
                "f_rest_29",
                "f_rest_30",
                "f_rest_31",
                "f_rest_32",
                "f_rest_33",
                "f_rest_34",
                "f_rest_35",
                "f_rest_36",
                "f_rest_37",
                "f_rest_38",
                "f_rest_39",
                "f_rest_40",
                "f_rest_41",
                "f_rest_42",
                "f_rest_43",
                "f_rest_44",
                "lang_feat_logit_0",
                "lang_feat_logit_1",
                "lang_feat_logit_2",
                "lang_feat_logit_3",
                "lang_feat_logit_4",
                "lang_feat_logit_5",
                "lang_feat_logit_6",
                "lang_feat_logit_7",
                "lang_feat_logit_8",
                "lang_feat_logit_9",
                "lang_feat_logit_10",
                "lang_feat_logit_11",
                "lang_feat_logit_12",
                "lang_feat_logit_13",
                "lang_feat_logit_14",
                "lang_feat_logit_15",
                "lang_feat_logit_16",
                "lang_feat_logit_17",
                "lang_feat_logit_18",
                "lang_feat_logit_19",
                "lang_feat_logit_20",
                "lang_feat_logit_21",
                "lang_feat_logit_22",
                "lang_feat_logit_23",
                "lang_feat_logit_24",
                "lang_feat_logit_25",
                "lang_feat_logit_26",
                "lang_feat_logit_27",
                "lang_feat_logit_28",
                "lang_feat_logit_29",
                "lang_feat_logit_30",
                "lang_feat_logit_31",
                "lang_feat_logit_32",
                "lang_feat_logit_33",
                "lang_feat_logit_34",
                "lang_feat_logit_35",
                "lang_feat_logit_36",
                "lang_feat_logit_37",
                "lang_feat_logit_38",
                "lang_feat_logit_39",
                "lang_feat_logit_40",
                "lang_feat_logit_41",
                "lang_feat_logit_42",
                "lang_feat_logit_43",
                "lang_feat_logit_44",
                "lang_feat_logit_45",
                "lang_feat_logit_46",
                "lang_feat_logit_47",
                "lang_feat_logit_48",
                "lang_feat_logit_49",
                "lang_feat_logit_50",
                "lang_feat_logit_51",
                "lang_feat_logit_52",
                "lang_feat_logit_53",
                "lang_feat_logit_54",
                "lang_feat_logit_55",
                "lang_feat_logit_56",
                "lang_feat_logit_57",
                "lang_feat_logit_58",
                "lang_feat_logit_59",
                "lang_feat_logit_60",
                "lang_feat_logit_61",
                "lang_feat_logit_62",
                "lang_feat_logit_63",
                "opacity",
                "scale_0",
                "scale_1",
                "scale_2",
                "rot_0",
                "rot_1",
                "rot_2",
                "rot_3",                
            };
            Assert.AreEqual(UnsafeUtility.SizeOf<InputSplatData>() / 4, splatAttributes.Length);
            NativeArray<int> srcOffsets = new NativeArray<int>(splatAttributes.Length, Allocator.Temp);
            for (int ai = 0; ai < splatAttributes.Length; ai++)
            {
                int attrIndex = attributes.IndexOf((splatAttributes[ai], PLYFileReader.ElementType.Float));
                int attrOffset = attrIndex >= 0 ? fileAttrOffsets[attrIndex] : -1;
                srcOffsets[ai] = attrOffset;
            }
            
            NativeArray<InputSplatData> dst = new NativeArray<InputSplatData>(count, Allocator.Persistent);
            ReorderPLYData(count, (byte*)input.GetUnsafeReadOnlyPtr(), stride, (byte*)dst.GetUnsafePtr(), UnsafeUtility.SizeOf<InputSplatData>(), (int*)srcOffsets.GetUnsafeReadOnlyPtr());
            return dst;
        }

        [BurstCompile]
        static unsafe void ReorderPLYData(int splatCount, byte* src, int srcStride, byte* dst, int dstStride, int* srcOffsets)
        {
            for (int i = 0; i < splatCount; i++)
            {
                for (int attr = 0; attr < dstStride / 4; attr++)
                {
                    if (srcOffsets[attr] >= 0)
                        *(int*)(dst + attr * 4) = *(int*)(src + srcOffsets[attr]);
                }
                src += srcStride;
                dst += dstStride;
            }
        }

        [BurstCompile]
        static unsafe void ReorderSHs(int splatCount, float* data)
        {
            int splatStride = UnsafeUtility.SizeOf<InputSplatData>() / 4;
            int shStartOffset = 9, shCount = 15;
            float* tmp = stackalloc float[shCount * 3];
            int idx = shStartOffset;
            for (int i = 0; i < splatCount; ++i)
            {
                for (int j = 0; j < shCount; ++j)
                {
                    tmp[j * 3 + 0] = data[idx + j];
                    tmp[j * 3 + 1] = data[idx + j + shCount];
                    tmp[j * 3 + 2] = data[idx + j + shCount * 2];
                }

                for (int j = 0; j < shCount * 3; ++j)
                {
                    data[idx + j] = tmp[j];
                }

                idx += splatStride;
            }
        }

        [BurstCompile]
        struct LinearizeDataJob : IJobParallelFor
        {
            public NativeArray<InputSplatData> splatData;
            public void Execute(int index)
            {
                var splat = splatData[index];

                // rot
                var q = splat.rot;
                var qq = GaussianUtils.NormalizeSwizzleRotation(new float4(q.x, q.y, q.z, q.w));
                qq = GaussianUtils.PackSmallest3Rotation(qq);
                splat.rot = new Quaternion(qq.x, qq.y, qq.z, qq.w);

                // scale
                splat.scale = GaussianUtils.LinearScale(splat.scale);

                // color
                splat.dc0 = GaussianUtils.SH0ToColor(splat.dc0);
                splat.opacity = GaussianUtils.Sigmoid(splat.opacity);

                splatData[index] = splat;
            }
        }

        static void LinearizeData(NativeArray<InputSplatData> splatData)
        {
            LinearizeDataJob job = new LinearizeDataJob();
            job.splatData = splatData;
            job.Schedule(splatData.Length, 4096).Complete();
        }
    }
}