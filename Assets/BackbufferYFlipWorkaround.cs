using UnityEngine;
using UnityEngine.Rendering;

[DisallowMultipleComponent]
[RequireComponent(typeof(Camera))]
public class BackbufferYFlipWorkaround : MonoBehaviour
{
    CommandBuffer _cb;

    void OnEnable()
    {
        _cb = new CommandBuffer { name = "Set _CameraTargetTexture for YFlip" };
        _cb.SetGlobalTexture("_CameraTargetTexture", BuiltinRenderTextureType.CameraTarget);

        var cam = GetComponent<Camera>();
        // Add in a couple of places to be safe
        cam.AddCommandBuffer(CameraEvent.BeforeForwardOpaque, _cb);
        cam.AddCommandBuffer(CameraEvent.BeforeSkybox, _cb);
    }

    void OnDisable()
    {
        var cam = GetComponent<Camera>();
        if (_cb != null)
        {
            cam.RemoveCommandBuffer(CameraEvent.BeforeForwardOpaque, _cb);
            cam.RemoveCommandBuffer(CameraEvent.BeforeSkybox, _cb);
            _cb.Release();
            _cb = null;
        }
    }
}
