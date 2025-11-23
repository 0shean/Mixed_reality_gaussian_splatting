using UnityEngine;
using UnityEngine.XR.Interaction.Toolkit;

public class HideControllerVisuals : MonoBehaviour
{
    void Start()
    {
        Debug.Log("=== HideControllerVisuals: Disabling controller visuals ===");

        // Find and disable all XRInteractorLineVisual components (teleportation rays)
        var lineVisuals = FindObjectsOfType<XRInteractorLineVisual>();
        foreach (var visual in lineVisuals)
        {
            visual.enabled = false;
            Debug.Log($"Disabled line visual on: {visual.gameObject.name}");
        }

        // Find and disable all MeshRenderer components on controller objects
        var controllers = GameObject.FindGameObjectsWithTag("Untagged");
        foreach (var obj in controllers)
        {
            if (obj.name.Contains("Controller") || obj.name.Contains("Hand"))
            {
                // Disable all mesh renderers (hand models)
                var meshRenderers = obj.GetComponentsInChildren<MeshRenderer>();
                foreach (var renderer in meshRenderers)
                {
                    renderer.enabled = false;
                    Debug.Log($"Disabled mesh renderer on: {renderer.gameObject.name}");
                }

                // Disable all skinned mesh renderers (animated hand models)
                var skinnedRenderers = obj.GetComponentsInChildren<SkinnedMeshRenderer>();
                foreach (var renderer in skinnedRenderers)
                {
                    renderer.enabled = false;
                    Debug.Log($"Disabled skinned mesh renderer on: {renderer.gameObject.name}");
                }
            }
        }

        Debug.Log("=== HideControllerVisuals: Complete ===");
    }
}
