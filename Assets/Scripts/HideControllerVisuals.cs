using UnityEngine;
using UnityEngine.XR.Interaction.Toolkit;

public class HideControllerVisuals : MonoBehaviour
{
    private int frameCount = 0;
    private const int CHECK_FRAMES = 10; // Check for first N frames to catch late-initialized objects

    void Start()
    {
        DisableAllVisuals();
    }

    void Update()
    {
        // Keep checking for a few frames to catch objects created after Start
        if (frameCount < CHECK_FRAMES)
        {
            DisableAllVisuals();
            frameCount++;
        }
    }

    void DisableAllVisuals()
    {
        // Find and disable all XRInteractorLineVisual components (XR Interaction Toolkit rays)
        var lineVisuals = FindObjectsOfType<XRInteractorLineVisual>(true);
        foreach (var visual in lineVisuals)
        {
            if (visual.enabled)
            {
                visual.enabled = false;
                Debug.Log($"Disabled XRInteractorLineVisual on: {visual.gameObject.name}");
            }
        }

        // Find and disable all LineRenderer components (generic rays)
        var lineRenderers = FindObjectsOfType<LineRenderer>(true);
        foreach (var lineRenderer in lineRenderers)
        {
            if (lineRenderer.enabled)
            {
                lineRenderer.enabled = false;
                Debug.Log($"Disabled LineRenderer on: {lineRenderer.gameObject.name}");
            }
        }

        // Find and disable all XRRayInteractor line visuals
        var rayInteractors = FindObjectsOfType<XRRayInteractor>(true);
        foreach (var rayInteractor in rayInteractors)
        {
            // Disable the line visual component if present
            var lineVisual = rayInteractor.GetComponent<XRInteractorLineVisual>();
            if (lineVisual != null && lineVisual.enabled)
            {
                lineVisual.enabled = false;
                Debug.Log($"Disabled ray interactor line visual on: {rayInteractor.gameObject.name}");
            }

            // Also try to disable any LineRenderer on the same object
            var lr = rayInteractor.GetComponent<LineRenderer>();
            if (lr != null && lr.enabled)
            {
                lr.enabled = false;
                Debug.Log($"Disabled ray interactor LineRenderer on: {rayInteractor.gameObject.name}");
            }
        }

        // Disable mesh renderers on controller objects
        var allObjects = FindObjectsOfType<GameObject>(true);
        foreach (var obj in allObjects)
        {
            if (obj.name.Contains("Controller") || obj.name.Contains("Hand") || obj.name.Contains("Ray"))
            {
                var meshRenderers = obj.GetComponentsInChildren<MeshRenderer>(true);
                foreach (var renderer in meshRenderers)
                {
                    if (renderer.enabled)
                    {
                        renderer.enabled = false;
                        Debug.Log($"Disabled mesh renderer on: {renderer.gameObject.name}");
                    }
                }

                var skinnedRenderers = obj.GetComponentsInChildren<SkinnedMeshRenderer>(true);
                foreach (var renderer in skinnedRenderers)
                {
                    if (renderer.enabled)
                    {
                        renderer.enabled = false;
                        Debug.Log($"Disabled skinned mesh renderer on: {renderer.gameObject.name}");
                    }
                }
            }
        }
    }
}
