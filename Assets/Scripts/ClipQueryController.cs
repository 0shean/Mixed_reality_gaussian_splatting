using UnityEngine;
using GaussianSplatting.Runtime;
using UnityEngine.XR;

public class ClipQueryController : MonoBehaviour
{
    [Header("References")]
    public GaussianSplatRenderer splatRenderer;

    [Header("Query Options")]
    public string[] queries = { "chair", "table", "panel", "tree" };

    [Header("Status")]
    public int currentQueryIndex = 0;
    public string currentQuery = "chair";

    private bool buttonWasPressed = false;

    void Start()
    {
        Debug.Log("=== ClipQueryController Start() called ===");

        if (splatRenderer == null)
        {
            Debug.Log("Searching for GaussianSplatRenderer...");
            splatRenderer = FindObjectOfType<GaussianSplatRenderer>();
            if (splatRenderer == null)
            {
                Debug.LogError("ClipQueryController: No GaussianSplatRenderer found in scene!");
            }
            else
            {
                Debug.Log("Found GaussianSplatRenderer!");
            }
        }

        // Check for ClipClient
        var clipClient = GetComponent<ClipClient>();
        if (clipClient == null)
        {
            Debug.LogError("ClipClient component NOT FOUND! Please add ClipClient to this GameObject.");
        }
        else
        {
            Debug.Log($"ClipClient found! Server URL: {clipClient.serverUrl}");
        }

        // Display initial query
        currentQuery = queries[currentQueryIndex];
        Debug.Log($"ClipQueryController initialized. Press TRIGGER (right controller) to query: {currentQuery}");
    }

    void Update()
    {
        // Check for trigger press on right controller (Quest)
        bool isPressed = false;

        // Try XR Input - using trigger instead of A button
        InputDevice rightController = InputDevices.GetDeviceAtXRNode(XRNode.RightHand);
        if (rightController.isValid)
        {
            // Try trigger
            if (rightController.TryGetFeatureValue(CommonUsages.triggerButton, out bool triggerValue))
            {
                isPressed = triggerValue;
            }
            // Fallback to primary button (A button)
            else if (rightController.TryGetFeatureValue(CommonUsages.primaryButton, out bool primaryButtonValue))
            {
                isPressed = primaryButtonValue;
            }
        }

        // Detect button press (rising edge)
        if (isPressed && !buttonWasPressed)
        {
            ExecuteQuery();
        }

        buttonWasPressed = isPressed;
    }

    void ExecuteQuery()
    {
        if (splatRenderer == null)
        {
            Debug.LogError("Cannot execute query: splatRenderer is null!");
            return;
        }

        currentQuery = queries[currentQueryIndex];
        Debug.Log($"========================================");
        Debug.Log($"Executing CLIP Query: \"{currentQuery}\"");
        Debug.Log($"========================================");

        // Call the ProcessCLIPQuery method
        splatRenderer.ProcessCLIPQuery(currentQuery);

        // Move to next query
        currentQueryIndex = (currentQueryIndex + 1) % queries.Length;
        string nextQuery = queries[currentQueryIndex];
        Debug.Log($"Next query will be: \"{nextQuery}\" (press A button again)");
    }

    // Allow manual testing from Inspector
    [ContextMenu("Execute Current Query")]
    void ExecuteQueryFromInspector()
    {
        ExecuteQuery();
    }
}
