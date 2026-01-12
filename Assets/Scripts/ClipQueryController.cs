using UnityEngine;
using GaussianSplatting.Runtime;
using UnityEngine.XR;
using UnityEngine.UI;
using TMPro;

/// <summary>
/// DEPRECATED: Keyboard input has been removed. Use VoiceInputController instead.
/// This component is kept for reference but is disabled.
/// Left trigger = start/stop voice recording
/// B button = cancel voice recording
/// </summary>
public class ClipQueryController : MonoBehaviour
{
    [Header("References")]
    public GaussianSplatRenderer splatRenderer;

    [Header("UI References (DEPRECATED - Not used)")]
    public GameObject inputCanvas;
    public TMP_InputField queryInputField;
    public Button sendQueryButton;

    [Header("UI Positioning")]
    [Tooltip("Distance from camera where the panel appears")]
    public float panelDistance = 2.0f;
    [Tooltip("Height offset from camera (positive = higher)")]
    public float panelHeightOffset = -0.3f;

    [Header("Status")]
    public string currentQuery = "";
    public bool isUIVisible = false;

    private Camera vrCamera;

    private bool buttonWasPressed = false;

    void Start()
    {
        Debug.Log("=== ClipQueryController: DEPRECATED - Keyboard input removed. Use voice input (left trigger) instead. ===");

        // Hide keyboard UI if it exists
        if (inputCanvas != null)
        {
            inputCanvas.SetActive(false);
        }
        isUIVisible = false;

        // Get VR camera (still needed for ExecuteQuery if called externally)
        vrCamera = Camera.main;
        if (vrCamera == null)
        {
            vrCamera = FindObjectOfType<Camera>();
        }

        if (splatRenderer == null)
        {
            splatRenderer = FindObjectOfType<GaussianSplatRenderer>();
        }
    }

    void Update()
    {
        // Keyboard input disabled - use VoiceInputController instead
    }

    void OpenInputUI()
    {
        if (inputCanvas == null)
        {
            Debug.LogError("Cannot open UI: inputCanvas is null!");
            return;
        }

        // Position the canvas in front of the camera
        PositionCanvasInFrontOfCamera();

        isUIVisible = true;
        inputCanvas.SetActive(true);

        Debug.Log("Input UI opened. Point at input field and click to type. Press B button to cancel.");

        // Clear previous text
        if (queryInputField != null)
        {
            queryInputField.text = "";
            // Don't auto-activate, let user click it
            Debug.Log("Input field ready. Click on it to start typing.");
        }
    }

    void PositionCanvasInFrontOfCamera()
    {
        if (vrCamera == null || inputCanvas == null)
        {
            Debug.LogError("Cannot position canvas: camera or canvas is null!");
            return;
        }

        Transform camTransform = vrCamera.transform;

        // Get forward direction (ignore vertical tilt for more comfortable viewing)
        Vector3 forward = camTransform.forward;
        forward.y = 0;  // Keep panel level, not tilted up/down
        forward.Normalize();

        // If looking straight up/down, use camera's actual forward
        if (forward.magnitude < 0.1f)
        {
            forward = camTransform.forward;
        }

        // Calculate position in front of camera
        Vector3 position = camTransform.position + forward * panelDistance;
        position.y = camTransform.position.y + panelHeightOffset;

        // Set position
        inputCanvas.transform.position = position;

        // Make the canvas face the camera
        inputCanvas.transform.rotation = Quaternion.LookRotation(forward);

        Debug.Log($"Canvas positioned at {position}, facing {forward}");
    }

    void CloseInputUI()
    {
        if (inputCanvas == null)
        {
            Debug.LogError("Cannot close UI: inputCanvas is null!");
            return;
        }

        isUIVisible = false;
        inputCanvas.SetActive(false);
        Debug.Log("Input UI closed.");
    }

    void OnSendQueryButtonClicked()
    {
        if (queryInputField == null)
        {
            Debug.LogError("Cannot send query: queryInputField is null!");
            return;
        }

        currentQuery = queryInputField.text.Trim();

        if (string.IsNullOrEmpty(currentQuery))
        {
            Debug.LogWarning("Query is empty! Please enter a search term.");
            return;
        }

        // Hide the UI
        CloseInputUI();

        // Execute the query
        ExecuteQuery();
    }

    void ExecuteQuery()
    {
        if (splatRenderer == null)
        {
            Debug.LogError("Cannot execute query: splatRenderer is null!");
            return;
        }

        if (string.IsNullOrEmpty(currentQuery))
        {
            Debug.LogError("Cannot execute query: currentQuery is empty!");
            return;
        }

        Debug.Log($"========================================");
        Debug.Log($"Executing CLIP Query: \"{currentQuery}\"");
        Debug.Log($"========================================");

        // Call the ProcessCLIPQuery method
        splatRenderer.ProcessCLIPQuery(currentQuery);
    }

    // Allow manual testing from Inspector
    [ContextMenu("Execute Current Query")]
    void ExecuteQueryFromInspector()
    {
        ExecuteQuery();
    }
}
