using UnityEngine;
using GaussianSplatting.Runtime;
using UnityEngine.XR;
using UnityEngine.UI;
using TMPro;

public class ClipQueryController : MonoBehaviour
{
    [Header("References")]
    public GaussianSplatRenderer splatRenderer;

    [Header("UI References")]
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
        Debug.Log("=== ClipQueryController Start() called ===");

        // Get VR camera
        vrCamera = Camera.main;
        if (vrCamera == null)
        {
            Debug.LogWarning("Main camera not found, searching for any camera...");
            vrCamera = FindObjectOfType<Camera>();
        }

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

        // Setup UI
        if (inputCanvas != null)
        {
            inputCanvas.SetActive(false);
            isUIVisible = false;
        }
        else
        {
            Debug.LogError("Input Canvas not assigned!");
        }

        if (sendQueryButton != null)
        {
            sendQueryButton.onClick.AddListener(OnSendQueryButtonClicked);
        }
        else
        {
            Debug.LogError("Send Query Button not assigned!");
        }

        Debug.Log("ClipQueryController initialized. Press TRIGGER (right controller) to open text input.");
    }

    void Update()
    {
        // Check for trigger press on right controller (Quest) - ONLY to open UI
        bool triggerPressed = false;
        bool bButtonPressed = false;

        InputDevice rightController = InputDevices.GetDeviceAtXRNode(XRNode.RightHand);
        if (rightController.isValid)
        {
            // Trigger to open UI
            if (rightController.TryGetFeatureValue(CommonUsages.triggerButton, out bool triggerValue))
            {
                triggerPressed = triggerValue;
            }

            // B button to close UI (secondary button on Quest)
            if (rightController.TryGetFeatureValue(CommonUsages.secondaryButton, out bool bButtonValue))
            {
                bButtonPressed = bButtonValue;
            }
        }

        // Detect trigger press (rising edge) - ONLY open UI when it's closed
        if (triggerPressed && !buttonWasPressed && !isUIVisible)
        {
            OpenInputUI();
        }

        // B button to close UI
        if (bButtonPressed && !buttonWasPressed && isUIVisible)
        {
            CloseInputUI();
        }

        buttonWasPressed = triggerPressed || bButtonPressed;
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
