using UnityEngine;
using UnityEngine.XR;
using UnityEngine.Networking;
using GaussianSplatting.Runtime;
using System;
using System.Collections;

#if PLATFORM_ANDROID
using UnityEngine.Android;
#endif

/// <summary>
/// Voice input controller for Meta Quest
/// Records audio and sends to server for speech-to-text using Whisper
/// Controls:
/// - LEFT TRIGGER: Start/stop voice recording
/// - RIGHT TRIGGER: Toggle render mode (Gaussian Splats / Occam Similarities)
/// - B BUTTON: Cancel recording
/// </summary>
public class VoiceInputController : MonoBehaviour
{
    [Header("References")]
    [Tooltip("Leave empty to auto-find")]
    public GaussianSplatRenderer splatRenderer;

    [Header("Server Settings")]
    [Tooltip("URL of your Python server")]
    public string serverUrl = "http://127.0.0.1:8000";

    [Header("Recording Settings")]
    public int recordingLength = 10;  // Max recording length in seconds
    public int sampleRate = 16000;    // Sample rate for audio

    [Header("Visual Feedback")]
    public GameObject listeningIndicator;
    public GameObject lastQueryPanel;

    [Header("UI Positioning (distance from camera)")]
    public float uiDistance = 1.5f;

    [Header("Status")]
    public bool isListening = false;
    public string lastRecognizedText = "";

    private bool leftTriggerWasPressed = false;
    private bool rightTriggerWasPressed = false;
    private bool bButtonWasPressed = false;
    private bool aButtonWasPressed = false;
    private Camera vrCamera;
    private AudioClip recordedClip;
    private string microphoneDevice;
    private bool permissionGranted = false;
    private GameObject uiContainer;
    private TMPro.TextMeshProUGUI lastQueryText;
    private TMPro.TextMeshProUGUI fpsText;
    private float fpsUpdateInterval = 0.5f;
    private float fpsTimer = 0f;
    private int frameCounter = 0;
    private Sprite roundedRectSprite;
    private GameObject renderModePanel;
    private TMPro.TextMeshProUGUI renderModeText;
    private int currentRenderModeIndex = 0;  // 0 = Splats, 1 = OccamSimilarities, 2 = OccamColoredSimilarities

    void Start()
    {
        Debug.Log("=== VoiceInputController Start() ===");

        // Get VR camera
        vrCamera = Camera.main;
        if (vrCamera == null)
        {
            vrCamera = FindObjectOfType<Camera>();
        }

        if (splatRenderer == null)
        {
            splatRenderer = FindObjectOfType<GaussianSplatRenderer>();
            if (splatRenderer == null)
            {
                Debug.LogError("VoiceInputController: No GaussianSplatRenderer found!");
            }
        }

        // Get server URL from ClipClient if available
        var clipClient = FindObjectOfType<ClipClient>();
        if (clipClient != null)
        {
            serverUrl = clipClient.serverUrl;
            Debug.Log($"Using server URL from ClipClient: {serverUrl}");
        }

        // Create UI container parented to camera (for screen-space effect in VR)
        CreateScreenSpaceUI();

        // Request microphone permission
        RequestMicrophonePermission();

        Debug.Log("VoiceInputController initialized. LEFT TRIGGER = voice input, RIGHT TRIGGER = toggle render mode, B = cancel recording.");
    }

    void CreateScreenSpaceUI()
    {
        if (vrCamera == null) return;

        // Create rounded rectangle sprite for panels
        roundedRectSprite = CreateRoundedRectSprite(128, 128, 32);

        // Create container that will be parented to camera
        uiContainer = new GameObject("VoiceInputUI");
        uiContainer.transform.SetParent(vrCamera.transform);
        uiContainer.transform.localPosition = new Vector3(0, 0, uiDistance);
        uiContainer.transform.localRotation = Quaternion.identity;

        // Create the last query panel (top-right corner)
        CreateLastQueryPanel();

        // Create the listening indicator (center)
        CreateListeningIndicator();

        // Create render mode display panel (bottom-center)
        CreateRenderModePanel();

        // Hide listening indicator initially, but keep last query panel and render mode button visible
        if (listeningIndicator != null)
        {
            listeningIndicator.SetActive(false);
        }
    }

    Sprite CreateRoundedRectSprite(int width, int height, int radius)
    {
        Texture2D texture = new Texture2D(width, height, TextureFormat.RGBA32, false);
        texture.filterMode = FilterMode.Bilinear;

        Color32 white = new Color32(255, 255, 255, 255);
        Color32 clear = new Color32(0, 0, 0, 0);

        // Fill with transparent
        Color32[] pixels = new Color32[width * height];
        for (int i = 0; i < pixels.Length; i++)
            pixels[i] = clear;

        // Draw rounded rectangle
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                bool inside = false;

                // Check if inside the main rectangle (excluding corners)
                if (x >= radius && x < width - radius)
                    inside = true;
                else if (y >= radius && y < height - radius)
                    inside = true;
                else
                {
                    // Check corners with circle formula
                    int cx, cy;

                    // Bottom-left corner
                    if (x < radius && y < radius)
                    {
                        cx = radius;
                        cy = radius;
                    }
                    // Bottom-right corner
                    else if (x >= width - radius && y < radius)
                    {
                        cx = width - radius - 1;
                        cy = radius;
                    }
                    // Top-left corner
                    else if (x < radius && y >= height - radius)
                    {
                        cx = radius;
                        cy = height - radius - 1;
                    }
                    // Top-right corner
                    else
                    {
                        cx = width - radius - 1;
                        cy = height - radius - 1;
                    }

                    float dist = Mathf.Sqrt((x - cx) * (x - cx) + (y - cy) * (y - cy));
                    inside = dist <= radius;
                }

                if (inside)
                    pixels[y * width + x] = white;
            }
        }

        texture.SetPixels32(pixels);
        texture.Apply();

        // Create sprite with 9-slice borders
        Vector4 border = new Vector4(radius, radius, radius, radius);
        Sprite sprite = Sprite.Create(texture, new Rect(0, 0, width, height), new Vector2(0.5f, 0.5f), 100f, 0, SpriteMeshType.FullRect, border);

        return sprite;
    }

    void CreateFPSPanel()
    {
        // Create Canvas for FPS (top-left)
        GameObject canvasObj = new GameObject("FPSCanvas");
        canvasObj.transform.SetParent(uiContainer.transform, false);
        Canvas canvas = canvasObj.AddComponent<Canvas>();
        canvas.renderMode = RenderMode.WorldSpace;

        RectTransform canvasRect = canvasObj.GetComponent<RectTransform>();
        canvasRect.sizeDelta = new Vector2(150, 50);
        canvasRect.localScale = new Vector3(0.001f, 0.001f, 0.001f);
        // Position in top-left corner
        canvasRect.localPosition = new Vector3(-0.35f, 0.25f, 0);

        // Add background panel
        GameObject panelObj = new GameObject("Background");
        panelObj.transform.SetParent(canvasObj.transform, false);

        RectTransform panelRect = panelObj.AddComponent<RectTransform>();
        panelRect.anchorMin = Vector2.zero;
        panelRect.anchorMax = Vector2.one;
        panelRect.sizeDelta = Vector2.zero;
        panelRect.offsetMin = Vector2.zero;
        panelRect.offsetMax = Vector2.zero;

        UnityEngine.UI.Image panelImage = panelObj.AddComponent<UnityEngine.UI.Image>();
        panelImage.color = new Color(0.1f, 0.1f, 0.1f, 0.8f);  // Dark semi-transparent

        // Add FPS text
        GameObject textObj = new GameObject("FPSText");
        textObj.transform.SetParent(canvasObj.transform, false);

        RectTransform textRect = textObj.AddComponent<RectTransform>();
        textRect.anchorMin = Vector2.zero;
        textRect.anchorMax = Vector2.one;
        textRect.sizeDelta = Vector2.zero;
        textRect.offsetMin = new Vector2(5, 5);
        textRect.offsetMax = new Vector2(-5, -5);

        fpsText = textObj.AddComponent<TMPro.TextMeshProUGUI>();
        fpsText.text = "FPS: --";
        fpsText.fontSize = 28;
        fpsText.color = Color.green;
        fpsText.alignment = TMPro.TextAlignmentOptions.Center;

        Debug.Log("Created FPS panel (top-left)");
    }

    void CreateLastQueryPanel()
    {
        // Create Canvas for last query (top-right) - Meta style
        GameObject canvasObj = new GameObject("LastQueryCanvas");
        canvasObj.transform.SetParent(uiContainer.transform, false);
        Canvas canvas = canvasObj.AddComponent<Canvas>();
        canvas.renderMode = RenderMode.WorldSpace;

        RectTransform canvasRect = canvasObj.GetComponent<RectTransform>();
        canvasRect.sizeDelta = new Vector2(500, 100);
        canvasRect.localScale = new Vector3(0.001f, 0.001f, 0.001f);
        // Position in top-right corner
        canvasRect.localPosition = new Vector3(0.38f, 0.22f, 0);

        // Add background panel with Meta-style color
        GameObject panelObj = new GameObject("Background");
        panelObj.transform.SetParent(canvasObj.transform, false);

        RectTransform panelRect = panelObj.AddComponent<RectTransform>();
        panelRect.anchorMin = Vector2.zero;
        panelRect.anchorMax = Vector2.one;
        panelRect.sizeDelta = Vector2.zero;
        panelRect.offsetMin = Vector2.zero;
        panelRect.offsetMax = Vector2.zero;

        UnityEngine.UI.Image panelImage = panelObj.AddComponent<UnityEngine.UI.Image>();
        // Meta-style dark background with slight blue tint
        panelImage.color = new Color(0.12f, 0.13f, 0.15f, 0.92f);
        // Use rounded sprite
        if (roundedRectSprite != null)
        {
            panelImage.sprite = roundedRectSprite;
            panelImage.type = UnityEngine.UI.Image.Type.Sliced;
        }

        // Add label text
        GameObject labelObj = new GameObject("Label");
        labelObj.transform.SetParent(canvasObj.transform, false);

        RectTransform labelRect = labelObj.AddComponent<RectTransform>();
        labelRect.anchorMin = new Vector2(0, 0.55f);
        labelRect.anchorMax = new Vector2(1, 1f);
        labelRect.sizeDelta = Vector2.zero;
        labelRect.offsetMin = new Vector2(24, 0);
        labelRect.offsetMax = new Vector2(-24, -10);

        TMPro.TextMeshProUGUI labelText = labelObj.AddComponent<TMPro.TextMeshProUGUI>();
        labelText.text = "Current Query";
        labelText.fontSize = 22;
        labelText.color = new Color(0.6f, 0.65f, 0.7f, 1f);  // Subtle gray-blue
        labelText.alignment = TMPro.TextAlignmentOptions.Left;
        labelText.fontStyle = TMPro.FontStyles.Normal;

        // Add query text
        GameObject textObj = new GameObject("QueryText");
        textObj.transform.SetParent(canvasObj.transform, false);

        RectTransform textRect = textObj.AddComponent<RectTransform>();
        textRect.anchorMin = new Vector2(0, 0);
        textRect.anchorMax = new Vector2(1, 0.55f);
        textRect.sizeDelta = Vector2.zero;
        textRect.offsetMin = new Vector2(24, 12);
        textRect.offsetMax = new Vector2(-24, 0);

        lastQueryText = textObj.AddComponent<TMPro.TextMeshProUGUI>();
        lastQueryText.text = "None";
        lastQueryText.fontSize = 30;
        lastQueryText.color = new Color(0.95f, 0.95f, 0.97f, 1f);  // Soft white
        lastQueryText.alignment = TMPro.TextAlignmentOptions.Left;
        lastQueryText.fontStyle = TMPro.FontStyles.Normal;

        lastQueryPanel = canvasObj;
        Debug.Log("Created last query panel (top-right) - Meta style");
    }

    void RequestMicrophonePermission()
    {
#if PLATFORM_ANDROID
        if (!Permission.HasUserAuthorizedPermission(Permission.Microphone))
        {
            Debug.Log("Requesting microphone permission...");
            Permission.RequestUserPermission(Permission.Microphone);
        }
        else
        {
            Debug.Log("Microphone permission already granted.");
            permissionGranted = true;
            InitializeMicrophone();
        }
#else
        permissionGranted = true;
        InitializeMicrophone();
#endif
    }

    void InitializeMicrophone()
    {
        // Get available microphone devices
        if (Microphone.devices.Length > 0)
        {
            microphoneDevice = Microphone.devices[0];
            Debug.Log($"Using microphone: {microphoneDevice}");
        }
        else
        {
            Debug.LogError("No microphone devices found!");
        }
    }

    void CreateListeningIndicator()
    {
        // Create Canvas (centered in screen) - Meta style
        GameObject canvasObj = new GameObject("ListeningIndicatorCanvas");
        canvasObj.transform.SetParent(uiContainer.transform, false);
        Canvas canvas = canvasObj.AddComponent<Canvas>();
        canvas.renderMode = RenderMode.WorldSpace;

        RectTransform canvasRect = canvasObj.GetComponent<RectTransform>();
        canvasRect.sizeDelta = new Vector2(550, 140);
        canvasRect.localScale = new Vector3(0.001f, 0.001f, 0.001f);
        // Position in center
        canvasRect.localPosition = Vector3.zero;

        // Add background panel - Meta style
        GameObject panelObj = new GameObject("Background");
        panelObj.transform.SetParent(canvasObj.transform, false);

        RectTransform panelRect = panelObj.AddComponent<RectTransform>();
        panelRect.anchorMin = Vector2.zero;
        panelRect.anchorMax = Vector2.one;
        panelRect.sizeDelta = Vector2.zero;
        panelRect.offsetMin = Vector2.zero;
        panelRect.offsetMax = Vector2.zero;

        UnityEngine.UI.Image panelImage = panelObj.AddComponent<UnityEngine.UI.Image>();
        // Meta-style dark background (will be tinted for recording state)
        panelImage.color = new Color(0.12f, 0.13f, 0.15f, 0.95f);
        // Use rounded sprite
        if (roundedRectSprite != null)
        {
            panelImage.sprite = roundedRectSprite;
            panelImage.type = UnityEngine.UI.Image.Type.Sliced;
        }

        // Add accent bar on left side (Meta style indicator) - with rounded ends
        GameObject accentObj = new GameObject("Accent");
        accentObj.transform.SetParent(canvasObj.transform, false);

        RectTransform accentRect = accentObj.AddComponent<RectTransform>();
        accentRect.anchorMin = new Vector2(0, 0);
        accentRect.anchorMax = new Vector2(0, 1);
        accentRect.pivot = new Vector2(0, 0.5f);
        accentRect.sizeDelta = new Vector2(8, 0);
        accentRect.anchoredPosition = new Vector2(0, 0);
        accentRect.offsetMin = new Vector2(16, 20);
        accentRect.offsetMax = new Vector2(24, -20);

        UnityEngine.UI.Image accentImage = accentObj.AddComponent<UnityEngine.UI.Image>();
        // Meta blue accent
        accentImage.color = new Color(0.2f, 0.5f, 1f, 1f);
        // Create small rounded sprite for accent bar
        Sprite accentSprite = CreateRoundedRectSprite(32, 64, 8);
        if (accentSprite != null)
        {
            accentImage.sprite = accentSprite;
            accentImage.type = UnityEngine.UI.Image.Type.Sliced;
        }

        // Add main text
        GameObject textObj = new GameObject("ListeningText");
        textObj.transform.SetParent(canvasObj.transform, false);

        RectTransform textRect = textObj.AddComponent<RectTransform>();
        textRect.anchorMin = Vector2.zero;
        textRect.anchorMax = Vector2.one;
        textRect.sizeDelta = Vector2.zero;
        textRect.offsetMin = new Vector2(40, 18);
        textRect.offsetMax = new Vector2(-24, -18);

        TMPro.TextMeshProUGUI text = textObj.AddComponent<TMPro.TextMeshProUGUI>();
        text.text = "Recording...";
        text.fontSize = 32;
        text.color = new Color(0.95f, 0.95f, 0.97f, 1f);  // Soft white
        text.alignment = TMPro.TextAlignmentOptions.Center;

        listeningIndicator = canvasObj;
        Debug.Log("Created listening indicator canvas (center, screen-locked) - Meta style");
    }

    void CreateRenderModePanel()
    {
        // Create Canvas for render mode display (top-right)
        GameObject canvasObj = new GameObject("RenderModePanelCanvas");
        canvasObj.transform.SetParent(uiContainer.transform, false);
        Canvas canvas = canvasObj.AddComponent<Canvas>();
        canvas.renderMode = RenderMode.WorldSpace;

        RectTransform canvasRect = canvasObj.GetComponent<RectTransform>();
        canvasRect.sizeDelta = new Vector2(400, 80);
        canvasRect.localScale = new Vector3(0.001f, 0.001f, 0.001f);
        // Position at top-right corner (mirroring the query panel on the left)
        canvasRect.localPosition = new Vector3(-0.38f, 0.22f, 0);

        // Add background panel with Meta-style color
        GameObject panelObj = new GameObject("Background");
        panelObj.transform.SetParent(canvasObj.transform, false);

        RectTransform panelRect = panelObj.AddComponent<RectTransform>();
        panelRect.anchorMin = Vector2.zero;
        panelRect.anchorMax = Vector2.one;
        panelRect.sizeDelta = Vector2.zero;
        panelRect.offsetMin = Vector2.zero;
        panelRect.offsetMax = Vector2.zero;

        UnityEngine.UI.Image panelImage = panelObj.AddComponent<UnityEngine.UI.Image>();
        // Meta-style dark background
        panelImage.color = new Color(0.12f, 0.13f, 0.15f, 0.92f);
        if (roundedRectSprite != null)
        {
            panelImage.sprite = roundedRectSprite;
            panelImage.type = UnityEngine.UI.Image.Type.Sliced;
        }

        // Add accent bar on left side
        GameObject accentObj = new GameObject("Accent");
        accentObj.transform.SetParent(canvasObj.transform, false);

        RectTransform accentRect = accentObj.AddComponent<RectTransform>();
        accentRect.anchorMin = new Vector2(0, 0);
        accentRect.anchorMax = new Vector2(0, 1);
        accentRect.pivot = new Vector2(0, 0.5f);
        accentRect.sizeDelta = new Vector2(8, 0);
        accentRect.offsetMin = new Vector2(16, 16);
        accentRect.offsetMax = new Vector2(24, -16);

        UnityEngine.UI.Image accentImage = accentObj.AddComponent<UnityEngine.UI.Image>();
        // Purple accent for Gaussian Splats mode
        accentImage.color = new Color(0.6f, 0.4f, 0.9f, 1f);
        Sprite accentSprite = CreateRoundedRectSprite(32, 64, 8);
        if (accentSprite != null)
        {
            accentImage.sprite = accentSprite;
            accentImage.type = UnityEngine.UI.Image.Type.Sliced;
        }

        // Add display text
        GameObject textObj = new GameObject("ModeText");
        textObj.transform.SetParent(canvasObj.transform, false);

        RectTransform textRect = textObj.AddComponent<RectTransform>();
        textRect.anchorMin = Vector2.zero;
        textRect.anchorMax = Vector2.one;
        textRect.sizeDelta = Vector2.zero;
        textRect.offsetMin = new Vector2(40, 10);
        textRect.offsetMax = new Vector2(-20, -10);

        renderModeText = textObj.AddComponent<TMPro.TextMeshProUGUI>();
        renderModeText.text = "Gaussian Splats";
        renderModeText.fontSize = 28;
        renderModeText.color = new Color(0.95f, 0.95f, 0.97f, 1f);
        renderModeText.alignment = TMPro.TextAlignmentOptions.Center;

        renderModePanel = canvasObj;
        Debug.Log("Created render mode display panel (top-right). Press RIGHT TRIGGER to toggle.");
    }

    void ToggleRenderMode()
    {
        if (splatRenderer == null)
        {
            Debug.LogWarning("Cannot toggle render mode: splatRenderer is null!");
            return;
        }

        // If in OccamColoredSimilarities (2), go to OccamSimilarities (1)
        // Otherwise toggle between Splats (0) and OccamSimilarities (1)
        if (currentRenderModeIndex == 2)
        {
            currentRenderModeIndex = 1;
        }
        else
        {
            currentRenderModeIndex = (currentRenderModeIndex == 0) ? 1 : 0;
        }

        ApplyRenderMode();
    }

    void ToggleColoredMode()
    {
        if (splatRenderer == null)
        {
            Debug.LogWarning("Cannot toggle colored mode: splatRenderer is null!");
            return;
        }

        // If already in OccamColoredSimilarities (2), go to OccamSimilarities (1)
        // Otherwise go to OccamColoredSimilarities (2)
        if (currentRenderModeIndex == 2)
        {
            currentRenderModeIndex = 1;
        }
        else
        {
            currentRenderModeIndex = 2;
        }

        ApplyRenderMode();
    }

    void ApplyRenderMode()
    {
        switch (currentRenderModeIndex)
        {
            case 0:
                splatRenderer.m_RenderMode = GaussianSplatRenderer.RenderMode.Splats;
                if (renderModeText != null)
                    renderModeText.text = "Gaussian Splats";
                Debug.Log("Switched to Gaussian Splats mode");
                break;
            case 1:
                splatRenderer.m_RenderMode = GaussianSplatRenderer.RenderMode.OccamSimilarities;
                if (renderModeText != null)
                    renderModeText.text = "Occam Similarities";
                Debug.Log("Switched to Occam Similarities mode");
                break;
            case 2:
                splatRenderer.m_RenderMode = GaussianSplatRenderer.RenderMode.OccamColoredSimilarities;
                if (renderModeText != null)
                    renderModeText.text = "Occam Colored";
                Debug.Log("Switched to Occam Colored Similarities mode");
                break;
        }

        // Update accent color based on mode
        UpdateRenderModeAccent();
    }

    void UpdateRenderModeAccent()
    {
        if (renderModePanel == null) return;

        var images = renderModePanel.GetComponentsInChildren<UnityEngine.UI.Image>();
        foreach (var image in images)
        {
            if (image.gameObject.name == "Accent")
            {
                // Different colors for each mode
                Color accentColor = currentRenderModeIndex switch
                {
                    0 => new Color(0.6f, 0.4f, 0.9f, 1f),   // Purple for Gaussian Splats
                    1 => new Color(0.2f, 0.7f, 0.8f, 1f),   // Cyan for Occam Similarities
                    2 => new Color(0.9f, 0.6f, 0.2f, 1f),   // Orange for Occam Colored
                    _ => new Color(0.6f, 0.4f, 0.9f, 1f)    // Default purple
                };
                image.color = accentColor;
            }
        }
    }

    void Update()
    {
        // Check for left controller trigger (voice input)
        bool leftTriggerPressed = false;
        // Check for right controller trigger (render mode toggle)
        bool rightTriggerPressed = false;
        bool bButtonPressed = false;
        bool aButtonPressed = false;  // A button on right controller for colored mode

        InputDevice leftController = InputDevices.GetDeviceAtXRNode(XRNode.LeftHand);
        if (leftController.isValid)
        {
            if (leftController.TryGetFeatureValue(CommonUsages.triggerButton, out bool triggerValue))
            {
                leftTriggerPressed = triggerValue;
            }
        }

        InputDevice rightController = InputDevices.GetDeviceAtXRNode(XRNode.RightHand);
        if (rightController.isValid)
        {
            // Check right trigger for render mode toggle
            if (rightController.TryGetFeatureValue(CommonUsages.triggerButton, out bool triggerValue))
            {
                rightTriggerPressed = triggerValue;
            }
            // Check A button (primaryButton) on right controller for colored mode toggle
            if (rightController.TryGetFeatureValue(CommonUsages.primaryButton, out bool aButtonValue))
            {
                aButtonPressed = aButtonValue;
            }
            // Check B button to cancel recording
            if (rightController.TryGetFeatureValue(CommonUsages.secondaryButton, out bool bButtonValue))
            {
                bButtonPressed = bButtonValue;
            }
        }

        // Detect LEFT trigger press (rising edge) - Voice input
        if (leftTriggerPressed && !leftTriggerWasPressed)
        {
            if (isListening)
            {
                StopRecording();
            }
            else
            {
                StartRecording();
            }
        }
        // Detect RIGHT trigger press (rising edge) - Toggle render mode (Splats <-> OccamSimilarities)
        else if (rightTriggerPressed && !rightTriggerWasPressed)
        {
            ToggleRenderMode();
        }
        // Detect A button press (rising edge) on left controller - Toggle colored mode
        else if (aButtonPressed && !aButtonWasPressed)
        {
            ToggleColoredMode();
        }
        // Detect B button press (rising edge) to cancel recording
        else if (bButtonPressed && !bButtonWasPressed && isListening)
        {
            CancelRecording();
        }

        leftTriggerWasPressed = leftTriggerPressed;
        rightTriggerWasPressed = rightTriggerPressed;
        bButtonWasPressed = bButtonPressed;
        aButtonWasPressed = aButtonPressed;

#if PLATFORM_ANDROID
        // Check permission status
        if (!permissionGranted && Permission.HasUserAuthorizedPermission(Permission.Microphone))
        {
            permissionGranted = true;
            InitializeMicrophone();
        }
#endif
    }

    public void CancelRecording()
    {
        Debug.Log("=== Cancelling voice recording ===");

        if (!isListening)
        {
            return;
        }

        isListening = false;

        // Stop recording without processing
        if (!string.IsNullOrEmpty(microphoneDevice) && Microphone.IsRecording(microphoneDevice))
        {
            Microphone.End(microphoneDevice);
        }

        // Show cancellation message briefly
        ShowMessage("Cancelled");
        UpdateIndicatorColor(new Color(0.6f, 0.6f, 0.65f, 1f));  // Gray accent
        StartCoroutine(HideIndicatorAfterDelay(1.0f));

        Debug.Log("Voice recording cancelled by user.");
    }

    System.Collections.IEnumerator HideIndicatorAfterDelay(float delay)
    {
        yield return new WaitForSeconds(delay);
        HideIndicator();
    }

    public void StartRecording()
    {
        Debug.Log("=== Starting voice recording ===");

#if PLATFORM_ANDROID
        if (!permissionGranted)
        {
            Debug.LogWarning("Microphone permission not granted!");
            RequestMicrophonePermission();
            return;
        }
#endif

        if (string.IsNullOrEmpty(microphoneDevice))
        {
            InitializeMicrophone();
            if (string.IsNullOrEmpty(microphoneDevice))
            {
                Debug.LogError("No microphone available!");
                ShowMessage("No microphone found!");
                return;
            }
        }

        // Stop any existing recording
        if (Microphone.IsRecording(microphoneDevice))
        {
            Microphone.End(microphoneDevice);
        }

        // Start recording
        recordedClip = Microphone.Start(microphoneDevice, false, recordingLength, sampleRate);

        if (recordedClip == null)
        {
            Debug.LogError("Failed to start microphone recording!");
            ShowMessage("Recording failed!");
            return;
        }

        isListening = true;

        // Show indicator
        if (listeningIndicator != null)
        {
            UpdateIndicatorText("Recording...\nTrigger = Send  •  B = Cancel");
            UpdateIndicatorColor(new Color(0.9f, 0.3f, 0.35f, 1f));  // Meta red accent
            listeningIndicator.SetActive(true);
        }

        Debug.Log("Recording started. Speak now! Press trigger to send, B to cancel.");
    }

    public void StopRecording()
    {
        Debug.Log("=== Stopping voice recording ===");

        if (!isListening)
        {
            return;
        }

        isListening = false;

        // Get recording position before stopping
        int recordingPosition = Microphone.GetPosition(microphoneDevice);

        // Stop recording
        Microphone.End(microphoneDevice);

        if (recordedClip == null || recordingPosition <= 0)
        {
            Debug.LogWarning("No audio recorded!");
            HideIndicator();
            return;
        }

        // Update indicator
        UpdateIndicatorText("Processing...");
        UpdateIndicatorColor(new Color(0.2f, 0.5f, 1f, 1f));  // Meta blue accent

        Debug.Log($"Recording stopped. Samples recorded: {recordingPosition}");

        // Send audio to server for transcription
        StartCoroutine(SendAudioToServer(recordedClip, recordingPosition));
    }

    IEnumerator SendAudioToServer(AudioClip clip, int samples)
    {
        Debug.Log("Sending audio to server for transcription...");

        // Convert AudioClip to WAV bytes
        byte[] wavData = AudioClipToWav(clip, samples);

        if (wavData == null || wavData.Length == 0)
        {
            Debug.LogError("Failed to convert audio to WAV!");
            HideIndicator();
            yield break;
        }

        Debug.Log($"WAV data size: {wavData.Length} bytes");

        // Create form data
        WWWForm form = new WWWForm();
        form.AddBinaryData("audio", wavData, "recording.wav", "audio/wav");

        string url = serverUrl + "/speech_to_text";
        Debug.Log($"Sending to: {url}");

        UnityWebRequest request = UnityWebRequest.Post(url, form);
        request.timeout = 30;

        yield return request.SendWebRequest();

        bool success = request.result == UnityWebRequest.Result.Success;
        string responseText = success ? request.downloadHandler.text : "";
        string errorMsg = success ? "" : request.error;
        long responseCode = request.responseCode;

        request.Dispose();

        if (!success)
        {
            Debug.LogError($"Speech-to-text request failed: {errorMsg}");
            Debug.LogError($"Response code: {responseCode}");
            ShowMessage("Connection failed");
            UpdateIndicatorColor(new Color(0.9f, 0.3f, 0.35f, 1f));  // Red accent
            yield return new WaitForSeconds(2f);
            HideIndicator();
            yield break;
        }

        // Parse response
        Debug.Log($"Server response: {responseText}");

        string transcribedText = "";
        bool parseSuccess = false;

        // Parse JSON without try-catch in coroutine
        var response = JsonUtility.FromJson<SpeechToTextResponse>(responseText);
        if (response != null && !string.IsNullOrEmpty(response.text))
        {
            transcribedText = response.text;
            parseSuccess = true;
        }

        if (parseSuccess && !string.IsNullOrEmpty(transcribedText))
        {
            lastRecognizedText = transcribedText;
            Debug.Log($"Transcribed text: {transcribedText}");

            // Show result briefly
            ShowMessage(transcribedText);
            UpdateIndicatorColor(new Color(0.3f, 0.75f, 0.45f, 1f));  // Green accent for success
            yield return new WaitForSeconds(1.2f);

            // Send as query
            SendVoiceQuery(transcribedText);
        }
        else
        {
            Debug.LogWarning("Empty transcription result");
            ShowMessage("Couldn't understand");
            UpdateIndicatorColor(new Color(0.9f, 0.6f, 0.2f, 1f));  // Orange accent
            yield return new WaitForSeconds(2f);
        }

        HideIndicator();
    }

    byte[] AudioClipToWav(AudioClip clip, int samples)
    {
        if (clip == null) return null;

        // Get audio data
        float[] audioData = new float[samples];
        clip.GetData(audioData, 0);

        // Convert to 16-bit PCM
        short[] intData = new short[samples];
        for (int i = 0; i < samples; i++)
        {
            intData[i] = (short)(audioData[i] * 32767f);
        }

        // Create WAV file
        byte[] wavData;
        using (var stream = new System.IO.MemoryStream())
        {
            using (var writer = new System.IO.BinaryWriter(stream))
            {
                int channels = clip.channels;
                int sampleRate = clip.frequency;

                // WAV header
                writer.Write(System.Text.Encoding.ASCII.GetBytes("RIFF"));
                writer.Write(36 + samples * 2);  // File size - 8
                writer.Write(System.Text.Encoding.ASCII.GetBytes("WAVE"));

                // Format chunk
                writer.Write(System.Text.Encoding.ASCII.GetBytes("fmt "));
                writer.Write(16);  // Chunk size
                writer.Write((short)1);  // Audio format (PCM)
                writer.Write((short)channels);  // Channels
                writer.Write(sampleRate);  // Sample rate
                writer.Write(sampleRate * channels * 2);  // Byte rate
                writer.Write((short)(channels * 2));  // Block align
                writer.Write((short)16);  // Bits per sample

                // Data chunk
                writer.Write(System.Text.Encoding.ASCII.GetBytes("data"));
                writer.Write(samples * 2);  // Data size

                // Audio data
                foreach (short sample in intData)
                {
                    writer.Write(sample);
                }

                wavData = stream.ToArray();
            }
        }

        return wavData;
    }

    void ShowMessage(string message)
    {
        if (listeningIndicator != null)
        {
            UpdateIndicatorText(message);
            listeningIndicator.SetActive(true);
        }
    }

    void UpdateIndicatorText(string text)
    {
        if (listeningIndicator != null)
        {
            var textComponent = listeningIndicator.GetComponentInChildren<TMPro.TextMeshProUGUI>();
            if (textComponent != null)
            {
                textComponent.text = text;
            }
        }
    }

    void UpdateIndicatorColor(Color accentColor)
    {
        if (listeningIndicator != null)
        {
            // Find the accent bar (second image component, named "Accent")
            var images = listeningIndicator.GetComponentsInChildren<UnityEngine.UI.Image>();
            foreach (var image in images)
            {
                if (image.gameObject.name == "Accent")
                {
                    image.color = accentColor;
                }
            }
        }
    }

    void HideIndicator()
    {
        if (listeningIndicator != null)
        {
            listeningIndicator.SetActive(false);
        }
    }

    void SendVoiceQuery(string query)
    {
        if (splatRenderer == null)
        {
            Debug.LogError("Cannot send voice query: splatRenderer is null!");
            return;
        }

        Debug.Log($"========================================");
        Debug.Log($"Executing VOICE Query: \"{query}\"");
        Debug.Log($"========================================");

        // Update the last query display
        UpdateLastQueryDisplay(query);

        splatRenderer.ProcessCLIPQuery(query);
    }

    void UpdateLastQueryDisplay(string query)
    {
        if (lastQueryText != null)
        {
            lastQueryText.text = query;
            lastQueryText.fontStyle = TMPro.FontStyles.Normal;
        }
    }

    void UpdateFPS()
    {
        frameCounter++;
        fpsTimer += Time.unscaledDeltaTime;

        if (fpsTimer >= fpsUpdateInterval)
        {
            float fps = frameCounter / fpsTimer;
            if (fpsText != null)
            {
                fpsText.text = $"FPS: {fps:F0}";

                // Color code: green = good, yellow = ok, red = bad
                if (fps >= 72)
                    fpsText.color = Color.green;
                else if (fps >= 45)
                    fpsText.color = Color.yellow;
                else
                    fpsText.color = Color.red;
            }

            frameCounter = 0;
            fpsTimer = 0f;
        }
    }

    void OnDestroy()
    {
        // Stop any ongoing recording
        if (!string.IsNullOrEmpty(microphoneDevice) && Microphone.IsRecording(microphoneDevice))
        {
            Microphone.End(microphoneDevice);
        }

        // Clean up UI
        if (uiContainer != null)
        {
            Destroy(uiContainer);
        }
    }

    [Serializable]
    private class SpeechToTextResponse
    {
        public string text;
        public string error;
    }
}
