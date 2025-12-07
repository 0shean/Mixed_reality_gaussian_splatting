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
/// Press left trigger to start/stop voice recording
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

    [Header("Indicator Positioning")]
    public float indicatorDistance = 1.5f;
    public float indicatorHeightOffset = 0.2f;

    [Header("Status")]
    public bool isListening = false;
    public string lastRecognizedText = "";

    private bool triggerWasPressed = false;
    private Camera vrCamera;
    private AudioClip recordedClip;
    private string microphoneDevice;
    private bool permissionGranted = false;

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

        // Create listening indicator if not assigned
        if (listeningIndicator == null)
        {
            CreateListeningIndicator();
        }

        // Hide listening indicator initially
        if (listeningIndicator != null)
        {
            listeningIndicator.SetActive(false);
        }

        // Request microphone permission
        RequestMicrophonePermission();

        Debug.Log("VoiceInputController initialized. Press LEFT TRIGGER to start/stop voice input.");
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
        // Create Canvas
        GameObject canvasObj = new GameObject("ListeningIndicatorCanvas");
        Canvas canvas = canvasObj.AddComponent<Canvas>();
        canvas.renderMode = RenderMode.WorldSpace;

        RectTransform canvasRect = canvasObj.GetComponent<RectTransform>();
        canvasRect.sizeDelta = new Vector2(500, 200);
        canvasRect.localScale = new Vector3(0.001f, 0.001f, 0.001f);

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
        panelImage.color = new Color(0.8f, 0.2f, 0.2f, 0.9f);  // Red background when recording

        // Add text
        GameObject textObj = new GameObject("ListeningText");
        textObj.transform.SetParent(canvasObj.transform, false);

        RectTransform textRect = textObj.AddComponent<RectTransform>();
        textRect.anchorMin = Vector2.zero;
        textRect.anchorMax = Vector2.one;
        textRect.sizeDelta = Vector2.zero;
        textRect.offsetMin = new Vector2(10, 10);
        textRect.offsetMax = new Vector2(-10, -10);

        TMPro.TextMeshProUGUI text = textObj.AddComponent<TMPro.TextMeshProUGUI>();
        text.text = "Recording...";
        text.fontSize = 48;
        text.color = Color.white;
        text.alignment = TMPro.TextAlignmentOptions.Center;

        listeningIndicator = canvasObj;
        Debug.Log("Created listening indicator canvas");
    }

    void PositionIndicatorInFrontOfCamera()
    {
        if (vrCamera == null || listeningIndicator == null) return;

        Transform camTransform = vrCamera.transform;

        Vector3 forward = camTransform.forward;
        forward.y = 0;
        forward.Normalize();

        if (forward.magnitude < 0.1f)
        {
            forward = camTransform.forward;
        }

        Vector3 position = camTransform.position + forward * indicatorDistance;
        position.y = camTransform.position.y + indicatorHeightOffset;

        listeningIndicator.transform.position = position;
        listeningIndicator.transform.rotation = Quaternion.LookRotation(forward);
    }

    void Update()
    {
        // Check for left controller trigger
        bool triggerPressed = false;

        InputDevice leftController = InputDevices.GetDeviceAtXRNode(XRNode.LeftHand);
        if (leftController.isValid)
        {
            if (leftController.TryGetFeatureValue(CommonUsages.triggerButton, out bool triggerValue))
            {
                triggerPressed = triggerValue;
            }
        }

        // Detect trigger press (rising edge)
        if (triggerPressed && !triggerWasPressed)
        {
            ToggleVoiceInput();
        }

        triggerWasPressed = triggerPressed;

#if PLATFORM_ANDROID
        // Check permission status
        if (!permissionGranted && Permission.HasUserAuthorizedPermission(Permission.Microphone))
        {
            permissionGranted = true;
            InitializeMicrophone();
        }
#endif
    }

    void ToggleVoiceInput()
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
            PositionIndicatorInFrontOfCamera();
            UpdateIndicatorText("Recording...\n(Press trigger to stop)");
            UpdateIndicatorColor(new Color(0.8f, 0.2f, 0.2f, 0.9f));  // Red
            listeningIndicator.SetActive(true);
        }

        Debug.Log("Recording started. Speak now!");
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
        UpdateIndicatorColor(new Color(0.2f, 0.2f, 0.8f, 0.9f));  // Blue

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
            ShowMessage("Transcription failed!\nCheck server.");
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
            ShowMessage($"Heard: \"{transcribedText}\"");
            yield return new WaitForSeconds(1.5f);

            // Send as query
            SendVoiceQuery(transcribedText);
        }
        else
        {
            Debug.LogWarning("Empty transcription result");
            ShowMessage("Couldn't understand.\nTry again.");
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
            PositionIndicatorInFrontOfCamera();
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

    void UpdateIndicatorColor(Color color)
    {
        if (listeningIndicator != null)
        {
            var image = listeningIndicator.GetComponentInChildren<UnityEngine.UI.Image>();
            if (image != null)
            {
                image.color = color;
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

        splatRenderer.ProcessCLIPQuery(query);
    }

    void OnDestroy()
    {
        // Stop any ongoing recording
        if (!string.IsNullOrEmpty(microphoneDevice) && Microphone.IsRecording(microphoneDevice))
        {
            Microphone.End(microphoneDevice);
        }
    }

    [Serializable]
    private class SpeechToTextResponse
    {
        public string text;
        public string error;
    }
}
