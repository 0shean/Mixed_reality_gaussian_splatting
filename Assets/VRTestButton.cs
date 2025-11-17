using UnityEngine;
using UnityEngine.XR.Interaction.Toolkit;

/// <summary>
/// VR button that sends a message to the server when pressed.
/// Attach this to a button GameObject with XRSimpleInteractable component.
/// </summary>
[RequireComponent(typeof(XRSimpleInteractable))]
public class VRTestButton : MonoBehaviour
{
    [Header("References")]
    [Tooltip("Reference to the ServerMessenger component")]
    public ServerMessenger serverMessenger;

    [Header("Settings")]
    [Tooltip("Message to send when button is pressed")]
    public string messageToSend = "Button pressed!";

    [Tooltip("Visual feedback - optional material to change color when pressed")]
    public Renderer buttonRenderer;
    public Color pressedColor = Color.green;
    public Color normalColor = Color.white;

    private XRSimpleInteractable interactable;

    void Awake()
    {
        interactable = GetComponent<XRSimpleInteractable>();

        // Find ServerMessenger if not assigned
        if (serverMessenger == null)
        {
            serverMessenger = FindObjectOfType<ServerMessenger>();
            if (serverMessenger == null)
            {
                Debug.LogError("[VRTestButton] No ServerMessenger found in scene!");
            }
        }
    }

    void OnEnable()
    {
        // Subscribe to interaction events
        if (interactable != null)
        {
            interactable.selectEntered.AddListener(OnButtonPressed);
            interactable.selectExited.AddListener(OnButtonReleased);
        }
    }

    void OnDisable()
    {
        // Unsubscribe from events
        if (interactable != null)
        {
            interactable.selectEntered.RemoveListener(OnButtonPressed);
            interactable.selectExited.RemoveListener(OnButtonReleased);
        }
    }

    private void OnButtonPressed(SelectEnterEventArgs args)
    {
        Debug.Log("[VRTestButton] Button pressed!");

        // Visual feedback
        if (buttonRenderer != null)
        {
            buttonRenderer.material.color = pressedColor;
        }

        // Send message to server
        if (serverMessenger != null)
        {
            serverMessenger.SendMessage(messageToSend);
        }
        else
        {
            Debug.LogError("[VRTestButton] ServerMessenger is not assigned!");
        }
    }

    private void OnButtonReleased(SelectExitEventArgs args)
    {
        Debug.Log("[VRTestButton] Button released!");

        // Reset visual feedback
        if (buttonRenderer != null)
        {
            buttonRenderer.material.color = normalColor;
        }
    }

    // Alternative: Use this for basic trigger button press (A/X button on Quest controllers)
    void Update()
    {
        // Optional: Allow testing with keyboard in editor
        if (Input.GetKeyDown(KeyCode.Space))
        {
            Debug.Log("[VRTestButton] Space key pressed (testing)");
            if (serverMessenger != null)
            {
                serverMessenger.SendMessage(messageToSend);
            }
        }
    }
}
