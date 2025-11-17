using UnityEngine;
using UnityEngine.XR;
using UnityEngine.XR.Interaction.Toolkit;

/// <summary>
/// Send message to server when pressing a button on the VR controller.
/// Much simpler than setting up 3D interactable buttons.
/// </summary>
public class ControllerButtonMessenger : MonoBehaviour
{
    [Header("References")]
    [Tooltip("Reference to the ServerMessenger component")]
    public ServerMessenger serverMessenger;

    [Header("Settings")]
    [Tooltip("Message to send when button is pressed")]
    public string messageToSend = "Button pressed!";

    [Tooltip("Which controller hand to use")]
    public XRNode controllerHand = XRNode.RightHand;

    [Header("Button Options")]
    [Tooltip("Use the primary button (A on right hand, X on left hand)")]
    public bool usePrimaryButton = true;

    [Tooltip("Use the trigger button")]
    public bool useTrigger = false;

    [Tooltip("Use the grip button")]
    public bool useGrip = false;

    private InputDevice targetDevice;
    private bool buttonWasPressedLastFrame = false;

    void Start()
    {
        // Find ServerMessenger if not assigned
        if (serverMessenger == null)
        {
            serverMessenger = FindObjectOfType<ServerMessenger>();
            if (serverMessenger == null)
            {
                Debug.LogError("[ControllerButtonMessenger] No ServerMessenger found in scene!");
            }
        }

        // Get the controller device
        targetDevice = InputDevices.GetDeviceAtXRNode(controllerHand);

        string handName = controllerHand == XRNode.RightHand ? "Right" : "Left";
        Debug.Log($"[ControllerButtonMessenger] Listening for {handName} controller button press");
    }

    void Update()
    {
        // Make sure we have a valid device
        if (!targetDevice.isValid)
        {
            targetDevice = InputDevices.GetDeviceAtXRNode(controllerHand);
            return;
        }

        bool buttonPressed = false;

        // Check Primary Button (A/X)
        if (usePrimaryButton)
        {
            bool primaryButtonValue;
            if (targetDevice.TryGetFeatureValue(CommonUsages.primaryButton, out primaryButtonValue))
            {
                buttonPressed |= primaryButtonValue;
            }
        }

        // Check Trigger
        if (useTrigger)
        {
            float triggerValue;
            if (targetDevice.TryGetFeatureValue(CommonUsages.trigger, out triggerValue))
            {
                buttonPressed |= (triggerValue > 0.9f); // Trigger fully pressed
            }
        }

        // Check Grip
        if (useGrip)
        {
            float gripValue;
            if (targetDevice.TryGetFeatureValue(CommonUsages.grip, out gripValue))
            {
                buttonPressed |= (gripValue > 0.9f); // Grip fully pressed
            }
        }

        // Detect button press (rising edge - wasn't pressed last frame, is pressed now)
        if (buttonPressed && !buttonWasPressedLastFrame)
        {
            OnButtonPressed();
        }

        buttonWasPressedLastFrame = buttonPressed;
    }

    private void OnButtonPressed()
    {
        string handName = controllerHand == XRNode.RightHand ? "Right" : "Left";
        Debug.Log($"[ControllerButtonMessenger] {handName} controller button pressed!");

        // Send message to server
        if (serverMessenger != null)
        {
            serverMessenger.SendMessage(messageToSend);
        }
        else
        {
            Debug.LogError("[ControllerButtonMessenger] ServerMessenger is not assigned!");
        }

        // Optional: haptic feedback
        HapticFeedback();
    }

    private void HapticFeedback()
    {
        // Give a brief vibration to confirm button press
        if (targetDevice.isValid)
        {
            HapticCapabilities capabilities;
            if (targetDevice.TryGetHapticCapabilities(out capabilities))
            {
                if (capabilities.supportsImpulse)
                {
                    targetDevice.SendHapticImpulse(0, 0.5f, 0.1f); // channel 0, 50% amplitude, 0.1 seconds
                }
            }
        }
    }
}
