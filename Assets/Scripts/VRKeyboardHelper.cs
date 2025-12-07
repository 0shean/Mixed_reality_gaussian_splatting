using UnityEngine;
using TMPro;
using System.Collections;

/// <summary>
/// Helper script to ensure VR keyboard works properly with TMP_InputField on Meta Quest
/// Attach this to the same GameObject as your TMP_InputField
/// </summary>
[RequireComponent(typeof(TMP_InputField))]
public class VRKeyboardHelper : MonoBehaviour
{
    private TMP_InputField inputField;
    private TouchScreenKeyboard keyboard;

    void Start()
    {
        inputField = GetComponent<TMP_InputField>();

        if (inputField != null)
        {
            // Configure for VR keyboard
            inputField.keyboardType = TouchScreenKeyboardType.Default;

            // For OpenXR/Meta Quest: DO NOT hide mobile input
            inputField.shouldHideMobileInput = false;

            // Make sure input field is interactive
            inputField.interactable = true;
            inputField.readOnly = false;

            // Add listeners
            inputField.onSelect.AddListener(OnInputFieldSelected);
            inputField.onDeselect.AddListener(OnInputFieldDeselected);

            Debug.Log($"VRKeyboardHelper initialized. Platform: {Application.platform}, TouchScreenKeyboard supported: {TouchScreenKeyboard.isSupported}");
        }
        else
        {
            Debug.LogError("VRKeyboardHelper: No TMP_InputField component found!");
        }
    }

    void OnInputFieldSelected(string text)
    {
        Debug.Log("VR Keyboard: Input field selected");
        StartCoroutine(OpenKeyboardCoroutine());
    }

    IEnumerator OpenKeyboardCoroutine()
    {
        // Wait a frame to ensure the input field is fully selected
        yield return null;

        // Explicitly open the system keyboard
        if (TouchScreenKeyboard.isSupported)
        {
            Debug.Log("VR Keyboard: Opening TouchScreenKeyboard");

            keyboard = TouchScreenKeyboard.Open(
                inputField.text,
                inputField.keyboardType,
                false,
                false,
                false,
                false
            );
        }
        else
        {
            Debug.LogWarning("VR Keyboard: TouchScreenKeyboard is not supported on this platform!");
        }
    }

    void Update()
    {
        // Update the input field with keyboard text
        if (keyboard != null)
        {
            if (keyboard.status == TouchScreenKeyboard.Status.Done)
            {
                inputField.text = keyboard.text;
                Debug.Log("VR Keyboard: Text entered: " + keyboard.text);
                keyboard = null;
            }
            else if (keyboard.status == TouchScreenKeyboard.Status.Canceled)
            {
                Debug.Log("VR Keyboard: Canceled");
                keyboard = null;
            }
            else if (keyboard.status == TouchScreenKeyboard.Status.Visible)
            {
                // Update text while typing
                inputField.text = keyboard.text;
            }
        }
    }

    void OnInputFieldDeselected(string text)
    {
        Debug.Log("VR Keyboard: Input field deselected");
    }

    void OnDestroy()
    {
        if (inputField != null)
        {
            inputField.onSelect.RemoveListener(OnInputFieldSelected);
            inputField.onDeselect.RemoveListener(OnInputFieldDeselected);
        }
    }
}
