using UnityEngine;
using TMPro;

/// <summary>
/// Simple on-screen keyboard for VR when system keyboard doesn't appear
/// Attach to a Canvas in your scene
/// </summary>
public class SimpleVRKeyboard : MonoBehaviour
{
    [Header("References")]
    public TMP_InputField targetInputField;
    public GameObject keyboardPanel;

    [Header("Keyboard Layout")]
    public Transform keysContainer;

    private string currentText = "";
    private const string KEYS = "qwertyuiopasdfghjklzxcvbnm ";

    void Start()
    {
        if (keyboardPanel != null)
        {
            keyboardPanel.SetActive(false);
        }
    }

    public void ShowKeyboard(TMP_InputField inputField)
    {
        targetInputField = inputField;
        currentText = inputField.text;

        if (keyboardPanel != null)
        {
            keyboardPanel.SetActive(true);
            Debug.Log("Simple VR Keyboard opened");
        }
    }

    public void HideKeyboard()
    {
        if (keyboardPanel != null)
        {
            keyboardPanel.SetActive(false);
        }
    }

    public void OnKeyPressed(string key)
    {
        if (targetInputField == null) return;

        if (key == "backspace")
        {
            if (currentText.Length > 0)
            {
                currentText = currentText.Substring(0, currentText.Length - 1);
            }
        }
        else if (key == "space")
        {
            currentText += " ";
        }
        else if (key == "clear")
        {
            currentText = "";
        }
        else
        {
            currentText += key;
        }

        targetInputField.text = currentText;
        Debug.Log("Current text: " + currentText);
    }

    public void OnDonePressed()
    {
        HideKeyboard();
        Debug.Log("Keyboard done, final text: " + currentText);
    }
}
