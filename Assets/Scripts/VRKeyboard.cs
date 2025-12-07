using UnityEngine;
using UnityEngine.UI;
using TMPro;
using System;

/// <summary>
/// Visual VR Keyboard that works with ray interactions
/// Creates keyboard buttons dynamically
/// </summary>
public class VRKeyboard : MonoBehaviour
{
    [Header("References")]
    public TMP_InputField targetInputField;
    public Transform keyboardContainer;
    public GameObject keyButtonPrefab;

    [Header("Layout Settings")]
    public float keyWidth = 100f;
    public float keyHeight = 100f;
    public float keySpacing = 10f;
    public float rowSpacing = 10f;

    [Header("Colors")]
    public Color normalKeyColor = new Color(0.2f, 0.2f, 0.2f, 1f);
    public Color specialKeyColor = new Color(0.3f, 0.3f, 0.5f, 1f);

    private string[] keyboardRows = new string[]
    {
        "1234567890",
        "qwertyuiop",
        "asdfghjkl",
        "zxcvbnm"
    };

    private bool isUpperCase = false;

    void Start()
    {
        if (keyButtonPrefab == null)
        {
            Debug.LogError("VRKeyboard: keyButtonPrefab is not assigned!");
            return;
        }

        CreateKeyboard();
    }

    void CreateKeyboard()
    {
        float startY = 0;

        // Create letter/number rows
        for (int row = 0; row < keyboardRows.Length; row++)
        {
            float startX = -(keyboardRows[row].Length * (keyWidth + keySpacing)) / 2f;
            float y = startY - (row * (keyHeight + rowSpacing));

            for (int col = 0; col < keyboardRows[row].Length; col++)
            {
                string key = keyboardRows[row][col].ToString();
                float x = startX + (col * (keyWidth + keySpacing));
                CreateKey(key, new Vector2(x, y), keyWidth, normalKeyColor);
            }
        }

        // Special keys row
        float specialRowY = startY - (keyboardRows.Length * (keyHeight + rowSpacing));

        // Shift key
        CreateKey("SHIFT", new Vector2(-300, specialRowY), keyWidth * 1.5f, specialKeyColor);

        // Space bar
        CreateKey("SPACE", new Vector2(0, specialRowY), keyWidth * 4f, normalKeyColor);

        // Backspace
        CreateKey("DEL", new Vector2(280, specialRowY), keyWidth * 1.5f, specialKeyColor);

        // Done row
        float doneRowY = specialRowY - (keyHeight + rowSpacing);
        CreateKey("CLEAR", new Vector2(-150, doneRowY), keyWidth * 2f, specialKeyColor);
        CreateKey("DONE", new Vector2(150, doneRowY), keyWidth * 2f, new Color(0.2f, 0.5f, 0.2f, 1f));
    }

    void CreateKey(string keyLabel, Vector2 position, float width, Color color)
    {
        GameObject keyObj = Instantiate(keyButtonPrefab, keyboardContainer);
        keyObj.name = "Key_" + keyLabel;

        RectTransform rect = keyObj.GetComponent<RectTransform>();
        rect.anchoredPosition = position;
        rect.sizeDelta = new Vector2(width, keyHeight);

        // Set button color
        Image img = keyObj.GetComponent<Image>();
        if (img != null)
        {
            img.color = color;
        }

        // Set button text
        TMP_Text text = keyObj.GetComponentInChildren<TMP_Text>();
        if (text != null)
        {
            text.text = keyLabel;
            text.fontSize = keyLabel.Length > 1 ? 24 : 36;
        }

        // Add click listener
        Button btn = keyObj.GetComponent<Button>();
        if (btn != null)
        {
            string capturedKey = keyLabel;
            btn.onClick.AddListener(() => OnKeyPressed(capturedKey));
        }
    }

    public void OnKeyPressed(string key)
    {
        if (targetInputField == null)
        {
            Debug.LogError("VRKeyboard: No target input field!");
            return;
        }

        switch (key)
        {
            case "SPACE":
                targetInputField.text += " ";
                break;

            case "DEL":
                if (targetInputField.text.Length > 0)
                {
                    targetInputField.text = targetInputField.text.Substring(0, targetInputField.text.Length - 1);
                }
                break;

            case "CLEAR":
                targetInputField.text = "";
                break;

            case "SHIFT":
                isUpperCase = !isUpperCase;
                UpdateKeyLabels();
                break;

            case "DONE":
                OnDonePressed();
                break;

            default:
                string charToAdd = isUpperCase ? key.ToUpper() : key.ToLower();
                targetInputField.text += charToAdd;
                break;
        }

        Debug.Log($"Key pressed: {key}, Current text: {targetInputField.text}");
    }

    void UpdateKeyLabels()
    {
        foreach (Transform child in keyboardContainer)
        {
            TMP_Text text = child.GetComponentInChildren<TMP_Text>();
            if (text != null && text.text.Length == 1)
            {
                text.text = isUpperCase ? text.text.ToUpper() : text.text.ToLower();
            }
        }
    }

    void OnDonePressed()
    {
        Debug.Log("VRKeyboard: Done pressed, text: " + targetInputField.text);
        // Find ClipQueryController and trigger send
        ClipQueryController controller = FindObjectOfType<ClipQueryController>();
        if (controller != null)
        {
            // Trigger the send button click
            controller.sendQueryButton?.onClick.Invoke();
        }
    }

    public void SetTargetInputField(TMP_InputField inputField)
    {
        targetInputField = inputField;
    }
}
