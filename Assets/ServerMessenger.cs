using System.Collections;
using System.Text;
using UnityEngine;
using UnityEngine.Networking;

/// <summary>
/// Sends messages to Flask server on desktop computer.
/// Quest and desktop must be on the same Wi-Fi network.
/// </summary>
public class ServerMessenger : MonoBehaviour
{
    [Header("Server Configuration")]
    [Tooltip("IP address of the desktop running Flask server")]
    public string serverIP = "192.168.1.100"; // Change this to your server IP

    [Tooltip("Port of the Flask server")]
    public int serverPort = 5001;

    [Header("Debug Settings")]
    [Tooltip("Send automatic test messages every X seconds")]
    public bool enablePeriodicTest = true;

    [Tooltip("Interval in seconds between test messages")]
    public float testInterval = 5f;

    private string ServerURL => $"http://{serverIP}:{serverPort}/message";
    private int messageCounter = 0;

    /// <summary>
    /// Send a message to the Flask server.
    /// </summary>
    /// <param name="message">The message to send</param>
    public void SendMessage(string message)
    {
        StartCoroutine(SendMessageCoroutine(message));
    }

    private IEnumerator SendMessageCoroutine(string message)
    {
        // Create JSON payload
        string jsonData = $"{{\"message\": \"{message}\"}}";
        byte[] bodyRaw = Encoding.UTF8.GetBytes(jsonData);

        Debug.Log($"[ServerMessenger] Sending to {ServerURL}: {message}");
        Debug.Log($"[ServerMessenger] JSON payload: {jsonData}");

        // Create web request
        using (UnityWebRequest www = new UnityWebRequest(ServerURL, "POST"))
        {
            www.uploadHandler = new UploadHandlerRaw(bodyRaw);
            www.uploadHandler.contentType = "application/json";
            www.downloadHandler = new DownloadHandlerBuffer();
            www.SetRequestHeader("Content-Type", "application/json");

            // Send request
            yield return www.SendWebRequest();

            // Check result
            if (www.result == UnityWebRequest.Result.Success)
            {
                Debug.Log($"[ServerMessenger] Success! Server response: {www.downloadHandler.text}");
            }
            else
            {
                Debug.LogError($"[ServerMessenger] Error: {www.error}");
                Debug.LogError($"[ServerMessenger] Response Code: {www.responseCode}");
                Debug.LogError($"[ServerMessenger] Make sure server is running at {ServerURL}");
            }
        }
    }

    /// <summary>
    /// Test connection to server by sending a ping.
    /// </summary>
    public void TestConnection()
    {
        StartCoroutine(PingServerCoroutine());
    }

    private IEnumerator PingServerCoroutine()
    {
        string pingURL = $"http://{serverIP}:{serverPort}/ping";

        using (UnityWebRequest www = UnityWebRequest.Get(pingURL))
        {
            Debug.Log($"[ServerMessenger] Pinging server at {pingURL}...");

            yield return www.SendWebRequest();

            if (www.result == UnityWebRequest.Result.Success)
            {
                Debug.Log($"[ServerMessenger] Ping successful! Server is alive.");
            }
            else
            {
                Debug.LogError($"[ServerMessenger] Ping failed: {www.error}");
            }
        }
    }

    // Test on start (optional)
    void Start()
    {
        Debug.Log($"[ServerMessenger] Initialized. Server: {ServerURL}");
        Debug.Log($"[ServerMessenger] Remember to set the correct server IP in the Inspector!");

        // Start periodic testing if enabled
        if (enablePeriodicTest)
        {
            Debug.Log($"[ServerMessenger] Periodic test enabled. Sending message every {testInterval} seconds.");
            InvokeRepeating(nameof(SendPeriodicTestMessage), testInterval, testInterval);
        }
    }

    private void SendPeriodicTestMessage()
    {
        messageCounter++;
        string message = $"Periodic test message #{messageCounter} - {System.DateTime.Now:HH:mm:ss}";
        Debug.Log($"[ServerMessenger] Sending periodic test: {message}");
        SendMessage(message);
    }
}
