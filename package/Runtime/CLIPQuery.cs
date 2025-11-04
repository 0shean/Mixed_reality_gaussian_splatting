using UnityEngine;
using UnityEngine.Networking;
using System.Text;
using System.Threading.Tasks;
using System;
using System.Linq;

public class ClipClient : MonoBehaviour
{
    [Header("Server Settings")]
    public string serverUrl = "http://127.0.0.1:8000/embed_text";

    public string textInput = "a photo of a wombat";

    /// <summary>
    /// Request a CLIP text embedding asynchronously.
    /// </summary>
    public async Task<float[]> RequestEmbeddingAsync(string text)
    {
        string jsonData = "{\"text\":\"" + text + "\"}";
        using (var request = new UnityWebRequest(serverUrl, "POST"))
        {
            byte[] bodyRaw = Encoding.UTF8.GetBytes(jsonData);
            request.uploadHandler = new UploadHandlerRaw(bodyRaw);
            request.downloadHandler = new DownloadHandlerBuffer();
            request.SetRequestHeader("Content-Type", "application/json");

            // Start request and await completion
            var operation = request.SendWebRequest();

            while (!operation.isDone)
                await Task.Yield(); // let Unity’s main thread keep running

            if (request.result == UnityWebRequest.Result.Success)
            {
                try
                {
                    // Parse {"embedding": [ ... ]}
                    var response = JsonUtility.FromJson<EmbeddingResponse>(request.downloadHandler.text);
                    return response.embedding;
                }
                catch (Exception e)
                {
                    Debug.LogError("JSON parse error: " + e.Message);
                    return null;
                }
            }
            else
            {
                Debug.LogError($"Request failed: {request.error}");
                return null;
            }
        }
    }

    [ContextMenu("Request Embedding")]
    private async void RequestEmbeddingFromInspector()
    {
        float[] embedding = await RequestEmbeddingAsync(textInput);
        if (embedding != null)
            Debug.Log("Successfully received embedding, length: " + embedding.Length);
    }

    [Serializable]
    private class EmbeddingResponse
    {
        public float[] embedding;
    }
}
