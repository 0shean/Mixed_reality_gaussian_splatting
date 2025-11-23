using UnityEngine;
using UnityEngine.Networking;
using System.Text;
using System.Threading.Tasks;
using System;
using System.Linq;

public class ClipClient : MonoBehaviour
{
    [Header("Server Settings")]
    public string serverUrl = "http://127.0.0.1:8000";

    public string textInput = "a photo of a wombat";

    public async Task<float[]> RequestSimilarityBufferAsync(string text)
    {
        string json = "{\"text\":\"" + text + "\"}";
        string fullUrl = serverUrl + "/similarity_binary";

        Debug.Log($"[ClipClient] Attempting to connect to: {fullUrl}");
        Debug.Log($"[ClipClient] Requesting similarity for: \"{text}\"");

        using (var request = new UnityWebRequest(fullUrl, "POST"))
        {
            byte[] body = Encoding.UTF8.GetBytes(json);
            request.uploadHandler = new UploadHandlerRaw(body);
            request.downloadHandler = new DownloadHandlerBuffer();
            request.SetRequestHeader("Content-Type", "application/json");

            var operation = request.SendWebRequest();
            while (!operation.isDone)
                await Task.Yield();

            if (request.result != UnityWebRequest.Result.Success)
            {
                Debug.LogError($"[ClipClient] Binary request failed!");
                Debug.LogError($"[ClipClient] URL was: {fullUrl}");
                Debug.LogError($"[ClipClient] Error: {request.error}");
                Debug.LogError($"[ClipClient] Response Code: {request.responseCode}");
                return null;
            }

            // Get raw bytes
            byte[] data = request.downloadHandler.data;

            // Convert byte[] → float[]
            int floatCount = data.Length / 4;  // float32 = 4 bytes
            float[] relevancy = new float[floatCount];

            Buffer.BlockCopy(data, 0, relevancy, 0, data.Length);

            return relevancy;
        }
    }

    [Serializable]
    public class RelevancyExtremaResponse
    {
        public float min_relevancy;
        public float max_relevancy;
    }

    public async Task<(float min, float max)> RequestRelevancyExtremaAsync(string text)
    {
        string json = "{\"text\":\"" + text + "\"}";
        string fullUrl = serverUrl + "/relevancy_extrema";

        Debug.Log($"[ClipClient] Requesting extrema from: {fullUrl}");

        using (var request = new UnityWebRequest(fullUrl, "POST"))
        {
            byte[] body = Encoding.UTF8.GetBytes(json);
            request.uploadHandler = new UploadHandlerRaw(body);
            request.downloadHandler = new DownloadHandlerBuffer();
            request.SetRequestHeader("Content-Type", "application/json");

            var op = request.SendWebRequest();
            while (!op.isDone)
                await Task.Yield();

            if (request.result != UnityWebRequest.Result.Success)
            {
                Debug.LogError($"[ClipClient] Extrema request failed!");
                Debug.LogError($"[ClipClient] URL was: {fullUrl}");
                Debug.LogError($"[ClipClient] Error: " + request.error);
                return (0f, 0f);
            }

            try
            {
                var response = JsonUtility.FromJson<RelevancyExtremaResponse>(request.downloadHandler.text);
                return (response.min_relevancy, response.max_relevancy);
            }
            catch (Exception e)
            {
                Debug.LogError("JSON parse error: " + e.Message);
                Debug.LogError("Response text was: " + request.downloadHandler.text);
                return (0f, 0f);
            }
        }
    }

    [ContextMenu("Request Similarity Scores")]
    private async void RequestSimilarityFromInspector()
    {
        float[] similarityBuf = await RequestSimilarityBufferAsync(textInput);
        if (similarityBuf != null)
            Debug.Log("Successfully received similarity scores for embedding, length: " + similarityBuf.Length);
    }

}
