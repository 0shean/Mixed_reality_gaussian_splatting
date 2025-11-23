#if UNITY_ANDROID
using UnityEditor.Android;
using UnityEngine;

public class AndroidManifestModifier : IPostGenerateGradleAndroidProject
{
    public int callbackOrder => 1;

    public void OnPostGenerateGradleAndroidProject(string path)
    {
        Debug.Log("AndroidManifestModifier: Modifying AndroidManifest.xml to allow cleartext traffic");

        var manifestPath = path + "/src/main/AndroidManifest.xml";
        var manifestDoc = new System.Xml.XmlDocument();
        manifestDoc.Load(manifestPath);

        var applicationNode = manifestDoc.SelectSingleNode("/manifest/application");
        if (applicationNode != null)
        {
            var cleartextAttr = manifestDoc.CreateAttribute("android", "usesCleartextTraffic", "http://schemas.android.com/apk/res/android");
            cleartextAttr.Value = "true";
            applicationNode.Attributes.Append(cleartextAttr);

            Debug.Log("AndroidManifestModifier: Added usesCleartextTraffic=true");
        }

        manifestDoc.Save(manifestPath);
        Debug.Log("AndroidManifestModifier: Saved modified AndroidManifest.xml");
    }
}
#endif
