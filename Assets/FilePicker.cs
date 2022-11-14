using System;
using System.Collections;
using System.IO;
using UnityEngine;


public class FilePicker : MonoBehaviour
{
    public string FinalPath;

    public void LoadFile(Action<string> onLoaded = null)
    {
        string FileType = NativeFilePicker.ConvertExtensionToFileType("*");

        NativeFilePicker.Permission permission = NativeFilePicker.PickFile((path) =>
        {
            if (path == null)
            {
                Debug.Log("Operation cancelled.");
            }
            else
            {
                FinalPath = path;
                Debug.Log("Picked: " + FinalPath);
                onLoaded?.Invoke(FinalPath);
            }
        }, new string[] { FileType });
    }
}
