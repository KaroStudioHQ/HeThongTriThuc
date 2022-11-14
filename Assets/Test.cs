using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Test : MonoBehaviour
{
    [SerializeField] FilePicker filePicker;
    [SerializeField] GameObject quad;

    private void Start()
    {
        filePicker.LoadFile((path) =>
        {
            //StartCoroutine(LoadTexture(path));
            StartCoroutine(LoadCSV(path));
        });
    }

    IEnumerator LoadTexture(string path)
    {
        WWW www = new WWW(path);
        while (!www.isDone)
            yield return null;
        quad.GetComponent<Renderer>().material.mainTexture = www.texture;
    }

    IEnumerator LoadCSV(string path)
    {
        WWW www = new WWW(path);
        while (!www.isDone)
            yield return null;
        Debug.Log(www.text);
    }
}
