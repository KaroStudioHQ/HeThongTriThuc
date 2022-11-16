using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using TMPro;

public class DataRow : MonoBehaviour
{
    public TextMeshProUGUI[] texts;
    public void SetData(float[] data, int info)
    {
        for (int i = 0; i < 4; i++)
        {
            texts[i].text = data[i].ToString();
        }
        texts[4].text = info.ToString();
    }
}
