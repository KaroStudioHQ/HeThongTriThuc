using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using TMPro;

public class UIManager : MonoBehaviour
{
    [SerializeField] DataGetter dataGetter;

    [SerializeField] DataRow dataRow;
    [SerializeField] Transform dataRowParent;

    [SerializeField] TextMeshProUGUI outputTxt;


    [SerializeField] TMP_InputField popInput, maxIter;
    private void Start()
    {
        dataRow.gameObject.SetActive(false);
        ShowData();
    }

    void ShowData()
    {
        var datas = dataGetter.GetTrainData();
        for (int i = 0; i < datas.Length; i++)
        {
            DataRow row = Instantiate(dataRow, dataRowParent);
            row.SetData(datas[i].info, datas[i].id);
            row.gameObject.SetActive(true);
        }
    }

    public void ClearOutput()
    {
        outputTxt.text = "";
    }

    public void SetOutput(string text)
    {
        outputTxt.text += "\n";
        outputTxt.text += text;
        outputTxt.text += "\n";
    }

    public int GetPopSize()
    {
        return int.Parse(popInput.text);
    }

    public int GetMaxIter()
    {
        return int.Parse(maxIter.text);
    }
}
