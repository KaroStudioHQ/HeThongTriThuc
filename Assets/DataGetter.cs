using System.Collections.Generic;
using UnityEngine;

public class DataGetter : MonoBehaviour
{
    [SerializeField] TextAsset data;

    IrisData[] trainDatas;
    IrisData[] testDatas;

    float max = 7.9f;
    float min = 4.3f;

    private void Awake()
    {
        GetData();
    }

    void GetData()
    {
        List<IrisData> trainDataList = new List<IrisData>();
        List<IrisData> testDataList = new List<IrisData>();

        string[] lines = data.text.Split('\n');
        int testCount = Mathf.RoundToInt(lines.Length * 2f / 10);
        for (int i = 0; i < lines.Length; i++)
        {
            if (lines[i].Trim() != "")
            {
                string[] lineDatas = lines[i].Split(',');

                IrisData irisData = new IrisData();

                irisData.info = new float[] {
                    (float.Parse(lineDatas[0]) - 4.3f)/(7.9f-4.3f),
                    (float.Parse(lineDatas[1]) - 2f)/(4.4f-2),
                    (float.Parse(lineDatas[2]) - 1)/(6.9f-1),
                    (float.Parse(lineDatas[3]) - 0.1f)/(2.5f-0.1f) };

                //irisData.info = new float[] {
                  //  (float.Parse(lineDatas[0])),
                    //(float.Parse(lineDatas[1])),
                    //(float.Parse(lineDatas[2])),
                    //(float.Parse(lineDatas[3])) };

                if (lineDatas[4] == "Iris-setosa")
                {
                    irisData.id = -1;
                }
                else if (lineDatas[4] == "Iris-versicolor")
                {
                    irisData.id = 0;
                }
                else
                {
                    irisData.id = 1;
                }
                trainDataList.Add(irisData);
            }
        }
        while (testDataList.Count < testCount)
        {
            int rand = Random.Range(0, trainDataList.Count);
            testDataList.Add(trainDataList[rand]);
            trainDataList.RemoveAt(rand);
        }
        trainDatas = trainDataList.ToArray();
        testDatas = testDataList.ToArray();
    }
    public IrisData[] GetTrainData()
    {
        return trainDatas;
    }

    public IrisData[] GetTestData()
    {
        return testDatas;
    }
}
