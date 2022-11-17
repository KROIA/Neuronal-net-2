using System.Collections;
using System.Collections.Generic;
using UnityEngine;

using NeuronalNet;
public class AI_XOR_agent : MonoBehaviour
{
    BackpropNet net = null;
    MultiSignalVector trainigsSet;
    MultiSignalVector resultSet;

    // Start is called before the first frame update
    void Start()
    {
        trainigsSet = new MultiSignalVector(4, 2);
        resultSet = new MultiSignalVector(4, 1);

        net = new BackpropNet();
        net.SetDimensions(2, 2, 5, 1);
        net.SetStreamSize(trainigsSet.Size());
        net.SetActivation(Net.Activation.sigmoid);
        net.SetHardware(Net.Hardware.cpu);
        net.SetLearnParameter(1.0f);
        net.EnableBias(true);
        net.Build();

        trainigsSet.SetElement(0,new SignalVector(new List<float>(new float[] { 0, 0 })));
        trainigsSet.SetElement(1,new SignalVector(new List<float>(new float[] { 0, 1 })));
        trainigsSet.SetElement(2,new SignalVector(new List<float>(new float[] { 1, 0 })));
        trainigsSet.SetElement(3,new SignalVector(new List<float>(new float[] { 1, 1 })));

        resultSet.SetElement(0, new SignalVector(new List<float>(new float[] { 0 })));
        resultSet.SetElement(1, new SignalVector(new List<float>(new float[] { 1 })));
        resultSet.SetElement(2, new SignalVector(new List<float>(new float[] { 1 })));
        resultSet.SetElement(3, new SignalVector(new List<float>(new float[] { 0 })));

        Debug.Log("Using neuronal net version: "+Net.GetVersion());
    }

    // Update is called once per frame
    void Update()
    {
        net.SetInputVector(trainigsSet);
        net.Calculate();
        net.Learn(resultSet);
        MultiSignalVector err = net.GetError();
        float currentError = err.GetRootMeanSquare();
        net.SetLearnParameter(currentError);
        Debug.Log(currentError);
    }
}
