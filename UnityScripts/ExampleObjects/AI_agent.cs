using System.Collections;
using System.Collections.Generic;
using UnityEngine;

using NeuronalNet;
public class AI_agent : MonoBehaviour
{
    Net net = null;
    // Start is called before the first frame update
    void Start()
    {
        net = new Net();
        net.SetDimensions(2, 4, 5, 1);
        Debug.Log("net.Build() returns: " + net.Build());
        Debug.Log("inputs  = " + net.GetInputCount());
        Debug.Log("hiddenX = " + net.GetHiddenXCount());
        Debug.Log("hiddenY = " + net.GetHiddenYCount());
        Debug.Log("outputs = " + net.GetOutputCount());
    }

    // Update is called once per frame
    void Update()
    {
        
    }
}
