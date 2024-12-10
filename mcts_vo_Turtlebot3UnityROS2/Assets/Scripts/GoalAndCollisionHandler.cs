using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class GoalAndCollisionHandler : MonoBehaviourRosNode
{
    public string NodeName = "goal_and_collision_handler";
    public float TfPublishingFrequency = 10.0f;


    protected override string nodeName { get { return NodeName; } }


    protected override void StartRos()
    {
        // Implement the StartRos method here
    }

    // Start is called before the first frame update
    void Start()
    {

    }

    // Update is called once per frame
    void Update()
    {

    }
}
