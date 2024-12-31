using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class FollowPlayer : MonoBehaviour {
    public GameObject target;
    private Transform target_transform;
    public bool isCustomOffset;
    public Vector3 offset;
    public float smoothSpeed = 0.3f;

    // Start is called before the first frame update
    void Start()
    {
        target_transform = target.transform;
         // You can also specify your own offset from inspector
        // by making isCustomOffset bool to true
        if (!isCustomOffset)
        {
            offset = transform.position - target_transform.position;
        }
    }

   private void LateUpdate()
    {
        SmoothFollow();   
    }

    public void SmoothFollow()
    {
        Vector3 targetPos = target_transform.position + offset;
        transform.position = Vector3.Lerp(transform.position, targetPos, smoothSpeed);
        // transform.LookAt(target_transform);
    }
}
