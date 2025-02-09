using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Test : MonoBehaviour
{
    // Start is called before the first frame update
    void Start()
    {
        MeshFilter meshFilter = gameObject.GetComponentInChildren<MeshFilter>();
        if (meshFilter != null)
        {
            Bounds meshBounds = meshFilter.mesh.bounds;
            Vector3 objectScale = gameObject.transform.lossyScale;
            Debug.Log("Object Scale: " + objectScale);

            float radius = Vector3.Scale(meshBounds.extents, objectScale).magnitude;
            float diameter = Vector3.Scale(meshBounds.size, objectScale).magnitude;

            Debug.Log("extents: " + meshBounds.extents);
            Debug.Log("size: " + meshBounds.size);
        }
        else
        {
            Debug.Log("No MeshFilter found.");
        }
    }

    // Update is called once per frame
    void Update()
    {
        
    }
}
