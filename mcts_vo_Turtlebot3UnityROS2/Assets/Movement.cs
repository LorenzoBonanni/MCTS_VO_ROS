using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Movement : MonoBehaviour
{
    float scale_x = 0.1f;
    float scale_y = 0.1f;
    float omega = 0.1f;
    public float dt = 0.3f; // Time interval for movement
    private float timer = 0f;
    private Vector3 startPosition;
    private Vector3 targetPosition;
    private int idx = 0;

    // Start is called before the first frame update
    void Start()
    {
        startPosition = transform.position;
        targetPosition = transform.position;      
    }

    // Update is called once per frame
    void Update()
    {
        timer += Time.deltaTime;

        if (timer >= dt){
            Debug.Log("Timer: " + timer);
            timer = 0f;
            // X python = Unity Z
            // Z python = Unity Y 
            // Y python = Unity -X
            Vector3 pos = transform.position;
            startPosition = transform.position;
            // random number btw -1 and 1
            float multiplier = (float) -0.1;
            float delta_x = multiplier * scale_x * (Mathf.Sin(omega * idx) + 2 * Mathf.Sin(2 * omega * idx));
            float delta_y = multiplier * scale_y * (Mathf.Cos(omega * idx) - 2 * Mathf.Cos(2 * omega * idx));
            float speed = Mathf.Sqrt(Mathf.Pow(delta_x, 2) + Mathf.Pow(delta_y, 2)) / dt;
            Debug.Log("Speed: " + speed);
            pos.z += delta_x;
            pos.x -= delta_y;    
            targetPosition = pos;
            idx += 1;
        }
        else
        {
            // Interpolate the position smoothly between the start and target positions
            float t = timer / dt;
            transform.position = Vector3.Lerp(startPosition, targetPosition, t);
        }

    }
}
