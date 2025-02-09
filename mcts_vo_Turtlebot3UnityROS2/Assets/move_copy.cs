using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class move_copy : MonoBehaviour
{

    public float dt = 0.3f; // Time interval for movement
    private float timer = 0f;
    private Vector3 startPosition;
    private Vector3 targetPosition;
    private int idx = 240-60;

    // PARAMETERS SINUSOIDAL
    public float amplitude = 0.01f; // Amplitude of the sinusoidal wave
    public  float frequency = 1f; // Frequency of the sinusoidal wave
    private float forwardSpeed = 0.08f; // Forward speed
    private int mulForwardSpeed = 1;


    // Start is called before the first frame update
    void Start()
    {
        Random.InitState(42);
        startPosition = transform.position;
        targetPosition = transform.position;     
    }

    // Update is called once per frame
    void Update()
    {
        timer += Time.deltaTime;

        if (timer >= dt){
            // Debug.Log("Current idx: " + idx);

            timer = 0f;
            // X python = Unity Z
            // Z python = Unity Y 
            // Y python = Unity -X
            Vector3 pos = transform.position;
            startPosition = transform.position;
            // Debug.Log("Current idx: " + idx);
            
            // Trefoil knot trajectory calculation
            // float t = idx * dt * 0.01f; // Reduce the speed by scaling down t
            // int multiplier = 1;
            // float omega = 0.1f;
            // float scale_x = 0.1f;
            // float scale_y = 0.1f;
            // float trefoilX = multiplier * scale_x * (Mathf.Sin(omega * idx) + 2 * Mathf.Sin(2 * omega * idx));
            // float trefoilZ = multiplier * scale_y * (Mathf.Cos(omega * idx) - 2 * Mathf.Cos(2 * omega * idx));
            // pos.x = startPosition.x + trefoilX;
            // pos.z = startPosition.z + trefoilZ;
            // float velocity = get_velocity(idx);
            // float angle = get_angle(idx) * Mathf.Deg2Rad;
            // float new_z = pos.z + velocity * dt * Mathf.Cos(angle);
            // float new_x = pos.x + velocity * dt * Mathf.Sin(angle);


            float offset = Mathf.Sin(idx * frequency * dt) * amplitude;
            pos.x += forwardSpeed * dt;
            pos.z += offset;

            targetPosition = pos;
            idx += 1;
        }
        else
        {
            // Interpolate the position smoothly between the start and target positions
            float t = timer / dt;
            transform.position = Vector3.Lerp(startPosition, targetPosition, t);
        }

        float speed = Random.Range(0.0f, 0.1f);
        forwardSpeed = mulForwardSpeed * speed;
        if (idx == 240){
            mulForwardSpeed *= -1;
            idx = 0;
        }
    }
}
