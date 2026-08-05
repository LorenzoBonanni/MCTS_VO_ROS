using System.Collections;
using System.Collections.Generic;
using UnityEngine;
public class move_4_copy : MonoBehaviour
{
    public float dt = 0.3f; // Time interval for movement
    private float timer = 0f;
    private Vector3 startPosition;
    private Vector3 targetPosition;
    private int idx = 0;
    // PARAMETERS SINUSOIDAL
    public float amplitude = 0.01f;
    public float frequency = 1f;
    public float forwardSpeed = 0.1f;
    private int mulForwardSpeed = 1;
    public int multiplier = 1;

    // The random speed is only redrawn every second completed movement step
    // and reused for the intermediate step in between.
    private int speedStepCount = 0;
    void Start()
    {
        Random.InitState(42);
        startPosition = transform.position;
        targetPosition = transform.position;
        // initial speed
        forwardSpeed = mulForwardSpeed * Random.Range(0.05f, 0.1f);
    }
    void Update()
    {
        timer += Time.deltaTime;
        while (timer >= dt)
        {
            // keep leftover time
            timer -= dt;
            // Start next segment from previous target
            startPosition = targetPosition;
            Vector3 pos = startPosition;
            // Randomize speed, only every second completed movement step;
            // reuse it for the step in between.
            if (speedStepCount % 100 == 0)
            {
                forwardSpeed = mulForwardSpeed * Random.Range(0.05f, 0.15f);
            }
            speedStepCount++;
            // Sinusoidal trajectory: apply the change in the sinusoidal
            // offset between consecutive steps, not the absolute offset,
            // so the path oscillates instead of drifting.
            float previousOffset = Mathf.Sin((idx - 1) * frequency * dt) * amplitude;
            float currentOffset  = Mathf.Sin(idx * frequency * dt) * amplitude;
            pos.x += forwardSpeed * dt;
            pos.z += currentOffset - previousOffset;
            targetPosition = pos;
            idx++;
            if (idx == (120 * multiplier))
            {
                mulForwardSpeed *= -1;
                idx = 0;
            }
        }
        // Smooth interpolation
        float t = timer / dt;
        transform.position = Vector3.Lerp(
            startPosition,
            targetPosition,
            t
        );
    }
}
