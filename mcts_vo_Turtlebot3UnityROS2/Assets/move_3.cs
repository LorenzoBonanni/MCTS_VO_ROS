using UnityEngine;

public class move_3 : MonoBehaviour
{
    // Public fixed simulation step (matches the original 'dt')
    // For determinism, keep this constant across runs.
    public float simulationDt = 0.3f;

    // Thresholds (step counts) – no scaling needed because they are kept as is.
    // (Original had no scaling logic, so we keep them fixed.)
    private const int angleSwitchIdx = 75;
    private const int minIdx = 40;
    private const int maxIdx = 130;

    // Pre‑computed speeds for each active step index
    private float[] precomputedSpeeds;

    // State
    private int idx = 0;
    private Vector3 currentPosition;   // exact position – no interpolation

    // Accumulator for fixed‑time stepping
    private float accumulator = 0f;

    void Start()
    {
        // 1. Fixed seed for full determinism
        Random.InitState(42);

        // 2. Pre‑compute speeds for all active steps (minIdx … maxIdx)
        int activeCount = maxIdx - minIdx + 1;
        precomputedSpeeds = new float[activeCount];
        for (int i = 0; i < activeCount; i++)
        {
            // Original code drew a new random speed for every step.
            // We replicate that exactly.
            precomputedSpeeds[i] = Random.Range(0.10f, 0.15f);
        }

        // 3. Initialize position
        currentPosition = transform.position;
        idx = 0;
    }

    void FixedUpdate()
    {
        // Accumulate real physics time (fixedDeltaTime is constant)
        accumulator += Time.fixedDeltaTime;

        // Advance as many whole simulation steps as possible
        while (accumulator >= simulationDt)
        {
            StepObstacle();
            accumulator -= simulationDt;
        }

        // No interpolation – sensors read the exact discrete position.
    }

    /// <summary>
    /// Performs exactly one simulation step (length = simulationDt).
    /// </summary>
    private void StepObstacle()
    {
        // Stop if we've passed the defined trajectory length
        if (idx > maxIdx)
            return;

        // Deterministic angle
        float angle = GetAngle(idx) * Mathf.Deg2Rad;

        // Pre‑computed speed – if outside active interval, speed is 0
        float speed = 0f;
        if (idx >= minIdx && idx <= maxIdx)
        {
            speed = precomputedSpeeds[idx - minIdx];
        }

        // Movement in X‑Z plane (angle measured from Z axis)
        float stepX = speed * simulationDt * Mathf.Sin(angle);
        float stepZ = speed * simulationDt * Mathf.Cos(angle);

        currentPosition.x += stepX;
        currentPosition.z += stepZ;

        // Update transform directly – no interpolation
        transform.position = currentPosition;

        idx++;
    }

    /// <summary>
    /// Deterministic angle function.
    /// </summary>
    private int GetAngle(int idx)
    {
        return (idx < angleSwitchIdx) ? 0 : -50;
    }
}