using UnityEngine;

public class move_4 : MonoBehaviour
{
    // Public simulation step (matches the original 'dt')
    // For determinism, keep this constant across runs.
    public float simulationDt = 0.3f;

    // Number of steps to pre‑compute (covers long runs). 
    // At dt=0.3, 10000 steps = 3000 seconds ≈ 50 minutes.
    [SerializeField] private int precomputeSteps = 10000;

    // State variables
    private int idx = 70;                // current index in the angle/position cycle
    private int globalStepCounter = 0;   // total steps taken since start (never resets)

    // Pre‑computed speeds for each step in time
    private float[] precomputedSpeeds;

    // Exact position – no interpolation
    private Vector3 currentPosition;

    // Accumulator for fixed‑time stepping
    private float accumulator = 0f;

    void Start()
    {
        // 1. Fixed seed for full determinism
        Random.InitState(42);

        // 2. Pre‑compute speeds for the entire run (or a long enough horizon)
        precomputedSpeeds = new float[precomputeSteps];
        for (int i = 0; i < precomputeSteps; i++)
        {
            // Original get_velocity always returns Random.Range for idx >= 70,
            // which is always true after start. So we generate a speed for every step.
            precomputedSpeeds[i] = Random.Range(0.10f, 0.15f);
        }

        // 3. Initialize position
        currentPosition = transform.position;
        transform.position = currentPosition;

        idx = 70;
        globalStepCounter = 0;
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
        // 1. Get deterministic angle based on the current idx
        float angle = GetAngle(idx) * Mathf.Deg2Rad;

        // 2. Get speed from the pre‑computed sequence (using global step counter)
        //    Wrap around if we exceed the precomputed array length.
        float speed = precomputedSpeeds[globalStepCounter % precomputeSteps];

        // 3. Move in X‑Z plane (angle measured from Z axis)
        float stepX = speed * simulationDt * Mathf.Sin(angle);
        float stepZ = speed * simulationDt * Mathf.Cos(angle);

        currentPosition.x += stepX;
        currentPosition.z += stepZ;

        // 4. Update transform directly – no interpolation
        transform.position = currentPosition;

        // 5. Advance indices
        idx++;
        globalStepCounter++;

        // 6. Reset idx when it exceeds the cycle limit (mirrors original logic)
        if (idx > 330)
        {
            idx = 70;
        }
    }

    /// <summary>
    /// Deterministic angle function (same as original).
    /// </summary>
    private int GetAngle(int idx)
    {
        if (idx < 100)
            return 0;
        else if (idx < 115)
            return 120;
        else if (idx < 210)
            return 90;
        else if (idx < 230)
            return 180;
        else
            return -90;
    }
}