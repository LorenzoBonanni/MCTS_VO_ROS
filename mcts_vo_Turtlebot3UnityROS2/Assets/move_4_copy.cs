using UnityEngine;

public class move_4_copy : MonoBehaviour
{
    // Public simulation step (matches the original 'dt')
    // For determinism, keep this constant across runs.
    public float simulationDt = 0.3f;

    // Sinusoidal parameters (same as original)
    public float amplitude = 0.01f;
    public float frequency = 1f;
    public float forwardSpeed = 0.1f;   // will be overwritten by random in Start
    public int multiplier = 1;
    public float maxSpeed = 0.15f;

    // State variables
    private int idx = 0;
    private int speedStepCount = 0;      // counts every step, never reset
    private int mulForwardSpeed = 1;     // flips sign when loop restarts

    // Exact position – no interpolation
    private Vector3 currentPosition;

    // Accumulator for fixed‑time stepping
    private float accumulator = 0f;

    void Start()
    {
        // 1. Fixed seed for full determinism
        Random.InitState(42);

        // 2. Initial random speed – this call is required to keep the
        //    random sequence in sync with the original (even though it
        //    will be redrawn on the first step).
        forwardSpeed = mulForwardSpeed * Random.Range(0.05f, 0.1f);

        // 3. Initialize position
        currentPosition = transform.position;
        transform.position = currentPosition;

        idx = 0;
        speedStepCount = 0;
        mulForwardSpeed = 1;
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
        // 1. Randomize speed every 100 steps (exactly as original)
        if (speedStepCount % 100 == 0)
        {
            forwardSpeed = mulForwardSpeed * Random.Range(0.05f, maxSpeed);
        }
        speedStepCount++;

        // 2. Compute sinusoidal offset for current and previous step
        //    The offset is applied to the Z coordinate.
        float previousOffset = Mathf.Sin((idx - 1) * frequency * simulationDt) * amplitude;
        float currentOffset  = Mathf.Sin(idx * frequency * simulationDt) * amplitude;

        // 3. Move in X (forward) and update Z by the delta of the sine
        currentPosition.x += forwardSpeed * simulationDt;
        currentPosition.z += currentOffset - previousOffset;

        // 4. Update transform directly – no interpolation
        transform.position = currentPosition;

        // 5. Increment index
        idx++;

        // 6. Check for loop reset (mirrors original condition)
        if (idx == 120 * multiplier)
        {
            mulForwardSpeed *= -1;   // reverse direction
            idx = 0;                 // restart sine phase
            // speedStepCount continues to increase (not reset)
        }
    }
}