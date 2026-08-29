using UnityEngine;

public class move_4_copy_int : MonoBehaviour
{
    // Public simulation step (matches the original 'dt')
    // For determinism, keep this constant across runs.
    public float simulationDt = 0.3f;

    // Goals (same as original)
    private static readonly Vector3 goal1 = new Vector3(-0.53f, 0.1f, -1.14f);
    private static readonly Vector3 goal2 = new Vector3(1.826f, 0.1f, -1.139f);
    private Vector3 goal = goal1;

    // State variables
    private int idx = 0;
    private float randNum = 0.0f;
    public float velocity = 0.15f;

    // Exact position (no interpolation)
    private Vector3 currentPosition;

    // Speed measurement (updated each step)
    public Vector3 currentVelocity { get; private set; }  // velocity in world units per second
    public float currentSpeed { get; private set; }       // magnitude of velocity

    // Accumulator for fixed‑time stepping
    private float accumulator = 0f;

    // Debug logging settings
    [Header("Debug Logging")]
    public bool enableLogging = true;          // enable/disable console logging
    public int logInterval = 10;               // log every N steps (e.g., 10)
    private int logCounter = 0;                // counts steps for logging
    private float maxSpeedSeen = 0f;

    void Start()
    {
        // 1. Fixed seed for full determinism
        Random.InitState(42);

        // 2. Initialize position
        currentPosition = transform.position;
        transform.position = currentPosition;

        // Set initial goal (as in original)
        goal = goal1;
        idx = 0;
        randNum = 0f;
        currentVelocity = Vector3.zero;
        currentSpeed = 0f;
        logCounter = 0;
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
        // 1. Check if we are close enough to the current goal → switch
        if (Vector3.Distance(currentPosition, goal) < 0.1f)
        {
            goal = (goal == goal1) ? goal2 : goal1;
        }

        // 2. Generate new random perturbation every 100 steps
        if (idx % 100 == 0)
        {
            randNum = Random.Range(-0.5f, 0.5f) * 2.5f;
        }

        // 3. Compute direction toward goal
        Vector3 direction = (goal - currentPosition).normalized;
        float goal_angle = Mathf.Atan2(direction.x, direction.z);
        float angle = goal_angle + randNum;

        // 4. Compute step displacement (X‑Z plane)
        float stepX = velocity * simulationDt * Mathf.Sin(angle);
        float stepZ = velocity * simulationDt * Mathf.Cos(angle);
        Vector3 step = new Vector3(stepX, 0f, stepZ);

        // 5. Update position
        currentPosition += step;
        transform.position = currentPosition;

        // 6. Compute velocity and speed from the step
        currentVelocity = step / simulationDt;   // units per second
        currentSpeed = currentVelocity.magnitude;
        if (currentSpeed > maxSpeedSeen)
            maxSpeedSeen = currentSpeed;

        // 7. Increment step index
        idx++;

        if (PositionLogger.Enabled)
        {
            PositionLogger.LogRow(gameObject.name, idx, idx * simulationDt,
                                   currentPosition.x, currentPosition.z, currentSpeed, maxSpeedSeen,
                                   transform.localScale.x / 2f);
        }

        // 8. Debug logging at configured interval
        if (enableLogging && idx % logInterval == 0)
        {
            // Log time (simulation time), idx, speed, and current position
            float simTime = idx * simulationDt;
            Debug.Log($"Step {idx} | Time: {simTime:F2}s | Speed: {currentSpeed:F4} m/s | " +
                      $"Position: ({currentPosition.x:F4}, {currentPosition.z:F4}) | " +
                      $"Goal: {(goal == goal1 ? "goal1" : "goal2")}");
        }
    }
}
