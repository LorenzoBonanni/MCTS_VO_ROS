using UnityEngine;

public class move_copy_int : MonoBehaviour
{
    // Public simulation step (matches the original 'dt')
    // For determinism, keep this constant across runs.
    public float simulationDt = 0.3f;

    // Goals (same as original)
    private static readonly Vector3 goal1 = new Vector3(1.731f, 0.1f, -2.018f);
    private static readonly Vector3 goal2 = new Vector3(-0.53f, 0.1f, -2.48f);
    private Vector3 goal = goal1;

    // Parameters
    public float velocity = 0.15f;

    // State variables
    private int idx = 0;
    private float randNum = 0f;

    // Pre‑computed random perturbations for each event (idx % 10 == 0)
    // We pre‑compute enough events to cover a long simulation.
    // Default: 100000 steps => 10000 events (≈ 8.3 hours at dt=0.3s)
    [SerializeField] private int maxSteps = 100000;
    private float[] precomputedRandoms;

    // Exact position – no interpolation
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
    public bool enableCsvLogging = false;
    private float maxSpeedSeen = 0f;

    void Start()
    {
        // 1. Fixed seed for full determinism
        Random.InitState(42);

        // 2. Pre‑compute random numbers for every event (idx multiple of 10)
        int eventCount = maxSteps / 10 + 1;
        precomputedRandoms = new float[eventCount];
        for (int i = 0; i < eventCount; i++)
        {
            // The original called: Random.Range(-0.5f, 0.5f) * 1.5f
            precomputedRandoms[i] = Random.Range(-0.5f, 0.5f) * 1.5f;
        }

        // 3. Initialize position and goal
        currentPosition = transform.position;
        transform.position = currentPosition;
        goal = goal1;          // start with goal1
        idx = 0;
        randNum = 0f;
        currentVelocity = Vector3.zero;
        currentSpeed = 0f;
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

        // 2. Update randNum every 10 steps (using pre‑computed values)
        if (idx % 10 == 0)
        {
            int eventIdx = idx / 10;
            // Wrap around if we exceed the pre‑computed array
            randNum = precomputedRandoms[eventIdx % precomputedRandoms.Length];
        }

        // 3. Compute direction toward the current goal
        Vector3 direction = (goal - currentPosition).normalized;
        float goal_angle = Mathf.Atan2(direction.x, direction.z);
        float angle = goal_angle + randNum;

        // 4. Move in X‑Z plane (angle measured from Z axis)
        float stepX = velocity * simulationDt * Mathf.Sin(angle);
        float stepZ = velocity * simulationDt * Mathf.Cos(angle);
        Vector3 step = new Vector3(stepX, 0f, stepZ);

        // 5. Update position
        currentPosition.x += stepX;
        currentPosition.z += stepZ;
        transform.position = currentPosition;

        // 6. Compute velocity and speed from the step
        currentVelocity = step / simulationDt;   // units per second
        currentSpeed = currentVelocity.magnitude;
        if (currentSpeed > maxSpeedSeen)
            maxSpeedSeen = currentSpeed;

        // 7. Increment step index
        idx++;

        ObstacleCsvLogger.LogRow(enableCsvLogging, gameObject.name, idx, idx * simulationDt,
                                  currentPosition.x, currentPosition.z, currentSpeed, maxSpeedSeen);

        // 8. Debug logging at configured interval
        if (enableLogging && idx % logInterval == 0)
        {
            float simTime = idx * simulationDt;
            Debug.Log($"Step {idx} | Time: {simTime:F2}s | Speed: {currentSpeed:F4} m/s | " +
                      $"Position: ({currentPosition.x:F4}, {currentPosition.z:F4}) | " +
                      $"Goal: {(goal == goal1 ? "goal1" : "goal2")}");
        }
    }
}
