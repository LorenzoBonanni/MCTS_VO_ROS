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

    // Lower bound of the forward-speed draw. Was hardcoded to 0.05, so setting
    // maxSpeed below that inverted the range and Unity drew in [maxSpeed, 0.05]
    // instead - the field silently stopped bounding anything, and the realised
    // speed exceeded the intended limit.
    public float minSpeed = 0.05f;

    // Direction of travel, in degrees, measured from +x towards +z. The step is
    // built as (forward, lateral) and then rotated by this angle, so the whole
    // motion turns together and the speed magnitude is unchanged - the bound
    // hypot(maxSpeed, amplitude*frequency) still holds at any angle.
    // 0 = advance along +x (the previous hardcoded behaviour), 180 = along -x,
    // 90 = along +z. transform.position is written in world coordinates, so
    // rotating the GameObject in the Inspector has no effect; this is the only
    // way to reorient the path.
    public float directionDeg = 0f;

    // Speed measurement, from the realised displacement: (p_t+1 - p_t) / dt.
    // move_1 and move_2 already expose these; move_copy/move_4_copy did not, so
    // the two obstacles that actually run in SIN_EASY could not be measured at
    // all and maxSpeed was mistaken for their speed. It is not: maxSpeed bounds
    // only the forward component, while the lateral term is the derivative of
    // amplitude*sin(i*frequency*dt) and contributes amplitude*frequency m/s on
    // its own.
    public Vector3 currentVelocity { get; private set; }
    public float currentSpeed { get; private set; }
    public float maxSpeedSeen { get; private set; }
    private int loggedSteps = 0;

    [Header("Debug Logging")]
    public bool enableLogging = true;
    public int logInterval = 10;

    // State variables
    private int idx = 0;
    private int speedStepCount = 0;      // counts every step, never reset
    private int mulForwardSpeed = 1;     // flips sign when loop restarts

    // Exact position – no interpolation
    private Vector3 currentPosition;

    // Accumulator for fixed‑time stepping
    private float accumulator = 0f;


    private Vector3 RotateStep(float forward, float lateral)
    {
        float a = directionDeg * Mathf.Deg2Rad;
        float c = Mathf.Cos(a);
        float s = Mathf.Sin(a);
        return new Vector3(forward * c - lateral * s, 0f, forward * s + lateral * c);
    }

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
            forwardSpeed = mulForwardSpeed * Random.Range(minSpeed, maxSpeed);
        }
        speedStepCount++;

        // 2. Compute sinusoidal offset for current and previous step
        //    The offset is applied to the Z coordinate.
        float previousOffset = Mathf.Sin((idx - 1) * frequency * simulationDt) * amplitude;
        float currentOffset  = Mathf.Sin(idx * frequency * simulationDt) * amplitude;

        // 3. Move in X (forward) and update Z by the delta of the sine
        Vector3 previousPosition = currentPosition;
        currentPosition += RotateStep(forwardSpeed * simulationDt,
                                      currentOffset - previousOffset);
        Vector3 s = currentPosition - previousPosition;

        // 4. Update transform directly – no interpolation
        transform.position = currentPosition;

        // ---- speed from the realised displacement ----
        currentVelocity = (currentPosition - previousPosition) / simulationDt;
        currentSpeed = currentVelocity.magnitude;
        if (currentSpeed > maxSpeedSeen)
            maxSpeedSeen = currentSpeed;

        loggedSteps++;
        if (PositionLogger.Enabled)
        {
            PositionLogger.LogRow(gameObject.name, loggedSteps, loggedSteps * simulationDt,
                                   currentPosition.x, currentPosition.z, currentSpeed, maxSpeedSeen,
                                   transform.localScale.x / 2f);
        }

        if (enableLogging && loggedSteps % logInterval == 0)
        {
            Debug.Log($"{GetType().Name} step {loggedSteps} | " +
                      $"Speed: {currentSpeed:F4} m/s | Max so far: {maxSpeedSeen:F4} m/s | " +
                      $"Step: ({s.x:F4}, {s.z:F4}) | " +
                      $"Position: ({currentPosition.x:F4}, {currentPosition.z:F4})");
        }


        // 5. Increment index
        idx++;

        // 6. Check for loop reset (mirrors original condition)
        if (idx == 120 * multiplier)
        {
            mulForwardSpeed *= -1;   // reverse direction
            // Apply the flip to forwardSpeed immediately. Bug fix: forwardSpeed
            // previously only picked up the new mulForwardSpeed sign at the next
            // "every 100 steps" redraw above, which is a different, unsynchronized
            // counter from the idx reset here (100 != 120*multiplier in general).
            // That let the obstacle keep travelling in the old direction for up
            // to ~100 extra steps past the intended turnaround - a large,
            // unbounded overshoot before it ever reversed.
            forwardSpeed = -forwardSpeed;
            idx = 0;                 // restart sine phase
            // speedStepCount continues to increase (not reset)
        }
    }
}