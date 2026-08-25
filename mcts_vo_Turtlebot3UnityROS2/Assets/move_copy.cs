using System.Collections.Generic;
using UnityEngine;

public class move_copy : MonoBehaviour
{
    // Public simulation step (matches the original 'dt')
    // For determinism, keep this constant across runs.
    public float simulationDt = 0.3f;

    // Sinusoidal parameters (same as original)
    public float amplitude = 0.01f;
    public float frequency = 1f;
    public int multiplier = 1;
    public float maxSpeed = 0.15f;

    // Lower bound of the forward-speed draw. Was hardcoded to 0, so the forward
    // speed averaged half of maxSpeed and varied per redraw: this obstacle
    // advanced roughly half as far as move_4_copy for the same settings, and by
    // an amount that depended on the RNG. Set it equal to maxSpeed for a
    // constant forward speed.
    public float minSpeed = 0f;

    // First index of the recorded leg; the leg is 240*multiplier - startIdx
    // steps long. Was a hardcoded 180, which left multiplier as the only lever
    // and made the leg jump 60 -> 300 -> 540 steps. Lower it to lengthen the
    // outward run without changing anything else.
    public int startIdx = 180;

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
    public bool enableCsvLogging = false;

    // State for replay
    private List<Vector3> steps = new List<Vector3>();
    private int phase = 0;              // 0 .. 2n-1 (out then back)

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

        // 2. Consume one random number to match the original's Start() call.
        //    (This value is not used for movement, but it ensures the random
        //     stream is in the same state as the original script.)
        float dummy = Random.Range(0.0f, 0.1f);

        // 3. Pre‑compute the entire outward leg.
        //    Recording starts at idx = 240 - 60 = 180 and continues until
        //    idx reaches 240 * multiplier. The step at each idx uses that
        //    idx for the sine and then increments idx.
        int endIdx = 240 * multiplier;   // exclusive (stop before this)
        int currentIdx = startIdx;
        int speedStepCount = 0;
        float forwardSpeed = 0f;          // will be set on first redraw
        int mulForwardSpeed = 1;

        while (currentIdx < endIdx)
        {
            // Redraw speed every 100 steps (matches original)
            if (speedStepCount % 100 == 0)
            {
                forwardSpeed = mulForwardSpeed * Random.Range(minSpeed, maxSpeed);
            }
            speedStepCount++;

            // Compute the sinusoidal offset delta
            float prevOffset = Mathf.Sin((currentIdx - 1) * frequency * simulationDt) * amplitude;
            float currOffset = Mathf.Sin(currentIdx * frequency * simulationDt) * amplitude;

            // Step vector: forward in X, sinusoidal change in Z
            Vector3 step = RotateStep(forwardSpeed * simulationDt, currOffset - prevOffset);
            steps.Add(step);

            currentIdx++;
        }

        // 4. Initialize position
        currentPosition = transform.position;
        transform.position = currentPosition;

        // 5. Start replay from the beginning (phase = 0 means forward replay)
        phase = 0;
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
        int n = steps.Count;
        if (n == 0)
            return;

        // phase runs from 0 to 2n-1:
        //   - 0 .. n-1        : forward replay of recorded steps
        //   - n .. 2n-1       : backward replay (negated steps in reverse)
        Vector3 s = phase < n ? steps[phase] : -steps[2 * n - 1 - phase];

        // Move
        Vector3 previousPosition = currentPosition;
        currentPosition += s;
        transform.position = currentPosition;

        // Advance phase and wrap around for continuous cycling
        phase = (phase + 1) % (2 * n);

        // ---- speed from the realised displacement ----
        currentVelocity = (currentPosition - previousPosition) / simulationDt;
        currentSpeed = currentVelocity.magnitude;
        if (currentSpeed > maxSpeedSeen)
            maxSpeedSeen = currentSpeed;

        loggedSteps++;
        ObstacleCsvLogger.LogRow(enableCsvLogging, gameObject.name, loggedSteps, loggedSteps * simulationDt,
                                  currentPosition.x, currentPosition.z, currentSpeed, maxSpeedSeen);

        if (enableLogging && loggedSteps % logInterval == 0)
        {
            Debug.Log($"{GetType().Name} step {loggedSteps} | " +
                      $"Speed: {currentSpeed:F4} m/s | Max so far: {maxSpeedSeen:F4} m/s | " +
                      $"Step: ({s.x:F4}, {s.z:F4}) | " +
                      $"Position: ({currentPosition.x:F4}, {currentPosition.z:F4})");
        }
    }
}