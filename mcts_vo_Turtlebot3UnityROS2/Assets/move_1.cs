using UnityEngine;
using System.Collections.Generic;

public class move_1 : MonoBehaviour
{
    public float simulationDt = 0.1f;

    private const float tunedDt = 0.3f;
    // Halved from the original 15/15/20/28/40/80 (tunedDt units, i.e. real
    // seconds independent of simulationDt): the original values gave a ~24 s
    // one-way recording leg before the obstacle starts bouncing within a
    // bounded path, which is longer than one ~15 s episode - watched in the
    // editor (no external reset) it just keeps expanding outward the whole
    // time, eventually leaving the arena / hitting static geometry. Halving
    // keeps the same relative segment shape but finishes recording by ~12 s,
    // so it bounces within a fixed, arena-sized path well inside one episode.
    private int shift = 8;
    private int seg1 = 8;
    private int seg2 = 10;
    private int seg3 = 14;
    private int seg4 = 20;
    private int maxIdx = 40;
    public float maxSpeed = 0.15f;

    // State for forward precomputation
    private int idx = 0;
    private Vector3 currentPosition;
    private float[] precomputedSpeeds;

    // Recording and replay
    private List<Vector3> steps = new List<Vector3>();
    private bool recording = true;
    private int phase = 0;   // 0..2n-1

    // Speed measurement
    public Vector3 currentVelocity { get; private set; }
    public float currentSpeed { get; private set; }
    private float maxSpeedSeen = 0f;

    private float accumulator = 0f;

    [Header("Debug Logging")]
    public bool enableLogging = true;
    public int logInterval = 10;

    void Start()
    {
        Random.InitState(42);

        float scale = tunedDt / simulationDt;
        shift = Mathf.RoundToInt(shift * scale);
        seg1 = Mathf.RoundToInt(seg1 * scale);
        seg2 = Mathf.RoundToInt(seg2 * scale);
        seg3 = Mathf.RoundToInt(seg3 * scale);
        seg4 = Mathf.RoundToInt(seg4 * scale);
        maxIdx = Mathf.RoundToInt(maxIdx * scale);

        precomputedSpeeds = new float[maxIdx + 1];
        int activeStepCount = 0;
        float lastSpeed = 0f;
        for (int i = 0; i <= maxIdx; i++)
        {
            if (i >= shift && i <= maxIdx)
            {
                if (activeStepCount % 100 == 0)
                    lastSpeed = Random.Range(0.10f, maxSpeed);
                activeStepCount++;
                precomputedSpeeds[i] = lastSpeed;
            }
            else
            {
                precomputedSpeeds[i] = 0f;
            }
        }

        currentPosition = transform.position;
        idx = 0;
        recording = true;
        phase = 0;
        steps.Clear();
        currentVelocity = Vector3.zero;
        currentSpeed = 0f;

        if (enableLogging)
            Debug.Log("move_1 started. Will record path, then replay backward-first.");
    }

    void FixedUpdate()
    {
        accumulator += Time.fixedDeltaTime;
        while (accumulator >= simulationDt)
        {
            StepObstacle();
            accumulator -= simulationDt;
        }
    }

    private void StepObstacle()
    {
        Vector3 step = Vector3.zero;

        if (recording)
        {
            // ---- Outward leg: record and move ----
            if (idx <= maxIdx)
            {
                float speed = precomputedSpeeds[idx];
                float angle = GetAngle(idx) * Mathf.Deg2Rad;
                float stepX = speed * simulationDt * Mathf.Sin(angle);
                float stepZ = speed * simulationDt * Mathf.Cos(angle);
                step = new Vector3(stepX, 0f, stepZ);

                steps.Add(step);

                currentPosition += step;
                transform.position = currentPosition;
                idx++;

                if (idx > maxIdx)
                {
                    recording = false;
                    phase = steps.Count;   // start replay by going backward (undo last step)
                    if (enableLogging)
                        Debug.Log($"Recording complete with {steps.Count} steps. Starting backward replay.");
                }
            }
        }
        else
        {
            // ---- Replay: phase starts at n (backward), then cycles ----
            int n = steps.Count;
            if (n == 0)
                return;

            if (phase < n)
                step = steps[phase];                  // forward
            else
                step = -steps[2 * n - 1 - phase];     // backward

            phase = (phase + 1) % (2 * n);            // cycle forever

            currentPosition += step;
            transform.position = currentPosition;
        }

        currentVelocity = step / simulationDt;
        currentSpeed = currentVelocity.magnitude;
        if (currentSpeed > maxSpeedSeen)
            maxSpeedSeen = currentSpeed;

        int csvStep = recording ? idx : phase;
        if (PositionLogger.Enabled)
        {
            PositionLogger.LogRow(gameObject.name, csvStep, csvStep * simulationDt,
                                   currentPosition.x, currentPosition.z, currentSpeed, maxSpeedSeen,
                                   transform.localScale.x / 2f);
        }

        if (enableLogging)
        {
            int logStep = recording ? idx : phase;
            if (logStep % logInterval == 0)
            {
                float simTime = idx * simulationDt;
                string mode = recording ? "Recording" : "Replay";
                Debug.Log($"{mode} Step {logStep} | Time: {simTime:F2}s | Speed: {currentSpeed:F4} m/s | " +
                          $"Position: ({currentPosition.x:F4}, {currentPosition.z:F4}) | " +
                          $"Step: ({step.x:F4}, {step.z:F4})");
            }
        }
    }

    private int GetAngle(int idx)
    {
        if (idx < shift + seg1)
            return 120;
        else if (idx < shift + seg2)
            return 90;
        else if (idx < shift + seg3)
            return 80;
        else if (idx < shift + seg4)
            return 360;
        else
            return 90;
    }
}
