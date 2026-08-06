using UnityEngine;
using System.Collections.Generic;

public class move_1 : MonoBehaviour
{
    public float simulationDt = 0.1f;

    private const float tunedDt = 0.3f;
    private int shift = 15;
    private int seg1 = 15;
    private int seg2 = 20;
    private int seg3 = 28;
    private int seg4 = 40;
    private int maxIdx = 80;
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
