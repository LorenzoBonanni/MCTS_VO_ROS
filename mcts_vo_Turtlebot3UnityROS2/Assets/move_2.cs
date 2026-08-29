using System.Collections.Generic;
using UnityEngine;

public class move_2 : MonoBehaviour
{
    public float simulationDt = 0.3f;

    private const float tunedDt = 0.3f;
    // Halved from the original 50/15/66 (tunedDt units) for the same reason
    // as move_1.cs: the original ~19.8 s recording leg outlives one ~15 s
    // episode, so watched for a while it keeps drifting rather than settling
    // into its bounded back-and-forth. Halved values finish recording by
    // ~9.9 s, well inside one episode.
    private int angleSwitchIdx = 25;
    [SerializeField] private int minIdx = 25;
    [SerializeField] private int maxIdx = 40;
    public float max_speed = 0.15f;

    private int idx = 0;
    private Vector3 currentPosition;
    private float[] precomputedSpeeds;

    private List<Vector3> steps = new List<Vector3>();
    private bool recording = true;
    private int phase = 0;
    private Vector3 replayStartPosition;   // position when replay began
    private Vector3 outwardStartPosition;  // absolute start of outward leg

    public Vector3 currentVelocity { get; private set; }
    public float currentSpeed { get; private set; }
    private float maxSpeedSeen = 0f;

    private float accumulator = 0f;
    private int totalSteps = 0;
    private int cycleCount = 0;            // number of completed replay cycles

    [Header("Debug Logging")]
    public bool enableLogging = true;
    public int logInterval = 10;

    [Header("Replay Settings")]
    public bool resetPositionEachCycle = true;  // correct floating‑point drift

    void Start()
    {
        Random.InitState(42);

        float scale = tunedDt / simulationDt;
        angleSwitchIdx = Mathf.RoundToInt(angleSwitchIdx * scale);
        minIdx = Mathf.RoundToInt(minIdx * scale);
        maxIdx = Mathf.RoundToInt(maxIdx * scale);

        int activeCount = maxIdx - minIdx + 1;
        precomputedSpeeds = new float[activeCount];
        int activeStepCount = 0;
        float lastSpeed = 0f;
        for (int i = 0; i < activeCount; i++)
        {
            if (activeStepCount % 100 == 0)
                lastSpeed = Random.Range(0.10f, max_speed);
            precomputedSpeeds[i] = lastSpeed;
            activeStepCount++;
        }

        outwardStartPosition = transform.position;
        currentPosition = outwardStartPosition;
        idx = 0;
        recording = true;
        phase = 0;
        steps.Clear();
        currentVelocity = Vector3.zero;
        currentSpeed = 0f;
        replayStartPosition = outwardStartPosition;
        totalSteps = 0;
        cycleCount = 0;

        if (enableLogging)
            Debug.Log("move_2 started. Logging enabled, interval = " + logInterval);
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
            step = ComputeRecordingStep();
            if (step != Vector3.zero)
                steps.Add(step);

            currentPosition += step;
            transform.position = currentPosition;
            idx++;

            if (idx > maxIdx)
            {
                recording = false;
                phase = steps.Count;          // start backward replay
                replayStartPosition = currentPosition;
                // cycleCount remains 0 until the first full replay cycle completes
            }
        }
        else
        {
            step = ComputeReplayStep();
            currentPosition += step;
            transform.position = currentPosition;

            // If we just wrapped to phase 0, we have completed one full replay cycle
            if (phase == 0 && resetPositionEachCycle)
            {
                // Reset to the exact outward start position to eliminate drift
                currentPosition = outwardStartPosition;
                transform.position = currentPosition;
                cycleCount++;
                if (enableLogging)
                    Debug.Log($"Cycle {cycleCount} completed. Position reset to start.");
            }
        }

        currentVelocity = step / simulationDt;
        currentSpeed = currentVelocity.magnitude;
        if (currentSpeed > maxSpeedSeen)
            maxSpeedSeen = currentSpeed;

        totalSteps++;
        if (PositionLogger.Enabled)
        {
            PositionLogger.LogRow(gameObject.name, totalSteps, totalSteps * simulationDt,
                                   currentPosition.x, currentPosition.z, currentSpeed, maxSpeedSeen,
                                   transform.localScale.x / 2f);
        }

        if (enableLogging && totalSteps % logInterval == 0)
        {
            float simTime = idx * simulationDt;
            string mode = recording ? "Recording" : "Replay";
            int stepCount = recording ? idx : phase;
            Vector3 displacement = currentPosition - replayStartPosition;

            Debug.Log(
                $"{mode} Step {stepCount} (total {totalSteps}) | Time: {simTime:F2}s | " +
                $"Speed: {currentSpeed:F4} m/s | " +
                $"Pos: ({currentPosition.x:F4}, {currentPosition.z:F4}) | " +
                $"Step: ({step.x:F4}, {step.z:F4}) | " +
                $"Disp from replay start: ({displacement.x:F4}, {displacement.z:F4}) | " +
                $"Angle: {GetAngle(idx)}° | Active: {(currentSpeed > 0.001f ? "Yes" : "No")}"
            );
        }
    }

    private Vector3 ComputeRecordingStep()
    {
        if (idx < minIdx || idx > maxIdx)
            return Vector3.zero;

        float angle = GetAngle(idx) * Mathf.Deg2Rad;
        float speed = precomputedSpeeds[idx - minIdx];
        return new Vector3(
            speed * simulationDt * Mathf.Sin(angle),
            0f,
            speed * simulationDt * Mathf.Cos(angle)
        );
    }

    private Vector3 ComputeReplayStep()
    {
        int n = steps.Count;
        if (n == 0)
            return Vector3.zero;

        // phase runs from 0 to 2n-1
        // 0..n-1: forward replay (same steps)
        // n..2n-1: backward replay (negated steps in reverse order)
        Vector3 s = phase < n ? steps[phase] : -steps[2 * n - 1 - phase];
        phase = (phase + 1) % (2 * n);
        return s;
    }

    private int GetAngle(int idx)
    {
        return idx < angleSwitchIdx ? -90 : -80;
    }
}
