using System.Collections;
using System.Collections.Generic;
using UnityEngine;
public class move_copy : MonoBehaviour
{
    public float dt = 0.3f;
    private float timer = 0f;
    private Vector3 startPosition;
    private Vector3 targetPosition;
    private int idx = 240 - 60;
    // PARAMETERS SINUSOIDAL
    public float amplitude = 0.01f;
    public float frequency = 1f;
    private float forwardSpeed;
    private int mulForwardSpeed = 1;
    public int multiplier = 1;
    private Vector3 old_pos;
    private Vector3 velocity;
    // The random speed is only redrawn every second completed movement step
    // and reused for the intermediate step in between.
    private int speedStepCount = 0;

    // The outward leg is recorded step by step, then replayed forwards and
    // backwards for ever, so the obstacle retraces exactly the path it drove
    // the first time instead of jumping at the point where it used to flip
    // direction and reset idx to 0 (which broke the sinusoid's continuity).
    private List<Vector3> steps = new List<Vector3>();
    private bool recording = true;
    private int phase = 0;
    void Start()
    {
        Random.InitState(42);
        startPosition = transform.position;
        targetPosition = transform.position;
        // initial speed
        forwardSpeed = mulForwardSpeed * Random.Range(0.0f, 0.1f);
    }

    Vector3 next_step()
    {
        if (recording)
        {
            // Randomize speed for this trajectory segment, only every second
            // completed movement step; reuse it for the step in between.
            if (speedStepCount % 100 == 0)
            {
                forwardSpeed = mulForwardSpeed * Random.Range(0.0f, 0.15f);
            }
            speedStepCount++;
            // Sinusoidal trajectory: apply the change in the sinusoidal
            // offset between consecutive steps, not the absolute offset.
            float previousOffset = Mathf.Sin((idx - 1) * frequency * dt) * amplitude;
            float currentOffset  = Mathf.Sin(idx * frequency * dt) * amplitude;
            Vector3 step = new Vector3(forwardSpeed * dt, 0f, currentOffset - previousOffset);
            steps.Add(step);
            idx++;
            if (idx == (240 * multiplier))
            {
                recording = false;      // outward leg done, start the way back
                phase = steps.Count;
            }
            return step;
        }

        int n = steps.Count;
        if (n == 0)
        {
            return Vector3.zero;
        }
        // phase runs 0..2n-1: out along the recorded steps, then back along the
        // same steps negated and in reverse order. Undoing exactly what was
        // done returns the obstacle to its starting point, so the cycle cannot
        // drift however long the run lasts.
        Vector3 s = phase < n ? steps[phase] : -steps[2 * n - 1 - phase];
        phase = (phase + 1) % (2 * n);
        return s;
    }

    void Update()
    {
        timer += Time.deltaTime;
        while (timer >= dt)
        {
            // preserve carryover
            timer -= dt;
            startPosition = targetPosition;
            targetPosition = startPosition + next_step();
        }
        // interpolation
        float t = timer / dt;
        old_pos = transform.position;
        transform.position = Vector3.Lerp(
            startPosition,
            targetPosition,
            t
        );
        // Compute velocity (m/s)
        velocity = (transform.position - old_pos) / Time.deltaTime;
    }
}
