using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class move_2 : MonoBehaviour
{

    public float dt = 0.1f; // Time interval for movement
    private float timer = 0f;
    // See move_1.cs: the period is set so the fastest step comes out at the
    // scene-wide peak speed, while dt stays the value used in the position
    // arithmetic, so the path is unchanged.
    private float stepPeriod;
    private const float MinVelocity = 0.10f;
    private const float MaxVelocity = 0.15f;
    private Vector3 startPosition;
    private Vector3 targetPosition;
    private int idx = 0;

    // Step counts, not seconds: idx advances once per dt, so these were sized
    // for dt = 0.3 and are scaled by 3 now that every obstacle runs at 0.1.
    // The obstacle therefore waits the same 4.5 s and covers the same distance
    // as before, in smaller steps.
    private int shift = 45;
    private int lastIdx = 300;

    // The outward leg is recorded step by step, then replayed forwards and
    // backwards for ever, so the obstacle retraces exactly the path it drove
    // the first time instead of stopping at the end of it. Recording is the
    // only way to retrace it: get_velocity draws a fresh Random.Range every
    // step, so the same idx does not produce the same displacement twice.
    private List<Vector3> steps = new List<Vector3>();
    private bool recording = true;
    private int phase = 0;

    // Start is called before the first frame update
    void Start()
    {
        Random.InitState(42);
        // See move_1.cs: the fast pair keeps its own peak, uniform on
        // (0.10, 0.15) m/s at scale 1, rather than the scene-wide 0.1.
        stepPeriod = ObstacleSpeed.PeriodFor(MaxVelocity * dt, MaxVelocity);
        startPosition = transform.position;
        targetPosition = transform.position;
    }

    int get_angle(int idx){
        if (idx < 150){
            return -90;
        }
        else {
            return -80;
        }
    }

    float get_velocity(int idx){
        if(idx < shift | idx > lastIdx){
            return 0f;
        }
        else {
            return Random.Range(MinVelocity, MaxVelocity);
        }
    }

    Vector3 next_step()
    {
        if (recording)
        {
            float velocity = get_velocity(idx);
            float angle = get_angle(idx) * Mathf.Deg2Rad;
            idx += 1;
            Vector3 step = Vector3.zero;
            if (velocity > 0f)
            {
                // X python = Unity Z
                // Z python = Unity Y
                // Y python = Unity -X
                step = new Vector3(velocity * dt * Mathf.Sin(angle), 0f,
                                   velocity * dt * Mathf.Cos(angle));
                steps.Add(step);
            }
            if (idx > lastIdx)
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

    // Update is called once per frame
    void Update()
    {
        // See move_1.cs: carrying the timer remainder and finishing the step
        // before computing the next one is what makes the speed independent of
        // the frame rate.
        timer += Time.deltaTime;
        while (timer >= stepPeriod)
        {
            timer -= stepPeriod;
            transform.position = targetPosition;
            startPosition = targetPosition;
            targetPosition = startPosition + next_step();
        }
        // Interpolate the position smoothly between the start and target positions
        transform.position = Vector3.Lerp(startPosition, targetPosition, timer / stepPeriod);
    }
}
