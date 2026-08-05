using System.Collections;
using System.Collections.Generic;
using UnityEngine;
public class move_2 : MonoBehaviour
{
    public float dt = 0.3f; // Time interval for movement
    private float timer = 0f;
    private Vector3 startPosition;
    private Vector3 targetPosition;
    private int idx = 0;
    // The thresholds below (50, 15, 100) used in get_angle/get_velocity are
    // step counts, not seconds. They were tuned assuming dt = 0.3s per step,
    // so at that dt they mark real times of 15s, 4.5s, and 30s. If dt
    // changes, the same step counts land at different real times, so they're
    // rescaled in Start() by tunedDt / dt to keep the same real-world
    // schedule at any dt.
    private const float tunedDt = 0.3f;
    private int angleSwitchIdx = 50;
    private int minIdx = 15;
    private int maxIdx = 100;
    // The random speed is only redrawn every second completed movement step
    // (velocity > 0 steps) and reused for the intermediate step, so
    // currentSpeed caches the last draw and velocityStepCount tracks parity.
    private float currentSpeed = 0f;
    private int velocityStepCount = 0;

    // The outward leg is recorded step by step, then replayed forwards and
    // backwards for ever, so the obstacle retraces exactly the path it drove
    // the first time instead of stopping once it reaches maxIdx. Recording
    // is the only way to retrace it: get_velocity draws a fresh Random.Range
    // periodically, so the same idx does not produce the same displacement
    // twice.
    private List<Vector3> steps = new List<Vector3>();
    private bool recording = true;
    private int phase = 0;
    // Start is called before the first frame update
    void Start()
    {
        Random.InitState(42);
        startPosition = transform.position;
        targetPosition = transform.position;
        // Rescale the step-count thresholds so they keep marking the same
        // real-world times regardless of dt.
        float scale = tunedDt / dt;
        angleSwitchIdx = Mathf.RoundToInt(angleSwitchIdx * scale);
        minIdx = Mathf.RoundToInt(minIdx * scale);
        maxIdx = Mathf.RoundToInt(maxIdx * scale);
    }
    int get_angle(int idx){
        if (idx < angleSwitchIdx){
            return -90;
        }
        else {
            return -80;
        }
    }
    float get_velocity(int idx){
        if(idx < minIdx | idx > maxIdx){
            return 0f;
        }
        else {
            if (velocityStepCount % 100 == 0)
            {
                currentSpeed = Random.Range(0.10f, 0.25f);
            }
            velocityStepCount += 1;
            return currentSpeed;
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
                step = new Vector3(velocity * dt * Mathf.Sin(angle), 0f,
                                   velocity * dt * Mathf.Cos(angle));
                steps.Add(step);
            }
            if (idx > maxIdx)
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
        timer += Time.deltaTime;
        while (timer >= dt){
            // Debug.Log("Index: " + idx);
            timer -= dt;
            transform.position = targetPosition;
            startPosition = targetPosition;
            targetPosition = startPosition + next_step();
        }
        // Interpolate the position smoothly between the start and target positions
        float t = timer / dt;
        transform.position = Vector3.Lerp(startPosition, targetPosition, t);
    }
}
