using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class move_1 : MonoBehaviour
{

    public float dt = 0.1f; // Time interval for movement
    private float timer = 0f;
    private Vector3 startPosition;
    private Vector3 targetPosition;
    private int idx = 0;

    // Step counts, not seconds: idx advances once per dt, so these were sized
    // for dt = 0.3 and are scaled by 3 now that every obstacle runs at 0.1.
    // The obstacle therefore waits the same 4.5 s and covers the same distance
    // as before, in smaller steps.
    private int shift = 45;
    private int lastIdx = 240;

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
        startPosition = transform.position;
        targetPosition = transform.position;
    }

    int get_angle(int idx){
        if(idx < shift+45){
            return 120;
        }
        else if(idx < shift+60){
            return 90;
        }
        else if(idx < shift+84){
            return 80;
        }
        else if(idx < shift+120){
            return 360;
        }
        else{
            return 90;
        }
    }

    float get_velocity(int idx){
        if(idx > lastIdx | idx < shift){
            return 0f;
        }
        else {
            return Random.Range(0.10f, 0.15f);
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
        // `while` rather than `if`, and `timer -= dt` rather than `timer = 0f`:
        // the timer is only inspected once per frame, so the deadline is always
        // crossed some way into a frame. Zeroing it threw that remainder away,
        // which stretched a step from dt to ceil(dt/frameTime)*frameTime, and
        // taking the next target from the interpolated position abandoned
        // whatever fraction of the step had not been covered yet. Together
        // those made the obstacle's speed a function of the frame rate: 0.05
        // m/s at 15 FPS against 0.10 m/s at 4500 FPS, for the same 0.10 m/s
        // written here. Carrying the remainder and finishing the step first
        // makes it dt per step at any frame rate.
        timer += Time.deltaTime;
        while (timer >= dt)
        {
            timer -= dt;
            transform.position = targetPosition;
            startPosition = targetPosition;
            targetPosition = startPosition + next_step();
        }
        // Interpolate the position smoothly between the start and target positions
        transform.position = Vector3.Lerp(startPosition, targetPosition, timer / dt);
    }
}
