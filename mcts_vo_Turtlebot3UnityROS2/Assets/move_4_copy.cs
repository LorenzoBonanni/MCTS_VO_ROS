using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class move_4_copy : MonoBehaviour
{

    public float dt = 0.1f; // Time interval for movement
    private float timer = 0f;
    // See move_1.cs: the period is set so the fastest step comes out at the
    // scene-wide peak speed, while dt stays the value used in the position
    // arithmetic, so the path is unchanged. That matters here in particular -
    // dt also sets the sinusoid's phase advance per step, so dividing the
    // lateral term by dt instead would have changed the spatial waveform.
    private float stepPeriod;
    private Vector3 startPosition;
    private Vector3 targetPosition;
    // Step count, not seconds. Sized for dt = 0.2 and doubled now that every
    // obstacle runs at 0.1, so the traverse covers the same ground in half-size
    // steps. How long it takes is no longer 240 * dt: a step lasts stepPeriod,
    // so at scale 1 the full out-and-back is 240 * 0.5099 = 122 s, and at scale
    // k it is that over k.
    private int period = 240;
    private int idx = 0;

    // PARAMETERS SINUSOIDAL
    public float amplitude = 0.01f; // Amplitude of the sinusoidal wave
    public  float frequency = 1f; // Frequency of the sinusoidal wave
    private float forwardSpeed = 0.1f; // Forward speed
    private int mulForwardSpeed = 1;
    // Bounds of the per-step speed draw below, named so the step-period
    // calculation in Start cannot drift away from the Random.Range it bounds.
    private const float MinForwardSpeed = 0.05f;
    private const float MaxForwardSpeed = 0.1f;

    // Start is called before the first frame update
    void Start()
    {
        Random.InitState(42);
        // A step is (forwardSpeed * dt) along x and the sinusoid offset along
        // z. The offset carries no dt, so at amplitude 0.05 and dt 0.1 it is
        // five times the forward term and dominates the peak.
        stepPeriod = ObstacleSpeed.PeriodFor(
            new Vector2(MaxForwardSpeed * dt, amplitude).magnitude);
        startPosition = transform.position;
        targetPosition = transform.position;     
    }

    // Update is called once per frame
    void Update()
    {
        // See move_1.cs: `while` + `timer -= dt` + finishing the step before the
        // next one is computed, so a step takes dt at any frame rate. The speed
        // draw also moves inside the loop - it used to run once per frame, so
        // it was re-rolled 15 times a second windowed and 4500 times a second
        // headless, and whichever value happened to be live when the timer
        // tripped was the one used.
        timer += Time.deltaTime;
        while (timer >= stepPeriod){
            timer -= stepPeriod;
            transform.position = targetPosition;
            startPosition = targetPosition;

            float speed = Random.Range(MinForwardSpeed, MaxForwardSpeed);
            forwardSpeed = mulForwardSpeed * speed;
            // X python = Unity Z
            // Z python = Unity Y 
            // Y python = Unity -X
            Vector3 pos = transform.position;

            // Trefoil knot trajectory calculation
            // float t = idx * dt * 0.01f; // Reduce the speed by scaling down t
            // int multiplier = 1;
            // float omega = 0.1f;
            // float scale_x = 0.1f;
            // float scale_y = 0.1f;
            // float trefoilX = multiplier * scale_x * (Mathf.Sin(omega * idx) + 2 * Mathf.Sin(2 * omega * idx));
            // float trefoilZ = multiplier * scale_y * (Mathf.Cos(omega * idx) - 2 * Mathf.Cos(2 * omega * idx));
            // pos.x = startPosition.x + trefoilX;
            // pos.z = startPosition.z + trefoilZ;
            // float velocity = get_velocity(idx);
            // float angle = get_angle(idx) * Mathf.Deg2Rad;
            // float new_z = pos.z + velocity * dt * Mathf.Cos(angle);
            // float new_x = pos.x + velocity * dt * Mathf.Sin(angle);


            float offset = Mathf.Sin(idx * frequency * dt) * amplitude;
            pos.x += forwardSpeed * dt;
            pos.z += offset;

            targetPosition = pos;
            idx += 1;
            if (idx == period){
                mulForwardSpeed *= -1;
                idx = 0;
            }
        }
        // Interpolate the position smoothly between the start and target positions
        transform.position = Vector3.Lerp(startPosition, targetPosition, timer / stepPeriod);
    }
}
