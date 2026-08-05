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
    private int shift = 15;

    // The offsets below (15, 20, 28, 40) and the max-idx cutoff (80) used in
    // get_angle/get_velocity are step counts, not seconds. They were tuned
    // assuming dt = 0.3s per step, so at that dt they mark real times of
    // 4.5s, 6s, 8.4s, 12s, and 24s. If dt changes, the same step counts land
    // at different real times, so they're rescaled in Start() by
    // tunedDt / dt to keep the same real-world schedule at any dt.
    private const float tunedDt = 0.3f;
    private int seg1 = 15;
    private int seg2 = 20;
    private int seg3 = 28;
    private int seg4 = 40;
    private int maxIdx = 80;

    // The random speed is only redrawn every second completed movement step
    // (velocity > 0 steps) and reused for the intermediate step, so
    // currentSpeed caches the last draw and velocityStepCount tracks parity.
    private float currentSpeed = 0f;
    private int velocityStepCount = 0;
    // Start is called before the first frame update
    void Start()
    {
        Random.InitState(42);
        startPosition = transform.position;
        targetPosition = transform.position;

        // Rescale the step-count thresholds so they keep marking the same
        // real-world times regardless of dt.
        float scale = tunedDt / dt;
        shift = Mathf.RoundToInt(shift * scale);
        seg1 = Mathf.RoundToInt(seg1 * scale);
        seg2 = Mathf.RoundToInt(seg2 * scale);
        seg3 = Mathf.RoundToInt(seg3 * scale);
        seg4 = Mathf.RoundToInt(seg4 * scale);
        maxIdx = Mathf.RoundToInt(maxIdx * scale);
    }
    int get_angle(int idx){
        if(idx < shift+seg1){
            return 120;
        }
        else if(idx < shift+seg2){
            return 90;
        }
        else if(idx < shift+seg3){
            return 80;
        }
        else if(idx < shift+seg4){
            return 360;
        }
        else{
            return 90;
        }
    }
    float get_velocity(int idx){
        if(idx > maxIdx | idx < shift){
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
    // Update is called once per frame
    void Update()
    {
        timer += Time.deltaTime;
        while (timer >= dt){
            timer -= dt;
            // X python = Unity Z
            // Z python = Unity Y 
            // Y python = Unity -X
            transform.position = targetPosition;
            Vector3 pos = transform.position;
            startPosition = transform.position;
            
            
            float velocity = get_velocity(idx);
            float angle = get_angle(idx) * Mathf.Deg2Rad;
            float new_z = pos.z + velocity * dt * Mathf.Cos(angle);
            float new_x = pos.x + velocity * dt * Mathf.Sin(angle);
            pos.z = new_z;
            pos.x = new_x;
            targetPosition = pos;
            idx += 1;
        }
        // Interpolate the position smoothly between the start and target positions
        float t = timer / dt;
        transform.position = Vector3.Lerp(startPosition, targetPosition, t);
    }
}
