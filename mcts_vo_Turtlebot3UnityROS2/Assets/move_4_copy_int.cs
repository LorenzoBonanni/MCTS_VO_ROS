using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class move_4_copy_int : MonoBehaviour
{

    public float dt = 0.3f; // Time interval for movement
    private float timer = 0f;
    // See move_1.cs: the period is set so the fastest step comes out at the
    // scene-wide peak speed, while dt stays the value used in the position
    // arithmetic. As in move_copy_int.cs this obstacle already ran at that
    // peak, so at scale 1 nothing about it changes; at scale k it reaches its
    // goal k times sooner along the same route.
    private float stepPeriod;
    // The flat speed used below, named so Start and Update cannot disagree.
    private const float MaxVelocity = 0.1f;
    private Vector3 startPosition;
    private Vector3 targetPosition;
    private static Vector3 goal1 = new Vector3(-0.53f, 0.1f, -1.14f);
    private static Vector3 goal2 = new Vector3(1.826f, 0.1f, -1.139f);
    private Vector3 goal = goal1;
    private int idx = 0;
    private float randNum = 0.0f;

    // Start is called before the first frame update
    void Start()
    {
        Random.InitState(42);
        stepPeriod = ObstacleSpeed.PeriodFor(MaxVelocity * dt);
        startPosition = transform.position;
        targetPosition = transform.position;     
    }

    // Update is called once per frame
    void Update()
    {
        // See move_1.cs: `while` + `timer -= dt` + finishing the step before the
        // next one is computed, so that a step takes dt at any frame rate. With
        // `if` + `timer = 0f` the obstacle ran at 0.05 m/s windowed and 0.10
        // m/s headless, for the 0.1 m/s written below.
        timer += Time.deltaTime;
        while (timer >= stepPeriod) {
            timer -= stepPeriod;
            transform.position = targetPosition;
            startPosition = targetPosition;

            if (Vector3.Distance(transform.position, goal) < 0.1f){
                if (goal == goal1){
                    goal = goal2;
                }
                else{
                    goal = goal1;
                }
            }

            if (idx % 10 == 0) {
                randNum = Random.Range(-0.5f, 0.5f) * 2.5f;
            }

            Vector3 pos = transform.position;
            float velocity = MaxVelocity;

            Vector3 direction = (goal - pos).normalized;
            float goal_angle = Mathf.Atan2(direction.x, direction.z);
            float angle = goal_angle + randNum;
            float new_z = pos.z + velocity * dt * Mathf.Cos(angle);
            float new_x = pos.x + velocity * dt * Mathf.Sin(angle);
            pos.z = new_z;
            pos.x = new_x;
            targetPosition = pos;
            idx+=1;
        }
        // Interpolate the position smoothly between the start and target positions
        transform.position = Vector3.Lerp(startPosition, targetPosition, timer / stepPeriod);
    }
}
