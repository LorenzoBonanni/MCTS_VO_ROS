using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class move_4_copy_int : MonoBehaviour
{

    public float dt = 0.3f; // Time interval for movement
    private float timer = 0f;
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
        startPosition = transform.position;
        targetPosition = transform.position;     
    }

    // Update is called once per frame
    void Update()
    {
        timer += Time.deltaTime;
        if (timer >= dt){
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

            // Debug.Log("Current idx: " + idx);
            timer = 0f;
            Vector3 pos = transform.position;
            startPosition = transform.position;
            
            float velocity = 0.1f;
            
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
        else {
            // Interpolate the position smoothly between the start and target positions
            float t = timer / dt;
            transform.position = Vector3.Lerp(startPosition, targetPosition, t);
        }
    }
}
