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

    // Start is called before the first frame update
    void Start()
    {
        Random.InitState(42);
        startPosition = transform.position;
        targetPosition = transform.position;     
    }

    int get_angle(int idx){
        if (idx < 50){
            return -90;
        }
        else {
            return -80;
        }
    }

    float get_velocity(int idx){
        if(idx < 15 | idx > 100){
            return 0f;
        }
        else {
            return Random.Range(0.10f, 0.15f);
        }
    }

    // Update is called once per frame
    void Update()
    {
        timer += Time.deltaTime;

        if (timer >= dt){
            // Debug.Log("Index: " + idx);
            timer = 0f;
            // X python = Unity Z
            // Z python = Unity Y 
            // Y python = Unity -X
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
        else
        {
            // Interpolate the position smoothly between the start and target positions
            float t = timer / dt;
            transform.position = Vector3.Lerp(startPosition, targetPosition, t);
        }
   
    }
}
