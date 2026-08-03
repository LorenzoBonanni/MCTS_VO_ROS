using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using UnityEngine;

// Ground-truth obstacle logger, for measuring whether obstacle motion depends on
// the frame rate.
//
// The movement scripts do `timer += Time.deltaTime; if (timer >= dt) { timer = 0f;
// ...move by velocity*dt... }`. Resetting the timer to zero throws away the
// overshoot, so a step is taken every ceil(dt / frameTime) * frameTime seconds
// rather than every dt. If that is real, obstacle speed changes with the frame
// rate, and a headless run is not the same experiment as a windowed one.
//
// This cannot be measured from the recorded LIDAR data: real motion is ~1 cm per
// control step while the RANSAC centre wobbles by ~4 cm, so the noise is several
// times the signal. Hence ground truth, straight from the transforms.
//
// Self-attaching, so no scene has to be modified to carry it. Writes a CSV and
// quits on its own:
//
//   env.x86_64 -probeOut /tmp/probe.csv -probeSeconds 40
//
public class ObstacleProbe : MonoBehaviour
{
    private readonly List<Transform> targets = new List<Transform>();
    private StringBuilder csv;
    private string outPath;
    private float stopAfter;

    [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
    private static void Bootstrap()
    {
        if (Arg("-probeOut", null) == null)
        {
            return;                      // not a probe run, stay out of the way
        }
        GameObject go = new GameObject("ObstacleProbe");
        go.AddComponent<ObstacleProbe>();
        DontDestroyOnLoad(go);
    }

    private static string Arg(string name, string fallback)
    {
        string[] args = System.Environment.GetCommandLineArgs();
        for (int i = 0; i < args.Length - 1; i++)
        {
            if (args[i] == name)
            {
                return args[i + 1];
            }
        }
        return fallback;
    }

    private void Start()
    {
        outPath = Arg("-probeOut", "/tmp/obstacle_probe.csv");
        float.TryParse(Arg("-probeSeconds", "40"), NumberStyles.Float,
                       CultureInfo.InvariantCulture, out stopAfter);
        if (stopAfter <= 0f)
        {
            stopAfter = 40f;
        }

        foreach (GameObject g in FindObjectsOfType<GameObject>())
        {
            if (g.name.Contains("Obstacle") && g.activeInHierarchy)
            {
                targets.Add(g.transform);
            }
        }
        targets.Sort((a, b) => string.CompareOrdinal(a.name, b.name));

        csv = new StringBuilder();
        csv.Append("time,frame,name,x,z\n");
        Debug.Log($"[ObstacleProbe] logging {targets.Count} obstacles -> {outPath}");
    }

    private void Update()
    {
        // Sampled every frame: the quantity of interest is exactly how motion
        // relates to frames, so sampling on any other clock would hide it.
        foreach (Transform t in targets)
        {
            csv.Append(Time.time.ToString("F5", CultureInfo.InvariantCulture)).Append(',')
               .Append(Time.frameCount).Append(',')
               .Append(t.name).Append(',')
               .Append(t.position.x.ToString("F5", CultureInfo.InvariantCulture)).Append(',')
               .Append(t.position.z.ToString("F5", CultureInfo.InvariantCulture)).Append('\n');
        }

        if (Time.time >= stopAfter)
        {
            Flush();
            Application.Quit();
        }
    }

    private void OnApplicationQuit()
    {
        Flush();
    }

    private void Flush()
    {
        if (csv == null)
        {
            return;
        }
        File.WriteAllText(outPath, csv.ToString());
        Debug.Log($"[ObstacleProbe] wrote {outPath}");
        csv = null;
    }
}
