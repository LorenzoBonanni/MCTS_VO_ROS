using System.Globalization;
using UnityEngine;

// One knob for how fast every obstacle in either scene moves, read once from the
// player command line:
//
//   env.x86_64 -obsSpeedScale 1.5
//
// Every obstacle peaks at exactly TargetMaxSpeed * scale, in both scenes, so the
// velocity obstacles can be sized from that one number.
//
// It works by setting the STEP PERIOD, not by touching any velocity. Every
// movement script is written as "each step, move by <some displacement>", with
// the waypoints, angle schedules, sinusoid phases and step counts all indexed by
// step number rather than by seconds. Scaling the displacements would therefore
// scale the path geometrically - the obstacle would sweep a larger region rather
// than the same region faster. Taking the same step over a different interval
// has none of that: the position arithmetic is untouched, so the sequence of
// positions is bit identical and the path is exactly the one the scene was
// designed around. Only the clock it is walked on changes.
//
// That also preserves the shape of every speed distribution rather than
// truncating it: move_copy draws Random.Range(0.0, 0.1) per step and stays
// uniform, rescaled, at any setting. A cap on the maximum would instead pile
// probability mass at the cap. Because the draws happen per step and not per
// second, the random sequence is unchanged too, so a run at scale 1.5 is the
// scale 1.0 run replayed faster rather than a different episode.
//
// Callers pass the largest distance one of their steps can cover, and get back
// how long that step must last for the peak speed to come out at
// TargetMaxSpeed * scale. This is what normalises the two scenes against each
// other, and they needed it: measured with ObstacleProbe before it existed, the
// intention obstacles peaked at 0.1001 m/s but the sinusoidal ones at 0.5078.
//
// The sinusoidal scene was the odd one out because move_copy and move_4_copy
// apply their lateral term as
//
//     pos.x += forwardSpeed * dt;   // a velocity, integrated over dt
//     pos.z += offset;              // a displacement, with no dt at all
//
// so the lateral speed was amplitude/dt = 0.05/0.1 = 0.5 m/s rather than the
// 0.1 m/s the forward term runs at. That is also why it drifted historically:
// the term scales with 1/dt, so when the scenes moved from dt = 0.2 to dt = 0.1
// (commit 20b7aac) the lateral speed doubled, silently, with amplitude
// unchanged at 0.05.
//
// Normalising the period rather than dividing the lateral term by dt is what
// keeps the trajectory identical: the sinusoid keeps its ~1.8 m sweep and its
// wavelength, and only takes 5.099x longer to walk it. Passing
// -obsSpeedScale 5.099 reproduces the old sinusoidal motion exactly.
public static class ObstacleSpeed
{
    // Peak speed of every obstacle at scale 1, m/s. loopHandler_copy.py sizes
    // --max-obs-vel from this same number; the two must not drift apart.
    public const float TargetMaxSpeed = 0.1f;

    private const float DefaultScale = 1.0f;

    private static float cached = -1f;

    // Resolved lazily rather than in a static constructor: the value is only
    // ever needed from Start(), by which point the command line is available,
    // and this keeps the parse and its log line off the class-load path.
    public static float Scale
    {
        get
        {
            if (cached < 0f)
            {
                cached = Parse();
                Debug.Log($"[ObstacleSpeed] scale={cached:F3}, " +
                          $"peak {TargetMaxSpeed * cached:F4} m/s");
            }
            return cached;
        }
    }

    // How long one movement step must last for an obstacle whose largest
    // possible step covers maxStepDistance metres to peak at
    // TargetMaxSpeed * Scale.
    public static float PeriodFor(float maxStepDistance)
    {
        return maxStepDistance / (TargetMaxSpeed * Scale);
    }

    private static float Parse()
    {
        string raw = Arg("-obsSpeedScale", null);
        if (raw == null)
        {
            return DefaultScale;
        }

        // Invariant culture explicitly: under a locale whose decimal separator
        // is a comma, "1.5" parses as 15 and the obstacles run ten times too
        // fast, silently.
        float value;
        if (!float.TryParse(raw, NumberStyles.Float, CultureInfo.InvariantCulture,
                            out value) || value <= 0f)
        {
            Debug.LogError($"[ObstacleSpeed] ignoring -obsSpeedScale '{raw}': " +
                           $"not a positive number, using {DefaultScale}");
            return DefaultScale;
        }
        return value;
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
}
