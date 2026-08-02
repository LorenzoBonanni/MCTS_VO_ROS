using System;
using UnityEditor;
using UnityEditor.Build.Reporting;
using UnityEngine;

// Headless build entry point, so the simulation environments can be rebuilt from
// the command line instead of the Editor UI:
//
//   Unity -batchmode -nographics -quit -projectPath <project> \
//         -executeMethod BuildScript.BuildLinux \
//         -buildScene Assets/Scenes/turtlebot3_COPY.unity \
//         -buildOutput ../env_build/sin_env/env.x86_64
//
// Scene and output are arguments rather than taken from Build Settings, which
// only ever lists the sinusoidal scene: the intention environment is the same
// project built from turtlebot3_COPY_INT.unity.
public class BuildScript
{
    private static string Arg(string name, string fallback)
    {
        string[] args = Environment.GetCommandLineArgs();
        for (int i = 0; i < args.Length - 1; i++)
        {
            if (args[i] == name)
            {
                return args[i + 1];
            }
        }
        return fallback;
    }

    public static void BuildLinux()
    {
        string scene = Arg("-buildScene", "Assets/Scenes/turtlebot3_COPY.unity");
        string output = Arg("-buildOutput", "../env_build/sin_env/env.x86_64");

        // Mono, matching the existing builds: their env_Data/Managed/*.dll are
        // managed assemblies, which an IL2CPP build would not produce.
        PlayerSettings.SetScriptingBackend(BuildTargetGroup.Standalone,
                                           ScriptingImplementation.Mono2x);

        BuildPlayerOptions options = new BuildPlayerOptions
        {
            scenes = new[] { scene },
            locationPathName = output,
            target = BuildTarget.StandaloneLinux64,
            options = BuildOptions.None,
        };

        Debug.Log($"[BuildScript] building {scene} -> {output}");
        BuildReport report = BuildPipeline.BuildPlayer(options);
        BuildSummary summary = report.summary;
        Debug.Log($"[BuildScript] result={summary.result} " +
                  $"size={summary.totalSize} time={summary.totalTime} " +
                  $"errors={summary.totalErrors} warnings={summary.totalWarnings}");

        EditorApplication.Exit(summary.result == BuildResult.Succeeded ? 0 : 1);
    }
}
