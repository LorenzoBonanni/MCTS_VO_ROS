using UnityEditor;
using UnityEngine;
using System.IO;

public class BuildScript
{
    static string[] scenes =
    {
        "Assets/Scenes/INT_EASY.unity",
        "Assets/Scenes/INT_COMPLEX.unity",
        "Assets/Scenes/INT_COMPLEX_SPEED.unity",
        "Assets/Scenes/SIN_EASY.unity",
        "Assets/Scenes/SIN_COMPLEX.unity",
        "Assets/Scenes/SIN_COMPLEX_SPEED.unity"
    };


    public static void BuildLinux()
    {
        string outputRoot = "env_build";

        foreach (string scene in scenes)
        {
            string name = Path.GetFileNameWithoutExtension(scene);

            string output = $"{outputRoot}/{name}/{name}.x86_64";

            BuildPipeline.BuildPlayer(
                new BuildPlayerOptions
                {
                    scenes = new[] { scene },
                    locationPathName = output,
                    target = BuildTarget.StandaloneLinux64,
                    options = BuildOptions.None
                }
            );

            FixDDSC(outputRoot + "/" + name);
        }

        Debug.Log("========== ALL SCENES BUILT SUCCESSFULLY ==========");
    }


    static void FixDDSC(string folder)
    {
        string pluginDir =
            Path.Combine(folder, folder.Substring(folder.LastIndexOf('/') + 1) + "_Data/Plugins");


        string src =
            Path.Combine(
                Application.dataPath,
                "Plugins/Linux/x86_64/libddsc.so"
            );


        string dst =
            Path.Combine(pluginDir, "libddsc.so");


        if (!File.Exists(dst))
        {
            File.Copy(src, dst, true);
        }


        string link =
            Path.Combine(pluginDir, "libddsc.so.0");


        if (!File.Exists(link))
        {
            System.Diagnostics.Process.Start(
                "ln",
                $"-s libddsc.so {link}"
            ).WaitForExit();
        }
    }
}
