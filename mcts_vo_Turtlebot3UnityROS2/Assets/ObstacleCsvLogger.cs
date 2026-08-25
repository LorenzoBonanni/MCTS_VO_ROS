using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using UnityEngine;
using UnityEngine.SceneManagement;

// Temporary instrumentation for verifying the SIN_COMPLEX redesign (obstacle
// crossing timing, peak speed, arena/static clearance). Off by default via
// each caller's own `enableCsvLogging` flag - remove once the redesign is
// verified.
public static class ObstacleCsvLogger
{
    private const string EnvVarName = "MCTSVO_OBS_LOG";
    private const string DefaultRelativeDir = "obstacle_log";
    private const string Header = "step,tempo,nome_oggetto,X,Z,velocita_istantanea,velocita_max_finora";

    private static readonly Dictionary<string, StreamWriter> _writers = new Dictionary<string, StreamWriter>();
    private static readonly string _runStamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");
    private static string _resolvedDir;
    private static string _sceneName;

    public static void LogRow(bool enabled, string objectName, int step, float tempo,
                               float x, float z, float velocitaIstantanea, float velocitaMaxFinora)
    {
        if (!enabled)
            return;

        StreamWriter writer = GetWriter(objectName);
        if (writer == null)
            return;

        writer.WriteLine(string.Format(CultureInfo.InvariantCulture,
            "{0},{1:F4},{2},{3:F4},{4:F4},{5:F4},{6:F4}",
            step, tempo, objectName, x, z, velocitaIstantanea, velocitaMaxFinora));
        writer.Flush();
    }

    private static StreamWriter GetWriter(string objectName)
    {
        if (_writers.TryGetValue(objectName, out var existing))
            return existing;

        try
        {
            string dir = ResolveLogDirectory();
            Directory.CreateDirectory(dir);

            string safeName = string.Join("_", objectName.Split(Path.GetInvalidFileNameChars()));
            string sceneName = ResolveSceneName();
            string path = Path.Combine(dir, $"{sceneName}_{safeName}_{_runStamp}.csv");

            var writer = new StreamWriter(path, append: true);
            writer.WriteLine(Header);
            writer.Flush();
            _writers[objectName] = writer;
            return writer;
        }
        catch (Exception e)
        {
            Debug.LogWarning($"ObstacleCsvLogger: failed to open log file for '{objectName}': {e.Message}");
            _writers[objectName] = null;
            return null;
        }
    }

    private static string ResolveLogDirectory()
    {
        if (_resolvedDir != null)
            return _resolvedDir;

        string envDir = Environment.GetEnvironmentVariable(EnvVarName);
        _resolvedDir = !string.IsNullOrEmpty(envDir) ? envDir : DefaultRelativeDir;
        return _resolvedDir;
    }

    private static string ResolveSceneName()
    {
        if (_sceneName != null)
            return _sceneName;

        string name = SceneManager.GetActiveScene().name;
        _sceneName = string.IsNullOrEmpty(name) ? "UnknownScene" : name;
        return _sceneName;
    }
}
